package main

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/ironpark/acp-go/acp1"

	"github.com/ironpark/deepseek-acp/internal/deepseek"
)

// compactHeadroom is room kept free in the context window for the next
// response, as in the harness's compaction.
const compactHeadroom = 65536

const summaryPrefix = "<conversation-summary>"

const compactPrompt = `The conversation is about to be compacted to free the context window. Write a summary that lets you continue the work without the earlier messages. Do not call any tools.

Include:
- The user's requests and goals, in their own words where it matters.
- Key decisions and constraints.
- Files read or changed, with their paths and the details that still matter.
- Commands run and their results, errors and how they were fixed.
- The current state of the work and the next steps.

Be thorough about details needed to continue; leave out everything else.`

// compactThreshold is the context size at which a model's conversation is
// compacted, or 0 if it never is.
func (a *deepseekAgent) compactThreshold(info modelInfo) int {
	if a.cfg.compactRatio == 0 {
		return 0
	}
	return max(min(int(a.cfg.compactRatio*float64(info.contextWindow)), info.contextWindow-compactHeadroom), 1)
}

// maybeCompact compacts the conversation when it fills the context window
// past the threshold.
func (a *deepseekAgent) maybeCompact(ctx context.Context, stream *acp1.SessionStream, sess *session, req deepseek.Request, info modelInfo, midTurn bool) error {
	threshold := a.compactThreshold(info)
	if threshold == 0 || sess.contextTokens() < threshold {
		return nil
	}
	_, err := a.compact(ctx, stream, sess, req, info, midTurn)
	return err
}

// compact summarizes the conversation and replaces the history with the
// summary, and reports whether it did. It must be called between exchanges,
// when every tool call has its result. midTurn says the model is in the
// middle of a task it should go on with.
//
// The summary request is the turn's request with one more instruction, so it
// shares the prompt prefix and hits DeepSeek's cache. A failed compaction is
// shown and logged but returns no error; only a cancelled turn or a broken
// connection does.
func (a *deepseekAgent) compact(ctx context.Context, stream *acp1.SessionStream, sess *session, req deepseek.Request, info modelInfo, midTurn bool) (bool, error) {
	id := acp1.GenerateToolCallID()
	if err := stream.StartToolCall(ctx, id, "Compacting conversation", acp1.ToolKindThink); err != nil {
		return false, err
	}
	summary, err := a.summarize(ctx, sess, req)
	if ctx.Err() != nil {
		return false, ctx.Err()
	}
	if err != nil {
		a.logger.Warn("compaction failed", "session", stream.SessionID(), "error", err)
		return false, stream.FailToolCall(ctx, id, acp1.WithToolContent(acp1.ToolText("Compaction failed: "+err.Error())))
	}

	var note strings.Builder
	fmt.Fprintf(&note, "%s\nThis session continues from an earlier conversation that ran out of context. Summary of it:\n\n%s\n", summaryPrefix, summary)
	sess.mu.Lock()
	todos := sess.todos
	sess.mu.Unlock()
	if len(todos) > 0 {
		note.WriteString("\nThe plan, as kept with todo_write:\n")
		for _, t := range todos {
			fmt.Fprintf(&note, "- [%s] %s\n", t.Status, t.Content)
		}
	}
	fmt.Fprintf(&note, "\n%s\n</conversation-summary>", sess.runtimeContext())
	if midTurn {
		note.WriteString("\nContinue the current task from where you left off.")
	}
	// Until the next call reports it, the summary's size stands in for the
	// conversation's.
	estimate := len(note.String()) / 3
	sess.replaceHistory([]deepseek.Message{{Role: "user", Content: []deepseek.Block{deepseek.TextBlock(note.String())}}}, estimate)
	if err := stream.SendUsage(ctx, uint64(estimate), uint64(info.contextWindow), nil); err != nil {
		return false, err
	}
	return true, stream.CompleteToolCall(ctx, id, acp1.WithToolContent(acp1.ToolText(summary)))
}

// summarize asks the model for a summary of the conversation.
func (a *deepseekAgent) summarize(ctx context.Context, sess *session, req deepseek.Request) (string, error) {
	instruction := deepseek.Message{Role: "user", Content: []deepseek.Block{deepseek.TextBlock(compactPrompt)}}
	req.Messages = sess.messages()
	if n := len(req.Messages); n > 0 && req.Messages[n-1].Role == "user" {
		last := req.Messages[n-1]
		req.Messages[n-1] = deepseek.Message{Role: "user", Content: append(last.Content[:len(last.Content):len(last.Content)], instruction.Content...)}
	} else {
		req.Messages = append(req.Messages, instruction)
	}
	reply, err := a.llm.Stream(ctx, req, deepseek.Handler{})
	if reply != nil {
		sess.recordUsage(reply.Usage)
	}
	if err != nil {
		return "", err
	}
	var summary strings.Builder
	for _, b := range reply.Message.Content {
		if b.Type == "text" {
			summary.WriteString(b.Text)
		}
	}
	if strings.TrimSpace(summary.String()) == "" {
		return "", errors.New("the model returned no summary")
	}
	return summary.String(), nil
}
