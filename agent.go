package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	acp "github.com/ironpark/acp-go"
	"github.com/ironpark/acp-go/acp1"
	"github.com/ironpark/acp-go/schema/optional"

	"github.com/ironpark/deepseek-acp/internal/deepseek"
)

// newAgent returns the constructor of the agent for a connection.
func newAgent(cfg *config, store acp1.SessionStore[*session], llm *deepseek.Client, logger *slog.Logger) func(*acp1.AgentSideConnection) acp1.Agent {
	manager := acp1.NewSessionManager(store,
		func(_ context.Context, params *acp1.NewSessionRequest) (acp1.SessionID, *session, error) {
			return acp1.GenerateSessionID(), newSession(params.Cwd, cfg), nil
		},
		// A turn always leaves the history whole, even when cancelled, so the
		// session is saved whenever one ends.
		acp1.WithAutoSave(func(id acp1.SessionID, err error) {
			logger.Error("save session", "session", id, "error", err)
		}))
	return func(c *acp1.AgentSideConnection) acp1.Agent {
		return &deepseekAgent{SessionManager: manager, client: c, llm: llm, cfg: cfg, logger: logger}
	}
}

// deepseekAgent embeds a SessionManager for the session lifecycle, turn
// cancellation and saving, and runs each prompt as a DeepSeek tool loop.
type deepseekAgent struct {
	*acp1.SessionManager[*session]
	client *acp1.AgentSideConnection
	llm    *deepseek.Client
	cfg    *config
	logger *slog.Logger
}

func (a *deepseekAgent) Initialize(context.Context, *acp1.InitializeRequest) (*acp1.InitializeResponse, error) {
	agentCaps := acp1.CapabilitiesOf(a)
	agentCaps.PromptCapabilities = &acp1.PromptCapabilities{Image: new(true), EmbeddedContext: new(true)}
	return &acp1.InitializeResponse{
		ProtocolVersion:   acp1.ProtocolVersion,
		AgentCapabilities: agentCaps,
		AgentInfo:         &acp1.Implementation{Name: "deepseek-acp", Title: new("DeepSeek"), Version: version},
	}, nil
}

// LoadSession replays the conversation to the client: the user's messages,
// the model's thinking and answers, and its tool calls.
func (a *deepseekAgent) LoadSession(ctx context.Context, params *acp1.LoadSessionRequest) (*acp1.LoadSessionResponse, error) {
	sess, err := a.Lookup(ctx, params.SessionID)
	if err != nil {
		return nil, err
	}
	if sess.setCwd(params.Cwd) {
		a.save(ctx, params.SessionID, sess)
	}
	stream := acp1.NewSessionStream(a.client, params.SessionID)
	if err := a.replay(ctx, stream, sess); err != nil {
		return nil, err
	}
	if err := a.sendContextUsage(ctx, stream, sess); err != nil {
		return nil, err
	}
	if err := stream.SendCommands(ctx, sess.AvailableCommands()); err != nil {
		return nil, err
	}
	return &acp1.LoadSessionResponse{Modes: sess.SessionModes(), ConfigOptions: sess.SessionConfigOptions()}, nil
}

// ResumeSession resumes the session through the manager, which also
// advertises the slash commands, and restores the client's context usage
// display.
func (a *deepseekAgent) ResumeSession(ctx context.Context, params *acp1.ResumeSessionRequest) (*acp1.ResumeSessionResponse, error) {
	resp, err := a.SessionManager.ResumeSession(ctx, params)
	if err != nil {
		return nil, err
	}
	if sess, err := a.Lookup(ctx, params.SessionID); err == nil {
		if err := a.sendContextUsage(ctx, acp1.NewSessionStream(a.client, params.SessionID), sess); err != nil {
			return nil, err
		}
	}
	return resp, nil
}

// sendContextUsage tells the client how full the context window is, as of
// the session's last model call.
func (a *deepseekAgent) sendContextUsage(ctx context.Context, stream *acp1.SessionStream, sess *session) error {
	used := sess.contextTokens()
	if used == 0 {
		return nil
	}
	model, _ := sess.settings()
	return stream.SendUsage(ctx, uint64(used), uint64(lookupModel(model).contextWindow), nil)
}

func (a *deepseekAgent) replay(ctx context.Context, stream *acp1.SessionStream, sess *session) error {
	results := map[string]deepseek.Block{}
	history := sess.messages()
	for _, m := range history {
		for _, b := range m.Content {
			if b.Type == "tool_result" {
				results[b.ToolUseID] = b
			}
		}
	}
	for _, m := range history {
		for _, b := range m.Content {
			var err error
			switch {
			case m.Role == "user" && b.Type == "text" && (strings.HasPrefix(b.Text, summaryPrefix) || strings.HasPrefix(b.Text, goalRoundPrefix)):
				err = stream.SendThought(ctx, b.Text)
			case m.Role == "user" && b.Type == "text" && !strings.HasPrefix(b.Text, runtimeContextPrefix):
				err = stream.SendUserMessage(ctx, b.Text)
			case m.Role == "assistant" && b.Type == "thinking" && b.Thinking != nil && *b.Thinking != "":
				err = stream.SendThought(ctx, *b.Thinking)
			case m.Role == "assistant" && b.Type == "text":
				err = stream.SendText(ctx, b.Text)
			case m.Role == "assistant" && b.Type == "tool_use":
				err = a.replayToolCall(ctx, stream, sess, b, results[b.ID])
			}
			if err != nil {
				return err
			}
		}
	}
	sess.mu.Lock()
	todos := sess.todos
	sess.mu.Unlock()
	if len(todos) > 0 {
		return stream.SendPlan(ctx, todos)
	}
	return nil
}

func (a *deepseekAgent) replayToolCall(ctx context.Context, stream *acp1.SessionStream, sess *session, call, result deepseek.Block) error {
	title, kind := call.Name, acp1.ToolKindOther
	var locations []acp1.ToolCallLocation
	if act, err := a.parseTool(sess, call); err == nil {
		if act.hidden {
			return nil // the plan is sent once, at the end
		}
		title, kind, locations = act.title, act.kind, act.locations
	}
	id := acp1.GenerateToolCallID()
	if err := stream.StartToolCall(ctx, id, title, kind,
		acp1.WithLocations(locations...), acp1.WithRawInput(call.Input)); err != nil {
		return err
	}
	content := acp1.WithToolContent(replayContent(sess, call, result)...)
	if result.IsError {
		return stream.FailToolCall(ctx, id, content)
	}
	return stream.CompleteToolCall(ctx, id, content)
}

// replayContent rebuilds what a tool call showed from its arguments and
// result: an edit's change as a diff, a new file's content, a command's
// output and a failure's error. The rest showed summaries the history does
// not keep.
func replayContent(sess *session, call, result deepseek.Block) []acp1.ToolCallContent {
	switch call.Name {
	case editTool.Name:
		args, err := decodeArgs[struct {
			FilePath  string `json:"file_path"`
			OldString string `json:"old_string"`
			NewString string `json:"new_string"`
		}](call)
		if err == nil && !result.IsError {
			return []acp1.ToolCallContent{acp1.ToolDiff(sess.resolve(args.FilePath), &args.OldString, args.NewString)}
		}
	case writeTool.Name:
		args, err := decodeArgs[struct {
			FilePath string `json:"file_path"`
			Content  string `json:"content"`
		}](call)
		// An overwritten file's old content is gone, so only a new file
		// shows as a diff.
		if err == nil && strings.HasPrefix(result.Content, "Created ") {
			return []acp1.ToolCallContent{acp1.ToolDiff(sess.resolve(args.FilePath), nil, args.Content)}
		}
	case bashTool.Name:
		if result.Content != "" {
			return []acp1.ToolCallContent{acp1.ToolText("```\n" + strings.TrimRight(result.Content, "\n") + "\n```")}
		}
	}
	if result.IsError && result.Content != "" {
		return []acp1.ToolCallContent{acp1.ToolText(result.Content)}
	}
	return nil
}

// Prompt runs the turn under the embedded manager, whose CancelSession
// cancels the turn's context, which also aborts the request to DeepSeek, and
// which saves the session when the turn ends. The response reports the
// session's token usage.
func (a *deepseekAgent) Prompt(ctx context.Context, params *acp1.PromptRequest) (*acp1.PromptResponse, error) {
	return a.RunTurnResponse(ctx, params.SessionID, func(ctx context.Context, sess *session) (*acp1.PromptResponse, error) {
		reason, err := a.handlePrompt(ctx, params.SessionID, sess, params.Prompt)
		return &acp1.PromptResponse{StopReason: reason, Usage: sess.acpUsage()}, err
	})
}

// handlePrompt runs a slash command, or answers the prompt with the model.
func (a *deepseekAgent) handlePrompt(ctx context.Context, sessionID acp1.SessionID, sess *session, prompt []acp1.ContentBlock) (acp1.StopReason, error) {
	if cmd, input, rest, ok := parseCommand(prompt); ok {
		return cmd.run(a, ctx, &commandCall{
			sessionID: sessionID, sess: sess, stream: acp1.NewSessionStream(a.client, sessionID),
			input: input, rest: rest,
		})
	}
	return a.promptTurn(ctx, sessionID, sess, prompt)
}

// promptTurn answers a prompt from the user, then works on an active goal.
func (a *deepseekAgent) promptTurn(ctx context.Context, sessionID acp1.SessionID, sess *session, prompt []acp1.ContentBlock) (acp1.StopReason, error) {
	if err := a.requireKey(); err != nil {
		return "", err
	}
	model, _ := sess.settings()
	if title, ok := sess.setTitleFrom(acp1.JoinTexts(prompt)); ok {
		// The client shows the title now rather than on its next session/list.
		update := acp1.SessionUpdateSessionInfoUpdate{
			Title: optional.Of(title), UpdatedAt: optional.Of(time.Now().UTC().Format(time.RFC3339Nano)),
		}
		if err := acp1.NewSessionStream(a.client, sessionID).Send(ctx, update); err != nil {
			return "", err
		}
	}
	reason, err := a.runTurn(ctx, sessionID, sess, promptBlocks(prompt, lookupModel(model).images))
	if err != nil {
		return "", err
	}
	return a.continueGoal(ctx, sessionID, sess, reason)
}

func (a *deepseekAgent) requireKey() error {
	if a.cfg.apiKey == "" {
		return acp.InternalError("DEEPSEEK_API_KEY is not set; add it to the agent's environment in your editor settings")
	}
	return nil
}

// turnRequest builds the request settings for a turn from the session's
// model and reasoning effort. They and the system prompt hold for the whole
// turn; a change made during it applies from the next one.
func (a *deepseekAgent) turnRequest(sess *session) (deepseek.Request, modelInfo) {
	model, effort := sess.settings()
	req := deepseek.Request{
		Model:     model,
		MaxTokens: a.cfg.maxTokens,
		System:    a.systemPrompt(sess, model),
		Tools:     tools,
		Thinking:  &deepseek.Thinking{Type: "enabled"},
	}
	if effort == "off" {
		req.Thinking.Type = "disabled"
	} else {
		req.OutputConfig = &deepseek.OutputConfig{Effort: effort}
	}
	return req, lookupModel(model)
}

// runTurn answers a user message of blocks: it calls the model, runs the tools it
// asks for, and calls it again with their results until it answers without
// a tool call.
//
// Every tool_use in the history is followed by its tool_result, even when the
// turn is cancelled halfway, or the next request would be rejected.
func (a *deepseekAgent) runTurn(ctx context.Context, sessionID acp1.SessionID, sess *session, blocks []deepseek.Block) (acp1.StopReason, error) {
	if err := a.requireKey(); err != nil {
		return "", err
	}
	stream := acp1.NewSessionStream(a.client, sessionID)
	req, info := a.turnRequest(sess)
	handler := deepseek.Handler{
		OnText:     func(text string) error { return stream.SendText(ctx, text) },
		OnThinking: func(text string) error { return stream.SendThought(ctx, text) },
	}

	if err := a.maybeCompact(ctx, stream, sess, req, info, false); err != nil {
		if ctx.Err() != nil {
			return acp1.StopReasonCancelled, nil
		}
		return "", err
	}
	// The message is made after any compaction, which restates the runtime
	// context itself.
	sess.commit(userTurn(sess, blocks))

	for range a.cfg.maxSteps {
		req.Messages = sess.messages()
		reply, err := a.llm.Stream(ctx, req, handler)
		if reply != nil {
			sess.recordUsage(reply.Usage)
		}
		if err != nil {
			// Keep what the user already saw of an interrupted answer, but
			// no tool calls, which would have no results.
			if reply != nil {
				sess.commit(withoutToolCalls(reply.Message))
			}
			if ctx.Err() != nil {
				return acp1.StopReasonCancelled, nil
			}
			return "", err
		}
		if used := reply.Usage.ContextTokens(); used > 0 {
			if err := stream.SendUsage(ctx, uint64(used), uint64(info.contextWindow), nil); err != nil {
				return "", err
			}
		}

		calls := reply.ToolCalls()
		if len(calls) == 0 {
			sess.commit(reply.Message)
			switch reply.StopReason {
			case "max_tokens":
				return acp1.StopReasonMaxTokens, nil
			case "refusal":
				return acp1.StopReasonRefusal, nil
			}
			return acp1.StopReasonEndTurn, nil
		}

		// Every call gets a result, however the loop ends: those not run
		// because the turn was cancelled or the connection failed are
		// reported as aborted.
		results := make([]deepseek.Block, 0, len(calls))
		var connErr error
		for _, call := range calls {
			if ctx.Err() != nil {
				break
			}
			result, isError, err := a.runTool(ctx, stream, sess, call)
			if err != nil {
				connErr = err
				break
			}
			results = append(results, deepseek.ToolResultBlock(call.ID, result, isError))
		}
		for _, call := range calls[len(results):] {
			results = append(results, deepseek.ToolResultBlock(call.ID, "Error: tool call aborted", true))
		}
		sess.commit(reply.Message, deepseek.Message{Role: "user", Content: results})
		switch {
		case ctx.Err() != nil:
			return acp1.StopReasonCancelled, nil
		case connErr != nil:
			return "", connErr
		case reply.StopReason == "max_tokens":
			return acp1.StopReasonMaxTokens, nil
		}
		if err := a.maybeCompact(ctx, stream, sess, req, info, true); err != nil {
			if ctx.Err() != nil {
				return acp1.StopReasonCancelled, nil
			}
			return "", err
		}
	}
	return acp1.StopReasonMaxTurnRequests, nil
}

func withoutToolCalls(m deepseek.Message) deepseek.Message {
	kept := deepseek.Message{Role: m.Role}
	for _, b := range m.Content {
		if b.Type != "tool_use" {
			kept.Content = append(kept.Content, b)
		}
	}
	return kept
}

const runtimeContextPrefix = "<runtime-context>"

// userTurn makes a user message of blocks. The first message after the
// permission or plan mode changes starts with a note of the new runtime
// context, so the system prompt never changes and stays cached.
func userTurn(sess *session, blocks []deepseek.Block) deepseek.Message {
	msg := deepseek.Message{Role: "user"}
	if text, changed := sess.contextChange(); changed {
		msg.Content = append(msg.Content, deepseek.TextBlock(fmt.Sprintf(
			"%s\nCurrent runtime context. This snapshot supersedes earlier ones.\n%s\n</runtime-context>",
			runtimeContextPrefix, text)))
	}
	msg.Content = append(msg.Content, blocks...)
	if len(msg.Content) == 0 {
		msg.Content = append(msg.Content, deepseek.TextBlock("(empty prompt)"))
	}
	return msg
}

// promptBlocks turns an ACP prompt into content blocks for the model.
func promptBlocks(prompt []acp1.ContentBlock, images bool) []deepseek.Block {
	msg := deepseek.Message{Role: "user"}
	for _, block := range prompt {
		switch b := block.Variant().(type) {
		case acp1.ContentBlockText:
			msg.Content = append(msg.Content, deepseek.TextBlock(b.Text))
		case acp1.ContentBlockImage:
			if images {
				msg.Content = append(msg.Content, deepseek.Block{Type: "image", Source: &deepseek.ImageSource{
					Type: "base64", MediaType: b.MIMEType, Data: b.Data,
				}})
			} else {
				msg.Content = append(msg.Content, deepseek.TextBlock("[an image was attached, but this model does not accept images]"))
			}
		case acp1.ContentBlockResource:
			if text, err := b.Resource.As[acp1.TextResourceContents](); err == nil {
				msg.Content = append(msg.Content, deepseek.TextBlock(fmt.Sprintf(
					"<context uri=%q>\n%s\n</context>", text.URI, text.Text)))
			}
		case acp1.ContentBlockResourceLink:
			msg.Content = append(msg.Content, deepseek.TextBlock(describeLink(b)))
		}
	}
	return msg.Content
}

// describeLink mentions a linked resource; a linked file is named by its path
// so the model can read it with its tools.
func describeLink(link acp1.ContentBlockResourceLink) string {
	if path, ok := strings.CutPrefix(link.URI, "file://"); ok {
		return fmt.Sprintf("@%s", path)
	}
	return fmt.Sprintf("[resource_link name=%s uri=%s]", link.Name, link.URI)
}

// systemPrompt follows the harness's ACP persona and tool guidance. It
// depends only on the model, the working directory and the date, so it stays
// byte-stable across a session and DeepSeek's prefix cache keeps hitting.
func (a *deepseekAgent) systemPrompt(sess *session, model string) string {
	var b strings.Builder
	fmt.Fprintf(&b, "You are a coding agent powered by the %s model, running inside the user's editor through the Agent Client Protocol.\n\n", model)
	b.WriteString(toolGuidance)
	fmt.Fprintf(&b, "\nYour working directory is %s. The platform is %s/%s. Today is %s.\n",
		sess.cwd, runtime.GOOS, runtime.GOARCH, time.Now().Format("2006-01-02"))
	if instructions := projectInstructions(sess.cwd); instructions != "" {
		b.WriteString("\n")
		b.WriteString(instructions)
	}
	return b.String()
}

const toolGuidance = `Use your tools to inspect and change the project instead of guessing.
- Use the read tool, not shell commands like cat, to inspect text files. Use offset and limit to continue reading large files.
- Read a file before editing it, and prefer edit for targeted changes. Read an existing file before overwriting it with write.
- Use the glob tool, not shell find, to find files by name, and the grep tool, not shell grep or rg, to search file contents.
- Use bash for builds, tests, git and other commands. Each call runs in a fresh shell; pass workdir instead of using cd. Check the [exit code: N] marker on every bash result and investigate failures before moving on.
- For multi-step work, keep a plan with todo_write and update it as you go.
- For a long-running objective that needs many turns, create a goal with create_goal; the session then keeps working on it in automatic rounds until you mark it complete with update_goal.
- The user may reject a change or a command. If so, do not retry it another way; ask the user or adjust your approach.
Be concise. Reference code as path:line.
`

// instructionFiles are read from the working directory into the system
// prompt, like the harness's agent-instructions plugin.
var instructionFiles = []string{"AGENTS.md", "CLAUDE.md"}

const instructionLimit = 64 << 10

func projectInstructions(cwd string) string {
	for _, name := range instructionFiles {
		data, err := os.ReadFile(filepath.Join(cwd, name))
		if err != nil || len(data) == 0 {
			continue
		}
		if len(data) > instructionLimit {
			data = data[:instructionLimit]
		}
		return fmt.Sprintf("Project instructions from %s:\n<instructions>\n%s\n</instructions>\n", name, data)
	}
	return ""
}

var errRejected = errors.New("the user rejected this")
