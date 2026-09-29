package main

import (
	"cmp"
	"context"
	"encoding/json/jsontext"
	"encoding/json/v2"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/ironpark/acp-go/acp1"

	"github.com/ironpark/deepseek-acp/internal/deepseek"
)

// Limits on what the tools return to the model, as in the harness.
const (
	readDefaultLimit = 2000     // lines
	readLineLimit    = 2000     // characters per line
	outputLimit      = 30 << 10 // bytes of command output kept, from the tail
	globLimit        = 100
	grepLimit        = 250
	bashTimeout      = 2 * time.Minute
	bashMaxTimeout   = 10 * time.Minute
)

// The tools offered to the model, after the harness's core tool set.
var (
	readTool = deepseek.Tool{
		Name:        "read",
		Description: "Read a UTF-8 text file and return line-numbered content. Reading a directory lists its entries.",
		InputSchema: jsontext.Value(`{"type":"object","properties":{
			"file_path":{"type":"string","description":"Path to read, absolute or relative to the working directory."},
			"offset":{"type":"number","description":"1-based first line to read; default 1."},
			"limit":{"type":"number","description":"Maximum number of lines to read; default 2000."}},
			"required":["file_path"]}`),
	}
	writeTool = deepseek.Tool{
		Name:        "write",
		Description: "Create or fully replace a UTF-8 text file.",
		InputSchema: jsontext.Value(`{"type":"object","properties":{
			"file_path":{"type":"string","description":"Path to write, absolute or relative to the working directory."},
			"content":{"type":"string","description":"The complete new content of the file."}},
			"required":["file_path","content"]}`),
	}
	editTool = deepseek.Tool{
		Name:        "edit",
		Description: "Edit an existing UTF-8 text file by replacing literal text. Unless replace_all is true, old_string must appear exactly once; include enough surrounding context to make it unique.",
		InputSchema: jsontext.Value(`{"type":"object","properties":{
			"file_path":{"type":"string","description":"Path to edit, absolute or relative to the working directory."},
			"old_string":{"type":"string","description":"The exact text to replace."},
			"new_string":{"type":"string","description":"The replacement text."},
			"replace_all":{"type":"boolean","description":"Replace every occurrence of old_string; default false."}},
			"required":["file_path","old_string","new_string"]}`),
	}
	globTool = deepseek.Tool{
		Name:        "glob",
		Description: "Find files, not directories, whose paths match a glob pattern such as **/*.go or src/**/*_test.ts, including hidden and ignored files. A pattern without a slash matches file names at any depth. Returns up to 100 paths, most recently modified first.",
		InputSchema: jsontext.Value(`{"type":"object","properties":{
			"pattern":{"type":"string","description":"The glob pattern. Supports *, **, ?, [abc] and {a,b}."},
			"path":{"type":"string","description":"Directory to search; default the working directory."}},
			"required":["pattern"]}`),
	}
	grepTool = deepseek.Tool{
		Name:        "grep",
		Description: "Search file contents with a regular expression (ripgrep syntax). Returns matching lines with line numbers, as path:line:text. Returns up to 250 matches.",
		InputSchema: jsontext.Value(`{"type":"object","properties":{
			"pattern":{"type":"string","description":"The regular expression to search for."},
			"path":{"type":"string","description":"File or directory to search; default the working directory."},
			"include":{"type":"string","description":"Only search files whose names match this glob, such as *.go."}},
			"required":["pattern"]}`),
	}
	bashTool = deepseek.Tool{
		Name:        "bash",
		Description: "Execute a bash command (bash -c) and return its output. Each call runs in a fresh shell; pass workdir instead of using cd. Long output is truncated to its tail. A non-zero exit is reported as [exit code: N].",
		InputSchema: jsontext.Value(`{"type":"object","properties":{
			"command":{"type":"string","description":"The bash command to execute."},
			"description":{"type":"string","description":"Clear, concise description of what this command does in active voice, 5-10 words (shown in the UI)."},
			"timeout_ms":{"type":"number","description":"Timeout in milliseconds; default 120000, at most 600000."},
			"workdir":{"type":"string","description":"Working directory for this command; default the working directory. A relative path is resolved against it."}},
			"required":["command","description"]}`),
	}
	todoWriteTool = deepseek.Tool{
		Name:        "todo_write",
		Description: "Replace the task plan shown to the user. Use it for multi-step work: list every step, keep exactly one in_progress while working, and mark steps completed as soon as they are done.",
		InputSchema: jsontext.Value(`{"type":"object","properties":{
			"todos":{"type":"array","items":{"type":"object","properties":{
				"content":{"type":"string","description":"What the step does."},
				"status":{"type":"string","enum":["pending","in_progress","completed"]},
				"priority":{"type":"string","enum":["high","medium","low"]}},
				"required":["content","status"]}}},
			"required":["todos"]}`),
	}
)

// tools is the catalog offered to the model. It never changes with the
// session's modes, so the request prefix stays cacheable.
var tools = []deepseek.Tool{
	readTool, writeTool, editTool, globTool, grepTool, bashTool, todoWriteTool,
	exitPlanModeTool, createGoalTool, getGoalTool, updateGoalTool,
}

// action is a tool call ready to show the client and run.
type action struct {
	title     string
	kind      acp1.ToolKind
	locations []acp1.ToolCallLocation
	// hidden actions are not shown as tool calls, like todo_write, which
	// shows as the plan.
	hidden bool
	// gated actions may wait for the user's permission, so they show as
	// pending until they run.
	gated bool
	// run does the work and returns the result for the model and the content
	// to show the client. With an error, the tool call fails, and the model
	// gets the result if there is one, the error otherwise.
	run func(ctx context.Context, stream *acp1.SessionStream, id acp1.ToolCallID) (string, []acp1.ToolCallContent, error)
}

// runTool runs one tool call from the model as an ACP tool call and returns
// the result for the model. A failing tool is reported to the model, which
// can react to it; only a failure to talk to the client returns an error.
//
// The ACP tool call id is generated rather than taken from the model: it must
// be unique in the session, and the model's ids need not be.
func (a *deepseekAgent) runTool(ctx context.Context, stream *acp1.SessionStream, sess *session, call deepseek.Block) (result string, isError bool, err error) {
	id := acp1.GenerateToolCallID()
	act, err := a.parseTool(sess, call)
	if err != nil {
		// A call that cannot be parsed still shows, as a tool call that fails.
		act = &action{
			title: call.Name,
			kind:  acp1.ToolKindOther,
			run: func(context.Context, *acp1.SessionStream, acp1.ToolCallID) (string, []acp1.ToolCallContent, error) {
				return "", nil, err
			},
		}
	}

	if act.hidden {
		result, _, err := act.run(ctx, stream, id)
		return result, false, err
	}

	show := stream.StartToolCall
	if act.gated {
		show = stream.ProposeToolCall
	}
	if err := show(ctx, id, act.title, act.kind,
		acp1.WithLocations(act.locations...), acp1.WithRawInput(call.Input)); err != nil {
		return "", false, err
	}
	result, content, runErr := act.run(ctx, stream, id)
	// The final status is sent even if the turn was cancelled meanwhile, so
	// the client does not show the call as still running.
	ctx = context.WithoutCancel(ctx)
	if runErr != nil {
		if content == nil {
			content = []acp1.ToolCallContent{acp1.ToolText(runErr.Error())}
		}
		if result == "" {
			result = "Error: " + runErr.Error()
		}
		return result, true, stream.FailToolCall(ctx, id, acp1.WithToolContent(content...))
	}
	return result, false, stream.CompleteToolCall(ctx, id, acp1.WithToolContent(content...))
}

// parseTool decodes the model's arguments into an action.
func (a *deepseekAgent) parseTool(sess *session, call deepseek.Block) (*action, error) {
	switch call.Name {
	case readTool.Name:
		args, err := decodeArgs[struct {
			FilePath string  `json:"file_path"`
			Offset   float64 `json:"offset"`
			Limit    float64 `json:"limit"`
		}](call)
		if err != nil {
			return nil, err
		}
		path := sess.resolve(args.FilePath)
		offset := max(int(args.Offset), 1)
		limit := cmp.Or(max(int(args.Limit), 0), readDefaultLimit)
		return &action{
			title:     "Read " + sess.display(path),
			kind:      acp1.ToolKindRead,
			locations: []acp1.ToolCallLocation{{Path: path, Line: new(uint32(offset))}},
			run: func(ctx context.Context, stream *acp1.SessionStream, _ acp1.ToolCallID) (string, []acp1.ToolCallContent, error) {
				return a.readFile(ctx, stream, path, offset, limit)
			},
		}, nil

	case writeTool.Name:
		args, err := decodeArgs[struct {
			FilePath string `json:"file_path"`
			Content  string `json:"content"`
		}](call)
		if err != nil {
			return nil, err
		}
		path := sess.resolve(args.FilePath)
		return &action{
			title:     "Write " + sess.display(path),
			kind:      acp1.ToolKindEdit,
			gated:     true,
			locations: []acp1.ToolCallLocation{{Path: path}},
			run: func(ctx context.Context, stream *acp1.SessionStream, id acp1.ToolCallID) (string, []acp1.ToolCallContent, error) {
				return a.writeFile(ctx, stream, sess, id, path, args.Content)
			},
		}, nil

	case editTool.Name:
		args, err := decodeArgs[struct {
			FilePath   string `json:"file_path"`
			OldString  string `json:"old_string"`
			NewString  string `json:"new_string"`
			ReplaceAll bool   `json:"replace_all"`
		}](call)
		if err != nil {
			return nil, err
		}
		path := sess.resolve(args.FilePath)
		return &action{
			title:     "Edit " + sess.display(path),
			kind:      acp1.ToolKindEdit,
			gated:     true,
			locations: []acp1.ToolCallLocation{{Path: path}},
			run: func(ctx context.Context, stream *acp1.SessionStream, id acp1.ToolCallID) (string, []acp1.ToolCallContent, error) {
				return a.editFile(ctx, stream, sess, id, path, args.OldString, args.NewString, args.ReplaceAll)
			},
		}, nil

	case globTool.Name:
		args, err := decodeArgs[struct {
			Pattern string `json:"pattern"`
			Path    string `json:"path"`
		}](call)
		if err != nil {
			return nil, err
		}
		root := sess.resolve(args.Path)
		return &action{
			title:     fmt.Sprintf("Find `%s`", args.Pattern),
			kind:      acp1.ToolKindSearch,
			locations: []acp1.ToolCallLocation{{Path: root}},
			run: func(ctx context.Context, _ *acp1.SessionStream, _ acp1.ToolCallID) (string, []acp1.ToolCallContent, error) {
				return globFiles(ctx, root, args.Pattern)
			},
		}, nil

	case grepTool.Name:
		args, err := decodeArgs[struct {
			Pattern string `json:"pattern"`
			Path    string `json:"path"`
			Include string `json:"include"`
		}](call)
		if err != nil {
			return nil, err
		}
		root := sess.resolve(args.Path)
		title := fmt.Sprintf("Search `%s`", args.Pattern)
		if args.Include != "" {
			title += fmt.Sprintf(" in `%s`", args.Include)
		}
		return &action{
			title:     title,
			kind:      acp1.ToolKindSearch,
			locations: []acp1.ToolCallLocation{{Path: root}},
			run: func(ctx context.Context, _ *acp1.SessionStream, _ acp1.ToolCallID) (string, []acp1.ToolCallContent, error) {
				return grepFiles(ctx, sess.cwd, root, args.Pattern, args.Include)
			},
		}, nil

	case bashTool.Name:
		args, err := decodeArgs[struct {
			Command     string  `json:"command"`
			Description string  `json:"description"`
			TimeoutMs   float64 `json:"timeout_ms"`
			Workdir     string  `json:"workdir"`
		}](call)
		if err != nil {
			return nil, err
		}
		if strings.TrimSpace(args.Command) == "" {
			return nil, errors.New("empty command")
		}
		timeout := bashTimeout
		if args.TimeoutMs > 0 {
			timeout = min(time.Duration(args.TimeoutMs)*time.Millisecond, bashMaxTimeout)
		}
		workdir := sess.resolve(args.Workdir)
		title := fmt.Sprintf("`%s`", oneLine(args.Command))
		if args.Description != "" {
			title = args.Description
		}
		return &action{
			title: title,
			kind:  acp1.ToolKindExecute,
			gated: true,
			run: func(ctx context.Context, stream *acp1.SessionStream, id acp1.ToolCallID) (string, []acp1.ToolCallContent, error) {
				return a.runBash(ctx, stream, sess, id, args.Command, workdir, timeout)
			},
		}, nil

	case todoWriteTool.Name:
		args, err := decodeArgs[struct {
			Todos []struct {
				Content  string `json:"content"`
				Status   string `json:"status"`
				Priority string `json:"priority"`
			} `json:"todos"`
		}](call)
		if err != nil {
			return nil, err
		}
		entries := make([]acp1.PlanEntry, 0, len(args.Todos))
		for _, t := range args.Todos {
			entries = append(entries, acp1.PlanEntry{
				Content:  t.Content,
				Status:   acp1.PlanEntryStatus(cmp.Or(t.Status, "pending")),
				Priority: acp1.PlanEntryPriority(cmp.Or(t.Priority, "medium")),
			})
		}
		return &action{
			hidden: true,
			run: func(ctx context.Context, stream *acp1.SessionStream, _ acp1.ToolCallID) (string, []acp1.ToolCallContent, error) {
				sess.setTodos(entries)
				if err := stream.SendPlan(ctx, entries); err != nil {
					return "", nil, err
				}
				return "Plan updated.", nil, nil
			},
		}, nil

	case exitPlanModeTool.Name:
		args, err := decodeArgs[struct {
			Plan string `json:"plan"`
		}](call)
		if err != nil {
			return nil, err
		}
		return &action{
			title: "Plan: " + planTitle(args.Plan),
			kind:  acp1.ToolKindSwitchMode,
			gated: true,
			run: func(ctx context.Context, stream *acp1.SessionStream, id acp1.ToolCallID) (string, []acp1.ToolCallContent, error) {
				return a.exitPlanMode(ctx, stream, sess, id, args.Plan)
			},
		}, nil

	case createGoalTool.Name:
		args, err := decodeArgs[struct {
			Objective     string  `json:"objective"`
			MaxGoalRounds float64 `json:"max_goal_rounds"`
		}](call)
		if err != nil {
			return nil, err
		}
		return &action{
			title: "Create goal: " + oneLine(args.Objective),
			kind:  acp1.ToolKindThink,
			run: func(context.Context, *acp1.SessionStream, acp1.ToolCallID) (string, []acp1.ToolCallContent, error) {
				result, err := sess.createGoal(args.Objective, int(args.MaxGoalRounds))
				return result, []acp1.ToolCallContent{acp1.ToolText(result)}, err
			},
		}, nil

	case getGoalTool.Name:
		return &action{
			title: "Read goal",
			kind:  acp1.ToolKindThink,
			run: func(context.Context, *acp1.SessionStream, acp1.ToolCallID) (string, []acp1.ToolCallContent, error) {
				g := sess.currentGoal()
				if g == nil {
					return "There is no goal.", nil, nil
				}
				return g.describe(), nil, nil
			},
		}, nil

	case updateGoalTool.Name:
		args, err := decodeArgs[struct {
			Action        string  `json:"action"`
			Objective     string  `json:"objective"`
			MaxGoalRounds float64 `json:"max_goal_rounds"`
			BlockedReason string  `json:"blocked_reason"`
		}](call)
		if err != nil {
			return nil, err
		}
		return &action{
			title: "Update goal: " + args.Action,
			kind:  acp1.ToolKindThink,
			run: func(context.Context, *acp1.SessionStream, acp1.ToolCallID) (string, []acp1.ToolCallContent, error) {
				g, err := sess.updateGoal(args.Action, args.Objective, int(args.MaxGoalRounds), args.BlockedReason)
				if err != nil {
					return "", nil, err
				}
				return "Goal updated.\n" + g.describe(), []acp1.ToolCallContent{acp1.ToolText(g.describe())}, nil
			},
		}, nil
	}
	return nil, fmt.Errorf("unknown tool %q", call.Name)
}

// planTitle is the plan's # heading, or a generic title.
func planTitle(plan string) string {
	for line := range strings.Lines(plan) {
		if title, ok := strings.CutPrefix(strings.TrimSpace(line), "# "); ok {
			return title
		}
	}
	return "review"
}

func decodeArgs[T any](call deepseek.Block) (T, error) {
	var args T
	if err := json.Unmarshal(call.Input, &args); err != nil {
		return args, fmt.Errorf("invalid arguments: %w", err)
	}
	return args, nil
}

// display shows a path relative to the working directory when it is inside.
func (s *session) display(path string) string {
	if rel, err := filepath.Rel(s.cwd, path); err == nil && filepath.IsLocal(rel) {
		return rel
	}
	return path
}

func oneLine(s string) string {
	s = strings.TrimSpace(s)
	if first, _, ok := strings.Cut(s, "\n"); ok {
		return first + " …"
	}
	return s
}

// Files.

// readText reads a file through the client when it can, which includes
// unsaved changes in the editor, and from disk otherwise.
func (a *deepseekAgent) readText(ctx context.Context, stream *acp1.SessionStream, path string) (string, error) {
	if a.client.ClientCapabilities().GetFS().GetReadTextFile() {
		return stream.ReadTextFile(ctx, path)
	}
	data, err := os.ReadFile(path)
	return string(data), err
}

// writeText writes a file through the client when it can, so the editor
// tracks the change, and to disk otherwise.
func (a *deepseekAgent) writeText(ctx context.Context, stream *acp1.SessionStream, path, content string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	if a.client.ClientCapabilities().GetFS().GetWriteTextFile() {
		return stream.WriteTextFile(ctx, path, content)
	}
	mode := os.FileMode(0o644)
	if info, err := os.Stat(path); err == nil {
		mode = info.Mode().Perm()
	}
	return os.WriteFile(path, []byte(content), mode)
}

func (a *deepseekAgent) readFile(ctx context.Context, stream *acp1.SessionStream, path string, offset, limit int) (string, []acp1.ToolCallContent, error) {
	if info, err := os.Stat(path); err == nil && info.IsDir() {
		return listDir(path)
	}
	text, err := a.readText(ctx, stream, path)
	if err != nil {
		return "", nil, err
	}
	lines := strings.Split(text, "\n")
	if n := len(lines); n > 1 && lines[n-1] == "" {
		lines = lines[:n-1] // the file's final newline
	}
	total := len(lines)
	if offset > total && total > 0 {
		return "", nil, fmt.Errorf("offset %d is past the end of the file (%d lines)", offset, total)
	}

	var b strings.Builder
	end := min(offset-1+limit, total)
	for i := offset - 1; i < end; i++ {
		line := lines[i]
		if len(line) > readLineLimit {
			line = line[:readLineLimit] + "… [line truncated]"
		}
		fmt.Fprintf(&b, "%d: %s\n", i+1, line)
	}
	if end < total {
		fmt.Fprintf(&b, "\n(Showing lines %d-%d of %d. Use offset=%d to continue.)", offset, end, total, end+1)
	} else {
		fmt.Fprintf(&b, "\n(End of file - total %d lines)", total)
	}
	summary := fmt.Sprintf("Read lines %d-%d of %d", min(offset, end), end, total)
	return b.String(), []acp1.ToolCallContent{acp1.ToolText(summary)}, nil
}

func listDir(path string) (string, []acp1.ToolCallContent, error) {
	entries, err := os.ReadDir(path)
	if err != nil {
		return "", nil, err
	}
	var b strings.Builder
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() {
			name += "/"
		}
		b.WriteString(name + "\n")
	}
	if len(entries) == 0 {
		b.WriteString("(empty directory)")
	}
	return b.String(), []acp1.ToolCallContent{acp1.ToolText(fmt.Sprintf("Listed %d entries", len(entries)))}, nil
}

// writeFile shows the change as a diff and writes it if the mode allows it.
func (a *deepseekAgent) writeFile(ctx context.Context, stream *acp1.SessionStream, sess *session, id acp1.ToolCallID, path, content string) (string, []acp1.ToolCallContent, error) {
	// A file that cannot be read is shown as a new one.
	var oldText *string
	if old, err := a.readText(ctx, stream, path); err == nil {
		oldText = &old
	}
	diff := []acp1.ToolCallContent{acp1.ToolDiff(path, oldText, content)}
	if err := a.permit(ctx, stream, sess, id, writeTool.Name, true, diff...); err != nil {
		return "", diff, err
	}
	if err := a.writeText(ctx, stream, path, content); err != nil {
		return "", diff, err
	}
	if oldText == nil {
		return "Created " + path, diff, nil
	}
	return "Wrote " + path, diff, nil
}

func (a *deepseekAgent) editFile(ctx context.Context, stream *acp1.SessionStream, sess *session, id acp1.ToolCallID, path, oldString, newString string, replaceAll bool) (string, []acp1.ToolCallContent, error) {
	if oldString == "" {
		return "", nil, errors.New("old_string is empty; use write to create a file")
	}
	if oldString == newString {
		return "", nil, errors.New("old_string and new_string are the same")
	}
	old, err := a.readText(ctx, stream, path)
	if err != nil {
		return "", nil, err
	}
	count := strings.Count(old, oldString)
	switch {
	case count == 0:
		return "", nil, errors.New("old_string was not found in the file; read the file again and copy the text exactly")
	case count > 1 && !replaceAll:
		return "", nil, fmt.Errorf("old_string appears %d times; add surrounding context to make it unique, or set replace_all", count)
	}
	// Here count is 1 unless replaceAll is set.
	updated := strings.ReplaceAll(old, oldString, newString)

	diff := []acp1.ToolCallContent{acp1.ToolDiff(path, &old, updated)}
	if err := a.permit(ctx, stream, sess, id, editTool.Name, true, diff...); err != nil {
		return "", diff, err
	}
	if err := a.writeText(ctx, stream, path, updated); err != nil {
		return "", diff, err
	}
	if count == 1 {
		return "Edited " + path, diff, nil
	}
	return fmt.Sprintf("Edited %s (%d replacements)", path, count), diff, nil
}

// Commands.

// runBash runs a command in a terminal the client owns when it can, so the
// client shows the output as it runs, and locally otherwise.
func (a *deepseekAgent) runBash(ctx context.Context, stream *acp1.SessionStream, sess *session, id acp1.ToolCallID, command, workdir string, timeout time.Duration) (string, []acp1.ToolCallContent, error) {
	if err := a.permit(ctx, stream, sess, id, bashTool.Name, false, acp1.ToolText("```sh\n"+command+"\n```")); err != nil {
		return "", nil, err
	}
	if a.client.ClientCapabilities().GetTerminal() {
		return a.runInTerminal(ctx, stream, id, command, workdir, timeout)
	}
	return a.runLocally(ctx, command, workdir, timeout)
}

func (a *deepseekAgent) runInTerminal(ctx context.Context, stream *acp1.SessionStream, id acp1.ToolCallID, command, workdir string, timeout time.Duration) (string, []acp1.ToolCallContent, error) {
	run, err := stream.RunTerminal(ctx, id, acp1.CreateTerminalRequest{
		Command:         a.cfg.shell,
		Args:            []string{"-c", command},
		Cwd:             &workdir,
		OutputByteLimit: new(uint64(outputLimit)),
	}, timeout)
	if run == nil {
		return "", nil, err
	}
	// The client keeps showing a released terminal's output, also for a
	// command the cancelled turn killed.
	content := []acp1.ToolCallContent{acp1.ToolTerminal(run.TerminalID)}
	if err != nil {
		return "", content, err
	}
	var code *int
	var signal string
	if exit := run.ExitStatus; exit != nil {
		if exit.ExitCode != nil {
			code = new(int(*exit.ExitCode))
		}
		if exit.Signal != nil {
			signal = *exit.Signal
		}
	}
	var timedOut time.Duration
	if run.TimedOut {
		timedOut = timeout
	}
	result, err := commandResult(run.Output, run.Truncated, code, signal, timedOut)
	return result, content, err
}

func (a *deepseekAgent) runLocally(ctx context.Context, command, workdir string, timeout time.Duration) (string, []acp1.ToolCallContent, error) {
	runCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	cmd := exec.CommandContext(runCtx, a.cfg.shell, "-c", command)
	cmd.Dir = workdir
	cmd.WaitDelay = 5 * time.Second
	setProcessGroup(cmd)
	out := &tailBuffer{limit: outputLimit}
	cmd.Stdout, cmd.Stderr = out, out
	err := cmd.Run()
	if ctx.Err() != nil {
		return "", nil, ctx.Err()
	}

	var (
		code     *int
		signal   string
		timedOut time.Duration
	)
	if runCtx.Err() != nil {
		timedOut = timeout
	}
	if err == nil {
		code = new(0)
	} else {
		exitErr, ok := errors.AsType[*exec.ExitError](err)
		switch {
		case ok && exitErr.ExitCode() >= 0:
			code = new(exitErr.ExitCode())
		case ok && timedOut == 0:
			signal = exitErr.String()
		case !ok && timedOut == 0:
			return "", nil, err // the shell could not start
		}
	}
	result, err := commandResult(string(out.buf), out.truncated, code, signal, timedOut)
	content := []acp1.ToolCallContent{acp1.ToolText("```\n" + strings.TrimRight(result, "\n") + "\n```")}
	return result, content, err
}

// tailBuffer keeps the last limit bytes written to it.
type tailBuffer struct {
	buf       []byte
	limit     int
	truncated bool
}

func (t *tailBuffer) Write(p []byte) (int, error) {
	t.buf = append(t.buf, p...)
	if over := len(t.buf) - t.limit; over > 0 {
		t.buf = append(t.buf[:0], t.buf[over:]...)
		t.truncated = true
	}
	return len(p), nil
}

var errCommandFailed = errors.New("command failed")

// commandResult formats a command's output for the model, with the harness's
// markers, and returns errCommandFailed if the command failed. timedOut is
// the timeout the command hit, or 0 if it did not.
func commandResult(output string, truncated bool, code *int, signal string, timedOut time.Duration) (string, error) {
	var b strings.Builder
	if truncated {
		b.WriteString("[output truncated; showing the last part]\n")
	}
	if strings.TrimSpace(output) == "" {
		b.WriteString("(no output)")
	} else {
		b.WriteString(strings.TrimRight(output, "\n"))
	}
	switch {
	case timedOut > 0:
		fmt.Fprintf(&b, "\n[timed out after %dms]", timedOut.Milliseconds())
	case signal != "":
		fmt.Fprintf(&b, "\n[killed by signal: %s]", signal)
	case code != nil && *code != 0:
		fmt.Fprintf(&b, "\n[exit code: %d]", *code)
	default:
		return b.String(), nil
	}
	return b.String(), errCommandFailed
}

// Permissions.

// permit decides whether a tool call may change files (edit) or run a
// command, by the session's mode, asking the user when the mode says to.
func (a *deepseekAgent) permit(ctx context.Context, stream *acp1.SessionStream, sess *session, id acp1.ToolCallID, tool string, edit bool, content ...acp1.ToolCallContent) error {
	if err := a.authorize(ctx, stream, sess, id, tool, edit, content...); err != nil {
		return err
	}
	// The call was pending while it might wait for the user; now it runs.
	return stream.UpdateToolCallStatus(ctx, id, acp1.ToolCallStatusInProgress)
}

// authorize is permit's decision.
func (a *deepseekAgent) authorize(ctx context.Context, stream *acp1.SessionStream, sess *session, id acp1.ToolCallID, tool string, edit bool, content ...acp1.ToolCallContent) error {
	if edit && sess.inPlanMode() {
		return errors.New("refused: plan mode is on; present the plan with exit_plan_mode instead of changing files")
	}
	switch mode := sess.currentMode(); {
	case mode == fullAccessMode:
		return nil
	case mode == readOnlyMode && edit:
		return errors.New("refused: the session is in read-only mode; the user must switch modes to allow file changes")
	case mode == acceptEditsMode && edit:
		return nil
	}
	if sess.isAllowed(tool) {
		return nil
	}
	toolCall := acp1.ToolCallUpdate{ToolCallID: id, Status: new(acp1.ToolCallStatusPending), Content: content}
	choice, allowed, err := stream.RequestPermission(ctx, toolCall,
		acp1.NewPermissionOption(acp1.PermissionOptionKindAllowOnce, "Allow"),
		acp1.NewPermissionOption(acp1.PermissionOptionKindAllowAlways, "Always allow "+tool),
		acp1.NewPermissionOption(acp1.PermissionOptionKindRejectOnce, "Reject"))
	if err != nil {
		return err
	}
	// Anything but an allowing choice, including a cancelled request, rejects.
	if !allowed {
		return errRejected
	}
	if choice.Kind == acp1.PermissionOptionKindAllowAlways {
		sess.allow(tool)
	}
	return nil
}
