package main

import (
	"context"
	"fmt"
	"slices"
	"strings"
	"time"

	"github.com/ironpark/acp-go/acp1"
)

// Slash commands, after the harness's human commands. A command runs without
// sending itself to the model; its result is shown to the user and kept out
// of the history. /plan and /goal may go on to start a model turn.

// command is one slash command.
type command struct {
	name, description, hint string
	run                     func(a *deepseekAgent, ctx context.Context, c *commandCall) (acp1.StopReason, error)
}

// commandCall is one invocation of a command.
type commandCall struct {
	sessionID acp1.SessionID
	sess      *session
	stream    *acp1.SessionStream
	input     string              // the text after the command name, trimmed
	rest      []acp1.ContentBlock // prompt blocks after the command's text
}

var commands = []command{
	{name: "compact", description: "Compact older conversation history", run: (*deepseekAgent).compactCommand},
	{name: "permission", description: "Switch the permission mode (read-only, ask, accept-edits, full-access)", hint: "<mode>", run: (*deepseekAgent).permissionCommand},
	{name: "plan", description: "Enter or leave plan mode", hint: "[off|message]", run: (*deepseekAgent).planCommand},
	{name: "goal", description: "Set or view the goal for a long-running task", hint: "[<objective>|clear|edit <objective>|pause|resume]", run: (*deepseekAgent).goalCommand},
}

func availableCommands() []acp1.AvailableCommand {
	list := make([]acp1.AvailableCommand, 0, len(commands))
	for _, c := range commands {
		ac := acp1.AvailableCommand{Name: c.name, Description: c.description}
		if c.hint != "" {
			ac.Input = &acp1.AvailableCommandInput{Hint: c.hint}
		}
		list = append(list, ac)
	}
	return list
}

// commandAdvertiseDelay lets a session/new or session/resume response reach
// the client before the commands for its session do.
const commandAdvertiseDelay = 50 * time.Millisecond

// advertiseCommands sends the session's commands once the response that
// creates or resumes it has gone out.
func (a *deepseekAgent) advertiseCommands(id acp1.SessionID) {
	time.AfterFunc(commandAdvertiseDelay, func() {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if err := acp1.NewSessionStream(a.client, id).SendCommands(ctx, availableCommands()); err != nil {
			a.logger.Warn("advertise commands", "session", id, "error", err)
		}
	})
}

// parseCommand finds the command a prompt starts with, if any. A prompt
// naming no known command, such as a path, goes to the model as text.
func parseCommand(prompt []acp1.ContentBlock) (*command, string, []acp1.ContentBlock, bool) {
	if len(prompt) == 0 {
		return nil, "", nil, false
	}
	text, ok := acp1.TextOf(prompt[0])
	if !ok {
		return nil, "", nil, false
	}
	line, ok := strings.CutPrefix(strings.TrimLeft(text, " \t\n"), "/")
	if !ok {
		return nil, "", nil, false
	}
	name, input, _ := strings.Cut(line, " ")
	if i := strings.IndexAny(name, "\n\t"); i >= 0 {
		name, input = name[:i], line[i:]
	}
	i := slices.IndexFunc(commands, func(c command) bool { return c.name == strings.ToLower(name) })
	if i < 0 {
		return nil, "", nil, false
	}
	return &commands[i], strings.TrimSpace(input), prompt[1:], true
}

// reply shows a command's result to the user.
func (c *commandCall) reply(ctx context.Context, text string) (acp1.StopReason, error) {
	return acp1.StopReasonEndTurn, c.stream.SendText(ctx, text)
}

func (a *deepseekAgent) compactCommand(ctx context.Context, c *commandCall) (acp1.StopReason, error) {
	if c.input != "" {
		return c.reply(ctx, "Usage: /compact (no arguments)")
	}
	if len(c.sess.messages()) == 0 {
		return c.reply(ctx, "No compactable history yet.")
	}
	if err := a.requireKey(); err != nil {
		return "", err
	}
	req, info := a.turnRequest(c.sess)
	before := c.sess.contextTokens()
	ok, err := a.compact(ctx, c.stream, c.sess, req, info, false)
	if err != nil || !ok {
		return acp1.StopReasonEndTurn, err
	}
	return c.reply(ctx, fmt.Sprintf("Compacted the conversation (about %d tokens before).", before))
}

// presetModes maps the harness's permission presets to this agent's modes.
var presetModes = map[string]acp1.SessionModeID{
	"workspace-write":    acceptEditsMode,
	"danger-full-access": fullAccessMode,
}

func (a *deepseekAgent) permissionCommand(ctx context.Context, c *commandCall) (acp1.StopReason, error) {
	var names []string
	for _, m := range modes {
		names = append(names, string(m.ID))
	}
	available := strings.Join(names, ", ")
	if c.input == "" {
		return c.reply(ctx, fmt.Sprintf("Current mode %s (available: %s)", c.sess.currentMode(), available))
	}
	mode := acp1.SessionModeID(c.input)
	if preset, ok := presetModes[c.input]; ok {
		mode = preset
	}
	if !validMode(mode) {
		return c.reply(ctx, fmt.Sprintf("Unknown mode %q (available: %s)", c.input, available))
	}
	if err := a.switchMode(ctx, c.stream, c.sess, mode); err != nil {
		return "", err
	}
	return c.reply(ctx, fmt.Sprintf("Permission mode: %s.", mode))
}

func (a *deepseekAgent) planCommand(ctx context.Context, c *commandCall) (acp1.StopReason, error) {
	switch {
	case c.input == "off":
		if !c.sess.inPlanMode() {
			return c.reply(ctx, "Plan mode is already off.")
		}
		c.sess.setPlanMode(false)
		return c.reply(ctx, "Plan mode off.")
	case c.input == "" && len(c.rest) == 0:
		c.sess.setPlanMode(true)
		return c.reply(ctx, "Plan mode on. Describe what to plan.")
	}
	c.sess.setPlanMode(true)
	prompt := c.rest
	if c.input != "" {
		prompt = append([]acp1.ContentBlock{acp1.TextBlock(c.input)}, c.rest...)
	}
	return a.promptTurn(ctx, c.sessionID, c.sess, prompt)
}

func (a *deepseekAgent) goalCommand(ctx context.Context, c *commandCall) (acp1.StopReason, error) {
	sess := c.sess
	switch word, objective, _ := strings.Cut(c.input, " "); {
	case c.input == "":
		g := sess.currentGoal()
		if g == nil {
			return c.reply(ctx, "No goal. Usage: /goal <objective> | clear | edit <objective> | pause | resume")
		}
		return c.reply(ctx, g.describe())
	case c.input == "clear":
		if !sess.clearGoal() {
			return c.reply(ctx, "No goal to clear.")
		}
		return c.reply(ctx, "Goal cleared.")
	case c.input == "pause" || c.input == "resume":
		g, err := sess.updateGoal(c.input, "", 0, "")
		if err != nil {
			return c.reply(ctx, "Cannot "+c.input+": "+err.Error())
		}
		if c.input == "pause" {
			return c.reply(ctx, "Goal paused.\n"+g.describe())
		}
		if err := c.stream.SendText(ctx, "Goal resumed.\n"+g.describe()+"\n\n"); err != nil {
			return "", err
		}
		return a.continueGoal(ctx, c.sessionID, sess, acp1.StopReasonEndTurn)
	case word == "edit":
		if strings.TrimSpace(objective) == "" {
			return c.reply(ctx, "Usage: /goal edit <objective>")
		}
		g, err := sess.updateGoal("edit", strings.TrimSpace(objective), 0, "")
		if err != nil {
			return c.reply(ctx, "Cannot edit: "+err.Error())
		}
		return c.reply(ctx, "Goal edited.\n"+g.describe())
	}
	if err := sess.setGoal(c.input, 0); err != nil {
		return c.reply(ctx, "Cannot set the goal: "+err.Error())
	}
	if err := c.stream.SendText(ctx, "Goal set.\n"+sess.currentGoal().describe()+"\n\n"); err != nil {
		return "", err
	}
	return a.continueGoal(ctx, c.sessionID, sess, acp1.StopReasonEndTurn)
}
