package main

import (
	"context"
	"errors"
	"fmt"
	"regexp"
	"slices"
	"strings"

	"github.com/ironpark/acp-go/acp1"
)

// rule is what an "always" answer to a permission request remembers.
type rule struct {
	// key is what an always choice saves.
	key string
	// match lists the saved keys that decide this call.
	match []string
	// label names what key covers, as in "Always allow <label>".
	label string
}

// toolRule covers every call of a tool.
func toolRule(tool string) rule {
	return rule{key: tool, match: []string{tool}, label: tool}
}

// commandRule covers a bash command. A simple command, one the shell runs as
// a single program, is covered by its program, such as `ls`, or for a
// program with subcommands by its subcommand, such as `go test`, which also
// covers `go test ./...`; any other command only by itself. The bare "bash" key, which allowed every command, still applies to
// sessions that saved it.
func commandRule(command string) rule {
	command = strings.TrimSpace(command)
	exact := "bash=" + command
	r := rule{key: exact, match: []string{"bash", exact}, label: "this command"}
	words, ok := simpleCommand(command)
	if !ok {
		return r
	}
	for i := range words {
		r.match = append(r.match, "bash:"+strings.Join(words[:i+1], " "))
	}
	prefix := words[:1]
	if len(words) > 1 && subcommandPrograms[words[0]] && subcommandPattern.MatchString(words[1]) {
		prefix = words[:2]
	}
	r.key = "bash:" + strings.Join(prefix, " ")
	r.label = fmt.Sprintf("`%s` commands", strings.Join(prefix, " "))
	return r
}

// subcommandPrograms are programs whose second word picks what they do, so
// a rule for them names it too.
var subcommandPrograms = map[string]bool{
	"git": true, "go": true, "gh": true, "npm": true, "pnpm": true, "yarn": true, "bun": true, "deno": true,
	"cargo": true, "rustup": true, "docker": true, "kubectl": true, "make": true, "uv": true, "pip": true,
	"pip3": true, "poetry": true, "dotnet": true, "mvn": true, "gradle": true, "swift": true, "brew": true,
	"terraform": true,
}

var subcommandPattern = regexp.MustCompile(`^[a-z][a-z0-9-]*$`)

// simpleCommand splits a command into words if it runs a single program with
// no expansions, redirections, subshells or other commands. Quotes and globs
// are allowed: without $ or backticks they expand to nothing else.
func simpleCommand(command string) ([]string, bool) {
	if command == "" || strings.ContainsAny(command, ";&|<>`$\\()\n") {
		return nil, false
	}
	words := strings.Fields(command)
	if strings.ContainsAny(words[0], `='"*?~`) {
		return nil, false // an assignment, or a program named by expansion
	}
	return words, true
}

// fullAccessOption is the permission choice that allows the call and
// switches the session to full access.
const fullAccessOption acp1.PermissionOptionID = "full_access"

// permit decides whether a tool call may change files (edit) or run a
// command, by the session's mode, asking the user when the mode says to.
func (a *deepseekAgent) permit(ctx context.Context, stream *acp1.SessionStream, sess *session, id acp1.ToolCallID, r rule, edit bool, content ...acp1.ToolCallContent) error {
	if err := a.authorize(ctx, stream, sess, id, r, edit, content...); err != nil {
		return err
	}
	// The call was pending while it might wait for the user; now it runs.
	return stream.UpdateToolCallStatus(ctx, id, acp1.ToolCallStatusInProgress)
}

// authorize is permit's decision.
func (a *deepseekAgent) authorize(ctx context.Context, stream *acp1.SessionStream, sess *session, id acp1.ToolCallID, r rule, edit bool, content ...acp1.ToolCallContent) error {
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
	switch sess.ruling(r) {
	case ruledAllow:
		return nil
	case ruledDeny:
		return fmt.Errorf("refused: the user chose to always reject %s; do not retry it another way", r.label)
	}
	toolCall := acp1.ToolCallUpdate{ToolCallID: id, Status: new(acp1.ToolCallStatusPending), Content: content}
	choice, allowed, err := stream.RequestPermission(ctx, toolCall,
		acp1.NewPermissionOption(acp1.PermissionOptionKindAllowOnce, "Allow"),
		acp1.NewPermissionOption(acp1.PermissionOptionKindAllowAlways, "Always allow "+r.label),
		acp1.PermissionOption{OptionID: fullAccessOption, Name: "Allow all (Full Access)", Kind: acp1.PermissionOptionKindAllowAlways},
		acp1.NewPermissionOption(acp1.PermissionOptionKindRejectOnce, "Reject"),
		acp1.NewPermissionOption(acp1.PermissionOptionKindRejectAlways, "Always reject "+r.label))
	if err != nil {
		return err
	}
	switch {
	case choice.OptionID == fullAccessOption:
		return a.switchMode(ctx, stream, sess, fullAccessMode)
	case choice.Kind == acp1.PermissionOptionKindAllowAlways:
		sess.remember(r.key, true)
	case choice.Kind == acp1.PermissionOptionKindRejectAlways:
		sess.remember(r.key, false)
	}
	// Anything but an allowing choice, including a cancelled request, rejects.
	if !allowed {
		return errRejected
	}
	return nil
}

type ruled int

const (
	ruledNone ruled = iota
	ruledAllow
	ruledDeny
)

// ruling returns what the user's saved choices decide for r. A rejection
// wins over an allowance.
func (s *session) ruling(r rule) ruled {
	s.mu.Lock()
	defer s.mu.Unlock()
	switch {
	case slices.ContainsFunc(r.match, func(k string) bool { return s.denied[k] }):
		return ruledDeny
	case slices.ContainsFunc(r.match, func(k string) bool { return s.allowed[k] }):
		return ruledAllow
	}
	return ruledNone
}

// remember saves an always choice for key.
func (s *session) remember(key string, allow bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	set := &s.denied
	if allow {
		set = &s.allowed
	}
	if *set == nil {
		*set = map[string]bool{}
	}
	(*set)[key] = true
}
