package main

import (
	"fmt"
	"slices"
	"testing"

	"github.com/ironpark/acp-go/acp1"
	"github.com/ironpark/acp-go/acp1/acp1test"
)

func TestCommandRule(t *testing.T) {
	tests := []struct {
		command, key string
	}{
		{"go test ./...", "bash:go test"},
		{"ls -la", "bash:ls"},
		{"git status", "bash:git status"},
		{"go test -run 'TestX' ./...", "bash:go test"},
		{"go test ./... && rm -rf /", "bash=go test ./... && rm -rf /"},
		{"echo $HOME", "bash=echo $HOME"},
		{"FOO=1 go test", "bash=FOO=1 go test"},
		{"cat a > b", "bash=cat a > b"},
	}
	for _, tt := range tests {
		if got := commandRule(tt.command).key; got != tt.key {
			t.Errorf("commandRule(%q).key = %q, want %q", tt.command, got, tt.key)
		}
	}

	sess := &session{}
	sess.remember("bash:go test", true)
	sess.remember("bash:go test -run", false)
	for command, want := range map[string]ruled{
		"go test ./...":         ruledAllow,
		"go test -run X":        ruledDeny,
		"go testify":            ruledNone,
		"go test ./... ; rm -r": ruledNone,
		"go vet ./...":          ruledNone,
	} {
		if got := sess.ruling(commandRule(command)); got != want {
			t.Errorf("ruling(%q) = %d, want %d", command, got, want)
		}
	}

	// A session that allowed every command before rules existed still does.
	legacy := &session{}
	legacy.remember("bash", true)
	if got := legacy.ruling(commandRule("rm -rf build")); got != ruledAllow {
		t.Errorf("legacy bash key: ruling = %d, want allow", got)
	}
}

// bashCall is a streamed reply that calls the bash tool once.
func bashCall(id, command string) string {
	return toolUseReply(id, "bash", fmt.Sprintf(`{"command":%q,"description":"run it"}`, command))
}

func TestPermissionChoices(t *testing.T) {
	dir := t.TempDir()
	fake := &fakeDeepSeek{responses: []string{
		bashCall("toolu_1", "echo one"),
		bashCall("toolu_2", "echo two"),
		bashCall("toolu_3", "pwd"),
		bashCall("toolu_4", "ls"),
		textReply("done", 10, 1),
	}}
	conn, client, store, newResp := startAgent(t, dir, fake)
	answers := []func(*acp1.RequestPermissionRequest) *acp1.RequestPermissionResponse{
		acp1test.AllowAlways, // echo one: allows `echo` commands
		func(*acp1.RequestPermissionRequest) *acp1.RequestPermissionResponse {
			return acp1.PermissionSelected(fullAccessOption) // pwd
		},
	}
	client.Permission = func(req *acp1.RequestPermissionRequest) *acp1.RequestPermissionResponse {
		answer := answers[0]
		answers = answers[1:]
		return answer(req)
	}

	resp, err := conn.Prompt(t.Context(), &acp1.PromptRequest{SessionID: newResp.SessionID, Prompt: []acp1.ContentBlock{acp1.TextBlock("go")}})
	if err != nil {
		t.Fatal(err)
	}
	if resp.StopReason != acp1.StopReasonEndTurn {
		t.Errorf("stop reason = %s", resp.StopReason)
	}
	// echo two is allowed by the first answer and ls by full access.
	perms := client.Permissions()
	if len(perms) != 2 {
		t.Fatalf("permission requests = %d, want 2", len(perms))
	}
	var names []string
	for _, o := range perms[0].Options {
		names = append(names, o.Name)
	}
	if want := []string{"Allow", "Always allow `echo` commands", "Allow all (Full Access)", "Reject", "Always reject `echo` commands"}; !slices.Equal(names, want) {
		t.Errorf("options = %q, want %q", names, want)
	}
	sess, _, _ := store.Get(t.Context(), newResp.SessionID)
	if sess.currentMode() != fullAccessMode {
		t.Errorf("mode = %s, want full-access", sess.currentMode())
	}
	if !client.seen("current_mode_update") {
		t.Errorf("mode change not sent: %v", client.tags())
	}
	for _, m := range fake.requests[4].Messages {
		for _, b := range m.Content {
			if b.Type == "tool_result" && b.IsError {
				t.Errorf("tool result %s failed: %s", b.ToolUseID, b.Content)
			}
		}
	}
}
