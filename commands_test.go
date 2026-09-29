package main

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/ironpark/acp-go/acp1"
)

// toolReply is a streamed response that calls tools, each given as name and
// JSON input.
func toolReply(calls ...[2]string) string {
	events := []string{`{"type":"message_start","message":{"usage":{"input_tokens":10,"output_tokens":0}}}`}
	for i, c := range calls {
		events = append(events,
			fmt.Sprintf(`{"type":"content_block_start","index":%d,"content_block":{"type":"tool_use","id":"toolu_%d","name":%q,"input":{}}}`, i, i, c[0]),
			fmt.Sprintf(`{"type":"content_block_delta","index":%d,"delta":{"type":"input_json_delta","partial_json":%q}}`, i, c[1]),
			fmt.Sprintf(`{"type":"content_block_stop","index":%d}`, i))
	}
	return sse(append(events,
		`{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":5}}`,
		`{"type":"message_stop"}`)...)
}

func prompt(t *testing.T, conn *acp1.ClientSideConnection, id acp1.SessionID, text string) *acp1.PromptResponse {
	t.Helper()
	resp, err := conn.Prompt(t.Context(), &acp1.PromptRequest{SessionID: id, Prompt: []acp1.ContentBlock{acp1.TextBlock(text)}})
	if err != nil {
		t.Fatal(err)
	}
	return resp
}

func TestParseCommand(t *testing.T) {
	tests := []struct {
		text, name, input string
	}{
		{"/goal ship the release", "goal", "ship the release"},
		{"  /PLAN\ndesign the cache", "plan", "design the cache"},
		{"/compact", "compact", ""},
		{"/usr/bin/env is broken", "", ""},
		{"hello /goal", "", ""},
	}
	for _, tt := range tests {
		cmd, input, _, ok := parseCommand([]acp1.ContentBlock{acp1.TextBlock(tt.text)})
		if got := ""; ok {
			got = cmd.name
			if got != tt.name || input != tt.input {
				t.Errorf("parseCommand(%q) = %q, %q; want %q, %q", tt.text, got, input, tt.name, tt.input)
			}
		} else if tt.name != "" {
			t.Errorf("parseCommand(%q) found no command; want %q", tt.text, tt.name)
		}
	}
}

func TestCommandsAdvertisedAndPermission(t *testing.T) {
	fake := &fakeDeepSeek{}
	conn, client, store, s := startAgent(t, t.TempDir(), fake)

	ctx, cancel := context.WithTimeout(t.Context(), 2*time.Second)
	defer cancel()
	if _, err := client.WaitFor(ctx, func(n *acp1.SessionNotification) bool {
		return n.SessionID == s.SessionID && n.Update.Tag() == "available_commands_update"
	}); err != nil {
		t.Error("commands were not advertised after session/new")
	}

	prompt(t, conn, s.SessionID, "/permission workspace-write")
	sess, _, _ := store.Get(t.Context(), s.SessionID)
	if sess.currentMode() != acceptEditsMode || !client.seen("current_mode_update") {
		t.Errorf("mode = %s, updates %v", sess.currentMode(), client.tags())
	}
	if len(fake.requests) != 0 {
		t.Errorf("a command reached the model: %d requests", len(fake.requests))
	}
}

func TestPlanMode(t *testing.T) {
	fake := &fakeDeepSeek{responses: []string{
		toolReply(
			[2]string{"write", `{"file_path":"x.txt","content":"x"}`},
			[2]string{"exit_plan_mode", `{"plan":"# Add caching\n\n1. Do it."}`},
		),
		textReply("Implementing.", 20, 5),
	}}
	dir := t.TempDir()
	conn, client, store, s := startAgent(t, dir, fake)

	prompt(t, conn, s.SessionID, "/plan add caching")
	if len(fake.requests) != 2 {
		t.Fatalf("requests = %d, want 2", len(fake.requests))
	}
	first := fake.requests[0].Messages[0].Content
	if !strings.Contains(first[0].Text, "Plan mode: on") || first[1].Text != "add caching" {
		t.Errorf("first request = %+v", first)
	}
	results := fake.requests[1].Messages[2].Content
	if !results[0].IsError || !strings.Contains(results[0].Content, "plan mode") {
		t.Errorf("write in plan mode = %+v, want refused", results[0])
	}
	if results[1].IsError || !strings.Contains(results[1].Content, "approved") {
		t.Errorf("exit_plan_mode = %+v, want approved", results[1])
	}
	sess, _, _ := store.Get(t.Context(), s.SessionID)
	if n := len(client.Permissions()); sess.inPlanMode() || n != 1 {
		t.Errorf("plan mode = %v, permission requests = %d; want off, 1", sess.inPlanMode(), n)
	}
}

func TestGoalRounds(t *testing.T) {
	fake := &fakeDeepSeek{responses: []string{
		textReply("Working.", 10, 5),
		toolReply([2]string{"update_goal", `{"action":"complete"}`}),
		textReply("Done.", 30, 5),
	}}
	conn, client, store, s := startAgent(t, t.TempDir(), fake)

	resp := prompt(t, conn, s.SessionID, "/goal ship the release")
	if resp.StopReason != acp1.StopReasonEndTurn {
		t.Errorf("stop reason = %s", resp.StopReason)
	}
	if len(fake.requests) != 3 {
		t.Fatalf("requests = %d, want 3 (two rounds)", len(fake.requests))
	}
	if last := fake.requests[1].Messages; !strings.Contains(last[len(last)-1].Content[0].Text, "Round: 2/256") {
		t.Errorf("second round prompt = %+v", last[len(last)-1])
	}
	sess, _, _ := store.Get(t.Context(), s.SessionID)
	if g := sess.currentGoal(); g == nil || g.Status != goalComplete || g.Rounds != 2 {
		t.Errorf("goal = %+v", g)
	}
	if text := client.Text(s.SessionID); !strings.HasPrefix(text, "Goal set.") {
		t.Errorf("reply = %q", text)
	}
}

func TestGoalPolicy(t *testing.T) {
	sess := &session{}
	if err := sess.setGoal("x", 0); err != nil {
		t.Fatal(err)
	}
	if err := sess.setGoal("y", 0); err == nil {
		t.Error("replaced an unfinished goal")
	}
	sess.goalRound = true
	if _, err := sess.updateGoal("pause", "", 0, ""); err == nil {
		t.Error("a goal round paused the goal")
	}
	if _, err := sess.updateGoal(goalBlocked, "", 0, "stuck"); err == nil {
		t.Errorf("blocked before %d rounds", minBlockedRounds)
	}
	sess.goal.Rounds = minBlockedRounds
	if g, err := sess.updateGoal(goalBlocked, "", 0, "stuck"); err != nil || g.Status != goalBlocked {
		t.Errorf("blocked = %+v, %v", g, err)
	}
	sess.goalRound = false
	sess.goal.MaxRounds = minBlockedRounds + 1
	if _, err := sess.updateGoal("resume", "", 0, ""); err != nil {
		t.Fatal(err)
	}
	if _, ok := sess.nextGoalRound(); !ok {
		t.Error("no round after resume")
	}
	if _, ok := sess.nextGoalRound(); ok || sess.goal.Status != goalBlocked {
		t.Errorf("round cap: goal = %+v", sess.goal)
	}
}

func TestPermissionModeConfigOption(t *testing.T) {
	conn, client, store, s := startAgent(t, t.TempDir(), &fakeDeepSeek{})
	if id := s.ConfigOptions[0].Variant().(acp1.SessionConfigOptionSelect).ID; id != modeOption {
		t.Errorf("first config option = %s, want %s", id, modeOption)
	}

	resp, err := conn.SetSessionConfigOption(t.Context(), new(acp1.NewSetSessionConfigOptionRequest(
		acp1.SetSessionConfigOptionRequestUntagged{SessionID: s.SessionID, ConfigID: modeOption, Value: "full-access"})))
	if err != nil {
		t.Fatal(err)
	}
	if got := resp.ConfigOptions[0].Variant().(acp1.SessionConfigOptionSelect).CurrentValue; got != "full-access" {
		t.Errorf("config option value = %s", got)
	}
	sess, _, _ := store.Get(t.Context(), s.SessionID)
	if sess.currentMode() != fullAccessMode || !client.seen("current_mode_update") {
		t.Errorf("mode = %s; mode update sent: %v", sess.currentMode(), client.seen("current_mode_update"))
	}

	if _, err := conn.SetSessionMode(t.Context(), &acp1.SetSessionModeRequest{SessionID: s.SessionID, ModeID: readOnlyMode}); err != nil {
		t.Fatal(err)
	}
	if !client.seen("config_option_update") {
		t.Error("session/set_mode sent no config option update")
	}
	if _, err := conn.SetSessionConfigOption(t.Context(), new(acp1.NewSetSessionConfigOptionRequest(
		acp1.SetSessionConfigOptionRequestUntagged{SessionID: s.SessionID, ConfigID: modeOption, Value: "yolo"}))); err == nil {
		t.Error("accepted an unknown mode")
	}
}
