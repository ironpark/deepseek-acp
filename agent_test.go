package main

import (
	"encoding/json/v2"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/ironpark/acp-go/acp1"
	"github.com/ironpark/acp-go/acp1/acp1test"

	"github.com/ironpark/deepseek-acp/internal/deepseek"
)

// sse renders Anthropic-style stream events.
func sse(events ...string) string {
	var b strings.Builder
	for _, e := range events {
		var typ struct {
			Type string `json:"type"`
		}
		_ = json.Unmarshal([]byte(e), &typ)
		fmt.Fprintf(&b, "event: %s\ndata: %s\n\n", typ.Type, e)
	}
	return b.String()
}

// fakeDeepSeek serves scripted responses in order and records the requests.
type fakeDeepSeek struct {
	mu        sync.Mutex
	responses []string
	requests  []deepseek.Request
}

func (f *fakeDeepSeek) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/anthropic/v1/messages" || r.Header.Get("x-api-key") != "test-key" {
		http.Error(w, `{"error":{"message":"bad request"}}`, http.StatusBadRequest)
		return
	}
	body, _ := io.ReadAll(r.Body)
	var req deepseek.Request
	if err := json.Unmarshal(body, &req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	f.mu.Lock()
	f.requests = append(f.requests, req)
	resp := f.responses[0]
	f.responses = f.responses[1:]
	f.mu.Unlock()
	w.Header().Set("content-type", "text/event-stream")
	io.WriteString(w, resp)
}

// testClient records the updates it receives and allows every permission
// request once.
type testClient struct{ acp1test.Client }

// tags lists the kinds of the updates received so far, in order.
func (c *testClient) tags() []string {
	var tags []string
	for _, n := range c.Updates() {
		tags = append(tags, n.Update.Tag())
	}
	return tags
}

func (c *testClient) seen(tag string) bool { return slices.Contains(c.tags(), tag) }

// startAgent connects a test client to an agent backed by fake, and opens a
// session in dir.
func startAgent(t *testing.T, dir string, fake *fakeDeepSeek) (*acp1.ClientSideConnection, *testClient, acp1.SessionStore[*session], *acp1.NewSessionResponse) {
	t.Helper()
	server := httptest.NewServer(fake)
	t.Cleanup(server.Close)
	cfg := &config{
		apiKey: "test-key", baseURL: server.URL + "/anthropic",
		defaultModel: "deepseek-v4-flash", defaultEffort: "high", defaultMode: askMode,
		maxTokens: 1000, maxSteps: 10, compactRatio: 0.8, shell: "bash",
	}
	store, err := acp1.NewFileStore[*session](filepath.Join(dir, "sessions"))
	if err != nil {
		t.Fatal(err)
	}
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))
	llm := &deepseek.Client{BaseURL: cfg.baseURL, APIKey: cfg.apiKey}
	client := &testClient{}
	conn := acp1test.Connect(t, newAgent(cfg, store, llm, logger), client)
	if _, err := conn.Initialize(t.Context(), &acp1.InitializeRequest{ProtocolVersion: acp1.ProtocolVersion}); err != nil {
		t.Fatal(err)
	}
	resp, err := conn.NewSession(t.Context(), &acp1.NewSessionRequest{Cwd: dir})
	if err != nil {
		t.Fatal(err)
	}
	return conn, client, store, resp
}

// textReply is a streamed answer of text alone.
func textReply(text string, input, output int) string {
	return sse(
		fmt.Sprintf(`{"type":"message_start","message":{"usage":{"input_tokens":%d,"output_tokens":0}}}`, input),
		`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		fmt.Sprintf(`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":%q}}`, text),
		`{"type":"content_block_stop","index":0}`,
		fmt.Sprintf(`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":%d}}`, output),
		`{"type":"message_stop"}`,
	)
}

func TestCompaction(t *testing.T) {
	dir := t.TempDir()
	fake := &fakeDeepSeek{responses: []string{
		textReply("SUMMARY OF EARLIER WORK", 900_000, 50),
		textReply("done", 2_000, 10),
	}}
	conn, client, store, newResp := startAgent(t, dir, fake)

	// A conversation that fills 90% of the context window.
	sess, _, err := store.Get(t.Context(), newResp.SessionID)
	if err != nil {
		t.Fatal(err)
	}
	sess.history = []deepseek.Message{
		{Role: "user", Content: []deepseek.Block{deepseek.TextBlock("earlier question")}},
		{Role: "assistant", Content: []deepseek.Block{deepseek.TextBlock("earlier answer")}},
	}
	sess.contextUsed = 900_000
	if err := store.Set(t.Context(), newResp.SessionID, sess); err != nil {
		t.Fatal(err)
	}

	resp, err := conn.Prompt(t.Context(), &acp1.PromptRequest{SessionID: newResp.SessionID, Prompt: []acp1.ContentBlock{acp1.TextBlock("next question")}})
	if err != nil {
		t.Fatal(err)
	}
	if resp.StopReason != acp1.StopReasonEndTurn || resp.Usage == nil || resp.Usage.TotalTokens != 902_060 {
		t.Errorf("response = %+v, usage %+v", resp, resp.Usage)
	}
	if len(fake.requests) != 2 {
		t.Fatalf("requests = %d, want 2", len(fake.requests))
	}
	summarize := fake.requests[0].Messages
	if len(summarize) != 3 || !strings.Contains(summarize[2].Content[0].Text, "about to be compacted") {
		t.Errorf("summary request = %+v", summarize)
	}
	next := fake.requests[1].Messages
	if len(next) != 1 || len(next[0].Content) != 2 ||
		!strings.Contains(next[0].Content[0].Text, "SUMMARY OF EARLIER WORK") || next[0].Content[1].Text != "next question" {
		t.Errorf("request after compaction = %+v", next)
	}
	if !client.seen("tool_call") {
		t.Errorf("compaction not shown: %v", client.tags())
	}
	sess, _, _ = store.Get(t.Context(), newResp.SessionID)
	if got := sess.contextTokens(); got != 2_010 {
		t.Errorf("context used after the turn = %d, want 2010", got)
	}
}

func TestPromptRunsToolLoop(t *testing.T) {
	dir := t.TempDir()
	fake := &fakeDeepSeek{responses: []string{
		sse(
			`{"type":"message_start","message":{"usage":{"input_tokens":100,"output_tokens":0}}}`,
			`{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`,
			`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"I should write the file."}}`,
			`{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig-1"}}`,
			`{"type":"content_block_stop","index":0}`,
			`{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_1","name":"write","input":{}}}`,
			`{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"file_path\":\"hello.txt\","}}`,
			`{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"\"content\":\"hi\\n\"}"}}`,
			`{"type":"content_block_stop","index":1}`,
			`{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":20}}`,
			`{"type":"message_stop"}`,
		),
		sse(
			`{"type":"message_start","message":{"usage":{"input_tokens":150,"output_tokens":0}}}`,
			`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
			`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Wrote "}}`,
			`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello.txt."}}`,
			`{"type":"content_block_stop","index":0}`,
			`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}`,
			`{"type":"message_stop"}`,
		),
	}}
	ctx := t.Context()
	conn, client, _, newResp := startAgent(t, dir, fake)
	if len(newResp.ConfigOptions) != 3 || newResp.Modes == nil {
		t.Fatalf("session/new: want modes and 3 config options, got %+v", newResp)
	}
	resp, err := conn.Prompt(ctx, &acp1.PromptRequest{SessionID: newResp.SessionID, Prompt: []acp1.ContentBlock{acp1.TextBlock("write hello.txt")}})
	if err != nil {
		t.Fatal(err)
	}
	if resp.StopReason != acp1.StopReasonEndTurn {
		t.Errorf("stop reason = %s, want end_turn", resp.StopReason)
	}
	if u := resp.Usage; u == nil || u.InputTokens != 250 || u.OutputTokens != 25 || u.TotalTokens != 275 {
		t.Errorf("usage = %+v, want 250 in, 25 out", u)
	}

	data, err := os.ReadFile(filepath.Join(dir, "hello.txt"))
	if err != nil || string(data) != "hi\n" {
		t.Errorf("hello.txt = %q, %v; want %q", data, err, "hi\n")
	}
	if n := len(client.Permissions()); n != 1 {
		t.Errorf("permission requests = %d, want 1", n)
	}
	if got := client.Text(newResp.SessionID); got != "Wrote hello.txt." {
		t.Errorf("streamed text = %q", got)
	}
	for _, want := range []string{"agent_thought_chunk", "tool_call", "tool_call_update", "agent_message_chunk", "usage_update", "session_info_update"} {
		if !client.seen(want) {
			t.Errorf("updates %v: missing %s", client.tags(), want)
		}
	}
	// The write waits for permission as a pending call, then runs.
	var statuses []acp1.ToolCallStatus
	for _, n := range client.Updates() {
		if call, ok := n.Update.As[acp1.SessionUpdateToolCall](); ok {
			statuses = append(statuses, call.GetStatus())
		}
		if call, ok := n.Update.As[acp1.SessionUpdateToolCallUpdate](); ok {
			statuses = append(statuses, call.GetStatus())
		}
	}
	if want := []acp1.ToolCallStatus{acp1.ToolCallStatusPending, acp1.ToolCallStatusInProgress, acp1.ToolCallStatusCompleted}; !slices.Equal(statuses, want) {
		t.Errorf("tool call statuses = %v, want %v", statuses, want)
	}

	// The second request replays the thinking with its signature and pairs
	// the tool call with its result.
	if len(fake.requests) != 2 {
		t.Fatalf("requests = %d, want 2", len(fake.requests))
	}
	second := fake.requests[1]
	if second.OutputConfig == nil || second.OutputConfig.Effort != "high" || second.Thinking.Type != "enabled" {
		t.Errorf("thinking config = %+v %+v", second.Thinking, second.OutputConfig)
	}
	if len(second.Messages) != 3 {
		t.Fatalf("second request messages = %d, want 3", len(second.Messages))
	}
	assistant := second.Messages[1]
	if b := assistant.Content[0]; b.Type != "thinking" || *b.Thinking != "I should write the file." || b.Signature != "sig-1" {
		t.Errorf("replayed thinking = %+v", b)
	}
	if b := assistant.Content[1]; b.Type != "tool_use" || b.ID != "toolu_1" {
		t.Errorf("replayed tool_use = %+v", b)
	}
	if b := second.Messages[2].Content[0]; b.Type != "tool_result" || b.ToolUseID != "toolu_1" || b.IsError {
		t.Errorf("tool_result = %+v", b)
	}
	if !strings.Contains(second.Messages[0].Content[0].Text, "Permission mode: ask") {
		t.Errorf("first user block should carry the runtime context, got %q", second.Messages[0].Content[0].Text)
	}

	// The session was saved and can be listed.
	list, err := conn.ListSessions(ctx, &acp1.ListSessionsRequest{})
	if err != nil {
		t.Fatal(err)
	}
	if len(list.Sessions) != 1 || list.Sessions[0].Title == nil || *list.Sessions[0].Title != "write hello.txt" {
		t.Errorf("sessions = %+v", list.Sessions)
	}
}

func TestGlobRegexp(t *testing.T) {
	tests := []struct {
		pattern, path string
		want          bool
	}{
		{"*.go", "main.go", true},
		{"*.go", "a/main.go", false},
		{"**/*.go", "main.go", true},
		{"**/*.go", "a/b/main.go", true},
		{"src/**/*_test.ts", "src/x/y/a_test.ts", true},
		{"*.{ts,tsx}", "a.tsx", true},
		{"*.{ts,tsx}", "a.js", false},
		{"file?.txt", "file1.txt", true},
		{"[ab].md", "b.md", true},
		{"[!ab].md", "c.md", true},
	}
	for _, tt := range tests {
		re, err := globRegexp(tt.pattern)
		if err != nil {
			t.Fatal(err)
		}
		if got := re.MatchString(tt.path); got != tt.want {
			t.Errorf("glob %q on %q = %v, want %v", tt.pattern, tt.path, got, tt.want)
		}
	}
}

func TestCommitMergesRoles(t *testing.T) {
	sess := &session{}
	sess.commit(deepseek.Message{Role: "user", Content: []deepseek.Block{deepseek.TextBlock("a")}})
	sess.commit(
		deepseek.Message{Role: "assistant"}, // an interrupted reply with nothing kept
		deepseek.Message{Role: "user", Content: []deepseek.Block{deepseek.TextBlock("b")}},
		deepseek.Message{Role: "assistant", Content: []deepseek.Block{deepseek.TextBlock("c")}},
	)
	got := sess.messages()
	if len(got) != 2 || len(got[0].Content) != 2 || got[1].Role != "assistant" {
		t.Errorf("history = %+v", got)
	}
}

func TestGoGrepAndLocalCommand(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "a.go"), []byte("package a\nfunc Hello() {}\n"), 0o644)
	os.WriteFile(filepath.Join(dir, "b.txt"), []byte("Hello\n"), 0o644)
	lines, err := goGrep(t.Context(), dir, `Hello`, "*.go")
	if err != nil || len(lines) != 1 || !strings.HasSuffix(lines[0], "a.go:2:func Hello() {}") {
		t.Errorf("goGrep = %q, %v", lines, err)
	}

	a := &deepseekAgent{cfg: &config{shell: "bash"}}
	result, _, err := a.runLocally(t.Context(), "echo out; echo err >&2; exit 3", dir, bashTimeout)
	if err != errCommandFailed || result != "out\nerr\n[exit code: 3]" {
		t.Errorf("runLocally = %q, %v", result, err)
	}
	result, _, err = a.runLocally(t.Context(), "head -c 40000 /dev/zero | tr '\\0' x", dir, bashTimeout)
	if err != nil || !strings.HasPrefix(result, "[output truncated") || len(result) > outputLimit+100 {
		t.Errorf("runLocally long output: %d bytes, %v", len(result), err)
	}
}
