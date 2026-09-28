package main

import (
	"context"
	"encoding/json/v2"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/ironpark/acp-go/acp1"

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

// testClient is an ACP client that records updates and allows every
// permission request.
type testClient struct {
	mu          sync.Mutex
	updates     []string
	text        strings.Builder
	permissions int
}

func (c *testClient) SessionUpdate(_ context.Context, n *acp1.SessionNotification) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.updates = append(c.updates, n.Update.Tag())
	if chunk, ok := n.Update.As[acp1.SessionUpdateAgentMessageChunk](); ok {
		if text, ok := acp1.TextOf(chunk.Content); ok {
			c.text.WriteString(text)
		}
	}
	return nil
}

func (c *testClient) RequestPermission(_ context.Context, p *acp1.RequestPermissionRequest) (*acp1.RequestPermissionResponse, error) {
	c.mu.Lock()
	c.permissions++
	c.mu.Unlock()
	return acp1.PermissionSelected(p.Options[0].OptionID), nil
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
	server := httptest.NewServer(fake)
	defer server.Close()

	cfg := &config{
		apiKey: "test-key", baseURL: server.URL + "/anthropic",
		defaultModel: "deepseek-v4-flash", defaultEffort: "high", defaultMode: askMode,
		maxTokens: 1000, maxSteps: 10, shell: "bash",
	}
	store, err := acp1.NewFileStore[*session](filepath.Join(dir, "sessions"))
	if err != nil {
		t.Fatal(err)
	}
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))
	client := &testClient{}

	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	llm := &deepseek.Client{BaseURL: cfg.baseURL, APIKey: cfg.apiKey}
	_, conn := acp1.Pipe(ctx, newAgent(cfg, store, llm, logger), func(*acp1.ClientSideConnection) acp1.Client { return client })

	if _, err := conn.Initialize(ctx, &acp1.InitializeRequest{ProtocolVersion: acp1.ProtocolVersion}); err != nil {
		t.Fatal(err)
	}
	newResp, err := conn.NewSession(ctx, &acp1.NewSessionRequest{Cwd: dir})
	if err != nil {
		t.Fatal(err)
	}
	if len(newResp.ConfigOptions) != 2 || newResp.Modes == nil {
		t.Fatalf("session/new: want modes and 2 config options, got %+v", newResp)
	}
	resp, err := conn.Prompt(ctx, &acp1.PromptRequest{SessionID: newResp.SessionID, Prompt: []acp1.ContentBlock{acp1.TextBlock("write hello.txt")}})
	if err != nil {
		t.Fatal(err)
	}
	if resp.StopReason != acp1.StopReasonEndTurn {
		t.Errorf("stop reason = %s, want end_turn", resp.StopReason)
	}

	data, err := os.ReadFile(filepath.Join(dir, "hello.txt"))
	if err != nil || string(data) != "hi\n" {
		t.Errorf("hello.txt = %q, %v; want %q", data, err, "hi\n")
	}
	if client.permissions != 1 {
		t.Errorf("permission requests = %d, want 1", client.permissions)
	}
	if got := client.text.String(); got != "Wrote hello.txt." {
		t.Errorf("streamed text = %q", got)
	}
	updates := strings.Join(client.updates, ",")
	for _, want := range []string{"agent_thought_chunk", "tool_call", "tool_call_update", "agent_message_chunk", "usage_update"} {
		if !strings.Contains(updates, want) {
			t.Errorf("updates %s: missing %s", updates, want)
		}
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
