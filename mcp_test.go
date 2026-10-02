package main

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"slices"
	"testing"

	"github.com/ironpark/acp-go/acp1"
	"github.com/modelcontextprotocol/go-sdk/mcp"

	"github.com/ironpark/deepseek-acp/internal/deepseek"
)

// toolUseReply is a streamed reply that calls one tool with input.
func toolUseReply(id, name, input string) string {
	return sse(
		`{"type":"message_start","message":{"usage":{"input_tokens":10,"output_tokens":0}}}`,
		fmt.Sprintf(`{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":%q,"name":%q,"input":{}}}`, id, name),
		fmt.Sprintf(`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":%q}}`, input),
		`{"type":"content_block_stop","index":0}`,
		`{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":5}}`,
		`{"type":"message_stop"}`,
	)
}

// TestMain runs the test binary as a stdio MCP server when asked to, for
// TestMCPStdio.
func TestMain(m *testing.M) {
	if os.Getenv("DEEPSEEK_ACP_TEST_MCP") == "1" {
		server := newMCPTestServer()
		if err := server.Run(context.Background(), &mcp.StdioTransport{}); err != nil {
			os.Exit(1)
		}
		os.Exit(0)
	}
	os.Exit(m.Run())
}

// mcpTestServer serves an MCP server over streamable HTTP with a read-only
// echo tool and a shout tool that changes nothing but does not say so.
func mcpTestServer(t *testing.T) string {
	server := newMCPTestServer()
	httpServer := httptest.NewServer(mcp.NewStreamableHTTPHandler(func(*http.Request) *mcp.Server { return server }, nil))
	t.Cleanup(httpServer.Close)
	return httpServer.URL
}

func newMCPTestServer() *mcp.Server {
	server := mcp.NewServer(&mcp.Implementation{Name: "test", Version: "1"}, nil)
	type input struct {
		Text string `json:"text"`
	}
	mcp.AddTool(server, &mcp.Tool{Name: "echo", Description: "Echo text.", Annotations: &mcp.ToolAnnotations{ReadOnlyHint: true}},
		func(_ context.Context, _ *mcp.CallToolRequest, in input) (*mcp.CallToolResult, any, error) {
			return &mcp.CallToolResult{Content: []mcp.Content{&mcp.TextContent{Text: "echo: " + in.Text}}}, nil, nil
		})
	mcp.AddTool(server, &mcp.Tool{Name: "shout", Description: "Shout text."},
		func(_ context.Context, _ *mcp.CallToolRequest, in input) (*mcp.CallToolResult, any, error) {
			return &mcp.CallToolResult{Content: []mcp.Content{&mcp.TextContent{Text: in.Text + "!"}}}, nil, nil
		})
	return server
}

func TestMCPTools(t *testing.T) {
	dir := t.TempDir()
	fake := &fakeDeepSeek{responses: []string{
		toolUseReply("toolu_1", "mcp__my-server__echo", `{"text":"hi"}`),
		toolUseReply("toolu_2", "mcp__my-server__shout", `{"text":"hey"}`),
		textReply("done", 10, 1),
	}}
	conn, client, store, _ := startAgent(t, dir, fake)
	resp, err := conn.NewSession(t.Context(), &acp1.NewSessionRequest{Cwd: dir, MCPServers: []acp1.MCPServer{
		acp1.NewMCPServer(acp1.MCPServerHTTP{Name: "my-server", URL: mcpTestServer(t)}),
	}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := conn.Prompt(t.Context(), &acp1.PromptRequest{SessionID: resp.SessionID, Prompt: []acp1.ContentBlock{acp1.TextBlock("go")}}); err != nil {
		t.Fatal(err)
	}

	var names []string
	for _, tool := range fake.requests[0].Tools {
		names = append(names, tool.Name)
	}
	if !slices.Contains(names, "mcp__my-server__echo") || !slices.Contains(names, "mcp__my-server__shout") {
		t.Errorf("tools offered = %v", names)
	}
	// Only shout, which is not read-only, asks.
	if n := len(client.Permissions()); n != 1 {
		t.Errorf("permission requests = %d, want 1", n)
	}
	results := map[string]deepseek.Block{}
	for _, m := range fake.requests[2].Messages {
		for _, b := range m.Content {
			if b.Type == "tool_result" {
				results[b.ToolUseID] = b
			}
		}
	}
	if r := results["toolu_1"]; r.IsError || r.Content != "echo: hi" {
		t.Errorf("echo result = %+v", r)
	}
	if r := results["toolu_2"]; r.IsError || r.Content != "hey!" {
		t.Errorf("shout result = %+v", r)
	}

	if _, err := conn.CloseSession(t.Context(), &acp1.CloseSessionRequest{SessionID: resp.SessionID}); err != nil {
		t.Fatal(err)
	}
	sess, _, _ := store.Get(t.Context(), resp.SessionID)
	if sess.mcpTools() != nil {
		t.Error("MCP connections still open after session/close")
	}
}

func TestMCPStdio(t *testing.T) {
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	sess := &session{cwd: t.TempDir()}
	a := &deepseekAgent{logger: slog.New(slog.NewTextHandler(io.Discard, nil))}
	a.setMCP(t.Context(), sess, []acp1.MCPServer{acp1.NewMCPServer(acp1.MCPServerStdio{
		Name: "local", Command: exe, Env: []acp1.EnvVariable{{Name: "DEEPSEEK_ACP_TEST_MCP", Value: "1"}},
	})})
	defer sess.closeMCP()
	tool, ok := sess.lookupMCPTool("mcp__local__echo")
	if !ok {
		t.Fatalf("tools = %+v", sess.mcpTools())
	}
	result, _, err := callMCPTool(t.Context(), tool, map[string]any{"text": "stdio"})
	if err != nil || result != "echo: stdio" {
		t.Errorf("echo = %q, %v", result, err)
	}
}
