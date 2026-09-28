package deepseek

import (
	"strings"
	"testing"
)

func TestReadStreamMalformedToolInput(t *testing.T) {
	stream := strings.Join([]string{
		`data: {"type":"message_start","message":{"usage":{"input_tokens":10}}}`,
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"t1","name":"bash"}}`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"command\":"}}`,
		`data: {"type":"message_delta","delta":{"stop_reason":"max_tokens"},"usage":{"output_tokens":3}}`,
		``,
	}, "\n\n")
	resp, err := readStream(strings.NewReader(stream), Handler{})
	if err != nil {
		t.Fatal(err)
	}
	calls := resp.ToolCalls()
	if len(calls) != 1 || string(calls[0].Input) != "{}" {
		t.Errorf("tool calls = %+v, want one with {} input", calls)
	}
	if resp.StopReason != "max_tokens" || resp.Usage.ContextTokens() != 13 {
		t.Errorf("stop = %s, usage = %+v", resp.StopReason, resp.Usage)
	}
}

func TestReadStreamError(t *testing.T) {
	stream := `data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":"par"}}` + "\n\n" +
		`data: {"type":"error","error":{"type":"overloaded_error","message":"busy"}}` + "\n\n"
	resp, err := readStream(strings.NewReader(stream), Handler{})
	if err == nil || !strings.Contains(err.Error(), "busy") {
		t.Fatalf("err = %v, want stream error", err)
	}
	if len(resp.Message.Content) != 1 || resp.Message.Content[0].Text != "par" {
		t.Errorf("partial response = %+v", resp.Message)
	}
}

func TestEndpoint(t *testing.T) {
	for base, want := range map[string]string{
		"":                                    "https://api.deepseek.com/anthropic/v1/messages",
		"https://api.deepseek.com/anthropic/": "https://api.deepseek.com/anthropic/v1/messages",
		"http://localhost:8080/v1":            "http://localhost:8080/v1/messages",
	} {
		if got := (&Client{BaseURL: base}).endpoint(); got != want {
			t.Errorf("endpoint(%q) = %q, want %q", base, got, want)
		}
	}
}
