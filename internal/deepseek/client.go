// Package deepseek is a minimal streaming client for DeepSeek's
// Anthropic-compatible Messages API (https://api.deepseek.com/anthropic),
// the endpoint deepseek-harness talks to. It uses only the standard library.
package deepseek

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json/jsontext"
	"encoding/json/v2"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// DefaultBaseURL is DeepSeek's Anthropic-compatible endpoint.
const DefaultBaseURL = "https://api.deepseek.com/anthropic"

// Client sends requests to the Messages API.
type Client struct {
	BaseURL    string
	APIKey     string
	HTTPClient *http.Client // nil means http.DefaultClient

	// Retries is how many times a request that failed before streaming
	// started (rate limit, overload, server error, network) is retried.
	Retries int
}

// Message is one entry of a conversation. Role is user or assistant.
type Message struct {
	Role    string  `json:"role"`
	Content []Block `json:"content"`
}

// Block is one content block. Type decides which fields are set:
//
//	text        Text
//	thinking    Thinking, Signature
//	tool_use    ID, Name, Input
//	tool_result ToolUseID, Content, IsError
//	image       Source
type Block struct {
	Type      string         `json:"type"`
	Text      string         `json:"text,omitzero"`
	Thinking  *string        `json:"thinking,omitzero"` // a pointer: sent even when empty, as the API requires
	Signature string         `json:"signature,omitzero"`
	ID        string         `json:"id,omitzero"`
	Name      string         `json:"name,omitzero"`
	Input     jsontext.Value `json:"input,omitzero"`
	ToolUseID string         `json:"tool_use_id,omitzero"`
	Content   string         `json:"content,omitzero"`
	IsError   bool           `json:"is_error,omitzero"`
	Source    *ImageSource   `json:"source,omitzero"`
}

// ImageSource is an inline base64 image.
type ImageSource struct {
	Type      string `json:"type"` // base64
	MediaType string `json:"media_type"`
	Data      string `json:"data"`
}

// TextBlock returns a text block.
func TextBlock(text string) Block { return Block{Type: "text", Text: text} }

// ToolResultBlock returns the result of the tool call id.
func ToolResultBlock(id, content string, isError bool) Block {
	return Block{Type: "tool_result", ToolUseID: id, Content: content, IsError: isError}
}

// Tool describes a function the model may call.
type Tool struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	InputSchema jsontext.Value `json:"input_schema"`
}

// Thinking turns the model's reasoning on or off.
type Thinking struct {
	Type string `json:"type"` // enabled or disabled
}

// OutputConfig sets the reasoning effort: low, high or max.
type OutputConfig struct {
	Effort string `json:"effort"`
}

// Request is a Messages API request.
type Request struct {
	Model        string        `json:"model"`
	MaxTokens    int           `json:"max_tokens"`
	System       string        `json:"system,omitzero"`
	Messages     []Message     `json:"messages"`
	Tools        []Tool        `json:"tools,omitzero"`
	Thinking     *Thinking     `json:"thinking,omitzero"`
	OutputConfig *OutputConfig `json:"output_config,omitzero"`
	Stream       bool          `json:"stream"`
}

// Usage counts the tokens of one response.
type Usage struct {
	InputTokens              int `json:"input_tokens"`
	OutputTokens             int `json:"output_tokens"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
}

// ContextTokens is how much of the context window the conversation fills
// after this response.
func (u Usage) ContextTokens() int {
	return u.InputTokens + u.CacheReadInputTokens + u.CacheCreationInputTokens + u.OutputTokens
}

// Response is a streamed response put back together.
type Response struct {
	Message    Message // the assistant message
	StopReason string  // end_turn, stop_sequence, tool_use, max_tokens, refusal
	Usage      Usage
}

// ToolCalls returns the tool_use blocks of the response.
func (r *Response) ToolCalls() []Block {
	var calls []Block
	for _, b := range r.Message.Content {
		if b.Type == "tool_use" {
			calls = append(calls, b)
		}
	}
	return calls
}

// Handler receives the response as it streams.
type Handler struct {
	OnText     func(string) error
	OnThinking func(string) error
}

// APIError is a failed request.
type APIError struct {
	Status     int
	Message    string
	RetryAfter time.Duration
}

func (e *APIError) Error() string {
	return fmt.Sprintf("deepseek: %d %s: %s", e.Status, http.StatusText(e.Status), e.Message)
}

func (e *APIError) retryable() bool {
	return e.Status == http.StatusTooManyRequests || e.Status >= 500
}

// Stream sends the request and streams the response to h. On an error after
// streaming started, it returns the partial response too, so the caller can
// keep what the user already saw. Cancelling ctx aborts the request.
func (c *Client) Stream(ctx context.Context, req Request, h Handler) (*Response, error) {
	req.Stream = true
	body, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}
	var resp *http.Response
	for attempt := 0; ; attempt++ {
		resp, err = c.post(ctx, body)
		if err == nil {
			break
		}
		wait, ok := c.retryDelay(err, attempt)
		if !ok {
			return nil, err
		}
		select {
		case <-ctx.Done():
			return nil, context.Cause(ctx)
		case <-time.After(wait):
		}
	}
	defer resp.Body.Close()
	return readStream(resp.Body, h)
}

func (c *Client) retryDelay(err error, attempt int) (time.Duration, bool) {
	if attempt >= c.Retries || errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return 0, false
	}
	wait := time.Second << attempt
	if apiErr, ok := errors.AsType[*APIError](err); ok {
		if !apiErr.retryable() {
			return 0, false
		}
		wait = max(wait, apiErr.RetryAfter)
	}
	return min(wait, 30*time.Second), true
}

func (c *Client) endpoint() string {
	base := strings.TrimRight(c.BaseURL, "/")
	if base == "" {
		base = DefaultBaseURL
	}
	if strings.HasSuffix(base, "/v1") {
		return base + "/messages"
	}
	return base + "/v1/messages"
}

func (c *Client) post(ctx context.Context, body []byte) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.endpoint(), bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("x-api-key", c.APIKey)
	req.Header.Set("anthropic-version", "2023-06-01")
	req.Header.Set("content-type", "application/json")
	req.Header.Set("accept", "text/event-stream")

	hc := c.HTTPClient
	if hc == nil {
		hc = http.DefaultClient
	}
	resp, err := hc.Do(req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode/100 == 2 {
		return resp, nil
	}
	defer resp.Body.Close()
	text, _ := io.ReadAll(io.LimitReader(resp.Body, 8<<10))
	apiErr := &APIError{Status: resp.StatusCode, Message: errorMessage(text)}
	if s, err := strconv.Atoi(resp.Header.Get("retry-after")); err == nil {
		apiErr.RetryAfter = time.Duration(s) * time.Second
	}
	return nil, apiErr
}

// errorMessage pulls the message out of an Anthropic-style error body.
func errorMessage(body []byte) string {
	var e struct {
		Error struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if json.Unmarshal(body, &e) == nil && e.Error.Message != "" {
		return e.Error.Message
	}
	return string(bytes.TrimSpace(body))
}

// event is one server-sent event of a streamed response.
type event struct {
	Type    string `json:"type"`
	Index   int    `json:"index"`
	Message *struct {
		Usage Usage `json:"usage"`
	} `json:"message"`
	ContentBlock *struct {
		Type     string `json:"type"`
		ID       string `json:"id"`
		Name     string `json:"name"`
		Text     string `json:"text"`
		Thinking string `json:"thinking"`
	} `json:"content_block"`
	Delta *struct {
		Type        string `json:"type"`
		Text        string `json:"text"`
		Thinking    string `json:"thinking"`
		Signature   string `json:"signature"`
		PartialJSON string `json:"partial_json"`
		StopReason  string `json:"stop_reason"`
	} `json:"delta"`
	Usage *Usage `json:"usage"`
	Error *struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"error"`
}

// partial is a content block being streamed. Its text is the block's text,
// thinking or tool input JSON, by its type.
type partial struct {
	block     Block
	text      strings.Builder
	signature strings.Builder
}

func (p *partial) finish() Block {
	b := p.block
	switch b.Type {
	case "text":
		b.Text = p.text.String()
	case "thinking":
		thinking := p.text.String()
		b.Thinking, b.Signature = &thinking, p.signature.String()
	case "tool_use":
		// Malformed arguments are replaced by an empty object, as the
		// harness does, so the history stays valid.
		input := jsontext.Value(p.text.String())
		if len(bytes.TrimSpace(input)) == 0 || !input.IsValid() {
			input = jsontext.Value("{}")
		}
		b.Input = input
	}
	return b
}

func readStream(body io.Reader, h Handler) (*Response, error) {
	var (
		resp   = &Response{Message: Message{Role: "assistant"}}
		blocks []*partial
	)
	// assemble puts the blocks streamed so far into the response.
	assemble := func() *Response {
		resp.Message.Content = resp.Message.Content[:0]
		for _, p := range blocks {
			if p == nil {
				continue
			}
			b := p.finish()
			if b.Type == "text" && b.Text == "" {
				continue
			}
			resp.Message.Content = append(resp.Message.Content, b)
		}
		return resp
	}

	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 0, 64<<10), 16<<20)
	for scanner.Scan() {
		data, ok := bytes.CutPrefix(scanner.Bytes(), []byte("data:"))
		if !ok {
			continue // event: lines, comments and blank lines
		}
		data = bytes.TrimSpace(data)
		if len(data) == 0 || string(data) == "[DONE]" {
			continue
		}
		var ev event
		if err := json.Unmarshal(data, &ev); err != nil {
			return assemble(), fmt.Errorf("deepseek: reading stream: %w", err)
		}
		switch ev.Type {
		case "message_start":
			if ev.Message != nil {
				resp.Usage = ev.Message.Usage
			}
		case "content_block_start":
			for len(blocks) <= ev.Index {
				blocks = append(blocks, nil)
			}
			p := &partial{}
			if cb := ev.ContentBlock; cb != nil {
				p.block = Block{Type: cb.Type, ID: cb.ID, Name: cb.Name}
				p.text.WriteString(cb.Text + cb.Thinking)
			}
			blocks[ev.Index] = p
		case "content_block_delta":
			if ev.Index >= len(blocks) || blocks[ev.Index] == nil || ev.Delta == nil {
				continue
			}
			p, d := blocks[ev.Index], ev.Delta
			switch d.Type {
			case "text_delta":
				p.text.WriteString(d.Text)
				if h.OnText != nil && d.Text != "" {
					if err := h.OnText(d.Text); err != nil {
						return assemble(), err
					}
				}
			case "thinking_delta":
				p.text.WriteString(d.Thinking)
				if h.OnThinking != nil && d.Thinking != "" {
					if err := h.OnThinking(d.Thinking); err != nil {
						return assemble(), err
					}
				}
			case "signature_delta":
				p.signature.WriteString(d.Signature)
			case "input_json_delta":
				p.text.WriteString(d.PartialJSON)
			}
		case "message_delta":
			if ev.Delta != nil && ev.Delta.StopReason != "" {
				resp.StopReason = ev.Delta.StopReason
			}
			if ev.Usage != nil {
				// message_delta carries the final output count; the input
				// counts come from message_start unless repeated here.
				resp.Usage.OutputTokens = ev.Usage.OutputTokens
				if ev.Usage.InputTokens > 0 {
					resp.Usage.InputTokens = ev.Usage.InputTokens
				}
				if ev.Usage.CacheReadInputTokens > 0 {
					resp.Usage.CacheReadInputTokens = ev.Usage.CacheReadInputTokens
				}
			}
		case "error":
			msg := "unknown error"
			if ev.Error != nil {
				msg = ev.Error.Type + ": " + ev.Error.Message
			}
			return assemble(), fmt.Errorf("deepseek: stream error: %s", msg)
		}
	}
	if err := scanner.Err(); err != nil {
		return assemble(), err
	}
	assemble()
	if resp.StopReason == "" {
		return resp, errors.New("deepseek: stream ended before the response finished")
	}
	return resp, nil
}
