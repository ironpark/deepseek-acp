package main

import (
	"cmp"
	"context"
	"encoding/json/jsontext"
	"encoding/json/v2"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/exec"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/ironpark/acp-go/acp1"
	"github.com/modelcontextprotocol/go-sdk/mcp"

	"github.com/ironpark/deepseek-acp/internal/deepseek"
)

// MCP servers: the client names them in session/new, load and resume, and
// their tools are offered to the model next to the built-in ones, as
// mcp__<server>__<tool>. The connections last while the session is open in
// this process; they are not saved.

const (
	// mcpConnectTimeout bounds connecting to a server and listing its tools.
	mcpConnectTimeout = 30 * time.Second
	// mcpResultLimit is how many bytes of a tool's result the model gets.
	mcpResultLimit = 100 << 10
	// toolNameLimit is the longest tool name the API takes.
	toolNameLimit = 64
)

// mcpTool is an MCP server's tool as the model sees it.
type mcpTool struct {
	server   string
	name     string // the tool's name on its server
	title    string
	readOnly bool
	session  *mcp.ClientSession
}

// mcpServers are a session's connected MCP servers and their tools.
type mcpServers struct {
	sessions []*mcp.ClientSession
	// tools are offered to the model in this order, which stays the same
	// for the connections' lifetime, so the request prefix stays cacheable.
	tools  []deepseek.Tool
	byName map[string]mcpTool
}

// connectMCP connects to the servers, in parallel, and lists their tools. A
// server that fails is logged and left out.
func connectMCP(ctx context.Context, servers []acp1.MCPServer, cwd string, logger *slog.Logger) *mcpServers {
	type connected struct {
		session *mcp.ClientSession
		tools   []*mcp.Tool
	}
	results := make([]connected, len(servers))
	var wg sync.WaitGroup
	for i, server := range servers {
		wg.Go(func() {
			name := mcpServerName(server)
			ctx, cancel := context.WithTimeout(ctx, mcpConnectTimeout)
			defer cancel()
			session, tools, err := connectMCPServer(ctx, server, cwd)
			if err != nil {
				logger.Warn("connect to an MCP server", "server", name, "error", err)
				return
			}
			results[i] = connected{session, tools}
		})
	}
	wg.Wait()

	m := &mcpServers{byName: map[string]mcpTool{}}
	for i, r := range results {
		if r.session == nil {
			continue
		}
		m.sessions = append(m.sessions, r.session)
		server := mcpServerName(servers[i])
		for _, tool := range r.tools {
			name := mcpToolName(server, tool.Name)
			if _, dup := m.byName[name]; dup {
				logger.Warn("MCP tool name taken; leaving the tool out", "server", server, "tool", tool.Name)
				continue
			}
			schema, err := json.Marshal(tool.InputSchema)
			if err != nil || tool.InputSchema == nil {
				schema = []byte(`{"type":"object"}`)
			}
			m.tools = append(m.tools, deepseek.Tool{
				Name:        name,
				Description: fmt.Sprintf("%s (from the %s MCP server)", strings.TrimSpace(tool.Description), server),
				InputSchema: jsontext.Value(schema),
			})
			t := mcpTool{server: server, name: tool.Name, title: tool.Title, session: r.session}
			if a := tool.Annotations; a != nil {
				t.readOnly = a.ReadOnlyHint
				if t.title == "" {
					t.title = a.Title
				}
			}
			m.byName[name] = t
		}
	}
	return m
}

func connectMCPServer(ctx context.Context, server acp1.MCPServer, cwd string) (*mcp.ClientSession, []*mcp.Tool, error) {
	var transport mcp.Transport
	switch s := server.Variant().(type) {
	case acp1.MCPServerStdio:
		cmd := exec.Command(s.Command, s.Args...)
		cmd.Dir = cwd
		cmd.Env = os.Environ()
		for _, v := range s.Env {
			cmd.Env = append(cmd.Env, v.Name+"="+v.Value)
		}
		transport = &mcp.CommandTransport{Command: cmd}
	case acp1.MCPServerHTTP:
		transport = &mcp.StreamableClientTransport{Endpoint: s.URL, HTTPClient: headerClient(s.Headers)}
	case acp1.MCPServerSSE:
		transport = &mcp.SSEClientTransport{Endpoint: s.URL, HTTPClient: headerClient(s.Headers)}
	default:
		return nil, nil, fmt.Errorf("unsupported MCP server type %q", server.Variant().Tag())
	}
	client := mcp.NewClient(&mcp.Implementation{Name: "deepseek-acp", Version: version}, nil)
	session, err := client.Connect(ctx, transport, nil)
	if err != nil {
		return nil, nil, err
	}
	var tools []*mcp.Tool
	for tool, err := range session.Tools(ctx, nil) {
		if err != nil {
			session.Close()
			return nil, nil, fmt.Errorf("list tools: %w", err)
		}
		tools = append(tools, tool)
	}
	return session, tools, nil
}

func mcpServerName(server acp1.MCPServer) string {
	switch s := server.Variant().(type) {
	case acp1.MCPServerStdio:
		return s.Name
	case acp1.MCPServerHTTP:
		return s.Name
	case acp1.MCPServerSSE:
		return s.Name
	case acp1.MCPServerACP:
		return s.Name
	}
	return "unknown"
}

// headerClient is an HTTP client that sends the server's headers.
func headerClient(headers []acp1.HTTPHeader) *http.Client {
	if len(headers) == 0 {
		return nil
	}
	return &http.Client{Transport: headerTransport{headers, http.DefaultTransport}}
}

type headerTransport struct {
	headers []acp1.HTTPHeader
	next    http.RoundTripper
}

func (t headerTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	req = req.Clone(req.Context())
	for _, h := range t.headers {
		req.Header.Set(h.Name, h.Value)
	}
	return t.next.RoundTrip(req)
}

var unsafeToolChars = regexp.MustCompile(`[^a-zA-Z0-9_-]+`)

// mcpToolName names a server's tool for the model, within the API's limits
// on tool names.
func mcpToolName(server, tool string) string {
	name := "mcp__" + unsafeToolChars.ReplaceAllString(server, "_") + "__" + unsafeToolChars.ReplaceAllString(tool, "_")
	return name[:min(len(name), toolNameLimit)]
}

// close ends the connections.
func (m *mcpServers) close() {
	if m == nil {
		return
	}
	for _, s := range m.sessions {
		s.Close()
	}
}

// setMCP connects the session to its client's MCP servers, replacing any
// earlier connections.
func (a *deepseekAgent) setMCP(ctx context.Context, sess *session, servers []acp1.MCPServer) {
	var m *mcpServers
	if len(servers) > 0 {
		m = connectMCP(ctx, servers, sess.cwd, a.logger)
	}
	sess.mu.Lock()
	old := sess.mcp
	sess.mcp = m
	sess.mu.Unlock()
	old.close()
}

// closeMCP ends the session's MCP connections.
func (s *session) closeMCP() {
	s.mu.Lock()
	m := s.mcp
	s.mcp = nil
	s.mu.Unlock()
	m.close()
}

// mcpTools are the session's MCP tools for the model.
func (s *session) mcpTools() []deepseek.Tool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.mcp == nil {
		return nil
	}
	return s.mcp.tools
}

func (s *session) lookupMCPTool(name string) (mcpTool, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.mcp == nil {
		return mcpTool{}, false
	}
	t, ok := s.mcp.byName[name]
	return t, ok
}

// mcpAction makes a call of an MCP tool into an action. A tool that does not
// declare itself read-only asks for permission like a command.
func (a *deepseekAgent) mcpAction(sess *session, call deepseek.Block, tool mcpTool) (*action, error) {
	var args map[string]any
	if len(call.Input) > 0 {
		if err := json.Unmarshal(call.Input, &args); err != nil {
			return nil, fmt.Errorf("invalid arguments: %w", err)
		}
	}
	title := fmt.Sprintf("%s: %s", tool.server, cmp.Or(tool.title, tool.name))
	kind := acp1.ToolKindOther
	if tool.readOnly {
		kind = acp1.ToolKindRead
	}
	return &action{
		title: title,
		kind:  kind,
		gated: !tool.readOnly,
		run: func(ctx context.Context, stream *acp1.SessionStream, id acp1.ToolCallID) (string, []acp1.ToolCallContent, error) {
			if !tool.readOnly {
				if err := a.permit(ctx, stream, sess, id, toolRule(call.Name), false); err != nil {
					return "", nil, err
				}
			}
			return callMCPTool(ctx, tool, args)
		},
	}, nil
}

var errMCPTool = errors.New("the MCP tool reported an error")

func callMCPTool(ctx context.Context, tool mcpTool, args map[string]any) (string, []acp1.ToolCallContent, error) {
	res, err := tool.session.CallTool(ctx, &mcp.CallToolParams{Name: tool.name, Arguments: args})
	if err != nil {
		return "", nil, err
	}
	var b strings.Builder
	for _, c := range res.Content {
		if b.Len() > 0 {
			b.WriteString("\n")
		}
		switch c := c.(type) {
		case *mcp.TextContent:
			b.WriteString(c.Text)
		case *mcp.ImageContent:
			fmt.Fprintf(&b, "[image: %s]", c.MIMEType)
		case *mcp.AudioContent:
			fmt.Fprintf(&b, "[audio: %s]", c.MIMEType)
		case *mcp.ResourceLink:
			fmt.Fprintf(&b, "[resource_link name=%s uri=%s]", c.Name, c.URI)
		case *mcp.EmbeddedResource:
			if r := c.Resource; r != nil && r.Text != "" {
				fmt.Fprintf(&b, "<resource uri=%q>\n%s\n</resource>", r.URI, r.Text)
			} else if r != nil {
				fmt.Fprintf(&b, "[resource uri=%s]", r.URI)
			}
		}
	}
	if b.Len() == 0 && res.StructuredContent != nil {
		if data, err := json.Marshal(res.StructuredContent); err == nil {
			b.Write(data)
		}
	}
	result := b.String()
	if result == "" {
		result = "(no output)"
	}
	if len(result) > mcpResultLimit {
		result = strings.ToValidUTF8(result[:mcpResultLimit], "") + "\n[output truncated]"
	}
	content := []acp1.ToolCallContent{acp1.ToolText(result)}
	if res.IsError {
		return result, content, errMCPTool
	}
	return result, content, nil
}
