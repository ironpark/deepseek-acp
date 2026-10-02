package main

import (
	"context"
	"encoding/json/v2"
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"time"

	acp "github.com/ironpark/acp-go"
	"github.com/ironpark/acp-go/acp1"

	"github.com/ironpark/deepseek-acp/internal/deepseek"
)

// titleLength is how many characters of the first prompt a session's title
// keeps.
const titleLength = 60

// session holds the state the agent keeps per ACP session.
type session struct {
	cwd string

	mu      sync.Mutex
	mode    acp1.SessionModeID
	model   string
	effort  string
	history []deepseek.Message // the conversation, without the system prompt
	updated time.Time

	// planMode is on while the model plans before acting (/plan).
	planMode bool
	// goal is the session's long-running objective (/goal), if any.
	goal *goal
	// goalArmed allows automatic goal rounds; like the harness, it is not
	// saved, so a restored goal waits for /goal resume.
	goalArmed bool
	// goalRound is set while the model works on an automatic goal round
	// rather than a direct request from the user.
	goalRound bool

	// toldContext is the runtime context the model was last told about, so
	// a note is added only when it changes. The system prompt stays
	// byte-stable, which keeps DeepSeek's prefix cache warm.
	toldContext string
	// allowed and denied hold the rule keys the user chose to always allow
	// or reject.
	allowed map[string]bool
	denied  map[string]bool
	// todos is the plan the model keeps with todo_write.
	todos []acp1.PlanEntry
	title string

	// usage sums the tokens of every model call in the session, and
	// contextUsed is how much of the context window the last call filled.
	usage       tokenUsage
	contextUsed int
}

// tokenUsage counts tokens across a session's model calls.
type tokenUsage struct {
	Input      uint64 `json:"input"` // not counting cache reads and writes
	Output     uint64 `json:"output"`
	CacheRead  uint64 `json:"cacheRead,omitzero"`
	CacheWrite uint64 `json:"cacheWrite,omitzero"`
}

func newSession(cwd string, cfg *config) *session {
	return &session{cwd: cwd, mode: cfg.defaultMode, model: cfg.defaultModel, effort: cfg.defaultEffort}
}

func (s *session) messages() []deepseek.Message {
	s.mu.Lock()
	defer s.mu.Unlock()
	return slices.Clone(s.history)
}

// commit appends messages to the conversation. A message with the same role
// as the last one is merged into it, which a cancelled turn can cause, so the
// history always alternates roles as the API requires. Tool results stay
// first in a user message: they only ever follow an assistant message.
func (s *session) commit(messages ...deepseek.Message) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, m := range messages {
		if len(m.Content) == 0 {
			continue
		}
		if n := len(s.history); n > 0 && s.history[n-1].Role == m.Role {
			s.history[n-1].Content = slices.Concat(s.history[n-1].Content, m.Content)
			continue
		}
		s.history = append(s.history, m)
	}
	s.updated = time.Now()
}

// settings returns the model and reasoning effort for the next request.
// recordUsage adds a model call's tokens to the session's totals.
func (s *session) recordUsage(u deepseek.Usage) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.usage.Input += uint64(u.InputTokens)
	s.usage.Output += uint64(u.OutputTokens)
	s.usage.CacheRead += uint64(u.CacheReadInputTokens)
	s.usage.CacheWrite += uint64(u.CacheCreationInputTokens)
	if used := u.ContextTokens(); used > 0 {
		s.contextUsed = used
	}
}

// contextTokens is how much of the context window the conversation fills,
// as of the last model call.
func (s *session) contextTokens() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.contextUsed
}

// acpUsage reports the session's token totals for a prompt response.
// Input counts cache reads and writes too, as they are input tokens.
func (s *session) acpUsage() *acp1.Usage {
	s.mu.Lock()
	defer s.mu.Unlock()
	u := s.usage
	input := u.Input + u.CacheRead + u.CacheWrite
	usage := &acp1.Usage{InputTokens: input, OutputTokens: u.Output, TotalTokens: input + u.Output}
	if u.CacheRead > 0 {
		usage.CachedReadTokens = new(u.CacheRead)
	}
	if u.CacheWrite > 0 {
		usage.CachedWriteTokens = new(u.CacheWrite)
	}
	return usage
}

// replaceHistory replaces the conversation with a compacted one, whose first
// message states the current mode, and records its estimated size.
func (s *session) replaceHistory(history []deepseek.Message, contextUsed int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.history = history
	s.toldContext = s.runtimeContextLocked()
	s.contextUsed = contextUsed
	s.updated = time.Now()
}

func (s *session) settings() (model, effort string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.model, s.effort
}

func (s *session) currentMode() acp1.SessionModeID {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.mode
}

func (s *session) setMode(mode acp1.SessionModeID) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.mode = mode
}

// runtimeContextLocked describes the permission and plan modes for the
// model. The caller holds s.mu.
func (s *session) runtimeContextLocked() string {
	text := "Permission mode: " + modeContext[s.mode]
	if s.planMode {
		text += "\n\n" + planModeContext
	} else {
		text += "\nPlan mode: off."
	}
	return text
}

// runtimeContext is runtimeContextLocked for callers without the lock.
func (s *session) runtimeContext() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.runtimeContextLocked()
}

// contextChange returns the runtime context if the model has not been told
// about it yet, and marks it told.
func (s *session) contextChange() (string, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	text := s.runtimeContextLocked()
	if text == s.toldContext {
		return "", false
	}
	s.toldContext = text
	return text, true
}

func (s *session) inPlanMode() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.planMode
}

func (s *session) setPlanMode(on bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.planMode = on
}

func (s *session) setTodos(todos []acp1.PlanEntry) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.todos = todos
}

// setModel switches the model. Thinking signatures are only valid for the
// model that produced them, so they are dropped from the history, as the
// harness does on replay to a different model.
func (s *session) setModel(model string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.model == model {
		return
	}
	s.model = model
	for i := range s.history {
		for j := range s.history[i].Content {
			s.history[i].Content[j].Signature = ""
		}
	}
}

// setCwd moves the session to dir and reports whether it changed.
func (s *session) setCwd(dir string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	changed := dir != "" && dir != s.cwd
	if changed {
		s.cwd = dir
	}
	return changed
}

func (s *session) setEffort(effort string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.effort = effort
}

// resolve makes a path from the model absolute, relative to the session's
// working directory; ACP file methods take absolute paths.
func (s *session) resolve(path string) string {
	if path == "" {
		return s.cwd
	}
	if strings.HasPrefix(path, "~/") {
		if home, err := os.UserHomeDir(); err == nil {
			path = filepath.Join(home, path[2:])
		}
	}
	if filepath.IsAbs(path) {
		return filepath.Clean(path)
	}
	return filepath.Join(s.cwd, path)
}

// Modes. They mirror the harness's sandbox modes and approval policy:
// read-only refuses changes, ask asks before changes and commands,
// accept-edits changes files without asking but asks before commands, and
// full-access asks for nothing.
const (
	readOnlyMode    acp1.SessionModeID = "read-only"
	askMode         acp1.SessionModeID = "ask"
	acceptEditsMode acp1.SessionModeID = "accept-edits"
	fullAccessMode  acp1.SessionModeID = "full-access"
)

var modes = []acp1.SessionMode{
	{ID: readOnlyMode, Name: "Read Only", Description: new("Read and search only; no file changes")},
	{ID: askMode, Name: "Ask", Description: new("Ask before changing files or running commands")},
	{ID: acceptEditsMode, Name: "Accept Edits", Description: new("Change files without asking; ask before commands")},
	{ID: fullAccessMode, Name: "Full Access", Description: new("Change files and run commands without asking")},
}

var modeContext = map[acp1.SessionModeID]string{
	readOnlyMode:    "read-only: file changes are refused; commands need the user's approval. Do not try to modify files.",
	askMode:         "ask: file changes and commands need the user's approval.",
	acceptEditsMode: "accept-edits: file changes are applied without asking; commands need the user's approval.",
	fullAccessMode:  "full-access: file changes and commands run without asking.",
}

func validMode(id acp1.SessionModeID) bool {
	return slices.ContainsFunc(modes, func(m acp1.SessionMode) bool { return m.ID == id })
}

// SessionModes reports the session's modes; the embedded manager includes
// them in session/new and session/resume responses.
func (s *session) SessionModes() *acp1.SessionModeState {
	return &acp1.SessionModeState{CurrentModeID: s.currentMode(), AvailableModes: modes}
}

// Config options: the permission mode, which is also a session mode for
// clients without config options, and the model and reasoning effort, like
// the harness's ACP server.
const (
	modeOption   acp1.SessionConfigID = "mode"
	modelOption  acp1.SessionConfigID = "model"
	effortOption acp1.SessionConfigID = "reasoning_effort"
)

var efforts = []acp1.SessionConfigSelectOption{
	{Value: "off", Name: "Off", Description: new("No thinking")},
	{Value: "low", Name: "Low"},
	{Value: "high", Name: "High"},
	{Value: "max", Name: "Max"},
}

func validEffort(effort string) bool {
	return slices.ContainsFunc(efforts, func(e acp1.SessionConfigSelectOption) bool { return string(e.Value) == effort })
}

// SessionConfigOptions reports the session's config options; the embedded
// manager includes them in session/new and session/resume responses.
func (s *session) SessionConfigOptions() []acp1.SessionConfigOption {
	model, effort := s.settings()
	var models []acp1.SessionConfigSelectOption
	for _, m := range offeredModels {
		models = append(models, acp1.SessionConfigSelectOption{Value: acp1.SessionConfigValueID(m.id), Name: m.name})
	}
	if !slices.ContainsFunc(offeredModels, func(m modelInfo) bool { return m.id == model }) {
		models = append(models, acp1.SessionConfigSelectOption{Value: acp1.SessionConfigValueID(model), Name: model})
	}
	var modeValues []acp1.SessionConfigSelectOption
	for _, m := range modes {
		modeValues = append(modeValues, acp1.SessionConfigSelectOption{Value: acp1.SessionConfigValueID(m.ID), Name: m.Name, Description: m.Description})
	}
	return []acp1.SessionConfigOption{
		acp1.NewSessionConfigOption(acp1.SessionConfigOptionSelect{
			ID: modeOption, Name: "Permission Mode", CurrentValue: acp1.SessionConfigValueID(s.currentMode()),
			Options: acp1.SelectOptions(modeValues...), Category: new(acp1.SessionConfigOptionCategoryMode),
		}),
		acp1.NewSessionConfigOption(acp1.SessionConfigOptionSelect{
			ID: modelOption, Name: "Model", CurrentValue: acp1.SessionConfigValueID(model),
			Options: acp1.SelectOptions(models...), Category: new(acp1.SessionConfigOptionCategoryModel),
		}),
		acp1.NewSessionConfigOption(acp1.SessionConfigOptionSelect{
			ID: effortOption, Name: "Reasoning Effort", CurrentValue: acp1.SessionConfigValueID(effort),
			Options: acp1.SelectOptions(efforts...), Category: new(acp1.SessionConfigOptionCategoryThoughtLevel),
		}),
	}
}

// switchMode changes the session's permission mode and tells the client,
// both as the session mode and as the mode config option, so the two agree
// whichever way the mode changed.
func (a *deepseekAgent) switchMode(ctx context.Context, stream *acp1.SessionStream, sess *session, mode acp1.SessionModeID) error {
	if !validMode(mode) {
		return acp.InvalidParams(fmt.Sprintf("unknown mode %q", mode))
	}
	sess.setMode(mode)
	if err := stream.SendModeUpdate(ctx, mode); err != nil {
		return err
	}
	return stream.SendConfigUpdate(ctx, sess.SessionConfigOptions())
}

func (a *deepseekAgent) SetSessionMode(ctx context.Context, params *acp1.SetSessionModeRequest) (*acp1.SetSessionModeResponse, error) {
	sess, err := a.Lookup(ctx, params.SessionID)
	if err != nil {
		return nil, err
	}
	if err := a.switchMode(ctx, acp1.NewSessionStream(a.client, params.SessionID), sess, params.ModeID); err != nil {
		return nil, err
	}
	a.save(ctx, params.SessionID, sess)
	return &acp1.SetSessionModeResponse{}, nil
}

func (a *deepseekAgent) SetSessionConfigOption(ctx context.Context, params *acp1.SetSessionConfigOptionRequest) (*acp1.SetSessionConfigOptionResponse, error) {
	req, ok := acp1.ConfigChangeOf(params)
	if !ok || req.Boolean != nil {
		return nil, acp.InvalidParams("expected a select value")
	}
	sess, err := a.Lookup(ctx, req.SessionID)
	if err != nil {
		return nil, err
	}
	value := string(req.Value)
	switch req.ConfigID {
	case modeOption:
		if err := a.switchMode(ctx, acp1.NewSessionStream(a.client, req.SessionID), sess, acp1.SessionModeID(value)); err != nil {
			return nil, err
		}
	case modelOption:
		if value == "" {
			return nil, acp.InvalidParams("empty model")
		}
		sess.setModel(value)
	case effortOption:
		if !validEffort(value) {
			return nil, acp.InvalidParams(fmt.Sprintf("unknown reasoning effort %q", value))
		}
		sess.setEffort(value)
	default:
		return nil, acp.InvalidParams(fmt.Sprintf("unknown config option %q", req.ConfigID))
	}
	a.save(ctx, req.SessionID, sess)
	return &acp1.SetSessionConfigOptionResponse{ConfigOptions: sess.SessionConfigOptions()}, nil
}

// Persistence.

// savedSession is a session as its file holds it.
type savedSession struct {
	Cwd     string             `json:"cwd"`
	Mode    acp1.SessionModeID `json:"mode"`
	Model   string             `json:"model"`
	Effort  string             `json:"effort"`
	History []deepseek.Message `json:"history,omitzero"`
	Allowed []string           `json:"allowed,omitzero"`
	Denied  []string           `json:"denied,omitzero"`
	Todos   []acp1.PlanEntry   `json:"todos,omitzero"`
	Title   string             `json:"title,omitzero"`
	Updated time.Time          `json:"updated,omitzero"`
	Usage   tokenUsage         `json:"usage,omitzero"`
	Context int                `json:"contextUsed,omitzero"`
	Plan    bool               `json:"planMode,omitzero"`
	Goal    *goal              `json:"goal,omitzero"`
}

// MarshalJSON saves the session for the store; a turn may be changing it.
// It copies the fields under the lock and encodes them outside it, so a save
// does not hold up the turn.
func (s *session) MarshalJSON() ([]byte, error) {
	s.mu.Lock()
	saved := savedSession{
		Cwd: s.cwd, Mode: s.mode, Model: s.model, Effort: s.effort, History: slices.Clone(s.history),
		Allowed: slices.Sorted(maps.Keys(s.allowed)), Denied: slices.Sorted(maps.Keys(s.denied)), Todos: s.todos, Title: s.title, Updated: s.updated,
		Usage: s.usage, Context: s.contextUsed, Plan: s.planMode, Goal: s.goal.clone(),
	}
	s.mu.Unlock()
	return json.Marshal(saved)
}

// UnmarshalJSON restores a session the store saved.
func (s *session) UnmarshalJSON(data []byte) error {
	var saved savedSession
	if err := json.Unmarshal(data, &saved); err != nil {
		return err
	}
	s.cwd, s.mode, s.model, s.effort = saved.Cwd, saved.Mode, saved.Model, saved.Effort
	s.history, s.todos, s.title, s.updated = saved.History, saved.Todos, saved.Title, saved.Updated
	s.usage, s.contextUsed = saved.Usage, saved.Context
	s.planMode, s.goal = saved.Plan, saved.Goal
	for _, key := range saved.Allowed {
		s.remember(key, true)
	}
	for _, key := range saved.Denied {
		s.remember(key, false)
	}
	return nil
}

// SessionInfo describes the session for session/list.
func (s *session) SessionInfo() acp1.SessionInfo {
	s.mu.Lock()
	defer s.mu.Unlock()
	info := acp1.SessionInfo{Cwd: s.cwd}
	if !s.updated.IsZero() {
		info.UpdatedAt = new(s.updated.UTC().Format(time.RFC3339Nano))
	}
	if s.title != "" {
		info.Title = new(s.title)
	}
	return info
}

// setTitleFrom names the session after its first prompt, and returns the
// title if it named it.
func (s *session) setTitleFrom(prompt string) (string, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.title != "" {
		return "", false
	}
	title := []rune(strings.Join(strings.Fields(prompt), " "))
	s.title = string(title[:min(len(title), titleLength)])
	return s.title, s.title != ""
}

// ListSessions lists the saved sessions, newest first.
func (a *deepseekAgent) ListSessions(ctx context.Context, params *acp1.ListSessionsRequest) (*acp1.ListSessionsResponse, error) {
	return a.List(ctx, params)
}

// save writes a session changed outside a turn to the store; the manager
// saves the changes turns make. A failure is logged rather than failing the
// request that changed the session.
func (a *deepseekAgent) save(ctx context.Context, id acp1.SessionID, sess *session) {
	if err := a.Save(ctx, id, sess); err != nil {
		a.logger.Error("save session", "session", id, "error", err)
	}
}
