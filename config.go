package main

import (
	"cmp"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/ironpark/acp-go/acp1"

	"github.com/ironpark/deepseek-acp/internal/deepseek"
)

// modelInfo describes a model offered in the model picker.
type modelInfo struct {
	id            string
	name          string
	contextWindow int
	images        bool // accepts image input
}

// knownModels is the catalog of the harness's ACP profile.
var knownModels = []modelInfo{
	{id: "deepseek-v4-flash", name: "DeepSeek V4 Flash", contextWindow: 1_000_000, images: true},
	{id: "deepseek-v4-pro", name: "DeepSeek V4 Pro", contextWindow: 1_000_000},
}

// offeredModels are the models in the model picker: knownModels, or
// DEEPSEEK_MODELS. It is set once at startup, before any session exists, so
// sessions restored from disk can report their options too.
var offeredModels = knownModels

// config is the agent's configuration, read from the environment.
type config struct {
	apiKey        string
	baseURL       string // empty means deepseek.DefaultBaseURL
	defaultModel  string
	defaultEffort string
	defaultMode   acp1.SessionModeID
	maxTokens     int
	maxSteps      int
	compactRatio  float64 // compact at this share of the context window; 0 never
	sessionDir    string
	shell         string
}

const envHelp = `Environment:
  DEEPSEEK_API_KEY         required; the DeepSeek API key
  DEEPSEEK_BASE_URL        default ` + deepseek.DefaultBaseURL + `
  DEEPSEEK_MODEL           default model; default deepseek-v4-flash
  DEEPSEEK_MODELS          comma-separated model ids to offer; default deepseek-v4-flash,deepseek-v4-pro
  DEEPSEEK_REASONING_EFFORT off, low, high or max; default high
  DEEPSEEK_MAX_TOKENS      max output tokens per response; default 256000
  DEEPSEEK_ACP_MODE        initial mode: read-only, ask, accept-edits or full-access; default ask
  DEEPSEEK_ACP_MAX_STEPS   max model calls per prompt; default 200
  DEEPSEEK_ACP_COMPACT_RATIO share of the context window at which the conversation is summarized; 0 turns it off; default 0.8
  DEEPSEEK_ACP_SESSIONS    session directory; default deepseek-acp/sessions in the user cache directory
  DEEPSEEK_ACP_SHELL       shell for the bash tool when run locally; default bash
`

func loadConfig() (*config, error) {
	cfg := &config{
		apiKey:        os.Getenv("DEEPSEEK_API_KEY"),
		baseURL:       os.Getenv("DEEPSEEK_BASE_URL"),
		defaultEffort: cmp.Or(os.Getenv("DEEPSEEK_REASONING_EFFORT"), "high"),
		defaultMode:   acp1.SessionModeID(cmp.Or(os.Getenv("DEEPSEEK_ACP_MODE"), string(askMode))),
		shell:         cmp.Or(os.Getenv("DEEPSEEK_ACP_SHELL"), "bash"),
	}
	if ids := os.Getenv("DEEPSEEK_MODELS"); ids != "" {
		var models []modelInfo
		for id := range strings.SplitSeq(ids, ",") {
			if id = strings.TrimSpace(id); id != "" {
				models = append(models, lookupModel(id))
			}
		}
		if len(models) == 0 {
			return nil, fmt.Errorf("DEEPSEEK_MODELS lists no models")
		}
		offeredModels = models
	}
	cfg.defaultModel = cmp.Or(os.Getenv("DEEPSEEK_MODEL"), offeredModels[0].id)

	if !validEffort(cfg.defaultEffort) {
		return nil, fmt.Errorf("DEEPSEEK_REASONING_EFFORT: want off, low, high or max, got %q", cfg.defaultEffort)
	}
	if !validMode(cfg.defaultMode) {
		return nil, fmt.Errorf("DEEPSEEK_ACP_MODE: unknown mode %q", cfg.defaultMode)
	}
	var err error
	if cfg.maxTokens, err = intEnv("DEEPSEEK_MAX_TOKENS", 256_000); err != nil {
		return nil, err
	}
	if cfg.maxSteps, err = intEnv("DEEPSEEK_ACP_MAX_STEPS", 200); err != nil {
		return nil, err
	}
	cfg.compactRatio = 0.8
	if s := os.Getenv("DEEPSEEK_ACP_COMPACT_RATIO"); s != "" {
		if cfg.compactRatio, err = strconv.ParseFloat(s, 64); err != nil || cfg.compactRatio < 0 || cfg.compactRatio >= 1 {
			return nil, fmt.Errorf("DEEPSEEK_ACP_COMPACT_RATIO: want a number from 0 up to 1, got %q", s)
		}
	}
	if cfg.sessionDir = os.Getenv("DEEPSEEK_ACP_SESSIONS"); cfg.sessionDir == "" {
		cache, err := os.UserCacheDir()
		if err != nil {
			return nil, fmt.Errorf("find the session directory; set DEEPSEEK_ACP_SESSIONS: %w", err)
		}
		cfg.sessionDir = filepath.Join(cache, "deepseek-acp", "sessions")
	}
	return cfg, nil
}

// lookupModel returns what is known about a model id. Unknown ids are passed
// through as text-only models with the default context window.
func lookupModel(id string) modelInfo {
	for _, m := range knownModels {
		if m.id == id {
			return m
		}
	}
	return modelInfo{id: id, name: id, contextWindow: 1_000_000}
}

func intEnv(name string, def int) (int, error) {
	s := os.Getenv(name)
	if s == "" {
		return def, nil
	}
	n, err := strconv.Atoi(s)
	if err != nil || n <= 0 {
		return 0, fmt.Errorf("%s: want a positive integer, got %q", name, s)
	}
	return n, nil
}
