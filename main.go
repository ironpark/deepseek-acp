// Command deepseek-acp is a coding agent backed by DeepSeek that speaks the
// Agent Client Protocol over stdio, so editors such as Zed can run it.
//
// It follows deepseek-harness: it talks to DeepSeek's Anthropic-compatible
// Messages API with thinking on, replays the model's thinking in the history,
// and offers the harness's core tools (read, write, edit, glob, grep, bash,
// todo_write). File reads and writes and commands go through the client when
// it supports them, so the editor shows unsaved buffers, diffs and live
// terminals; otherwise they run locally.
package main

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"syscall"

	acp "github.com/ironpark/acp-go"
	"github.com/ironpark/acp-go/acp1"

	"github.com/ironpark/deepseek-acp/internal/deepseek"
)

const version = "0.1.0"

func main() {
	showVersion := flag.Bool("version", false, "print the version and exit")
	debug := flag.Bool("debug", false, "log every ACP message to stderr")
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: deepseek-acp [flags]\n\nRuns a DeepSeek coding agent over ACP on stdin/stdout.\n\nFlags:\n")
		flag.PrintDefaults()
		fmt.Fprint(os.Stderr, "\n"+envHelp)
	}
	flag.Parse()
	if *showVersion {
		fmt.Println("deepseek-acp", version)
		return
	}

	// Stdout carries the protocol, so logs go to stderr.
	level := slog.LevelInfo
	if *debug {
		level = slog.LevelDebug
	}
	logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: level}))

	cfg, err := loadConfig()
	if err != nil {
		logger.Error("configuration", "error", err)
		os.Exit(2)
	}
	if cfg.apiKey == "" {
		// Keep running so the editor can show the error on the first prompt.
		logger.Warn("DEEPSEEK_API_KEY is not set; prompts will fail until it is")
	}

	store, err := acp1.NewFileStore[*session](cfg.sessionDir)
	if err != nil {
		logger.Error("open the session store", "dir", cfg.sessionDir, "error", err)
		os.Exit(1)
	}
	llm := &deepseek.Client{BaseURL: cfg.baseURL, APIKey: cfg.apiKey, Retries: 4}

	opts := []acp.Option{acp.WithErrorHandler(func(err error) { logger.Error("acp", "error", err) })}
	if *debug {
		opts = append(opts, acp.WithMiddleware(acp.LoggingMiddleware(logger)))
	}
	conn := acp1.NewAgentSideConnection(newAgent(cfg, store, llm, logger), acp.NewStdioTransport(os.Stdin, os.Stdout), opts...)

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	logger.Info("deepseek-acp ready", "version", version, "model", cfg.defaultModel, "sessions", cfg.sessionDir)
	err = conn.Start(ctx)
	closeAllMCP(store)
	if err != nil && ctx.Err() == nil {
		logger.Error("connection ended", "error", err)
		os.Exit(1)
	}
}

// closeAllMCP ends every session's MCP connections, which stops the servers
// the agent started.
func closeAllMCP(store acp1.SessionStore[*session]) {
	ctx := context.Background()
	ids, _ := store.List(ctx)
	for _, id := range ids {
		if sess, ok, _ := store.Get(ctx, id); ok {
			sess.closeMCP()
		}
	}
}
