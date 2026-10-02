# deepseek-acp

English | [한국어](README.ko.md)

A DeepSeek coding agent that speaks the [Agent Client Protocol](https://agentclientprotocol.com) over stdio, built on
[ironpark/acp-go](https://github.com/ironpark/acp-go) and modeled on
[deepseek-harness](https://github.com/deepseek-ai/deepseek-harness).

- Talks to DeepSeek's Anthropic-compatible Messages API (`https://api.deepseek.com/anthropic`) with thinking enabled,
  and streams thinking/answers as `agent_thought_chunk` / `agent_message_chunk`. Thinking blocks (with signatures) are
  replayed in history, as the harness does.
- Tools, after the harness's core set: `read`, `write`, `edit`, `glob`, `grep`, `bash`, `todo_write` (shown as the ACP plan).
- File reads/writes and commands go through the client (`fs/*`, `terminal/*`) when it supports them — so the editor
  shows diffs, unsaved buffers and live terminals — and run locally otherwise.
- Permission modes: `read-only`, `ask` (default), `accept-edits`, `full-access`. They can be picked like the model and
  reasoning effort, which are config options too; the mode is offered both as a config option and as an ACP session mode.
  A permission request can allow or reject once or always, or switch the session to `full-access`. "Always" remembers a
  file tool, or a command by its program (`ls`) or subcommand (`go test`, which also covers `go test ./...`); a command
  with pipes, redirections, `$` expansions or several commands is remembered only as written.
- MCP servers the client names in `session/new`, `load` and `resume` (stdio, HTTP and SSE) are connected, and their
  tools are offered to the model as `mcp__<server>__<tool>`. A tool that does not declare itself read-only asks for
  permission like a command.
- Sessions are saved to disk and support `session/list`, `load`, `resume`, `close`, `delete`.
- `AGENTS.md` (or `CLAUDE.md`) in the working directory is added to the system prompt.
- Reports context window usage (`usage_update`, also restored on `session/load` and `resume`) and the session's token
  totals in each prompt response. When the conversation reaches 80% of the context window (keeping 64K tokens free), it
  is summarized and replaced by the summary, as the harness's compaction does.

## Commands

Slash commands from the harness's human commands. They run without going to the model, and their result is shown
without entering the conversation; `/plan` and `/goal` can go on to start a model turn.

| Command | What it does |
| --- | --- |
| `/compact` | Summarizes the conversation now and replaces it with the summary. |
| `/permission [mode]` | Shows or switches the mode: `read-only`, `ask`, `accept-edits`, `full-access` (the harness preset names `workspace-write` and `danger-full-access` work too). |
| `/plan [off\|message]` | Turns plan mode on, optionally with a first request, or off. In plan mode the model explores and presents a plan with `exit_plan_mode` for approval; file writes and edits are refused until it is approved. |
| `/goal [<objective>\|clear\|edit <objective>\|pause\|resume]` | Sets, shows or changes a long-running goal. The agent keeps working on an active goal in automatic rounds (up to 256) until the model marks it complete or blocked with `update_goal`. The model can also create one with `create_goal`. |

The harness's `/feedback` (sends feedback to its telemetry backend) and `/export` (a Web download) are not included.

## Build

```sh
go build -o bin/deepseek-acp .
```

## Zed

With a built binary:

```json
{
  "agent_servers": {
    "DeepSeek": {
      "command": "/path/to/bin/deepseek-acp",
      "args": [],
      "env": { "DEEPSEEK_API_KEY": "sk-..." }
    }
  }
}
```

Or with `go run` from a checkout, which builds the current source on each start (the first start is slower while it compiles; `go` must be on Zed's `PATH`):

```json
{
  "agent_servers": {
    "DeepSeek": {
      "command": "go",
      "args": ["-C", "/path/to/deepseek-acp", "run", "."],
      "env": { "DEEPSEEK_API_KEY": "sk-..." }
    }
  }
}
```

Or from the module path, without a checkout:

```json
{
  "agent_servers": {
    "DeepSeek": {
      "command": "go",
      "args": ["run", "github.com/ironpark/deepseek-acp@latest"],
      "env": { "DEEPSEEK_API_KEY": "sk-..." }
    }
  }
}
```

## Configuration

| Variable | Default |
| --- | --- |
| `DEEPSEEK_API_KEY` | required |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com/anthropic` |
| `DEEPSEEK_MODEL` | first of `DEEPSEEK_MODELS` |
| `DEEPSEEK_MODELS` | `deepseek-v4-flash,deepseek-v4-pro` |
| `DEEPSEEK_REASONING_EFFORT` | `high` (`off`, `low`, `high`, `max`) |
| `DEEPSEEK_MAX_TOKENS` | `256000` |
| `DEEPSEEK_ACP_MODE` | `ask` |
| `DEEPSEEK_ACP_MAX_STEPS` | `200` model calls per prompt |
| `DEEPSEEK_ACP_COMPACT_RATIO` | `0.8` share of the context window that triggers compaction; `0` turns it off |
| `DEEPSEEK_ACP_SESSIONS` | `<user cache dir>/deepseek-acp/sessions` |
| `DEEPSEEK_ACP_SHELL` | `bash` |

Run with `-debug` to log every ACP message to stderr.
