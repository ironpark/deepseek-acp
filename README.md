# deepseek-acp

A DeepSeek coding agent that speaks the [Agent Client Protocol](https://agentclientprotocol.com) over stdio, built on
[ironpark/acp-go](https://github.com/ironpark/acp-go) and modeled on
[deepseek-harness](https://github.com/deepseek-ai/deepseek-harness).

- Talks to DeepSeek's Anthropic-compatible Messages API (`https://api.deepseek.com/anthropic`) with thinking enabled,
  and streams thinking/answers as `agent_thought_chunk` / `agent_message_chunk`. Thinking blocks (with signatures) are
  replayed in history, as the harness does.
- Tools, after the harness's core set: `read`, `write`, `edit`, `glob`, `grep`, `bash`, `todo_write` (shown as the ACP plan).
- File reads/writes and commands go through the client (`fs/*`, `terminal/*`) when it supports them — so the editor
  shows diffs, unsaved buffers and live terminals — and run locally otherwise.
- Modes: `read-only`, `ask` (default), `accept-edits`, `full-access`. Config options: model and reasoning effort.
- Sessions are saved to disk and support `session/list`, `load`, `resume`, `close`, `delete`.
- `AGENTS.md` (or `CLAUDE.md`) in the working directory is added to the system prompt.

## Build

```sh
go build -o bin/deepseek-acp .
```

## Zed

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
| `DEEPSEEK_ACP_SESSIONS` | `<user cache dir>/deepseek-acp/sessions` |
| `DEEPSEEK_ACP_SHELL` | `bash` |

Run with `-debug` to log every ACP message to stderr.
