# deepseek-acp

[English](README.md) | 한국어

[Agent Client Protocol](https://agentclientprotocol.com)(ACP)을 stdio로 제공하는 DeepSeek 코딩 에이전트입니다.
[ironpark/acp-go](https://github.com/ironpark/acp-go)로 만들었고,
[deepseek-harness](https://github.com/deepseek-ai/deepseek-harness)를 참고해 설계했습니다.

- DeepSeek의 Anthropic 호환 Messages API(`https://api.deepseek.com/anthropic`)를 thinking을 켠 상태로 호출합니다. 사고 과정과 답변은
  `agent_thought_chunk` / `agent_message_chunk`로 스트리밍합니다. harness와 마찬가지로 thinking 블록은 서명과 함께 히스토리에 다시 보냅니다.
- 도구는 harness의 핵심 도구 세트를 따릅니다: `read`, `write`, `edit`, `glob`, `grep`, `bash`, `todo_write`(ACP plan으로 표시).
- 파일 읽기·쓰기와 명령 실행은 클라이언트가 지원하면 클라이언트(`fs/*`, `terminal/*`)를 거치고, 지원하지 않으면 로컬에서 실행합니다.
  클라이언트를 거치면 에디터에서 diff, 저장하지 않은 버퍼, 실시간 터미널을 볼 수 있습니다.
- 권한 모드: `read-only`, `ask`(기본), `accept-edits`, `full-access`. 모델·reasoning effort와 같은 설정 옵션으로 선택할 수 있으며,
  ACP 세션 모드로도 함께 제공됩니다.
  권한 요청에서는 한 번 또는 항상 허용·거부하거나, 세션을 `full-access`로 전환할 수 있습니다. "항상"은 파일 도구 단위로, 명령은
  프로그램(`ls`)이나 하위 명령(`go test`, `go test ./...`도 포함) 단위로 기억합니다. 파이프·리다이렉션·`$` 확장이 있거나 여러 명령을
  잇는 명령은 그 명령 그대로만 기억합니다.
- 세션은 디스크에 저장되며 `session/list`, `load`, `resume`, `close`, `delete`를 지원합니다.
- 작업 디렉터리의 `AGENTS.md`(없으면 `CLAUDE.md`)를 시스템 프롬프트에 추가합니다.
- 컨텍스트 창 사용량을 `usage_update`로 알리고(`session/load`, `resume` 때도 복원), 프롬프트 응답마다 세션 누적 토큰 사용량을 담습니다.
  대화가 컨텍스트 창의 80%에 이르면(64K 토큰은 비워 둠) harness의 compaction처럼 대화를 요약해 요약본으로 바꿉니다.

## 명령어

harness의 명령어(human commands)를 옮긴 슬래시 명령어입니다. 모델에 보내지 않고 실행되며, 결과는 화면에만 표시되고 대화 기록에는
남지 않습니다. `/plan`과 `/goal`은 이어서 모델 턴을 시작할 수 있습니다.

| 명령어 | 동작 |
| --- | --- |
| `/compact` | 지금 대화를 요약해 요약본으로 바꿉니다. |
| `/permission [mode]` | 모드를 보여 주거나 바꿉니다: `read-only`, `ask`, `accept-edits`, `full-access`. harness 프리셋 이름 `workspace-write`, `danger-full-access`도 받습니다. |
| `/plan [off\|message]` | plan 모드를 켜거나(첫 요청을 함께 줄 수 있음) 끕니다. plan 모드에서 모델은 코드를 살펴본 뒤 `exit_plan_mode`로 계획을 제출해 승인을 받고, 승인 전에는 파일 쓰기·수정이 거부됩니다. |
| `/goal [<objective>\|clear\|edit <objective>\|pause\|resume]` | 장기 목표를 설정·확인·변경합니다. 목표가 활성 상태면 모델이 `update_goal`로 완료나 막힘을 표시할 때까지 자동 라운드(최대 256회)로 계속 작업합니다. 모델이 `create_goal`로 직접 목표를 만들 수도 있습니다. |

harness의 `/feedback`(harness 텔레메트리 백엔드로 피드백 전송)과 `/export`(웹 다운로드)는 옮기지 않았습니다.

## 빌드

```sh
go build -o bin/deepseek-acp .
```

## Zed 설정

빌드한 바이너리로 실행:

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

소스를 받아 둔 경우 `go run`으로 실행할 수도 있습니다. 시작할 때마다 현재 소스를 빌드하므로, 처음 시작할 때는 컴파일하느라 느립니다. `go`가 Zed의 `PATH`에 있어야 합니다.

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

소스를 받지 않고 모듈 경로로 실행:

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

## 환경 변수

| 변수 | 기본값 |
| --- | --- |
| `DEEPSEEK_API_KEY` | 필수 |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com/anthropic` |
| `DEEPSEEK_MODEL` | `DEEPSEEK_MODELS`의 첫 번째 모델 |
| `DEEPSEEK_MODELS` | `deepseek-v4-flash,deepseek-v4-pro` |
| `DEEPSEEK_REASONING_EFFORT` | `high` (`off`, `low`, `high`, `max`) |
| `DEEPSEEK_MAX_TOKENS` | `256000` |
| `DEEPSEEK_ACP_MODE` | `ask` |
| `DEEPSEEK_ACP_MAX_STEPS` | 프롬프트당 모델 호출 `200`회 |
| `DEEPSEEK_ACP_COMPACT_RATIO` | `0.8` (요약을 시작할 컨텍스트 창 비율, `0`이면 끔) |
| `DEEPSEEK_ACP_SESSIONS` | `<사용자 캐시 디렉터리>/deepseek-acp/sessions` |
| `DEEPSEEK_ACP_SHELL` | `bash` (bash 도구에서 쓰는 셸) |

`-debug` 플래그를 붙여 실행하면 모든 ACP 메시지를 stderr에 기록합니다.
