package main

import (
	"context"
	"encoding/json/jsontext"
	"errors"

	"github.com/ironpark/acp-go/acp1"

	"github.com/ironpark/deepseek-acp/internal/deepseek"
)

// Plan mode, after the harness's dsh-plan-mode: the model explores and
// designs first, then presents the plan with exit_plan_mode for approval.
// The tool catalog stays the same in both modes, for prompt-cache stability;
// unlike the harness, file writes and edits are refused while planning.

const planModeContext = `Plan mode: on. Stay in plan mode until exit_plan_mode succeeds or the user switches plan mode off. Imperative language to implement changes means plan the implementation, not execute it. A user's conversational agreement approves nothing and does not end plan mode; fold the confirmed decision into the plan and submit it.

Explore first. Use non-mutating reads, searches, static analysis, and checks to ground the plan in the actual repository. Do not edit or write files, change configuration, run formatters or code generation that rewrites tracked files, commit, or otherwise carry out the plan; file writes and edits are refused. Prefer existing functions and patterns over new machinery. Do not use todo_write to track this planning phase: it tracks implementation after an approved plan.

Make the plan decision-complete: state the goal and success criteria; group implementation changes by subsystem; identify public API, schema, and data-flow changes; cover edge cases, failure modes, tests, acceptance criteria, and explicit assumptions. Keep it concise enough to review but detailed enough that another engineer can implement it without making design decisions.

When ready, call exit_plan_mode with the complete plan markdown, starting with a # title. Make exit_plan_mode the only and final tool call in that response: it presents the plan for approval, and implementation begins only after approval. Do not paste the final plan as a plain reply or ask "should I proceed?" in prose. If the user keeps planning, ask what to change, then present the plan again.`

var exitPlanModeTool = deepseek.Tool{
	Name:        "exit_plan_mode",
	Description: "Use only in plan mode. Present your plan for the user's review and, on approval, leave plan mode. The user may approve (carry out the plan from your next step) or keep planning; revise and present again.",
	InputSchema: jsontext.Value(`{"type":"object","properties":{
		"plan":{"type":"string","description":"The complete plan, as markdown, starting with a # heading that names it."}},
		"required":["plan"]}`),
}

var errNotPlanning = errors.New("not in plan mode; exit_plan_mode is only for presenting a plan in plan mode")

// exitPlanMode shows the plan and asks the user to approve it. On approval
// plan mode ends and the model carries out the plan in the same turn.
func (a *deepseekAgent) exitPlanMode(ctx context.Context, stream *acp1.SessionStream, sess *session, id acp1.ToolCallID, plan string) (string, []acp1.ToolCallContent, error) {
	content := []acp1.ToolCallContent{acp1.ToolText(plan)}
	if !sess.inPlanMode() {
		return "", content, errNotPlanning
	}
	toolCall := acp1.ToolCallUpdate{ToolCallID: id, Status: new(acp1.ToolCallStatusPending), Content: content}
	_, approved, err := stream.RequestPermission(ctx, toolCall,
		acp1.NewPermissionOption(acp1.PermissionOptionKindAllowOnce, "Approve plan"),
		acp1.NewPermissionOption(acp1.PermissionOptionKindRejectOnce, "Keep planning"))
	if err != nil {
		return "", content, err
	}
	if !approved {
		return "The user chose to keep planning. Ask what they want changed, then revise the plan and present it again with exit_plan_mode.", content, nil
	}
	sess.setPlanMode(false)
	return "The user approved the plan. Plan mode is off: carry out the plan now, tracking progress with todo_write.", content, nil
}
