package main

import (
	"context"
	"encoding/json/jsontext"
	"errors"
	"fmt"
	"strings"

	"github.com/ironpark/acp-go/acp1"

	"github.com/ironpark/deepseek-acp/internal/deepseek"
)

// Goals, after the harness's dsh-goal: one long-running objective per
// session, which the agent keeps working on in automatic rounds until the
// model marks it complete or blocked, the user pauses it, or the round cap
// runs out.

const (
	goalActive   = "active"
	goalPaused   = "paused"
	goalComplete = "complete"
	goalBlocked  = "blocked"
)

// defaultGoalRounds caps the automatic rounds of a goal, as in the harness.
const defaultGoalRounds = 256

// minBlockedRounds is how many rounds a goal must have run before a round
// may mark it blocked; the harness's default is three.
const minBlockedRounds = 3

const goalRoundPrefix = "<goal_round>"

// goal is a session's long-running objective.
type goal struct {
	Objective     string `json:"objective"`
	Status        string `json:"status"`
	Rounds        int    `json:"rounds"`
	MaxRounds     int    `json:"maxRounds"`
	BlockedReason string `json:"blockedReason,omitzero"`
}

func (g *goal) clone() *goal {
	if g == nil {
		return nil
	}
	c := *g
	return &c
}

func (g *goal) finished() bool {
	return g.Status == goalComplete
}

// describe renders the goal for the user and the model.
func (g *goal) describe() string {
	var b strings.Builder
	fmt.Fprintf(&b, "Objective: %s\nStatus: %s\nRounds: %d/%d", g.Objective, g.Status, g.Rounds, g.MaxRounds)
	if g.BlockedReason != "" {
		fmt.Fprintf(&b, "\nBlocked: %s", g.BlockedReason)
	}
	return b.String()
}

// currentGoal returns a copy of the session's goal, or nil.
func (s *session) currentGoal() *goal {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.goal.clone()
}

// setGoal creates or replaces the goal. It refuses to replace an unfinished
// goal, which /goal edit or update_goal edit changes instead.
func (s *session) setGoal(objective string, maxRounds int) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.goal != nil && !s.goal.finished() {
		return errors.New("an unfinished goal exists; edit, clear or complete it first")
	}
	if maxRounds <= 0 {
		maxRounds = defaultGoalRounds
	}
	s.goal = &goal{Objective: objective, Status: goalActive, MaxRounds: maxRounds}
	s.goalArmed = true
	return nil
}

// updateGoal applies an action to the goal and returns it. edit, pause and
// resume need a direct request from the user; a goal round may only
// complete the goal or, after minBlockedRounds, mark it blocked.
func (s *session) updateGoal(action, objective string, maxRounds int, reason string) (*goal, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	g := s.goal
	if g == nil {
		return nil, errors.New("there is no goal")
	}
	if s.goalRound && action != goalComplete && action != goalBlocked {
		return nil, fmt.Errorf("%s needs a direct request from the user, not a goal round", action)
	}
	switch action {
	case "edit":
		if objective == "" && maxRounds <= 0 {
			return nil, errors.New("edit needs an objective or max_goal_rounds")
		}
		if objective != "" {
			g.Objective = objective
		}
		if maxRounds > 0 {
			g.MaxRounds = maxRounds
		}
	case "pause":
		if g.Status != goalActive {
			return nil, fmt.Errorf("the goal is %s, not active", g.Status)
		}
		g.Status, s.goalArmed = goalPaused, false
	case "resume":
		if g.finished() {
			return nil, errors.New("the goal is complete")
		}
		if g.Rounds >= g.MaxRounds {
			return nil, fmt.Errorf("the goal has used all %d rounds; raise the cap with edit first", g.MaxRounds)
		}
		g.Status, g.BlockedReason, s.goalArmed = goalActive, "", true
	case goalComplete:
		g.Status, g.BlockedReason, s.goalArmed = goalComplete, "", false
	case goalBlocked:
		if reason == "" {
			return nil, errors.New("blocked needs a blocked_reason")
		}
		if s.goalRound && g.Rounds < minBlockedRounds {
			return nil, fmt.Errorf("a goal round can mark the goal blocked only after %d rounds; keep working on it", minBlockedRounds)
		}
		g.Status, g.BlockedReason, s.goalArmed = goalBlocked, reason, false
	default:
		return nil, fmt.Errorf("unknown action %q", action)
	}
	return g.clone(), nil
}

func (s *session) clearGoal() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	had := s.goal != nil
	s.goal, s.goalArmed = nil, false
	return had
}

// nextGoalRound starts the next automatic round of an armed, active goal and
// returns its prompt. When the round cap runs out, the goal is blocked.
func (s *session) nextGoalRound() (deepseek.Block, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	g := s.goal
	if g == nil || g.Status != goalActive || !s.goalArmed {
		return deepseek.Block{}, false
	}
	if g.Rounds >= g.MaxRounds {
		g.Status, g.BlockedReason, s.goalArmed = goalBlocked, fmt.Sprintf("used all %d goal rounds", g.MaxRounds), false
		return deepseek.Block{}, false
	}
	g.Rounds++
	return deepseek.TextBlock(goalRoundPrompt(g)), true
}

func (s *session) setGoalRound(on bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.goalRound = on
}

// goalRoundPrompt is the harness's goal-round instruction.
func goalRoundPrompt(g *goal) string {
	return goalRoundPrefix + "\n" +
		fmt.Sprintf("Objective: %q\nRound: %d/%d\n\n", g.Objective, g.Rounds, g.MaxRounds) +
		"Continue working toward the objective in this same session. Treat the current workspace, " +
		"tool results, and durable session state as authoritative; inspect them instead of assuming " +
		"earlier narration is still current. Make concrete progress and verify the result. Before " +
		"claiming completion, gather evidence that the whole objective is achieved, read the current " +
		"goal, and mark it complete with update_goal. If work remains, leave the goal active for the next round. " +
		fmt.Sprintf("Mark it blocked only when the same condition has blocked progress for at least %d rounds.\n", minBlockedRounds) +
		"</goal_round>"
}

// continueGoal runs automatic goal rounds after a turn that ended normally,
// while the goal is active and armed and has rounds left.
func (a *deepseekAgent) continueGoal(ctx context.Context, sessionID acp1.SessionID, sess *session, reason acp1.StopReason) (acp1.StopReason, error) {
	stream := acp1.NewSessionStream(a.client, sessionID)
	ran := false
	for reason == acp1.StopReasonEndTurn && ctx.Err() == nil {
		block, ok := sess.nextGoalRound()
		if !ok {
			break
		}
		if err := stream.SendThought(ctx, block.Text); err != nil {
			return "", err
		}
		ran = true
		sess.setGoalRound(true)
		var err error
		reason, err = a.runTurn(ctx, sessionID, sess, []deepseek.Block{block})
		sess.setGoalRound(false)
		a.save(ctx, sessionID, sess)
		if err != nil {
			return "", err
		}
	}
	// Only a goal that these rounds blocked is reported, not one that was
	// already blocked before the prompt.
	if g := sess.currentGoal(); ran && g != nil && g.Status == goalBlocked && reason == acp1.StopReasonEndTurn {
		if err := stream.SendText(ctx, "\n\nGoal blocked: "+g.BlockedReason); err != nil {
			return "", err
		}
	}
	return reason, nil
}

// The goal tools, after the harness's dsh-tool-goal. The model's view has no
// ids or revisions: a session has one goal and one writer.
var (
	createGoalTool = deepseek.Tool{
		Name:        "create_goal",
		Description: "Create a persisted goal that keeps this session working across automatic continuation rounds. Use it when the direct human request is a long-running objective, even if the user did not say \"goal\"; not for single-turn work.",
		InputSchema: jsontext.Value(`{"type":"object","properties":{
			"objective":{"type":"string","description":"The concrete completion objective inferred from the direct human request."},
			"max_goal_rounds":{"type":"number","description":"Optional positive limit on automatic continuation rounds; default 256."}},
			"required":["objective"]}`),
	}
	getGoalTool = deepseek.Tool{
		Name:        "get_goal",
		Description: "Read the current session goal: its objective, status and rounds.",
		InputSchema: jsontext.Value(`{"type":"object","properties":{}}`),
	}
	updateGoalTool = deepseek.Tool{
		Name:        "update_goal",
		Description: "Update the current goal.",
		InputSchema: jsontext.Value(`{"type":"object","properties":{
			"action":{"type":"string","enum":["edit","pause","resume","complete","blocked"],"description":"edit, pause, and resume require a direct top-level human request. complete and blocked are also allowed during an automatic continuation of this goal; blocked is rejected before 3 rounds."},
			"objective":{"type":"string","description":"Replacement objective; valid only with action edit."},
			"max_goal_rounds":{"type":"number","description":"Replacement cap; valid only with action edit."},
			"blocked_reason":{"type":"string","description":"Required only with action blocked: the concrete condition that persisted across rounds and blocks progress."}},
			"required":["action"]}`),
	}
)

func (s *session) inGoalRound() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.goalRound
}

// createGoal backs the create_goal tool.
func (s *session) createGoal(objective string, maxRounds int) (string, error) {
	if s.inGoalRound() {
		return "", errors.New("create_goal needs a direct request from the user, not a goal round")
	}
	if strings.TrimSpace(objective) == "" {
		return "", errors.New("empty objective")
	}
	if err := s.setGoal(objective, maxRounds); err != nil {
		return "", err
	}
	return "Goal created. The session continues working on it in automatic rounds after this turn.\n" + s.currentGoal().describe(), nil
}
