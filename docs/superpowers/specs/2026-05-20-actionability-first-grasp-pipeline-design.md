# Actionability-First Grasp Pipeline Design

> Date: 2026-05-20
> Status: Draft for review
> Scope: design only. This document does not authorize implementation until an implementation plan is written and reviewed.

## 1. Problem Statement

Phase 1-4 improved grasp observability and added a gated `small_round_slippery` candidate-source policy, but the Phase 4 GPU run shows that candidate ordering alone does not solve grasp generalization.

Observed Phase 4 evidence:

```text
phase4-profiled-smround-gen50-20260520_013128:
  total: 50
  success: 16/50 = 32.0%
  errors: 0
  timeouts: 0

  failure breakdown:
    clarification_loop: 8
    slipped_descend: 8
    ik_unreachable: 7
    slipped_lift: 6
    safety_loop: 4
    gripper_empty: 1

  candidate source policy:
    prefer_selected_strategy_candidate:applied: 2
    prefer_selected_strategy_candidate:not_applied: 4
    legacy:not_applied: 31

  transition:
    vlm_top_grasp -> strategy_gentle_side: 2
```

The two Phase 4 policy-applied cases were exactly the intended transitions:

```text
random_seed_4 lemon_wedge:
  selected_strategy: gentle_side
  executed_strategy: gentle_side
  legacy_first_candidate_source: vlm_top_grasp
  final_first_candidate_source: strategy_gentle_side
  failure: ik_unreachable
  pre_grasp_reason: axis_gap_too_large

random_seed_13 orange:
  selected_strategy: gentle_side
  executed_strategy: gentle_side
  legacy_first_candidate_source: vlm_top_grasp
  final_first_candidate_source: strategy_gentle_side
  failure: ik_unreachable
  pre_grasp_reason: axis_gap_too_small
```

Interpretation:

1. Phase 4 correctly reduced selected/executed strategy mismatch for the targeted cases.
2. The promoted `gentle_side` candidates were not physically actionable.
3. The next root problem is not squeeze, depth, or another candidate-source patch.
4. The system needs a shared actionability contract between target selection, grasp planning, and execution.

## 2. Root Cause

The current pipeline has a planning/execution contract gap.

In `src/grasp_planner.py`, candidates are filtered with:

```python
env.is_reachable(c.point_3d, c.approach_dir)
```

But `src/env_wrapper.py:is_reachable()` currently returns `True` unconditionally. It is explicitly documented as a placeholder for future geometric / IK / navigation-aware filtering.

Meanwhile, the real geometric handoff check already exists later in the execution path:

- `src/grasp_execution.py`
  - `decompose_pre_grasp_error(...)`
  - `evaluate_pre_grasp_handoff(...)`
  - reasons such as `axis_gap_too_small`, `axis_gap_too_large`, `lateral_misaligned`
- `src/env_wrapper.py`
  - `move_to_pre_grasp_diagnostic(...)`
  - `evaluate_pre_grasp_at_current(...)`
- `src/action_executor.py`
  - consumes pre-grasp diagnostics and fails with structured `ik_unreachable`

This means planner and executor do not agree on what "reachable" means. The planner can rank a candidate first, while executor later proves that the same candidate cannot reach a valid pre-grasp handoff pose.

## 3. Phase 1-4 Assessment

### Phase 1: Observability

Useful and necessary.

It added final-pose and attempt diagnostics so reports can explain whether the robot actually lifted the object, which strategy executed, which candidate source was used, and which failure mode occurred.

It was not expected to improve success rate directly.

### Phase 2: Diagnostic Profile Classification

Useful as analysis infrastructure.

It introduced profiles such as `small_round_slippery`, `wide_ungraspable`, `thin_flat`, and `handled`, allowing success/failure grouping by object family.

However, profile classification is not actionability. For example, Phase 4 still shows `wide_ungraspable` with both successes and failures. A profile should be treated as a prior, not as proof that a grasp is executable.

### Phase 3: Gated Profile Policy

Useful as an engineering safety gate.

It introduced:

```yaml
grasp_policy:
  mode: legacy | profiled
  enabled_profiles: [...]
```

Default `legacy` behavior remains protected. Profiled behavior can be tested without changing production defaults.

### Phase 4: Candidate-Source Policy

Useful as a targeted experiment, but not a success-rate solution.

It proved that the candidate-source mismatch existed:

```text
selected_strategy=gentle_side
candidate_source=vlm_top_grasp
executed_strategy=top_down
```

It also proved that fixing this mismatch is insufficient. Promoting `strategy_gentle_side` exposed a deeper pre-grasp feasibility failure.

Phase 4 should remain a controlled diagnostic/profiled behavior. It should not be broadened into a general fix.

## 4. Goals

1. Add a shared actionability model before candidate ranking and execution.
2. Make planner candidate ordering depend on physical feasibility, not only source score or profile policy.
3. Prevent known-infeasible candidates from being promoted just because their semantic strategy matches the LLM-selected strategy.
4. Separate target-selection failures from grasp-actionability failures in evaluation.
5. Preserve legacy defaults until actionability gating is validated with GPU evidence.
6. Keep Phase 1-4 diagnostics and extend them instead of replacing them.

## 5. Non-Goals

This design does not:

1. Tune `depth_margin_m`.
2. Tune `squeeze_extra_steps`.
3. Change close-gripper, descend, lift, micro-lift, or post-lift success criteria.
4. Add new neural models.
5. Add new dependencies.
6. Make memory override hard geometry checks.
7. Apply profile behavior globally.
8. Treat `small_round_slippery` as the only remaining problem.

## 6. Proposed Architecture

Introduce an actionability-first pipeline:

```text
observe
  -> target resolution
  -> candidate generation
  -> actionability audit
  -> candidate ranking
  -> execution
  -> structured failure feedback
  -> evaluation/reporting
```

The main change is a new actionability layer between candidate generation and candidate ranking.

### 6.1 Target Resolution Contract

Every planned grasp should carry an explicit target-body resolution result.

Recommended data:

```python
@dataclass(frozen=True)
class TargetResolution:
    requested_label: str
    selected_label: str
    target_body: str | None
    body_category: str | None
    source: str
    confidence: float
    used_fallback: bool
    reason: str
```

Purpose:

- Make it explicit whether the robot is acting on `obj_main`, a distractor body, or a semantically matched category.
- Avoid silent fallback to `obj_main` when a named target was not resolved.
- Make `glass_cup` / `glass cup`, `hotdog_bun` / `hot dog bun`, and similar normalization issues visible.

Design rule:

For manipulation, exact raw string equality is too brittle. Body/category matching should use the same normalized label key used elsewhere in perception and belief matching.

### 6.2 Candidate Actionability Contract

Every `GraspCandidate` should be evaluated before policy-based reordering.

Recommended data:

```python
@dataclass(frozen=True)
class CandidateActionability:
    source: str
    selected_strategy: str | None
    executed_strategy: str
    target_body: str | None
    actionable: bool
    hard_reject: bool
    reason: str
    total_error_m: float | None
    lateral_error_m: float | None
    axis_error_m: float | None
    approach_gap_m: float | None
    lateral_limit_m: float | None
    object_size_m: tuple[float, float, float] | None
    score_modifier: float
```

Purpose:

- Tell planner whether a candidate can reach a safe pre-grasp handoff pose.
- Preserve the reason for rejection or downranking.
- Give evaluation a direct answer to: "Was this a bad target, bad candidate, or bad execution?"

### 6.3 Diagnostic-Only First

The first implementation slice should not change candidate ordering.

It should:

1. Compute actionability diagnostics for each candidate where possible.
2. Attach actionability diagnostics to candidates/attempts.
3. Add aggregate report fields.
4. Run gen50 and compare against Phase 4.

This prevents another policy patch from being mistaken for a root-cause fix.

### 6.4 Profiled Actionability Gating

Only after diagnostic-only evidence is reviewed, enable actionability gating under:

```yaml
grasp_policy:
  mode: profiled
  enabled_profiles:
    - small_round_slippery
  actionability_gate: true
```

Candidate ranking should become:

```text
1. Generate legacy candidates.
2. Compute actionability for each candidate.
3. Preserve legacy ordering in legacy mode.
4. In profiled actionability mode:
   a. hard-reject candidates with known unsafe/unreachable handoff.
   b. rank actionable candidates first.
   c. apply profile/candidate-source policy only among actionable candidates.
   d. if no actionable candidate exists, preserve legacy ordering but record "no_actionable_candidate".
```

This avoids the Phase 4 failure mode where `strategy_gentle_side` is promoted before proving it can execute.

### 6.5 Structured Failure Feedback

Executor already emits useful failure details. Planner should consume them as constraints.

Examples:

```text
pre_grasp_reason=axis_gap_too_large
  -> do not retry the same source/approach geometry unchanged.

pre_grasp_reason=axis_gap_too_small
  -> same candidate is too close along approach axis; do not promote unchanged side approach.

failure=slipped_descend with object_displaced_before_close
  -> treat candidate as contact-unstable; prefer less lateral pushing or replan from live object pose.

failure=gripper_empty
  -> distinguish "missed target" from "object too thin/wide" before changing squeeze.
```

This should replace the current weak behavior where failures are stored but the next plan mostly relies on memory advice and candidate de-duplication.

## 7. Evaluation Taxonomy

Long-generalization reports should separate failure families:

```text
target_selection_failure:
  clarification_loop
  target_not_resolved
  object_mismatch

safety_decision_failure:
  safety_loop
  safety_refusal

planning_actionability_failure:
  no_candidate
  no_actionable_candidate
  target_body_unresolved
  pre_grasp_axis_gap_too_small
  pre_grasp_axis_gap_too_large
  pre_grasp_lateral_misaligned

execution_failure:
  slipped_descend
  slipped_lift
  gripper_empty
  hit_z_floor
  ik_unreachable_after_actionable_plan
```

This matters because the Phase 4 run contains at least three independent failure classes:

1. target/safety loops (`clarification_loop`, `safety_loop`)
2. candidate actionability failures (`ik_unreachable`)
3. physical grasp failures (`slipped_descend`, `slipped_lift`, `gripper_empty`)

They should not be fixed by the same policy.

## 8. Diagnostics To Add

Per attempt:

```text
target_resolution_source
target_body
target_body_category
target_resolution_used_fallback
candidate_actionability_policy
candidate_actionability_actionable
candidate_actionability_hard_reject
candidate_actionability_reason
candidate_actionability_total_error_m
candidate_actionability_lateral_error_m
candidate_actionability_axis_error_m
candidate_actionability_approach_gap_m
candidate_actionability_lateral_limit_m
legacy_first_candidate_actionable
final_first_candidate_actionable
no_actionable_candidate
```

Per long-generalization summary:

```text
failure_family_breakdown
failure_mode_by_actionability_reason
candidate_actionability_usage
candidate_actionability_transition_usage
target_resolution_source_usage
success_rate_by_actionability_reason
no_actionable_candidate_count
ik_unreachable_after_actionable_plan_count
```

## 9. Compatibility And Rollout

### Default behavior

Default `legacy` must remain unchanged.

```yaml
grasp_policy:
  mode: legacy
  enabled_profiles: []
```

In legacy mode:

- candidate ordering stays score-based as today.
- actionability may be recorded only if explicitly diagnostic-only and non-invasive.
- no candidate should be filtered or promoted by actionability.

### Profiled behavior

Profiled behavior should be opt-in.

Recommended staged flags:

```yaml
grasp_policy:
  mode: profiled
  enabled_profiles:
    - small_round_slippery
  actionability_diagnostics: true
  actionability_gate: false
```

Then later:

```yaml
grasp_policy:
  mode: profiled
  enabled_profiles:
    - small_round_slippery
  actionability_diagnostics: true
  actionability_gate: true
```

This gives two GPU checkpoints:

1. Can actionability diagnostics explain current failures without changing behavior?
2. Does actionability gating improve success when enabled?

## 10. Testing Strategy

### Unit tests

Required tests:

1. Target resolution uses normalized category matching.
2. Target resolution records fallback instead of silently treating fallback as a normal match.
3. Candidate actionability records `axis_gap_too_large`.
4. Candidate actionability records `axis_gap_too_small`.
5. Candidate actionability records `lateral_misaligned`.
6. Legacy mode preserves candidate ordering even when actionability diagnostics exist.
7. Profiled actionability diagnostics do not change ordering when `actionability_gate=false`.
8. Profiled actionability gate rejects or downranks a known-infeasible `strategy_gentle_side`.
9. Candidate-source policy only promotes selected-strategy candidates among actionable candidates.
10. If no actionable candidate exists, legacy ordering is preserved and `no_actionable_candidate=True` is recorded.
11. Oracle summary preserves actionability diagnostics.
12. Long-generalization summary aggregates actionability diagnostics.

### Integration tests

Use fake envs with deterministic pre-grasp diagnostics:

```text
candidate A: vlm_top_grasp, actionable=True
candidate B: strategy_gentle_side, actionable=False, reason=axis_gap_too_large
```

Expected behavior:

- legacy: A remains first by score.
- profiled diagnostics only: A remains first, diagnostics recorded.
- profiled gate enabled: B cannot be promoted ahead of A.

### GPU validation

Run sequence:

1. Fixed lemon smoke, legacy mode.
2. Fixed lemon smoke, profiled diagnostics-only.
3. Fixed lemon smoke, profiled actionability gate.
4. gen50, profiled diagnostics-only.
5. gen50, profiled actionability gate.

Compare against:

```text
Phase 3:
  gen50: 19/50 = 38.0%
  small_round_slippery: 4/6

Phase 4:
  gen50: 16/50 = 32.0%
  small_round_slippery: 4/6
```

Primary success criteria:

```text
fixed lemon remains stable
false success remains 0
post_lift_verified remains true for successes
ik_unreachable caused by promoted gentle_side decreases
candidate-source mismatch does not reappear for small_round_slippery
no increase in clarification_loop or safety_loop from grasp changes
```

Secondary success criteria:

```text
gen50 success improves over Phase 4
small_round_slippery improves over 4/6
failure_family_breakdown shows fewer planning_actionability failures
```

## 11. Risks

### Risk: actionability check duplicates executor motion

Mitigation:

Start diagnostic-only. Do not run expensive or state-changing simulation motion in the planner. Prefer pure geometric checks and existing body/EEF state. If a check would move the robot, it belongs in executor, not planner.

### Risk: rejecting too many candidates

Mitigation:

Use a staged policy:

1. diagnostics-only
2. soft score modifier
3. hard reject only for reasons proven by GPU logs

### Risk: target-body fallback hides object mismatch

Mitigation:

Make fallback explicit in diagnostics. Treat `used_fallback=True` as a separate evaluation bucket.

### Risk: profile policy becomes another patch layer

Mitigation:

Actionability must be profile-independent infrastructure. Profiles may decide when to enable policy, but they must not override hard actionability checks.

### Risk: memory reinforces stale failures

Mitigation:

Memory can suggest strategy priors, but it must not override current target resolution or actionability rejection.

## 12. Recommended Implementation Order

1. Add target resolution diagnostics without changing behavior.
2. Add candidate actionability dataclasses and pure evaluator helpers.
3. Attach actionability diagnostics to candidates and attempts.
4. Extend oracle and long-generalization reports.
5. Run profiled diagnostics-only GPU gen50.
6. Review whether actionability explains `ik_unreachable`, `slipped_descend`, and `gripper_empty`.
7. Add profiled actionability gating for `small_round_slippery` only.
8. Run fixed lemon and gen50 again.
9. Only then decide whether to broaden beyond `small_round_slippery`.

## 13. Design Decision

The recommended path is not another profile-specific patch.

The next phase should build an actionability-first contract that makes candidate ranking aware of physical feasibility before execution. Phase 1-4 should be kept because they provide the measurement and gating infrastructure required to validate this safely, but Phase 4 should be treated as evidence that semantic strategy alignment is not enough.

The system should only execute or promote candidates after it can answer:

```text
Which body are we acting on?
Why is this candidate physically actionable?
If it is not actionable, which measurable condition failed?
If execution fails anyway, how does that failure constrain the next plan?
```

Until those questions are represented in code and reports, additional squeeze/depth/candidate-source tuning is likely to keep producing local fixes and new failure modes.
