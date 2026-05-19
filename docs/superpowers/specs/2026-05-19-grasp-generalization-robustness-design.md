# Grasp Generalization And Robustness Design

Date: 2026-05-19

Status: Design, pre-implementation

Owner: EmboSight grasp pipeline

## 1. Context

The fixed lemon validation is now stable for `fixed_lemon_001` with `seed=42`:

```text
Successes: 5/5
pre_close_abort: 0
grasp_confirmed: 1 per run
micro_lift_ok: 1 per run
post_lift_verified: 1 per run
steps: 4
time: about 112-116s
```

This proves the local lemon failure mode was not an unavoidable gripper limitation. The root issue was the first top-down attempt pushing a round slippery object before closure. The recent fix changed high-slip top-down descent from `0.025m` to `0.010m` and refreshed live object XY before initial descent.

That result does not prove generalization. It is one object category, one seed, one layout, one successful policy profile. The next risk is overfitting the system to lemon by turning a successful local parameter into a global behavior.

## 2. Problem Statement

The current grasp system still has four generalization risks:

1. Evaluation output does not expose enough final execution evidence to compare policies across objects.
2. Grasp strategy parameters are only partially conditioned on object properties.
3. The execution loop is still mostly open-loop between pre-grasp, descend, close, and lift.
4. Memory can become useful for transfer only if it records verified outcomes with the right physical context.

The goal is not to find one universal `depth_margin_m`. The goal is to make the grasp pipeline choose and verify behavior from object state, object properties, and live execution feedback.

## 3. Research Basis

The proposed direction is supported, but only if adapted conservatively to this codebase.

- DeliGrasp shows that LLM-inferred physical properties such as mass, friction, and compliance can drive adaptive grasp policies. Its hardware can control gripper torque/current, so this project must adapt the idea into position-controller parameters such as `squeeze_extra_steps`, descent depth, and lift profile rather than copying force equations directly. Source: [DeliGrasp arXiv:2403.07832](https://arxiv.org/abs/2403.07832).
- Closed-loop grasping work such as GG-CNN shows that perception-driven feedback improves grasping under object movement and control inaccuracies. This supports expanding the current live-object refresh and pre-close alignment checks rather than relying on a one-shot target. Source: [Closing the Loop for Robotic Grasping arXiv:1804.05172](https://arxiv.org/abs/1804.05172).
- Slip-control work shows that trajectory modulation can matter as much as grip force. That does not justify adding a learned tactile controller here, because this environment does not expose real tactile sensing, but it supports future conservative lift-profile limits. Source: [Nature Machine Intelligence slip-control article](https://www.nature.com/articles/s42256-025-01062-2).

## 4. Goals

1. Improve robustness across object type, seed, and layout without weakening current lemon success.
2. Prevent false success by requiring post-lift object evidence for every accepted grasp.
3. Make each grasp attempt diagnosable from logs, oracle summary, and episode JSON.
4. Add object profile classification before allowing profile-specific policy changes.
5. Keep all behavior changes behind small, testable commits with explicit rollback points.

## 5. Non-goals

1. Do not train or import a neural grasp model.
2. Do not rewrite `ActionExecutor` or replace the current planner in one step.
3. Do not assume tactile sensors or force-controlled gripper APIs.
4. Do not let LLM output directly control low-level parameters without clamps and verification.
5. Do not enable profile-specific execution by default until diagnostic-only profile labels have been validated.

## 6. Safety Constraints

These constraints are mandatory for implementation:

1. Default behavior stays legacy until a profile flag is explicitly enabled.
2. Diagnostic-only stages must not change robot behavior.
3. Every reported success must include `post_lift_verified=1` or an equivalent object-height check.
4. Memory learns only from verified outcomes.
5. Selected strategy and executed strategy remain separate.
6. Any profile-specific policy must be reversible by one config switch or one commit revert.
7. Wide or physically ungraspable objects must not be turned into false successes.

## 7. Proposed Architecture

### 7.1 Grasp Diagnostics Extension

Extend oracle and multi-run summaries before changing more behavior.

Required per-episode fields:

```text
post_lift_obj_pos
post_lift_obj_delta_z
post_lift_eef_pos
selected_strategy
executed_strategy
candidate_source
depth_margin_m
squeeze_extra_steps
finger_width_m
pre_close_abort_count
live_xy_drift_max_m
attempts_count
post_lift_verified
```

Rationale:

The system cannot safely improve generalization if success logs hide whether the object truly lifted, which strategy actually executed, or how far the live object drifted during approach.

### 7.2 Diagnostic-only Object Profile Classifier

Add an object profile classifier that initially writes labels only to diagnostics.

Profiles:

```text
small_round_slippery
thin_flat
wide_ungraspable
handled
fragile_soft
default_rigid
unknown
```

Inputs:

```text
label
visible_features
safety_dist
GraspStrategy.slip_risk
GraspStrategy.mass_g
AABB size if available
candidate approach direction and finger width
pose_estimate if available
```

Diagnostic-only means:

```text
profile is recorded
profile does not change candidate ordering
profile does not change depth margin
profile does not change squeeze
profile does not change success criteria
```

### 7.3 Policy Profile Registry

After diagnostic-only validation, profiles can map to bounded policy defaults.

Initial registry:

```yaml
small_round_slippery:
  allowed_strategies: [top_down, tilted_grasp]
  depth_margin_m: 0.010
  squeeze_extra_steps_min: 16
  drift_abort_m: 0.020
  requires_micro_lift: true

thin_flat:
  reject_naive_top_down: true
  preferred_strategies: [gentle_side, tilted_grasp, scoop_under]
  requires_pre_close_alignment: true

wide_ungraspable:
  reject_if_width_exceeds_gripper: true
  report_ungraspable_instead_of_false_success: true

handled:
  preferred_strategies: [handle_grasp, gentle_side]
  top_down_is_advisory_only: true

fragile_soft:
  squeeze_extra_steps_max: 8
  depth_margin_m_max: 0.010
  lift_profile: gentle

default_rigid:
  use_legacy_policy: true
```

The registry is not a replacement for live checks. It only provides safe priors. Execution must still verify object movement and lift.

### 7.4 Closed-loop Execution Guards

The existing execution loop already contains useful guards:

```text
pre_initial_descend live object refresh
pre_z_stall_nudge live object refresh
post_z_stall_nudge live object refresh
pre_close alignment abort
micro-lift verification
post-lift object-height verification
```

Future closed-loop extensions should be incremental:

1. During approach, stop descending if live XY drift exceeds the profile threshold.
2. If early contact is detected and the target is close enough, close instead of continuing to push downward.
3. If z-stall repeats, refresh pose and replan rather than continuing repeated max-step moves.
4. During lift, allow optional conservative lift profile for fragile or slippery objects.

### 7.5 Experience Memory

Memory should become condition-aware only after diagnostics are reliable.

Future memory records should include:

```yaml
label: lemon
profile: small_round_slippery
object_size_m: [0.058, 0.055, 0.058]
selected_strategy: top_down
executed_strategy: top_down
candidate_source: strategy_top_down
depth_margin_m: 0.010
squeeze_extra_steps: 18
post_lift_obj_delta_z: 0.091
failure_mode: success
verified_success: true
```

Retrieval should transfer only across similar profiles and compatible size ranges. For example, lemon experience can advise lime or small orange, but it should not advise a mug, tray, or bread.

## 8. Rollout Strategy

### Phase 1: Observability only

Add oracle and multi-run summary fields. No execution behavior changes.

Success criteria:

```text
all existing unit tests pass
fixed_lemon_001 still 5/5
summary.csv exposes final pose and executed parameters
```

### Phase 2: Diagnostic-only profile

Add `src/grasp_profile.py` and record profile in attempt diagnostics. No execution behavior changes.

Success criteria:

```text
lemon -> small_round_slippery
tupperware-like wide object -> wide_ungraspable
mug-like handled object -> handled or default_rigid with reason
bread-like object -> fragile_soft
unknown object -> unknown or default_rigid
```

### Phase 3: Flagged profile policy

Add `grasp_policy_mode: legacy | profiled`. Default is `legacy`. Enable only for controlled runs.

Success criteria:

```text
legacy mode produces identical behavior to current code
profiled mode changes only profiles explicitly enabled in tests
```

### Phase 4: One-profile execution rollout

Enable `small_round_slippery` policy only. This generalizes the lemon fix to similar round slippery objects.

Success criteria:

```text
fixed lemon: 10/10 success, pre_close_abort=0
round fruit matrix: at least 80% success
false success: 0
post_lift_verified: every success
```

### Phase 5: Condition-aware memory

Store verified parameter outcomes and retrieve them for similar profiles only.

Success criteria:

```text
memory stores profile and executed parameters
memory ignores stale code_version entries
memory advice does not override hard safety or geometry checks
```

## 9. Evaluation Matrix

The minimum matrix should include:

```text
fixed_lemon_001 seed=42
round fruits across multiple seeds
thin flat objects
wide ungraspable objects
handled objects
fragile or soft objects
distractor-heavy scenes
```

Required metrics:

```text
success_rate
false_success_count
post_lift_verified_rate
avg_attempts
pre_close_abort_count
object_not_lifted_count
gripper_empty_count
slipped_descend_count
ik_unreachable_count
avg_time_s
final_pose_variance
failure_mode_distribution
```

## 10. Risk Assessment

### Risk: profile classifier misclassifies objects

Mitigation:

Run profile in diagnostic-only mode first. Do not allow it to change execution until profile labels match logs across a matrix.

### Risk: LLM over-controls low-level parameters

Mitigation:

Keep hard clamps in code. Treat LLM output as an advisory prior, not as a command.

### Risk: memory reinforces wrong behavior

Mitigation:

Record only verified success, keep selected/executed strategy separation, and bump memory code version when semantics change.

### Risk: profile-specific policy regresses non-lemon objects

Mitigation:

Default to `legacy`. Enable one profile at a time. Require matrix evidence before broad enablement.

### Risk: execution loop grows too complex

Mitigation:

Add small guard functions with narrow tests. Do not rewrite `ActionExecutor` in a single change.

## 11. Recommended Next Step

Start with Phase 1 and Phase 2 only:

```text
1. Extend oracle summary and multi-run CSV.
2. Add diagnostic-only object profile labels.
3. Run fixed lemon plus a small object matrix.
4. Review logs before any profile controls execution.
```

This gives enough evidence to decide whether `small_round_slippery` should become the first profile-controlled policy without risking the rest of the grasp stack.
