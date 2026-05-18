# Lemon Grasp Follow-up Optimization Plan

Date: 2026-05-19

Baseline commit: `6583446 Report actual post-lift grasp coordinates`

Baseline validation command:

```bash
bash scripts/validate_lemon_grasp.sh | tee lemon_real_pos.log
```

Baseline result:

```text
success : True
steps   : 6
time    : 351.5s
grasp_failure_mode: success
```

The baseline is a real grasp success, but it is recovery-dependent and slow.

## Current Implementation Status

Implemented in the current follow-up change set:

- P0 strategy attribution separation.
- P1/P3 live object XY refresh after z-stall base nudge and before post-nudge re-align.

Still pending GPU validation:

- Whether `pre_close_abort` drops from `1` to `0` for `fixed_lemon_001`.
- Whether episode runtime decreases from the baseline `351.5s`.
- Whether the first attempt succeeds or still requires recovery.

Local verification completed:

```text
python -m pytest tests/test_action_executor_v1.py::TestZStallLateralReAlign::test_post_nudge_realign_uses_live_object_xy_not_stale_candidate tests/test_agent_speech.py::test_grasp_memory_payload_separates_selected_strategy_from_executed_shape -q
2 passed

python -m pytest tests/test_action_executor_v1.py tests/test_agent_speech.py tests/test_agent_run.py tests/test_memory_integration.py tests/test_memory_manager.py -q
113 passed

python -m ruff check src/action_executor.py src/agent.py tests/test_action_executor_v1.py tests/test_agent_speech.py
All checks passed

python -m pytest tests -q
462 passed
```

## Optimization Goals

Primary goals:

1. Keep true grasp success high.
2. Reduce first-attempt lemon failure.
3. Eliminate misleading strategy attribution.
4. Reduce validation runtime.
5. Preserve the no-false-positive grasp checks.

Concrete target metrics:

- Fixed seed `42`: `success=True`, `pre_close_abort=0`, `post_lift_verified=1`.
- Repeated fixed-seed validation: at least `5/5` success.
- Episode runtime: reduce from about `340-351s` toward `<180s`.
- `move_arm_to max_steps reached`: reduce from many repeated warnings toward `<=2` per successful episode.
- No success is accepted without either micro-lift follow evidence or post-lift object-height verification.

## Non-goals

Do not start by tuning gripper force alone.

The current evidence shows that once the gripper contacts the lemon, the object follows the micro-lift and final lift:

```text
[close_gripper] grasp confirmed + squeezed at step 33
[micro_lift] eef Δz=0.0103 obj Δz=0.0098 follows=True
[act] post-lift verified: obj Δz=0.089 (0.947→1.036)
```

The bigger issue is approach quality before contact, not post-contact squeeze strength.

## P0: Fix Strategy And Candidate Attribution

Status: implemented locally; GPU log validation pending.

Problem:

The second attempt logs `gentle_side`, but the successful candidate source is `vlm_top_grasp`:

```text
[agent] grasp strategy: gentle_side
```

```json
"grasp_candidate_source": "vlm_top_grasp"
```

Risk:

The memory system can learn the wrong lesson, such as "gentle_side succeeded", even when the executed candidate was top-grasp shaped.

Recommended change:

- Store selected LLM strategy and executed candidate source separately in `GraspAttempt.diagnostic`.
- Record approach direction and depth margin in the attempt diagnostic.
- Update memory events to say which executed candidate actually succeeded.

Acceptance criteria:

- Grasp attempt diagnostics include `selected_strategy`, `executed_strategy`,
  `candidate_source`, `approach_dir`, `finger_width_m`, and `depth_margin_m`.
- Memory records the executed strategy in `context["strategy"]`, while keeping
  `selected_strategy` separately for analysis.
- A selected `gentle_side` strategy plus a `vlm_top_grasp` top-down candidate is
  recorded as `selected_strategy=gentle_side`, `candidate_source=vlm_top_grasp`,
  `executed_strategy=top_down`.
- Covered by
  `tests/test_agent_speech.py::test_grasp_memory_payload_separates_selected_strategy_from_executed_shape`.

## P1: Make The First Attempt Use The Live Lemon Pose

Status: partially implemented locally; GPU validation pending.

Problem:

The first attempt gets stale after recovery:

```text
[pre_grasp_align] eef=(0.123,-2.860,0.940) obj=(0.188,-2.836,0.947) lateral=0.0693m
[pre_close_align] abort ... lateral=0.0693m > limit=0.0200m
```

This means the final close point is about 6.9 cm away from the live lemon body.

Recommended change:

- Refresh live object position before every close decision.
- Also refresh before the second descent after a base nudge, not only immediately before closing.
- If live object XY has moved more than a small threshold, update the candidate or force replan before continuing descent.

Suggested thresholds:

- live-object drift warning: `> 0.015m`
- forced refresh/replan: `> 0.025m`
- hard pre-close abort: keep the current finger-width-based limit

Acceptance criteria:

- In seed `42`, the first attempt no longer reaches a `pre_close_align abort`.
- If the lemon moves, logs show an early refresh before another long descent loop.
- `selected_target_position` is refreshed before the next grasp plan, not after a wasted close window.
- Covered locally by
  `tests/test_action_executor_v1.py::TestZStallLateralReAlign::test_post_nudge_realign_uses_live_object_xy_not_stale_candidate`.
- Expected GPU signal:

```text
[live_obj_refresh] stage=post_z_stall_nudge drift=...
```

## P2: Reduce Top-down Depth And Z-stall Waste

Status: not implemented yet.

Problem:

The first top-down attempt descends with a deeper margin:

```text
[approach] margin=0.025m along [ 0.  0. -1.], adjusted target=[... 0.92514914]
```

Then it repeatedly stalls:

```text
[descend] z stalled at 0.943 for 3 steps (Δ=0.018m above target). contact=False, close_enough=False
```

The successful second attempt uses a shallower margin:

```text
[approach] margin=0.010m along [ 0.  0. -1.], adjusted target=[... 0.9369208]
```

Recommended change:

- For lemon-like round/high-slip objects, do not use `0.025m` top-down depth on the first attempt.
- Prefer a shallower top grasp margin around `0.010m`, or choose a tilted/side strategy only if the executed candidate is actually compatible.
- Add a z-stall budget. After repeated stalls, stop descending, refresh object pose, and replan instead of running several `move_arm_to` max-step loops.

Acceptance criteria:

- No repeated z-stall loops against the same stale candidate.
- `move_arm_to max_steps reached` count drops significantly.
- Episode time drops below the baseline without reducing true success.

## P3: Improve Recovery Policy After Base Nudge

Status: partially implemented locally; GPU validation pending.

Problem:

The base nudge is useful, but the first attempt still ends with stale target geometry:

```text
[act] base nudge: residual=(-0.011,-0.065) |Δ|=0.066m
[nudge] base translated by world Δ=(-0.011, -0.065)m |Δ|=0.066m
```

Later:

```text
[act] post-nudge lateral re-align: offset=0.049m, target_xy=(0.125, -2.858) z=0.948
```

The re-align target uses the old candidate location, while the live object has moved to about `(0.188, -2.836)`.

Recommended change:

- After every nudge, immediately read the target body position from sim.
- If live object and candidate differ, realign to the live object, not the old candidate.
- If live object moved more than the safe limit, cancel this attempt and replan instead of continuing.

Acceptance criteria:

- Post-nudge logs include both `candidate_xy` and `live_obj_xy`.
- The system does not descend toward old `(0.125, -2.858)` after the live object is known near `(0.188, -2.836)`.
- Implemented behavior: after z-stall base nudge, the executor reads the live
  simulator body pose and refreshes candidate XY before lateral re-align.
- Remaining improvement: expand the same live-pose refresh to other recovery
  branches, not only the z-stall nudge path.

## P4: Extend Validation Beyond Repeated Fixed Seed

Status: not implemented yet.

Problem:

`validate_lemon_grasp_multi.sh 5` proves deterministic repeatability for the current fixed setup, but it does not prove broader robustness if all runs use the same seed and scenario placement.

Recommended change:

- Add a multi-seed or multi-scenario lemon validation script.
- Keep the current fixed-seed script as a regression test.
- Add summary columns for:
  - `pre_close_abort`
  - `grasp_confirmed`
  - `micro_lift_ok`
  - `post_lift_verified`
  - `post_lift_obj_pos`
  - `candidate_source`
  - `selected_strategy`
  - `move_arm_to_max_steps_count`
  - elapsed time

Acceptance criteria:

- Fixed seed stays `5/5`.
- Multi-seed report shows success rate, common failure modes, and whether failures are approach, contact, or lift failures.
- The CSV contains enough evidence to distinguish true success from speech-only success.

## P5: Add Final Pose To Oracle Summary

Status: not implemented yet.

Problem:

The oracle summary currently includes pre-lift selected target position:

```json
"selected_target_position": [
  0.18830765783786774,
  -2.8358876705169678,
  0.9469208121299744
]
```

But the final speech reports post-lift object position:

```text
x=0.192m, y=-2.831m, z=1.036m
```

Recommended change:

- Add `post_lift_obj_pos` to oracle summary when available.
- Add `post_lift_obj_delta_z`.
- Add `final_pose_source`, such as `sim_body_post_lift`, `eef_post_lift`, or `belief_estimate_fallback`.

Acceptance criteria:

- Oracle summary itself can prove whether the final position is real or a fallback.
- The final speech and oracle final pose agree within rounding tolerance.

## Suggested Implementation Order

1. Add diagnostics for selected strategy, candidate source, approach direction, depth margin, and post-lift position to summaries.
2. Fix memory attribution so it records the executed candidate, not only the selected LLM strategy.
3. Refresh live object pose after base nudge and before post-nudge re-align.
4. Reduce lemon first-attempt top-down margin or route to a shallower live top grasp.
5. Add z-stall early-exit and replan policy.
6. Add multi-seed validation.

## Regression Commands

Use this after each optimization:

```bash
bash scripts/validate_lemon_grasp.sh | tee lemon_single.log
bash scripts/validate_lemon_grasp_multi.sh 5 | tee lemon_multi.log
```

For a change to count as an improvement, require:

- `success: True`
- `grasp_failure_mode: success`
- `[close_gripper] grasp confirmed + squeezed`
- `[micro_lift] ... follows=True`
- `[act] post-lift verified`
- no increase in `no_grasp` or `object NOT lifted`
- lower or equal `pre_close_abort`
- lower episode runtime or fewer max-step warnings
