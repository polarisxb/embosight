# Lemon Grasp Validation Summary

Date: 2026-05-19

Commit under test: `6583446 Report actual post-lift grasp coordinates`

Command:

```bash
bash scripts/validate_lemon_grasp.sh | tee lemon_real_pos.log
```

Scenario: `fixed_lemon_001`

Query: `pick up the lemon`

## Verdict

This run is a real physical grasp success. It is not just a speech or oracle false positive.

However, the run did not succeed on the first top-down attempt. The first attempt pushed or lost alignment with the lemon and was correctly aborted before closing. The episode succeeded on the second grasp attempt after the target position had been refreshed.

In short:

- True final grasp: yes.
- First attempt success: no.
- Success path quality: acceptable for recovery, not yet good enough for first-try reliability.
- Remaining risk: the system can still report some intermediate states in misleading ways unless downstream summaries distinguish planned target, live object pose, and final post-lift pose.

## Object And Target Evidence

The environment placed the task target as a counter distractor, not as `obj_main`:

```text
[obj_types] runtime object categories: {'obj_main': 'tupperware', 'distr_counter_main': 'lemon', 'distr_cab_main': 'steak'}
```

The oracle summary still reports:

```json
"actual_object": "tupperware",
"selected_target_label": "lemon",
"selected_target_position": [
  0.18830765783786774,
  -2.8358876705169678,
  0.9469208121299744
]
```

This is not contradictory. `actual_object` is the scenario's `obj_main`, while the requested target is the lemon on the counter, `distr_counter_main`.

The `selected_target_position` is the target estimate before the final lift. It is not the final held object position.

## First Attempt: Correctly Aborted, Not Successful

The first strategy selected was top-down:

```text
[agent] grasp strategy: top_down
[grasp_pose] 'distr_counter_main' AABB z=[0.918,0.976] h=0.058m -> wrist_z=0.950
[grasp_planner] strategy=top_down -> approach=[ 0.  0. -1.] width=0.04m
```

The pre-grasp motion failed to reach clean alignment:

```text
[move_arm_to] max_steps reached, dist=0.0665m ori_err=0.9237rad
[pre_grasp_diag] move_ok=False handoff=False reason=lateral_misaligned total=0.066 lateral=0.066 axis=0.006 gap=0.044 lateral_limit=0.020
```

The executor tried a base nudge and continued descending, but the object ended up far from the old candidate point:

```text
[pre_grasp_align] eef=(0.123,-2.860,0.940) obj=(0.188,-2.836,0.947) lateral=0.0693m z_diff=-0.0068m
[pre_close_align] abort: eef=(0.123,-2.860,0.940) obj=(0.188,-2.836,0.947) candidate_xy=(0.125,-2.858) lateral=0.0693m > limit=0.0200m
```

This is a successful safety check, not a successful grasp. The system avoided closing on a stale target that was about 6.9 cm away laterally.

## Second Attempt: Real Grasp Success

After the first abort, the selected target position was refreshed to the live object position:

```json
"selected_target_position": [
  0.18830765783786774,
  -2.8358876705169678,
  0.9469208121299744
]
```

The second strategy log says `gentle_side`:

```text
[agent] grasp strategy: gentle_side | reason: Top_down failed previously due to slipping...
```

But the successful grasp candidate is still reported as:

```json
"grasp_candidate_source": "vlm_top_grasp"
```

So the physical grasp that succeeded should be interpreted as a refreshed top grasp candidate with the second strategy's safer parameters, not as a pure side grasp.

The gripper made contact and squeezed:

```text
[close_gripper] contact at step 6, squeezing 28 more
[close_gripper] grasp confirmed + squeezed at step 33
```

The micro-lift check showed the object followed the gripper:

```text
[micro_lift] eef Δz=0.0103 obj Δz=0.0098 follows=True
```

The full lift verification showed a meaningful object height change:

```text
[act] post-lift verified: obj Δz=0.089 (0.947→1.036)
```

These three signals together are the main evidence for a real grasp:

1. Contact and squeeze were confirmed.
2. The object followed a micro-lift.
3. The object rose by about 8.9 cm after the final lift.

## Final Speech Evidence

The final speech now reports the post-lift simulator body position:

```text
speech  : 已为您拿到lemon，当前物体世界坐标约 x=0.192m，y=-2.831m，z=1.036m。
```

This differs from the pre-lift target estimate:

```json
"selected_target_position": [
  0.18830765783786774,
  -2.8358876705169678,
  0.9469208121299744
]
```

The `z` value changed from about `0.947m` to `1.036m`, matching the post-lift verification. That means the speech is no longer reusing the stale belief target coordinate.

The x/y values also shifted slightly from the selected target estimate:

- selected target x/y: `(0.188, -2.836)`
- post-lift object x/y: `(0.192, -2.831)`

That small shift is expected after contact, squeezing, micro-lift, and lift.

## Misleading Signals To Treat Carefully

### `success: True`

The episode-level success is valid in this run, but it hides the important fact that the first attempt failed and recovery was required.

### `selected_target_position`

This is a selected target estimate. It is not the final object pose after grasp. Use `post_lift_obj_pos` / final speech / post-lift logs for final position.

### `pre_grasp_eval ... safe_handoff`

The log says:

```text
[pre_grasp_eval] at_current handoff=True reason=safe_handoff total=0.011 lateral=0.009 axis=0.006 gap=0.044 lateral_limit=0.020
```

But later, live object alignment showed:

```text
lateral=0.0693m > limit=0.0200m
```

This suggests `pre_grasp_eval` can be evaluating the handoff against the stale candidate or end-effector state, not the latest live object body pose. The final pre-close alignment check is currently more trustworthy.

### Strategy Attribution

The agent selected `gentle_side` for the second attempt, but the successful candidate source was `vlm_top_grasp`.

This can pollute memory if the system records "gentle_side succeeded" while the executed physical candidate was top-grasp shaped. Future logs and memory should record both:

- selected LLM strategy
- executed candidate source / approach direction

## Current Failure Pattern

The fixed-seed lemon run currently follows this pattern:

1. Top-down attempt plans against the initial lemon pose.
2. Pre-grasp motion is laterally misaligned.
3. Recovery and descent push or lose alignment with the lemon.
4. `pre_close_align` detects stale target and aborts before closing.
5. The target pose is refreshed.
6. The second grasp attempt succeeds.

This is a valid recovery-based success, but not a robust first-try grasp.

## Follow-up Fix Implemented After This Run

After reviewing the baseline log, two follow-up fixes were implemented in the
current change set.

### Live Object Refresh After Z-stall Base Nudge

Before the fix, z-stall recovery could nudge the base and then re-align toward
the old candidate XY. The baseline evidence was:

```text
[act] post-nudge lateral re-align: offset=0.049m, target_xy=(0.125, -2.858) z=0.948
[pre_grasp_align] eef=(0.123,-2.860,0.940) obj=(0.188,-2.836,0.947) lateral=0.0693m
```

The executor now refreshes candidate XY from the live simulator body pose after
z-stall nudge and before post-nudge lateral re-align. When the live object and
candidate differ by more than `0.025m`, it updates:

- `candidate.point_3d[:2]`
- `target.position_3d`
- `target.pose_estimate.position`, when present

Expected GPU evidence after this fix:

```text
[live_obj_refresh] stage=post_z_stall_nudge drift=...
```

The goal is to prevent the old pattern where the arm keeps descending toward
`candidate_xy=(0.125,-2.858)` after the lemon has moved near
`obj=(0.188,-2.836)`.

### Strategy Attribution Separation

Before the fix, the run selected `gentle_side` for the second attempt, but the
successful candidate source was `vlm_top_grasp`. This could cause memory to
learn the wrong lesson.

The agent now records these fields separately in successful and failed grasp
attempt diagnostics:

- `selected_strategy`
- `executed_strategy`
- `candidate_source`
- `approach_dir`
- `finger_width_m`
- `depth_margin_m`

For the baseline pattern, this should be represented as:

```text
selected_strategy = gentle_side
candidate_source  = vlm_top_grasp
executed_strategy = top_down
```

This means memory no longer has to treat "LLM selected gentle_side" as the same
thing as "the physical executed candidate was a side grasp".

### Local Verification Evidence

The follow-up fixes were verified locally with:

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

GPU validation is still required to confirm whether `pre_close_abort` drops from
`1` to `0` in `fixed_lemon_001`.

## Validation Criteria Going Forward

For future lemon validation runs, do not use episode `success` alone. Track these signals together:

- `success: True`
- `grasp_failure_mode: success`
- `[close_gripper] grasp confirmed + squeezed`
- `[micro_lift] ... follows=True`
- `[act] post-lift verified`
- final speech includes post-lift `x/y/z`
- `pre_close_abort` count
- number of `move_arm_to max_steps reached`
- total episode time

The current run passes the true-grasp checks, but still has optimization debt:

- `pre_close_abort = 1`
- many `move_arm_to max_steps reached`
- total runtime: `351.5s`
