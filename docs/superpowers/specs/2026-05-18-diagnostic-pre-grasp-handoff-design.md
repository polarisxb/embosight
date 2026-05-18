---
title: Diagnostic Pre-Grasp Handoff and Recovery Design
date: 2026-05-18
status: draft-for-review
---

# Diagnostic Pre-Grasp Handoff and Recovery Design

## Goal

Replace the current fixed-distance pre-grasp success rule with a diagnostic, approach-frame handoff and recovery system for EmboSight grasp execution.

The immediate trigger is the fixed lemon validation: after restoring base-frame arm OSC commands and safe base navigation, the arm reliably reaches a residual of roughly 6-7 cm from the top-down pre-grasp target but fails before entering `descend`. A simple threshold change can unblock that case, but it does not generalize across object sizes, grasp strategies, or approach directions.

The goal is to make pre-grasp execution answer a more meaningful question:

```text
Is the robot safely positioned to hand off to the next contact-aware approach stage?
```

rather than:

```text
Is the end effector within one fixed Euclidean distance of the pre-grasp point?
```

## Non-Goals

- Do not implement full whole-body motion planning, contact MPC, or a global IK planner in this phase.
- Do not tune a single global threshold such as `0.06`, `0.08`, or `0.10` as the final solution.
- Do not remove the existing staged pipeline: navigate → pre-grasp → approach/descend → close → lift.
- Do not break legacy mocks or tests that only implement `move_to_pre_grasp(candidate) -> bool`.
- Do not change semantic grasp strategy selection in this phase.

## Current Evidence

Recent GPU probes and validation runs show this failure chain:

1. `navigate_base_to(offset=0.30)` places PandaOmron in an arm-control-degenerate state. Arm OSC pulses collapse to tiny common drift. `reset_goal()` does not fix it.
2. Offsets below `0.55m` are unsafe for the lemon case; `0.55m` restores useful arm motion, while `0.65m` resembles the reset baseline.
3. The right-arm OSC expects mobile-base-frame delta commands, not world-frame commands.
4. After fixing frame handling and safe offset, `move_arm_to` can reduce pre-grasp error from roughly `0.48m` to roughly `0.06-0.09m`.
5. A fixed pre-grasp Euclidean threshold then blocks the pipeline before `descend`, even though the next stage is designed to be contact-aware.
6. However, the observed residual is mostly lateral in the top-down approach frame, so simply accepting all `0.08m` residuals would be unsafe for small objects.

Therefore the root issue is not one parameter. It is a missing execution diagnostic layer between pre-grasp motion and contact-aware approach.

## Recommended Approach

Implement a C+ design:

```text
Diagnostic pre-grasp handoff
+ approach-frame error decomposition
+ object/skill-scale normalized gates
+ base/arm recovery
+ explicit failure taxonomy
```

This keeps the current pipeline structure but changes pre-grasp from a Boolean black box into an explainable guarded primitive.

## Architecture

### Components

1. **Pre-grasp diagnostic types**
   - Add a small execution-telemetry module: `src/grasp_execution.py`.
   - Define `PreGraspResult` and reason enums/strings.
   - Keep these types independent of RoboCasa internals so they are easy to test.

2. **Approach-frame error decomposition**
   - Add a pure helper that decomposes final EEF error into:
     - lateral error: error orthogonal to the approach direction
     - approach-axis error: error along the approach direction
     - approach gap: distance from current EEF to grasp point along the approach direction
   - This works for top-down, side, and tilted approaches.

3. **Diagnostic pre-grasp primitive**
   - Add `EnvWrapper.move_to_pre_grasp_diagnostic(candidate, height_m=0.05) -> PreGraspResult`.
   - Keep `EnvWrapper.move_to_pre_grasp(candidate, ...) -> bool` as a compatibility wrapper.
   - The diagnostic method performs the same physical motion but returns structured evidence.

4. **ActionExecutor recovery policy**
   - `ActionExecutor.act()` should prefer the diagnostic method when available.
   - If pre-grasp succeeds or handoff is safe, continue to `approach`.
   - If lateral error is too large, run a bounded base/arm recovery and retry pre-grasp once.
   - If recovery still fails, return a specific failure mode instead of generic `ik_unreachable`.

5. **Telemetry and oracle support**
   - Log diagnostic fields in a compact, parseable format.
   - Include pre-grasp failure reasons in `GraspActionResult.details` so evaluation can report meaningful failure modes.

## Data Model

### `PreGraspResult`

Proposed fields:

```python
@dataclass
class PreGraspResult:
    ok: bool
    handoff_ok: bool
    needs_recovery: bool
    reason: str
    final_eef: np.ndarray
    pre_pos: np.ndarray
    grasp_point: np.ndarray
    approach_dir: np.ndarray
    total_error_m: float
    lateral_error_m: float
    axis_error_m: float
    approach_gap_m: float
    lateral_limit_m: float
    min_approach_gap_m: float
    max_approach_gap_m: float
    move_ok: bool
```

Field meanings:

- `ok`: strict pre-grasp motion met the normal precise threshold.
- `handoff_ok`: not strictly converged, but safe to continue into `approach` / `descend`.
- `needs_recovery`: the error pattern is recoverable by base/arm adjustment.
- `reason`: machine-readable reason string.
- `final_eef`: EEF position after the pre-grasp attempt.
- `pre_pos`: intended pre-grasp point.
- `grasp_point`: target grasp point on or near the object.
- `approach_dir`: normalized direction from pre-grasp toward the object.
- `lateral_error_m`: offset orthogonal to `approach_dir`.
- `axis_error_m`: offset along `approach_dir` relative to `pre_pos`.
- `approach_gap_m`: distance from current EEF to the grasp point along `approach_dir`.
- `*_limit_m`: the active guard thresholds used for this candidate.
- `move_ok`: raw result from `move_arm_to`.

### Failure / handoff reasons

Use stable reason strings:

```text
strict_ok
safe_handoff
lateral_misaligned
axis_gap_too_small
axis_gap_too_large
below_grasp_point
orientation_blocked
base_recovery_failed
pre_grasp_unreachable
legacy_bool_failed
```

The exact string list should remain small and documented. These strings can be used by tests, logs, and oracle summaries.

## Approach-Frame Error Decomposition

Given:

```text
p = final_eef
pre = pre_grasp target
g = grasp point
ad = unit approach direction from pre-grasp toward object
```

Compute:

```text
pre_error = pre - p
axis_error_signed = dot(pre_error, ad)
lateral_error = norm(pre_error - axis_error_signed * ad)
axis_error = abs(axis_error_signed)
approach_gap = dot(g - p, ad)
total_error = norm(pre_error)
```

For top-down:

```text
ad = [0, 0, -1]
lateral_error ≈ XY offset
approach_gap ≈ EEF height above grasp point
```

For side grasp:

```text
lateral_error captures vertical and sideways offset around the side approach ray
approach_gap captures remaining forward distance to the grasp point
```

This avoids treating all residual directions as equally important.

## Handoff Gates

A pre-grasp attempt may hand off to `approach` only when all required guards pass.

The raw `move_arm_to` convergence threshold remains an internal strict-motion
criterion, not the semantic handoff criterion:

```text
top_down_strict_pre_thresh_m = 0.06
side_or_tilted_strict_pre_thresh_m = 0.12
```

This preserves the old meaning of strict pre-grasp success while allowing
diagnostics to decide whether a non-strict result is safe to hand off or must
recover.

### Lateral gate

The EEF must be close enough to the approach ray:

```text
lateral_error_m <= lateral_limit_m
```

Use candidate scale rather than a single global value:

```text
finger_width = candidate.finger_width_m or 0.04
lateral_limit_m = clamp(0.5 * finger_width, min=0.015, max=0.045)
```

Rationale:

- Small objects require tight lateral alignment.
- Wider grasp strategies can tolerate more lateral error.
- The clamp prevents extreme values from making the guard useless.

### Approach-gap gate

The EEF must still be on the safe approach side of the object:

```text
min_approach_gap_m <= approach_gap_m <= max_approach_gap_m
```

Initial implementation values:

```text
min_approach_gap_m = 0.010
max_approach_gap_m = max(height_m + 0.030, 0.080)
```

Rationale:

- If the gap is too small or negative, the EEF may already be in contact, below, or past the grasp point.
- If the gap is too large, the arm has not established a useful pre-grasp configuration.

### Orientation gate

Orientation should not block top-down handoff unless it creates an unsafe geometry condition.

Recommended policy:

- For top-down, prioritize lateral and approach-gap gates; defer fine orientation to later stages when possible.
- For side or tilted approaches, keep a looser orientation guard because wrong orientation can cause side collisions.
- If orientation is known to be the dominant failure, return `orientation_blocked` rather than `ik_unreachable`.

## Recovery Policy

### When to recover

Run one bounded recovery attempt when:

```text
move_ok is False
handoff_ok is False
reason == lateral_misaligned
```

Do not recover when:

- `approach_gap_m` is negative or below the safety minimum.
- The EEF is below the grasp point for top-down.
- The previous base navigation already placed the arm in a known degenerate close-offset state.

### Recovery actions

Use a two-level policy:

1. **Lateral base nudge**
   - Compute residual lateral direction in world XY.
   - Move the mobile base by a clipped correction with initial cap `max_base_nudge_m = 0.06`.
   - Preserve safe target distance bounds; for PandaOmron, do not move closer than `0.55m` to the grasp point.
   - Sync observations after teleport or base movement.

2. **Safe-offset retry**
   - If lateral nudge is unavailable or unsafe, retry `navigate_base_to` with `offset_m = 0.65`.
   - Retry pre-grasp once.
   - Do not loop indefinitely.

The recovery should be generic: it responds to measured residual geometry rather than object label.

## ActionExecutor Flow

New preferred flow:

```text
candidate = next unused candidate
navigate_base_to(candidate.point_3d[:2], offset >= safe minimum)

if env has move_to_pre_grasp_diagnostic:
    result = env.move_to_pre_grasp_diagnostic(candidate)
    if result.ok or result.handoff_ok:
        continue to approach
    if result.needs_recovery:
        run one recovery and retry diagnostic pre-grasp
        if retry.ok or retry.handoff_ok:
            continue to approach
    return failed_result(candidate, result.reason, result details)
else:
    use legacy move_to_pre_grasp(candidate) bool path

approach(...)
close_gripper(...)
lift(...)
```

The legacy path keeps existing unit tests and fake environments working.

## Compatibility Plan

- Keep `move_to_pre_grasp(candidate) -> bool`.
- Add `move_to_pre_grasp_diagnostic` as an optional richer API.
- Update real `EnvWrapper` to implement the diagnostic API.
- Update `ActionExecutor` to use the diagnostic API only when present.
- Existing fake envs can continue to implement only the bool method.
- Tests should cover both the diagnostic path and legacy bool fallback.

## Testing Plan

### Unit tests for pure math

Add tests for approach-frame decomposition:

- top-down: XY residual maps to lateral error.
- top-down: Z residual maps to approach-axis error.
- side approach: horizontal approach residual maps to approach-axis error.
- zero or invalid approach direction normalizes to top-down.

### EnvWrapper diagnostic tests

Add mocked `EnvWrapper` tests for:

- strict convergence returns `ok=True`, `reason=strict_ok`.
- top-down lateral error within candidate-scaled limit returns `handoff_ok=True`.
- top-down lateral error above limit returns `needs_recovery=True`, `reason=lateral_misaligned`.
- negative approach gap returns no handoff.
- legacy `move_to_pre_grasp()` returns `ok or handoff_ok`.

### ActionExecutor tests

Add tests for:

- Diagnostic pre-grasp handoff proceeds to `approach`.
- Lateral misalignment triggers exactly one recovery retry.
- Recovery success proceeds to `approach`.
- Recovery failure returns a specific failure mode, not generic `ik_unreachable`.
- Legacy envs without diagnostic API still work.

### GPU validation

Run after implementation:

```bash
bash scripts/validate_lemon_grasp.sh
```

Expected improvement:

- The pipeline no longer fails immediately at pre-grasp due to a fixed Euclidean threshold.
- If the residual is mostly lateral and too large, logs show `lateral_misaligned` and a bounded recovery attempt.
- If recovery succeeds, logs show transition into `[descend]`.
- If recovery fails, the oracle reports a meaningful pre-grasp failure reason.

## Logging and Evaluation

Log one compact line after diagnostic pre-grasp:

```text
[pre_grasp_diag] move_ok=false handoff=false reason=lateral_misaligned total=0.069 lateral=0.064 axis=0.005 gap=0.046 lateral_limit=0.020
```

Include the same values in `GraspActionResult.details` when pre-grasp fails.

This improves scientific reporting because failures can be counted by cause:

- base-control degeneracy
- pre-grasp lateral misalignment
- approach-axis / z limit
- descend z stall
- contact missing
- gripper empty
- slipped lift

## Rollback Plan

If the diagnostic design introduces regressions:

1. Disable diagnostic use in `ActionExecutor` and fall back to the legacy bool path.
2. Keep the pure helper tests and diagnostic dataclass if they are harmless.
3. Re-run fixed lemon validation and the full unit suite.
4. Do not return to global threshold tuning as the final design; use diagnostics to identify the next failing guard.

## Success Criteria

The design is successful when:

- Pre-grasp no longer depends on a single global Euclidean threshold.
- Failures distinguish lateral misalignment, approach-axis problems, orientation blocking, and base recovery failure.
- Small-object handoff is stricter than large-object handoff through candidate scale.
- Existing mocks and legacy tests remain compatible.
- Lemon validation reaches `descend` or reports a specific recovery failure instead of generic `ik_unreachable`.
- The full unit test suite passes.

## Implementation Phases

1. Add pure diagnostic types and approach-frame error helper.
2. Add `move_to_pre_grasp_diagnostic` in `EnvWrapper` while preserving the bool wrapper.
3. Update `ActionExecutor` to consume diagnostics and keep legacy fallback.
4. Add bounded base recovery for `lateral_misaligned`.
5. Add tests and run GPU lemon validation.

## Design Decision

The final design intentionally avoids claiming that `0.08m` is universally safe. A coarse Euclidean threshold may remain as an internal raw move threshold, but it must not be the semantic success criterion. The semantic criterion is guarded handoff safety in the grasp approach frame, with object/skill-scale normalization and explicit recovery.
