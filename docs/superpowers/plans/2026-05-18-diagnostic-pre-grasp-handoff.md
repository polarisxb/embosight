# Diagnostic Pre-Grasp Handoff Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace fixed pre-grasp Euclidean threshold semantics with diagnostic approach-frame handoff, candidate-scale guards, and bounded recovery.

**Architecture:** Add a focused `src/grasp_execution.py` module for pure dataclasses and geometry helpers. Extend `EnvWrapper` with `move_to_pre_grasp_diagnostic()` while preserving `move_to_pre_grasp()` as a bool compatibility wrapper. Update `ActionExecutor` to consume diagnostics when available, recover once from lateral misalignment, and keep legacy fallback for mocks.

**Tech Stack:** Python 3.10, dataclasses, NumPy, pytest, existing EmboSight `EnvWrapper`, `ActionExecutor`, and `GraspCandidate` data model.

---

## File Structure

- Create `src/grasp_execution.py`
  - Owns `PreGraspResult`, reason constants, `normalize_approach_dir()`, `decompose_pre_grasp_error()`, and `evaluate_pre_grasp_handoff()`.
  - No RoboCasa imports.

- Create `tests/test_grasp_execution.py`
  - Pure unit tests for approach-frame decomposition and candidate-scale handoff gates.

- Modify `src/env_wrapper.py`
  - Import helpers from `src.grasp_execution`.
  - Add `move_to_pre_grasp_diagnostic(candidate, height_m=0.05)`.
  - Change `move_to_pre_grasp()` into a compatibility wrapper returning `diagnostic.ok or diagnostic.handoff_ok`.
  - Restore strict top-down raw move threshold to `0.06`; non-strict handoff is handled by diagnostics.

- Modify `src/action_executor.py`
  - Prefer `move_to_pre_grasp_diagnostic()` when available.
  - Continue when `ok` or `handoff_ok`.
  - On `needs_recovery`, call a bounded pre-grasp recovery method once, retry diagnostic pre-grasp, then continue or fail with a specific reason.
  - Keep bool fallback for env mocks.

- Modify `tests/test_env_wrapper_grasp.py`
  - Replace the temporary `0.08` threshold expectation with diagnostic handoff/recovery tests.

- Modify `tests/test_action_executor_v1.py`
  - Add diagnostic path tests for handoff, recovery success, recovery failure, and legacy fallback.

## Task 1: Pure diagnostic model and geometry helper

**Files:**
- Create: `src/grasp_execution.py`
- Create: `tests/test_grasp_execution.py`

- [ ] **Step 1: Write failing tests for top-down decomposition and lateral gate**

Add `tests/test_grasp_execution.py` with tests equivalent to:

```python
import numpy as np

from src.grasp_execution import (
    PRE_GRASP_LATERAL_MISALIGNED,
    PRE_GRASP_SAFE_HANDOFF,
    decompose_pre_grasp_error,
    evaluate_pre_grasp_handoff,
)


def test_top_down_xy_error_maps_to_lateral_error():
    final_eef = np.array([0.13, -2.80, 0.98], dtype=np.float32)
    pre_pos = np.array([0.125, -2.86, 0.982], dtype=np.float32)
    grasp_point = np.array([0.125, -2.86, 0.932], dtype=np.float32)
    approach_dir = np.array([0.0, 0.0, -1.0], dtype=np.float32)

    d = decompose_pre_grasp_error(final_eef, pre_pos, grasp_point, approach_dir)

    assert d.lateral_error_m > 0.055
    assert d.lateral_error_m < 0.065
    assert d.approach_gap_m > 0.04
    assert d.approach_gap_m < 0.06


def test_small_object_lateral_error_requires_recovery():
    final_eef = np.array([0.13, -2.80, 0.98], dtype=np.float32)
    pre_pos = np.array([0.125, -2.86, 0.982], dtype=np.float32)
    grasp_point = np.array([0.125, -2.86, 0.932], dtype=np.float32)
    approach_dir = np.array([0.0, 0.0, -1.0], dtype=np.float32)

    result = evaluate_pre_grasp_handoff(
        move_ok=False,
        final_eef=final_eef,
        pre_pos=pre_pos,
        grasp_point=grasp_point,
        approach_dir=approach_dir,
        finger_width_m=0.04,
        height_m=0.05,
    )

    assert result.handoff_ok is False
    assert result.needs_recovery is True
    assert result.reason == PRE_GRASP_LATERAL_MISALIGNED


def test_candidate_scaled_lateral_error_can_handoff_when_inside_limit():
    final_eef = np.array([0.505, 0.01, 1.00], dtype=np.float32)
    pre_pos = np.array([0.50, 0.00, 1.00], dtype=np.float32)
    grasp_point = np.array([0.50, 0.00, 0.95], dtype=np.float32)
    approach_dir = np.array([0.0, 0.0, -1.0], dtype=np.float32)

    result = evaluate_pre_grasp_handoff(
        move_ok=False,
        final_eef=final_eef,
        pre_pos=pre_pos,
        grasp_point=grasp_point,
        approach_dir=approach_dir,
        finger_width_m=0.04,
        height_m=0.05,
    )

    assert result.handoff_ok is True
    assert result.needs_recovery is False
    assert result.reason == PRE_GRASP_SAFE_HANDOFF
```

- [ ] **Step 2: Run tests to verify RED**

Run: `python -m pytest tests/test_grasp_execution.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'src.grasp_execution'`.

- [ ] **Step 3: Implement `src/grasp_execution.py` minimally**

Implement:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PRE_GRASP_STRICT_OK = "strict_ok"
PRE_GRASP_SAFE_HANDOFF = "safe_handoff"
PRE_GRASP_LATERAL_MISALIGNED = "lateral_misaligned"
PRE_GRASP_AXIS_GAP_TOO_SMALL = "axis_gap_too_small"
PRE_GRASP_AXIS_GAP_TOO_LARGE = "axis_gap_too_large"
PRE_GRASP_BELOW_GRASP_POINT = "below_grasp_point"
PRE_GRASP_UNREACHABLE = "pre_grasp_unreachable"


@dataclass
class PreGraspDecomposition:
    total_error_m: float
    lateral_error_m: float
    axis_error_m: float
    approach_gap_m: float


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


def normalize_approach_dir(approach_dir) -> np.ndarray:
    ad = np.asarray(approach_dir, dtype=np.float32)
    if ad.shape[0] > 3:
        ad = ad[:3]
    norm = float(np.linalg.norm(ad))
    if norm < 1e-6:
        return np.array([0.0, 0.0, -1.0], dtype=np.float32)
    return ad / norm


def decompose_pre_grasp_error(final_eef, pre_pos, grasp_point, approach_dir) -> PreGraspDecomposition:
    p = np.asarray(final_eef, dtype=np.float32)[:3]
    pre = np.asarray(pre_pos, dtype=np.float32)[:3]
    g = np.asarray(grasp_point, dtype=np.float32)[:3]
    ad = normalize_approach_dir(approach_dir)
    pre_error = pre - p
    axis_signed = float(np.dot(pre_error, ad))
    lateral_vec = pre_error - axis_signed * ad
    approach_gap = float(np.dot(g - p, ad))
    return PreGraspDecomposition(
        total_error_m=float(np.linalg.norm(pre_error)),
        lateral_error_m=float(np.linalg.norm(lateral_vec)),
        axis_error_m=abs(axis_signed),
        approach_gap_m=approach_gap,
    )


def lateral_limit_for_finger_width(finger_width_m: float | None) -> float:
    width = 0.04 if finger_width_m is None else float(finger_width_m)
    return float(np.clip(0.5 * width, 0.015, 0.045))


def evaluate_pre_grasp_handoff(
    *,
    move_ok: bool,
    final_eef,
    pre_pos,
    grasp_point,
    approach_dir,
    finger_width_m: float | None,
    height_m: float,
) -> PreGraspResult:
    ad = normalize_approach_dir(approach_dir)
    d = decompose_pre_grasp_error(final_eef, pre_pos, grasp_point, ad)
    lateral_limit = lateral_limit_for_finger_width(finger_width_m)
    min_gap = 0.010
    max_gap = max(float(height_m) + 0.030, 0.080)
    p = np.asarray(final_eef, dtype=np.float32)[:3]
    g = np.asarray(grasp_point, dtype=np.float32)[:3]

    if move_ok:
        reason = PRE_GRASP_STRICT_OK
        handoff_ok = True
        needs_recovery = False
    elif d.approach_gap_m < min_gap:
        reason = PRE_GRASP_AXIS_GAP_TOO_SMALL
        handoff_ok = False
        needs_recovery = False
    elif ad[2] < -0.9 and p[2] < g[2]:
        reason = PRE_GRASP_BELOW_GRASP_POINT
        handoff_ok = False
        needs_recovery = False
    elif d.approach_gap_m > max_gap:
        reason = PRE_GRASP_AXIS_GAP_TOO_LARGE
        handoff_ok = False
        needs_recovery = False
    elif d.lateral_error_m <= lateral_limit:
        reason = PRE_GRASP_SAFE_HANDOFF
        handoff_ok = True
        needs_recovery = False
    else:
        reason = PRE_GRASP_LATERAL_MISALIGNED
        handoff_ok = False
        needs_recovery = True

    return PreGraspResult(
        ok=bool(move_ok),
        handoff_ok=handoff_ok,
        needs_recovery=needs_recovery,
        reason=reason,
        final_eef=p,
        pre_pos=np.asarray(pre_pos, dtype=np.float32)[:3],
        grasp_point=g,
        approach_dir=ad,
        total_error_m=d.total_error_m,
        lateral_error_m=d.lateral_error_m,
        axis_error_m=d.axis_error_m,
        approach_gap_m=d.approach_gap_m,
        lateral_limit_m=lateral_limit,
        min_approach_gap_m=min_gap,
        max_approach_gap_m=max_gap,
        move_ok=bool(move_ok),
    )
```

- [ ] **Step 4: Run tests to verify GREEN**

Run: `python -m pytest tests/test_grasp_execution.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/grasp_execution.py tests/test_grasp_execution.py
git commit -m "feat(grasp): add pre-grasp diagnostic geometry"
```

## Task 2: EnvWrapper diagnostic pre-grasp API

**Files:**
- Modify: `src/env_wrapper.py`
- Modify: `tests/test_env_wrapper_grasp.py`

- [ ] **Step 1: Write failing tests for diagnostic API and wrapper compatibility**

Add tests to `tests/test_env_wrapper_grasp.py` that assert:

```python
def test_move_to_pre_grasp_diagnostic_reports_lateral_misalignment():
    env = PregraspThresholdEnv()
    env.final_eef = np.array([0.56, 0.2, 0.95], dtype=np.float32)
    candidate = GraspCandidate(
        point_3d=np.array([0.5, 0.2, 0.9], dtype=np.float32),
        approach_dir=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        finger_width_m=0.04,
        score=0.8,
    )

    result = env.move_to_pre_grasp_diagnostic(candidate)

    assert result.ok is False
    assert result.handoff_ok is False
    assert result.needs_recovery is True
    assert result.reason == "lateral_misaligned"


def test_move_to_pre_grasp_bool_accepts_safe_diagnostic_handoff():
    env = PregraspThresholdEnv()
    env.final_eef = np.array([0.51, 0.2, 0.95], dtype=np.float32)
    candidate = GraspCandidate(
        point_3d=np.array([0.5, 0.2, 0.9], dtype=np.float32),
        approach_dir=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        finger_width_m=0.04,
        score=0.8,
    )

    assert env.move_to_pre_grasp(candidate) is True
    assert env.move_calls[-1][2] == 0.06
```

Update `PregraspThresholdEnv.get_eef_pos()` so it returns `self.final_eef` if present.

- [ ] **Step 2: Run tests to verify RED**

Run: `python -m pytest tests/test_env_wrapper_grasp.py::test_move_to_pre_grasp_diagnostic_reports_lateral_misalignment tests/test_env_wrapper_grasp.py::test_move_to_pre_grasp_bool_accepts_safe_diagnostic_handoff -q`

Expected: FAIL because `move_to_pre_grasp_diagnostic` does not exist or `threshold_m` is still `0.08`.

- [ ] **Step 3: Implement diagnostic method in `EnvWrapper`**

Add imports at the top of `src/env_wrapper.py`:

```python
from src.grasp_execution import evaluate_pre_grasp_handoff, normalize_approach_dir
```

Add method `move_to_pre_grasp_diagnostic()` by extracting current logic from `move_to_pre_grasp()`, calling `move_arm_to()` with strict threshold `0.06` for top-down and `0.12` otherwise, then evaluating handoff from final EEF.

Change `move_to_pre_grasp()` to:

```python
def move_to_pre_grasp(self, candidate, height_m: float = 0.05) -> bool:
    result = self.move_to_pre_grasp_diagnostic(candidate, height_m=height_m)
    return bool(result.ok or result.handoff_ok)
```

- [ ] **Step 4: Run tests to verify GREEN**

Run: `python -m pytest tests/test_env_wrapper_grasp.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/env_wrapper.py tests/test_env_wrapper_grasp.py
git commit -m "feat(grasp): add diagnostic pre-grasp primitive"
```

## Task 3: ActionExecutor diagnostic consumption and bounded recovery hook

**Files:**
- Modify: `src/action_executor.py`
- Modify: `tests/test_action_executor_v1.py`

- [ ] **Step 1: Write failing ActionExecutor tests**

Add fake envs in `tests/test_action_executor_v1.py` that implement `move_to_pre_grasp_diagnostic()` returning simple objects with fields `ok`, `handoff_ok`, `needs_recovery`, `reason`, and numeric diagnostics.

Add tests asserting:

```python
def test_act_continues_when_diagnostic_handoff_ok():
    env = _DiagnosticPreGraspEnv(result_reason="safe_handoff", handoff_ok=True)
    exe = ActionExecutor(scene_describer=None)
    h, _ = _hyp_with_candidate()

    result = exe.act(h, DecomposedTask(primary_target="apple"), env)

    assert result.success is True
    assert "approach" in env.calls


def test_act_retries_once_for_lateral_pre_grasp_recovery():
    env = _DiagnosticPreGraspEnv(
        first_reason="lateral_misaligned",
        first_needs_recovery=True,
        second_reason="safe_handoff",
        second_handoff_ok=True,
    )
    exe = ActionExecutor(scene_describer=None)
    h, _ = _hyp_with_candidate()

    result = exe.act(h, DecomposedTask(primary_target="apple"), env)

    assert result.success is True
    assert env.recovery_calls == 1
    assert env.pre_grasp_diag_calls == 2


def test_act_reports_specific_pre_grasp_reason_after_recovery_failure():
    env = _DiagnosticPreGraspEnv(
        first_reason="lateral_misaligned",
        first_needs_recovery=True,
        second_reason="base_recovery_failed",
        second_handoff_ok=False,
    )
    exe = ActionExecutor(scene_describer=None)
    h, _ = _hyp_with_candidate()

    result = exe.act(h, DecomposedTask(primary_target="apple"), env)

    assert result.success is False
    assert result.failure_mode == "base_recovery_failed"
    assert result.details["stage"] == "pre_grasp"
```

- [ ] **Step 2: Run tests to verify RED**

Run: `python -m pytest tests/test_action_executor_v1.py -q`

Expected: FAIL because `ActionExecutor` ignores diagnostic pre-grasp.

- [ ] **Step 3: Implement diagnostic path and recovery helper**

In `src/action_executor.py`:

- Prefer `env.move_to_pre_grasp_diagnostic(candidate)` when present.
- Add `_pre_grasp_details(result)` to serialize diagnostic fields.
- Add `_recover_pre_grasp(env, candidate, result)`:
  - If env has `recover_pre_grasp`, call it.
  - Else if env has `navigate_base_to`, call it with `offset_m=0.65`.
  - Return `True` if recovery call did not raise.
- Retry diagnostic once after recovery.
- Return `self._failed_result(candidate, result.reason, details, env)` on failure.

- [ ] **Step 4: Run tests to verify GREEN**

Run: `python -m pytest tests/test_action_executor_v1.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/action_executor.py tests/test_action_executor_v1.py
git commit -m "feat(grasp): consume pre-grasp diagnostics in executor"
```

## Task 4: Verification and cleanup

**Files:**
- Modify if needed: `docs/superpowers/specs/2026-05-18-diagnostic-pre-grasp-handoff-design.md`

- [ ] **Step 1: Run focused tests**

Run:

```bash
python -m pytest tests/test_grasp_execution.py tests/test_env_wrapper_grasp.py tests/test_action_executor_v1.py tests/test_env_wrapper_orientation.py tests/test_env_wrapper_real_base_ori.py -q
```

Expected: PASS.

- [ ] **Step 2: Run full tests**

Run:

```bash
python -m pytest tests/ -q --no-header
```

Expected: PASS.

- [ ] **Step 3: Check diffs**

Run:

```bash
git diff --check
git status --short
git log --oneline -5
```

Expected: no whitespace errors; only planned files modified or committed.

- [ ] **Step 4: Final commit if cleanup was needed**

Run only if Step 3 shows uncommitted cleanup changes:

```bash
git add <changed files>
git commit -m "test(grasp): verify diagnostic pre-grasp handoff"
```

- [ ] **Step 5: GPU validation handoff**

Ask the user to run:

```bash
git pull --rebase
bash scripts/validate_lemon_grasp.sh
```

Expected: the run either reaches `[descend]` or reports a specific diagnostic pre-grasp failure reason such as `lateral_misaligned` / `base_recovery_failed`, not generic `ik_unreachable`.

## Self-Review

- Spec coverage: diagnostic data model, approach-frame decomposition, handoff gates, compatibility, ActionExecutor flow, testing, logging, and rollback are represented in the tasks.
- Placeholder scan: no task uses TBD/TODO/fill-in instructions.
- Type consistency: plan consistently uses `PreGraspResult`, `move_to_pre_grasp_diagnostic`, `handoff_ok`, `needs_recovery`, and `reason`.
