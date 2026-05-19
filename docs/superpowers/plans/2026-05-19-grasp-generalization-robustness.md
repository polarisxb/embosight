# Grasp Generalization Robustness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add observability and diagnostic-only object profiling so grasp robustness can be improved across objects without immediately changing execution behavior.

**Architecture:** The first implementation slice is deliberately non-invasive: extend oracle/multi-run summaries, add a standalone object profile classifier, and record profile diagnostics on grasp attempts. Profile-driven execution is reserved for a separate follow-up feature-flag plan and is not enabled in the initial tasks.

**Tech Stack:** Python dataclasses, pytest, ruff, existing EmboSight `WorldBelief`, `ActionExecutor`, `GraspPlanner`, `MemoryManager`, shell validation scripts.

---

## Scope

This plan implements Phase 1 and Phase 2 from the design:

1. Grasp diagnostics extension.
2. Multi-run summary extension.
3. Diagnostic-only object profile classification.
4. Recording profile diagnostics without changing grasp behavior.

This plan does not implement profile-controlled execution, memory parameter transfer, new neural models, or controller rewrites.

## File Structure

### Create

- `src/grasp_profile.py`
  - Defines `GraspProfile`, `GraspProfileResult`, and `classify_grasp_profile(...)`.
  - Pure function module with no simulator side effects.

- `tests/test_grasp_profile.py`
  - Unit tests for profile classification.

### Modify

- `src/eval_oracle.py`
  - Add final pose and executed-parameter fields to `OracleSummary`.

- `tests/test_eval_oracle.py`
  - Cover new oracle fields from an episode result payload.

- `scripts/validate_lemon_grasp_multi.sh`
  - Extract additional fields from logs or episode JSON for `summary.csv`.

- `src/action_executor.py`
  - Add profile metadata into attempt diagnostics after classification is available.
  - No profile-controlled behavior changes.

- `src/agent.py`
  - Ensure profile fields pass through memory payload if already present in attempt diagnostics.
  - No memory decision behavior changes.

- `tests/test_action_executor_v1.py`
  - Verify profile diagnostics are attached and do not change execution flow.

- `tests/test_agent_speech.py`
  - Verify memory payload preserves profile diagnostic fields without replacing executed strategy.

## Task 1: Extend Oracle Summary With Final Grasp Evidence

**Files:**

- Modify: `src/eval_oracle.py`
- Test: `tests/test_eval_oracle.py`

- [ ] **Step 1: Write the failing oracle test**

Add this test to `tests/test_eval_oracle.py`:

```python
def test_oracle_summary_includes_final_grasp_evidence():
    from src.eval_oracle import summarize_episode

    episode = {
        "success": True,
        "failure_reason": None,
        "target": {
            "label": "lemon",
            "position_3d": [0.125, -2.857, 0.947],
            "position_std_m": 0.02,
            "grasp_attempts": [
                {
                    "failure_mode": "success",
                    "candidate_source": "strategy_top_down",
                    "diagnostic": {
                        "post_lift_obj_pos": [0.134, -2.855, 1.038],
                        "post_lift_eef_pos": [0.127, -2.860, 1.055],
                        "obj_z_before": 0.947,
                        "obj_z_after": 1.038,
                        "selected_strategy": "top_down",
                        "executed_strategy": "top_down",
                        "depth_margin_m": 0.010,
                        "squeeze_extra_steps": 18,
                        "grasp_profile": "small_round_slippery",
                    },
                },
            ],
        },
        "hypotheses": [],
        "action_history": [],
        "planning_blockers": [],
    }

    summary = summarize_episode(episode, scenario_id="fixed_lemon_001")
    data = summary.to_dict()

    assert data["post_lift_obj_pos"] == [0.134, -2.855, 1.038]
    assert data["post_lift_eef_pos"] == [0.127, -2.860, 1.055]
    assert data["post_lift_obj_delta_z"] == 0.091
    assert data["selected_strategy"] == "top_down"
    assert data["executed_strategy"] == "top_down"
    assert data["depth_margin_m"] == 0.010
    assert data["squeeze_extra_steps"] == 18
    assert data["grasp_profile"] == "small_round_slippery"
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
python -m pytest tests/test_eval_oracle.py::test_oracle_summary_includes_final_grasp_evidence -q
```

Expected:

```text
FAILED ... KeyError or AttributeError for post_lift_obj_pos
```

- [ ] **Step 3: Extend `OracleSummary`**

In `src/eval_oracle.py`, add fields to the dataclass:

```python
    post_lift_obj_pos: list[float] | None = None
    post_lift_eef_pos: list[float] | None = None
    post_lift_obj_delta_z: float | None = None
    selected_strategy: str | None = None
    executed_strategy: str | None = None
    depth_margin_m: float | None = None
    squeeze_extra_steps: int | None = None
    grasp_profile: str | None = None
```

Update `to_dict()` if it manually enumerates fields. If `to_dict()` already uses `asdict`, no extra serialization code is needed.

- [ ] **Step 4: Extract diagnostic fields in `summarize_episode`**

In `src/eval_oracle.py`, after resolving `grasp_attempt`, add:

```python
    diagnostic = grasp_attempt.get("diagnostic", {}) if grasp_attempt else {}
    obj_z_before = _float_or_none(diagnostic.get("obj_z_before"))
    obj_z_after = _float_or_none(diagnostic.get("obj_z_after"))
    obj_delta_z = (
        obj_z_after - obj_z_before
        if obj_z_before is not None and obj_z_after is not None
        else None
    )
```

Pass these into `OracleSummary(...)`:

```python
        post_lift_obj_pos=diagnostic.get("post_lift_obj_pos"),
        post_lift_eef_pos=diagnostic.get("post_lift_eef_pos"),
        post_lift_obj_delta_z=obj_delta_z,
        selected_strategy=diagnostic.get("selected_strategy"),
        executed_strategy=diagnostic.get("executed_strategy"),
        depth_margin_m=_float_or_none(diagnostic.get("depth_margin_m")),
        squeeze_extra_steps=(
            int(diagnostic["squeeze_extra_steps"])
            if diagnostic.get("squeeze_extra_steps") is not None
            else None
        ),
        grasp_profile=diagnostic.get("grasp_profile"),
```

- [ ] **Step 5: Run the targeted test**

Run:

```bash
python -m pytest tests/test_eval_oracle.py::test_oracle_summary_includes_final_grasp_evidence -q
```

Expected:

```text
1 passed
```

- [ ] **Step 6: Run oracle tests**

Run:

```bash
python -m pytest tests/test_eval_oracle.py -q
```

Expected:

```text
all tests passed
```

- [ ] **Step 7: Commit**

Commit only the oracle files:

```bash
git add src/eval_oracle.py tests/test_eval_oracle.py
git commit -m "Expose final grasp evidence in oracle summaries" -m "The fixed lemon validation needs final object pose, executed parameters, and profile diagnostics in the oracle output so generalization work can be evaluated without reading raw logs.

Constraint: This change must not alter execution behavior
Rejected: Parse final pose from speech | speech is user-facing and localized, not a stable evaluation contract
Confidence: high
Scope-risk: narrow
Directive: Keep oracle fields sourced from attempt diagnostics, not formatted speech
Tested: python -m pytest tests/test_eval_oracle.py -q
Not-tested: GPU lemon run with the new oracle fields"
```

## Task 2: Extend Multi-run Summary Output

**Files:**

- Modify: `scripts/validate_lemon_grasp_multi.sh`

- [ ] **Step 1: Add CSV columns**

Change the CSV header from:

```bash
echo "run,exit_code,success,pre_close_abort,grasp_confirmed,micro_lift_ok,post_lift_verified,no_grasp,object_not_lifted,steps,time,grasp_failure_mode,log" \
    > "${SUMMARY_CSV}"
```

to:

```bash
echo "run,exit_code,success,pre_close_abort,grasp_confirmed,micro_lift_ok,post_lift_verified,no_grasp,object_not_lifted,steps,time,grasp_failure_mode,post_lift_obj_pos,post_lift_obj_delta_z,selected_strategy,executed_strategy,depth_margin_m,squeeze_extra_steps,grasp_profile,log" \
    > "${SUMMARY_CSV}"
```

- [ ] **Step 2: Add JSON field extractors**

Below `extract_failure_mode()`, add:

```bash
extract_json_scalar() {
    local key="$1"
    local file="$2"
    python - "$key" "$file" <<'PY'
import json
import sys

key = sys.argv[1]
path = sys.argv[2]
value = ""
try:
    text = open(path, "r", encoding="utf-8", errors="ignore").read()
    marker = "========== ORACLE SUMMARY =========="
    if marker in text:
        block = text.split(marker, 1)[1]
        start = block.find("{")
        end = block.find("\nepisode:")
        raw = block[start:end].strip() if start >= 0 and end >= 0 else ""
        if raw:
            data = json.loads(raw)
            item = data.get(key)
            if isinstance(item, list):
                value = ";".join(f"{float(x):.6f}" for x in item)
            elif item is not None:
                value = str(item)
except Exception:
    value = ""
print(value)
PY
}
```

- [ ] **Step 3: Extract new fields inside the run loop**

After `failure_mode="$(extract_failure_mode "${log_path}")"`, add:

```bash
    post_lift_obj_pos="$(extract_json_scalar 'post_lift_obj_pos' "${log_path}")"
    post_lift_obj_delta_z="$(extract_json_scalar 'post_lift_obj_delta_z' "${log_path}")"
    selected_strategy="$(extract_json_scalar 'selected_strategy' "${log_path}")"
    executed_strategy="$(extract_json_scalar 'executed_strategy' "${log_path}")"
    depth_margin_m="$(extract_json_scalar 'depth_margin_m' "${log_path}")"
    squeeze_extra_steps="$(extract_json_scalar 'squeeze_extra_steps' "${log_path}")"
    grasp_profile="$(extract_json_scalar 'grasp_profile' "${log_path}")"
```

- [ ] **Step 4: Append new CSV fields**

Change the CSV append line to:

```bash
    echo "${run_idx},${exit_code},${success},${pre_close_abort},${grasp_confirmed},${micro_lift_ok},${post_lift_verified},${no_grasp},${object_not_lifted},${steps},${time_s},${failure_mode},${post_lift_obj_pos},${post_lift_obj_delta_z},${selected_strategy},${executed_strategy},${depth_margin_m},${squeeze_extra_steps},${grasp_profile},${log_path}" \
        >> "${SUMMARY_CSV}"
```

- [ ] **Step 5: Syntax-check the script**

Run:

```bash
bash -n scripts/validate_lemon_grasp_multi.sh
```

Expected:

```text
no output, exit 0
```

- [ ] **Step 6: Commit**

```bash
git add scripts/validate_lemon_grasp_multi.sh
git commit -m "Report final grasp diagnostics in lemon multi-run CSV" -m "The multi-run validator now preserves final pose and executed parameter fields from oracle summaries, making repeated grasp runs comparable without scanning raw logs.

Constraint: The script must remain usable on the GPU environment without extra dependencies
Rejected: Depend on jq | jq may not be installed in the target conda environment
Confidence: medium
Scope-risk: narrow
Directive: Keep CSV extraction tolerant of old logs that lack the new oracle fields
Tested: bash -n scripts/validate_lemon_grasp_multi.sh
Not-tested: Full GPU multi-run after oracle field rollout"
```

## Task 3: Add Diagnostic-only Grasp Profile Classifier

**Files:**

- Create: `src/grasp_profile.py`
- Create: `tests/test_grasp_profile.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_grasp_profile.py`:

```python
import numpy as np

from src.grasp_profile import GraspProfile, classify_grasp_profile
from src.world_belief import GraspCandidate, GraspStrategy, Hypothesis


def _hyp(label: str, visible: str = "", slip: str = "medium") -> Hypothesis:
    return Hypothesis(
        object_id=f"{label}_main",
        label=label,
        label_alternatives=[(label, 0.95)],
        label_entropy=0.1,
        position_3d=np.array([0.5, 0.0, 0.9], dtype=np.float32),
        position_std_m=0.02,
        visible_features=visible,
        grasp_strategy=GraspStrategy(strategy="top_down", slip_risk=slip),
    )


def _candidate(width: float = 0.04) -> GraspCandidate:
    return GraspCandidate(
        point_3d=np.array([0.5, 0.0, 0.9], dtype=np.float32),
        approach_dir=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        finger_width_m=width,
        score=1.0,
        source="strategy_top_down",
    )


def test_lemon_is_small_round_slippery():
    result = classify_grasp_profile(
        _hyp("lemon", "round yellow smooth waxy fruit", "high"),
        _candidate(),
        object_size_m=(0.058, 0.055, 0.058),
    )

    assert result.profile == GraspProfile.SMALL_ROUND_SLIPPERY
    assert result.confidence >= 0.7
    assert "round" in result.reasons


def test_wide_object_is_ungraspable_when_width_exceeds_gripper():
    result = classify_grasp_profile(
        _hyp("tupperware", "wide rectangular container", "medium"),
        _candidate(width=0.04),
        object_size_m=(0.14, 0.10, 0.05),
        gripper_max_width_m=0.08,
    )

    assert result.profile == GraspProfile.WIDE_UNGRASPABLE
    assert "width_exceeds_gripper" in result.reasons


def test_mug_with_handle_is_handled():
    result = classify_grasp_profile(
        _hyp("mug", "ceramic cup with handle", "medium"),
        _candidate(width=0.04),
        object_size_m=(0.07, 0.09, 0.10),
    )

    assert result.profile == GraspProfile.HANDLED


def test_unknown_object_defaults_without_side_effects():
    result = classify_grasp_profile(
        _hyp("object", "", "medium"),
        _candidate(width=0.04),
        object_size_m=None,
    )

    assert result.profile == GraspProfile.DEFAULT_RIGID
    assert result.execution_overrides == {}
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
python -m pytest tests/test_grasp_profile.py -q
```

Expected:

```text
FAILED ... ModuleNotFoundError: No module named 'src.grasp_profile'
```

- [ ] **Step 3: Create `src/grasp_profile.py`**

Add:

```python
"""Diagnostic-only object profile classification for grasp analysis."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

import numpy as np


class GraspProfile(str, Enum):
    SMALL_ROUND_SLIPPERY = "small_round_slippery"
    THIN_FLAT = "thin_flat"
    WIDE_UNGRASPABLE = "wide_ungraspable"
    HANDLED = "handled"
    FRAGILE_SOFT = "fragile_soft"
    DEFAULT_RIGID = "default_rigid"
    UNKNOWN = "unknown"


@dataclass
class GraspProfileResult:
    profile: GraspProfile
    confidence: float
    reasons: list[str] = field(default_factory=list)
    execution_overrides: dict = field(default_factory=dict)

    def to_diagnostic(self) -> dict:
        return {
            "grasp_profile": self.profile.value,
            "grasp_profile_confidence": float(self.confidence),
            "grasp_profile_reasons": list(self.reasons),
        }


def _text_tokens(*parts: object) -> set[str]:
    text = " ".join(str(p or "").lower() for p in parts)
    for ch in ",.;:/()[]{}_-":
        text = text.replace(ch, " ")
    return {tok for tok in text.split() if tok}


def _size_tuple(object_size_m: Iterable[float] | None) -> tuple[float, float, float] | None:
    if object_size_m is None:
        return None
    arr = np.asarray(list(object_size_m), dtype=np.float32).reshape(-1)
    if arr.shape[0] < 3 or not np.all(np.isfinite(arr[:3])):
        return None
    return float(arr[0]), float(arr[1]), float(arr[2])


def classify_grasp_profile(
    hyp,
    candidate=None,
    object_size_m: Iterable[float] | None = None,
    gripper_max_width_m: float = 0.08,
) -> GraspProfileResult:
    """Classify the target for diagnostics without changing execution."""
    label = getattr(hyp, "label", "")
    visible = getattr(hyp, "visible_features", "")
    strategy = getattr(hyp, "grasp_strategy", None)
    slip = str(getattr(strategy, "slip_risk", "medium") or "medium").lower()
    tokens = _text_tokens(label, visible, slip)
    size = _size_tuple(object_size_m)
    reasons: list[str] = []

    if size is not None:
        max_xy = max(size[0], size[1])
        min_dim = min(size)
        if max_xy > gripper_max_width_m:
            return GraspProfileResult(
                profile=GraspProfile.WIDE_UNGRASPABLE,
                confidence=0.9,
                reasons=["width_exceeds_gripper"],
            )
        if min_dim <= 0.015 and max_xy >= 0.05:
            return GraspProfileResult(
                profile=GraspProfile.THIN_FLAT,
                confidence=0.75,
                reasons=["thin_aabb"],
            )

    if {"handle", "handled", "mug", "cup", "pan"} & tokens:
        return GraspProfileResult(
            profile=GraspProfile.HANDLED,
            confidence=0.8,
            reasons=["handle_semantics"],
        )

    if {"bread", "cake", "soft", "fragile", "delicate"} & tokens:
        return GraspProfileResult(
            profile=GraspProfile.FRAGILE_SOFT,
            confidence=0.75,
            reasons=["fragile_or_soft_semantics"],
        )

    round_terms = {"round", "sphere", "spherical", "lemon", "orange", "lime", "apple"}
    slippery_terms = {"smooth", "slippery", "waxy", "glossy", "high"}
    is_small = size is None or max(size) <= 0.09
    if is_small and (round_terms & tokens) and (slip == "high" or slippery_terms & tokens):
        reasons.extend(["round", "slippery", "small"])
        return GraspProfileResult(
            profile=GraspProfile.SMALL_ROUND_SLIPPERY,
            confidence=0.85,
            reasons=reasons,
        )

    return GraspProfileResult(
        profile=GraspProfile.DEFAULT_RIGID,
        confidence=0.5,
        reasons=["default"],
    )
```

- [ ] **Step 4: Run profile tests**

Run:

```bash
python -m pytest tests/test_grasp_profile.py -q
```

Expected:

```text
4 passed
```

- [ ] **Step 5: Run ruff**

Run:

```bash
python -m ruff check src/grasp_profile.py tests/test_grasp_profile.py
```

Expected:

```text
All checks passed!
```

- [ ] **Step 6: Commit**

```bash
git add src/grasp_profile.py tests/test_grasp_profile.py
git commit -m "Classify grasp profiles for diagnostics" -m "A pure diagnostic classifier labels targets as small-round-slippery, wide-ungraspable, handled, fragile-soft, thin-flat, or default-rigid without changing execution behavior.

Constraint: Initial profile rollout must be diagnostic-only
Rejected: Put profile rules directly inside GraspPlanner | separate pure module is easier to test and keep side-effect free
Confidence: medium
Scope-risk: narrow
Directive: Do not let profile classification change execution until the feature flag task is implemented
Tested: python -m pytest tests/test_grasp_profile.py -q
Tested: python -m ruff check src/grasp_profile.py tests/test_grasp_profile.py
Not-tested: Real GPU object matrix profile distribution"
```

## Task 4: Attach Profile Diagnostics To Grasp Attempts

**Files:**

- Modify: `src/action_executor.py`
- Modify: `tests/test_action_executor_v1.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_action_executor_v1.py`:

```python
def test_success_diagnostic_contains_grasp_profile_without_changing_execution():
    from src.action_executor import ActionExecutor
    from src.world_belief import DecomposedTask, GraspStrategy

    exe = ActionExecutor(scene_describer=None)
    env = FakeEnv()
    h, _ = _hyp_with_candidate()
    h.label = "lemon"
    h.visible_features = "round yellow smooth waxy fruit"
    h.grasp_strategy = GraspStrategy(strategy="top_down", slip_risk="high")

    result = exe.act(h, DecomposedTask(primary_target="lemon"), env)

    assert result.success is True
    diag = result.attempt.diagnostic
    assert diag["grasp_profile"] == "small_round_slippery"
    assert diag["grasp_profile_confidence"] >= 0.7
    assert "close" in env.calls
    assert "lift" in env.calls
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```bash
python -m pytest tests/test_action_executor_v1.py::test_success_diagnostic_contains_grasp_profile_without_changing_execution -q
```

Expected:

```text
FAILED ... KeyError: 'grasp_profile'
```

- [ ] **Step 3: Add profile diagnostic helper**

In `src/action_executor.py`, add a private helper near other diagnostic helpers:

```python
    def _classify_profile_diagnostic(self, target, candidate, env) -> dict:
        try:
            from src.grasp_profile import classify_grasp_profile

            object_size = None
            body = self._resolve_target_body(target, env)
            if body is not None and hasattr(env, "_get_body_aabb"):
                aabb = env._get_body_aabb(body)
                if aabb is not None:
                    lo, hi = aabb
                    object_size = np.asarray(hi, dtype=np.float32) - np.asarray(lo, dtype=np.float32)
            result = classify_grasp_profile(
                target,
                candidate,
                object_size_m=object_size,
            )
            return result.to_diagnostic()
        except Exception as e:
            logger.debug("[grasp_profile] diagnostic skipped: %s", e)
            return {}
```

If the environment method is named differently, inspect `src/env_wrapper.py` and use the existing AABB method name. Keep failures best-effort.

- [ ] **Step 4: Merge profile diagnostics into success attempt**

In `src/action_executor.py`, where the success `diagnostic = {...}` dict is built, add:

```python
        diagnostic.update(
            self._classify_profile_diagnostic(target, candidate, env)
        )
```

Place it before constructing `GraspAttempt`.

- [ ] **Step 5: Run targeted test**

Run:

```bash
python -m pytest tests/test_action_executor_v1.py::test_success_diagnostic_contains_grasp_profile_without_changing_execution -q
```

Expected:

```text
1 passed
```

- [ ] **Step 6: Run action executor tests**

Run:

```bash
python -m pytest tests/test_action_executor_v1.py -q
```

Expected:

```text
all tests passed
```

- [ ] **Step 7: Commit**

```bash
git add src/action_executor.py tests/test_action_executor_v1.py
git commit -m "Attach diagnostic grasp profiles to attempts" -m "Successful grasp attempts now include best-effort object profile diagnostics so matrix evaluations can compare behavior by object class without changing execution.

Constraint: Profile classification must remain best-effort and side-effect free
Rejected: Abort grasp when profile classification fails | diagnostics must not affect execution
Confidence: medium
Scope-risk: narrow
Directive: Keep profile diagnostic failures non-fatal
Tested: python -m pytest tests/test_action_executor_v1.py -q
Not-tested: GPU object matrix profile labels"
```

## Task 5: Preserve Profile Fields In Memory Payload

**Files:**

- Modify: `src/agent.py`
- Modify: `tests/test_agent_speech.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_agent_speech.py`:

```python
def test_grasp_memory_payload_preserves_grasp_profile_diagnostic():
    from src.agent import EmboSightAgent
    from src.world_belief import GraspAttempt, GraspCandidate, Hypothesis

    candidate = GraspCandidate(
        point_3d=np.array([0.134, -2.855, 0.947], dtype=np.float32),
        approach_dir=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        finger_width_m=0.04,
        score=1.0,
        source="strategy_top_down",
    )
    attempt = GraspAttempt(
        timestamp=0.0,
        candidate=candidate,
        failure_mode="success",
        end_effector_pose_reached=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        diagnostic={
            "grasp_profile": "small_round_slippery",
            "grasp_profile_confidence": 0.85,
            "grasp_profile_reasons": ["round", "slippery", "small"],
        },
    )
    hyp = Hypothesis(
        object_id="distr_counter_main",
        label="lemon",
        label_alternatives=[("lemon", 0.95)],
        label_entropy=0.1,
        position_3d=np.array([0.134, -2.855, 0.947], dtype=np.float32),
        position_std_m=0.02,
        grasp_candidates=[candidate],
        grasp_attempts=[attempt],
    )

    context, _ = EmboSightAgent._grasp_memory_payload(hyp, attempt)

    assert context["grasp_profile"] == "small_round_slippery"
    assert context["grasp_profile_confidence"] == 0.85
    assert context["grasp_profile_reasons"] == ["round", "slippery", "small"]
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```bash
python -m pytest tests/test_agent_speech.py::test_grasp_memory_payload_preserves_grasp_profile_diagnostic -q
```

Expected:

```text
FAILED ... KeyError: 'grasp_profile'
```

- [ ] **Step 3: Extend `_grasp_memory_payload`**

In `src/agent.py`, inside `_grasp_memory_payload`, after the existing diagnostic-derived fields are added to `context`, add:

```python
        for key in (
            "grasp_profile",
            "grasp_profile_confidence",
            "grasp_profile_reasons",
        ):
            if key in diag:
                context[key] = diag[key]
```

Use the local diagnostic variable name already present in the function. If it is named `diagnostic`, use `diagnostic` instead of `diag`.

- [ ] **Step 4: Run targeted test**

Run:

```bash
python -m pytest tests/test_agent_speech.py::test_grasp_memory_payload_preserves_grasp_profile_diagnostic -q
```

Expected:

```text
1 passed
```

- [ ] **Step 5: Run related tests**

Run:

```bash
python -m pytest tests/test_agent_speech.py tests/test_agent_run.py::TestVerifyMismatchFlow -q
```

Expected:

```text
all tests passed
```

- [ ] **Step 6: Commit**

```bash
git add src/agent.py tests/test_agent_speech.py
git commit -m "Carry grasp profile diagnostics into memory events" -m "Memory events preserve profile labels and reasons from verified grasp diagnostics while keeping selected and executed strategies separate.

Constraint: Memory must not infer profile independently from stale belief data
Rejected: Reclassify profile in Agent | the executor already owns attempt diagnostics
Confidence: high
Scope-risk: narrow
Directive: Store profile only from attempt diagnostics until profile semantics are versioned
Tested: python -m pytest tests/test_agent_speech.py tests/test_agent_run.py::TestVerifyMismatchFlow -q
Not-tested: Long-term memory consolidation with profile analytics"
```

## Task 6: Verification Sweep

**Files:**

- No new files.
- Verify all files touched in Tasks 1-5.

- [ ] **Step 1: Run targeted tests**

Run:

```bash
python -m pytest tests/test_eval_oracle.py tests/test_grasp_profile.py tests/test_action_executor_v1.py tests/test_agent_speech.py -q
```

Expected:

```text
all tests passed
```

- [ ] **Step 2: Run full tests**

Run:

```bash
python -m pytest tests/ -q --no-header
```

Expected:

```text
all tests passed
```

- [ ] **Step 3: Run ruff**

Run:

```bash
python -m ruff check src/eval_oracle.py src/grasp_profile.py src/action_executor.py src/agent.py tests/test_eval_oracle.py tests/test_grasp_profile.py tests/test_action_executor_v1.py tests/test_agent_speech.py
```

Expected:

```text
All checks passed!
```

- [ ] **Step 4: Run whitespace check**

Run:

```bash
git diff --check
```

Expected:

```text
exit 0
```

CRLF warnings on Windows are acceptable if there are no whitespace errors.

- [ ] **Step 5: Manual GPU validation command**

Run on the GPU environment:

```bash
RUNS=5 MEMORY_MODE=isolated bash scripts/validate_lemon_grasp_multi.sh 5
```

Expected:

```text
Successes: 5/5
pre_close_abort: 0 for every run
post_lift_verified: 1 for every run
summary.csv includes post_lift_obj_pos and executed diagnostics
```

- [ ] **Step 6: Route verification failures back to their owning task**

If Task 6 finds a failure, do not make a generic cleanup commit. Return to the task that owns the failing file, add or update the smallest test for that failure, rerun that task's verification command, and amend that task's commit before rerunning Task 6.

## Implementation Order

Execute tasks in this order:

```text
Task 1 -> Task 2 -> Task 3 -> Task 4 -> Task 5 -> Task 6
```

Do not start Task 4 before Task 3 is committed. Do not start profile-controlled execution until this diagnostic-only plan has been implemented, verified, and reviewed against GPU logs.

## Self-review Checklist

- Each task has exact files.
- Each behavior change has a failing test first.
- Initial rollout does not change execution policy.
- Profile classification is diagnostic-only.
- Oracle and CSV expose final verified evidence.
- Memory stores profile diagnostics only after attempt diagnostics exist.
- No new dependencies are introduced.
- Every commit uses Lore-style trailers.
