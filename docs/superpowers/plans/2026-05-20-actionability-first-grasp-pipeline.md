# Actionability-First Grasp Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a shared target-resolution and candidate-actionability contract so the system can explain and gate physically infeasible grasp candidates before more profile tuning.

**Architecture:** Add small pure modules for target resolution and actionability diagnostics, then wire them through planner, executor, oracle, and long-generalization reports. The first rollout is diagnostics-only; the behavior-changing gate is profiled and uses the existing pre-grasp diagnostic as the first truthful non-closure actionability check.

**Tech Stack:** Python dataclasses, existing `WorldBelief` / `GraspPlanner` / `ActionExecutor`, `src.grasp_execution` pre-grasp diagnostics, pytest, ruff, Bash eval scripts.

---

## Review Result

The design spec is implementable, with one important execution constraint:

`GraspPlanner` must not invent a fake IK preview from the current EEF pose. The current code already proves that `env.is_reachable()` is too weak. The first authoritative physical gate available today is `env.move_to_pre_grasp_diagnostic(candidate)`, which happens before descend, close, and lift. This plan therefore stages the work as:

1. diagnostics-only target/actionability contracts;
2. reporting and failure-family taxonomy;
3. profiled pre-grasp gate in `ActionExecutor` that can skip a candidate before gripper closure and try the next candidate;
4. GPU validation before broadening beyond `small_round_slippery`.

## Scope

This plan implements one feature in staged slices:

- Target resolution diagnostics.
- Candidate actionability diagnostics.
- Long-run summary fields.
- Profiled actionability gate for `small_round_slippery`.

This plan does not tune squeeze, depth, descend, close, lift, memory retrieval, or success criteria.

## File Structure

### Create

- `src/target_resolution.py`
  - Owns normalized target body resolution and the `TargetResolution` dataclass.
  - Does not move the robot.

- `src/grasp_actionability.py`
  - Owns `CandidateActionability`, pre-grasp reason mapping, diagnostic serialization, and actionability gate decisions.
  - Does not move the robot by itself.

- `tests/test_target_resolution.py`
  - Unit tests for normalized body/category matching and fallback diagnostics.

- `tests/test_grasp_actionability.py`
  - Unit tests for actionability dataclass serialization and pre-grasp reason classification.

### Modify

- `configs/agent.yaml`
  - Add default disabled flags:
    - `actionability_diagnostics: false`
    - `actionability_gate: false`

- `src/grasp_policy.py`
  - Add config readers for actionability diagnostics and gate.

- `src/grasp_planner.py`
  - Attach target resolution and candidate actionability diagnostics to candidates.
  - Preserve legacy ordering unless profiled gate is explicitly enabled.

- `src/action_executor.py`
  - Preserve actionability diagnostics on attempts.
  - Add profiled pre-grasp actionability gate that can skip a candidate after pre-grasp failure and try the next candidate before descend/close.

- `src/eval_oracle.py`
  - Preserve new diagnostics in oracle summaries.

- `eval/run_long_generalization.py`
  - Parse and summarize actionability fields and failure families.

- `scripts/validate_lemon_grasp_multi.sh`
  - Add new diagnostics to smoke CSV.

- `scripts/run_grasp_baseline.sh`
  - Include new summary sections in report excerpt.

- Tests:
  - `tests/test_grasp_policy.py`
  - `tests/test_grasp_strategy.py`
  - `tests/test_action_executor_v1.py`
  - `tests/test_eval_oracle.py`
  - `tests/test_long_generalization_runner.py`
  - `tests/test_grasp_baseline_script.py`

## Config Contract

Default config must remain non-invasive:

```yaml
grasp_policy:
  mode: legacy
  enabled_profiles: []
  actionability_diagnostics: false
  actionability_gate: false
```

Diagnostics-only GPU config:

```yaml
grasp_policy:
  mode: profiled
  enabled_profiles:
    - small_round_slippery
  actionability_diagnostics: true
  actionability_gate: false
```

Gate GPU config:

```yaml
grasp_policy:
  mode: profiled
  enabled_profiles:
    - small_round_slippery
  actionability_diagnostics: true
  actionability_gate: true
```

---

### Task 1: Add Target Resolution Helper

**Files:**

- Create: `src/target_resolution.py`
- Create: `tests/test_target_resolution.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_target_resolution.py`:

```python
from __future__ import annotations

from src.target_resolution import resolve_target_body


class _Env:
    def __init__(self, mapping):
        self.mapping = mapping

    def _get_obj_type_map(self):
        return dict(self.mapping)


def test_resolve_target_body_matches_normalized_category():
    env = _Env({
        "obj_main": "glass_cup",
        "distr_counter_main": "hotdog_bun",
    })

    result = resolve_target_body(
        requested_label="glass cup",
        selected_label="glass cup",
        env=env,
    )

    assert result.target_body == "obj_main"
    assert result.body_category == "glass_cup"
    assert result.source == "normalized_category"
    assert result.confidence == 0.9
    assert result.used_fallback is False
    assert result.reason == "matched selected label to body category"


def test_resolve_target_body_records_unresolved_without_fallback():
    env = _Env({"obj_main": "tupperware"})

    result = resolve_target_body(
        requested_label="lemon",
        selected_label="lemon",
        env=env,
        allow_fallback=False,
    )

    assert result.target_body is None
    assert result.body_category is None
    assert result.source == "unresolved"
    assert result.confidence == 0.0
    assert result.used_fallback is False
    assert result.reason == "no matching body category"


def test_resolve_target_body_records_explicit_fallback():
    env = _Env({"obj_main": "tupperware"})

    result = resolve_target_body(
        requested_label="lemon",
        selected_label="lemon",
        env=env,
        allow_fallback=True,
    )

    assert result.target_body == "obj_main"
    assert result.body_category == "tupperware"
    assert result.source == "fallback_obj_main"
    assert result.confidence == 0.5
    assert result.used_fallback is True
    assert result.reason == "fallback to obj_main"


def test_target_resolution_diagnostic_keys_are_stable():
    env = _Env({"distr_counter_main": "lemon_wedge"})

    result = resolve_target_body(
        requested_label="lemon wedge",
        selected_label="lemon wedge",
        env=env,
    )

    assert result.to_diagnostic() == {
        "target_resolution_requested_label": "lemon wedge",
        "target_resolution_selected_label": "lemon wedge",
        "target_body": "distr_counter_main",
        "target_body_category": "lemon_wedge",
        "target_resolution_source": "normalized_category",
        "target_resolution_confidence": 0.9,
        "target_resolution_used_fallback": False,
        "target_resolution_reason": "matched selected label to body category",
    }
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```bash
python -m pytest tests/test_target_resolution.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.target_resolution'`.

- [ ] **Step 3: Implement minimal helper**

Create `src/target_resolution.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def label_key(text: object) -> str:
    return "".join(ch for ch in str(text or "").lower() if ch.isalnum())


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

    def to_diagnostic(self) -> dict[str, object]:
        return {
            "target_resolution_requested_label": self.requested_label,
            "target_resolution_selected_label": self.selected_label,
            "target_body": self.target_body,
            "target_body_category": self.body_category,
            "target_resolution_source": self.source,
            "target_resolution_confidence": float(self.confidence),
            "target_resolution_used_fallback": bool(self.used_fallback),
            "target_resolution_reason": self.reason,
        }


def resolve_target_body(
    *,
    requested_label: str | None,
    selected_label: str | None,
    env: Any,
    allow_fallback: bool = False,
) -> TargetResolution:
    requested = str(requested_label or "").strip()
    selected = str(selected_label or "").strip()
    selected_key = label_key(selected)

    type_map: dict[str, str] = {}
    if env is not None and hasattr(env, "_get_obj_type_map"):
        try:
            raw = env._get_obj_type_map()
            if isinstance(raw, dict):
                type_map = {str(body): str(cat) for body, cat in raw.items()}
        except Exception:
            type_map = {}

    if selected_key:
        for body, category in type_map.items():
            if selected_key == label_key(category):
                return TargetResolution(
                    requested_label=requested,
                    selected_label=selected,
                    target_body=body,
                    body_category=category,
                    source="normalized_category",
                    confidence=0.9,
                    used_fallback=False,
                    reason="matched selected label to body category",
                )

    if allow_fallback and "obj_main" in type_map:
        return TargetResolution(
            requested_label=requested,
            selected_label=selected,
            target_body="obj_main",
            body_category=type_map.get("obj_main"),
            source="fallback_obj_main",
            confidence=0.5,
            used_fallback=True,
            reason="fallback to obj_main",
        )

    return TargetResolution(
        requested_label=requested,
        selected_label=selected,
        target_body=None,
        body_category=None,
        source="unresolved",
        confidence=0.0,
        used_fallback=False,
        reason="no matching body category",
    )
```

- [ ] **Step 4: Run tests and confirm pass**

Run:

```bash
python -m pytest tests/test_target_resolution.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/target_resolution.py tests/test_target_resolution.py
git commit -m "Make target body resolution explicit" \
  -m "Target/category matching was split across perception, planner, and executor. A small normalized helper records whether a body was resolved or only reached through fallback." \
  -m "Constraint: Does not change manipulation behavior yet; this task adds pure diagnostics only." \
  -m "Confidence: high" \
  -m "Scope-risk: narrow" \
  -m "Tested: python -m pytest tests/test_target_resolution.py -q"
```

---

### Task 2: Add Candidate Actionability Types

**Files:**

- Create: `src/grasp_actionability.py`
- Create: `tests/test_grasp_actionability.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_grasp_actionability.py`:

```python
from __future__ import annotations

import numpy as np

from src.grasp_execution import evaluate_pre_grasp_handoff
from src.world_belief import GraspCandidate


def _candidate() -> GraspCandidate:
    return GraspCandidate(
        point_3d=np.array([0.5, 0.0, 0.9], dtype=np.float32),
        approach_dir=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        finger_width_m=0.06,
        score=0.70,
        source="strategy_gentle_side",
    )


def test_actionability_from_pre_grasp_axis_gap_too_large():
    from src.grasp_actionability import actionability_from_pre_grasp_result

    c = _candidate()
    result = evaluate_pre_grasp_handoff(
        move_ok=False,
        final_eef=np.array([0.34, 0.0, 0.9], dtype=np.float32),
        pre_pos=np.array([0.45, 0.0, 0.9], dtype=np.float32),
        grasp_point=np.array([0.5, 0.0, 0.9], dtype=np.float32),
        approach_dir=c.approach_dir,
        finger_width_m=c.finger_width_m,
        height_m=0.05,
    )

    actionability = actionability_from_pre_grasp_result(
        c,
        result,
        selected_strategy="gentle_side",
        target_body="obj_main",
    )

    assert actionability.actionable is False
    assert actionability.hard_reject is True
    assert actionability.reason == "axis_gap_too_large"
    assert actionability.source == "strategy_gentle_side"
    assert actionability.executed_strategy == "gentle_side"
    assert actionability.target_body == "obj_main"


def test_actionability_from_pre_grasp_lateral_misaligned_is_recoverable():
    from src.grasp_actionability import actionability_from_pre_grasp_result

    c = _candidate()
    result = evaluate_pre_grasp_handoff(
        move_ok=False,
        final_eef=np.array([0.45, 0.08, 0.9], dtype=np.float32),
        pre_pos=np.array([0.45, 0.0, 0.9], dtype=np.float32),
        grasp_point=np.array([0.5, 0.0, 0.9], dtype=np.float32),
        approach_dir=c.approach_dir,
        finger_width_m=c.finger_width_m,
        height_m=0.05,
    )

    actionability = actionability_from_pre_grasp_result(
        c,
        result,
        selected_strategy="gentle_side",
        target_body="obj_main",
    )

    assert actionability.actionable is False
    assert actionability.hard_reject is False
    assert actionability.reason == "lateral_misaligned"


def test_unknown_actionability_is_not_a_hard_reject():
    from src.grasp_actionability import unknown_actionability

    c = _candidate()
    actionability = unknown_actionability(
        c,
        selected_strategy="gentle_side",
        target_body="obj_main",
        reason="not_evaluated",
    )

    assert actionability.actionable is True
    assert actionability.hard_reject is False
    assert actionability.reason == "not_evaluated"


def test_actionability_diagnostic_keys_are_stable():
    from src.grasp_actionability import unknown_actionability

    c = _candidate()
    data = unknown_actionability(
        c,
        selected_strategy="gentle_side",
        target_body="obj_main",
        reason="not_evaluated",
    ).to_diagnostic(prefix="candidate_actionability")

    assert data["candidate_actionability_source"] == "strategy_gentle_side"
    assert data["candidate_actionability_selected_strategy"] == "gentle_side"
    assert data["candidate_actionability_executed_strategy"] == "gentle_side"
    assert data["candidate_actionability_target_body"] == "obj_main"
    assert data["candidate_actionability_actionable"] is True
    assert data["candidate_actionability_hard_reject"] is False
    assert data["candidate_actionability_reason"] == "not_evaluated"
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```bash
python -m pytest tests/test_grasp_actionability.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.grasp_actionability'`.

- [ ] **Step 3: Implement actionability module**

Create `src/grasp_actionability.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.grasp_execution import (
    PRE_GRASP_AXIS_GAP_TOO_LARGE,
    PRE_GRASP_AXIS_GAP_TOO_SMALL,
    PRE_GRASP_BELOW_GRASP_POINT,
    PRE_GRASP_LATERAL_MISALIGNED,
    PRE_GRASP_SAFE_HANDOFF,
    PRE_GRASP_STRICT_OK,
)


@dataclass(frozen=True)
class CandidateActionability:
    source: str
    selected_strategy: str | None
    executed_strategy: str
    target_body: str | None
    actionable: bool
    hard_reject: bool
    reason: str
    total_error_m: float | None = None
    lateral_error_m: float | None = None
    axis_error_m: float | None = None
    approach_gap_m: float | None = None
    lateral_limit_m: float | None = None
    object_size_m: tuple[float, float, float] | None = None
    score_modifier: float = 0.0

    def to_diagnostic(self, prefix: str = "candidate_actionability") -> dict[str, Any]:
        return {
            f"{prefix}_source": self.source,
            f"{prefix}_selected_strategy": self.selected_strategy,
            f"{prefix}_executed_strategy": self.executed_strategy,
            f"{prefix}_target_body": self.target_body,
            f"{prefix}_actionable": bool(self.actionable),
            f"{prefix}_hard_reject": bool(self.hard_reject),
            f"{prefix}_reason": self.reason,
            f"{prefix}_total_error_m": self.total_error_m,
            f"{prefix}_lateral_error_m": self.lateral_error_m,
            f"{prefix}_axis_error_m": self.axis_error_m,
            f"{prefix}_approach_gap_m": self.approach_gap_m,
            f"{prefix}_lateral_limit_m": self.lateral_limit_m,
            f"{prefix}_object_size_m": (
                list(self.object_size_m) if self.object_size_m is not None else None
            ),
            f"{prefix}_score_modifier": float(self.score_modifier),
        }


def executed_strategy_name(candidate: Any, selected_strategy: str | None = None) -> str:
    source = str(getattr(candidate, "source", "") or "")
    if source.startswith("strategy_"):
        return source[len("strategy_"):]
    if source == "vlm_top_grasp":
        return "top_down"
    if source == "geometric_centroid":
        return "geometric_centroid"
    return selected_strategy or "unknown"


def unknown_actionability(
    candidate: Any,
    *,
    selected_strategy: str | None,
    target_body: str | None,
    reason: str,
) -> CandidateActionability:
    return CandidateActionability(
        source=str(getattr(candidate, "source", "unknown")),
        selected_strategy=selected_strategy,
        executed_strategy=executed_strategy_name(candidate, selected_strategy),
        target_body=target_body,
        actionable=True,
        hard_reject=False,
        reason=reason,
    )


def actionability_from_pre_grasp_result(
    candidate: Any,
    result: Any,
    *,
    selected_strategy: str | None,
    target_body: str | None,
) -> CandidateActionability:
    reason = str(getattr(result, "reason", "unknown"))
    actionable = bool(getattr(result, "ok", False) or getattr(result, "handoff_ok", False))
    recoverable = bool(getattr(result, "needs_recovery", False))
    hard_reject = (
        not actionable
        and not recoverable
        and reason in {
            PRE_GRASP_AXIS_GAP_TOO_SMALL,
            PRE_GRASP_AXIS_GAP_TOO_LARGE,
            PRE_GRASP_BELOW_GRASP_POINT,
        }
    )
    if reason in {PRE_GRASP_STRICT_OK, PRE_GRASP_SAFE_HANDOFF}:
        score_modifier = 0.05
    elif reason == PRE_GRASP_LATERAL_MISALIGNED:
        score_modifier = -0.05
    elif hard_reject:
        score_modifier = -1.0
    else:
        score_modifier = 0.0

    return CandidateActionability(
        source=str(getattr(candidate, "source", "unknown")),
        selected_strategy=selected_strategy,
        executed_strategy=executed_strategy_name(candidate, selected_strategy),
        target_body=target_body,
        actionable=actionable,
        hard_reject=hard_reject,
        reason=reason,
        total_error_m=_float_or_none(getattr(result, "total_error_m", None)),
        lateral_error_m=_float_or_none(getattr(result, "lateral_error_m", None)),
        axis_error_m=_float_or_none(getattr(result, "axis_error_m", None)),
        approach_gap_m=_float_or_none(getattr(result, "approach_gap_m", None)),
        lateral_limit_m=_float_or_none(getattr(result, "lateral_limit_m", None)),
        score_modifier=score_modifier,
    )


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
```

- [ ] **Step 4: Run tests and confirm pass**

Run:

```bash
python -m pytest tests/test_grasp_actionability.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/grasp_actionability.py tests/test_grasp_actionability.py
git commit -m "Represent candidate actionability explicitly" \
  -m "Pre-grasp diagnostics already produce the geometry facts needed to distinguish actionable candidates from hard rejects. This adds a pure serialization and classification layer without changing execution." \
  -m "Constraint: No robot motion or candidate ordering changes in this slice." \
  -m "Confidence: high" \
  -m "Scope-risk: narrow" \
  -m "Tested: python -m pytest tests/test_grasp_actionability.py -q"
```

---

### Task 3: Add Actionability Policy Flags

**Files:**

- Modify: `configs/agent.yaml`
- Modify: `src/grasp_policy.py`
- Modify: `tests/test_grasp_policy.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_grasp_policy.py`:

```python

def test_actionability_flags_default_to_disabled():
    from src.grasp_policy import (
        actionability_diagnostics_enabled,
        actionability_gate_enabled,
    )

    assert actionability_diagnostics_enabled(None) is False
    assert actionability_gate_enabled(None, "small_round_slippery") is False


def test_actionability_diagnostics_can_be_enabled_in_profiled_mode():
    from src.grasp_policy import actionability_diagnostics_enabled

    assert actionability_diagnostics_enabled({
        "mode": "profiled",
        "enabled_profiles": ["small_round_slippery"],
        "actionability_diagnostics": True,
    }) is True


def test_actionability_gate_requires_profiled_enabled_profile():
    from src.grasp_policy import actionability_gate_enabled

    config = {
        "mode": "profiled",
        "enabled_profiles": ["small_round_slippery"],
        "actionability_gate": True,
    }

    assert actionability_gate_enabled(config, "small_round_slippery") is True
    assert actionability_gate_enabled(config, "thin_flat") is False
    assert actionability_gate_enabled({"mode": "legacy", "actionability_gate": True}, "small_round_slippery") is False
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```bash
python -m pytest tests/test_grasp_policy.py -q
```

Expected: FAIL because the new helper functions do not exist.

- [ ] **Step 3: Implement policy helpers**

Add to `src/grasp_policy.py` after `_enabled_profiles(...)`:

```python

def actionability_diagnostics_enabled(config: dict[str, Any] | None) -> bool:
    if not isinstance(config, dict):
        return False
    return _mode(config) == "profiled" and bool(config.get("actionability_diagnostics"))


def actionability_gate_enabled(
    config: dict[str, Any] | None,
    grasp_profile: str | None,
) -> bool:
    profile = _profile_name(grasp_profile)
    return (
        isinstance(config, dict)
        and _mode(config) == "profiled"
        and bool(config.get("actionability_gate"))
        and profile in _enabled_profiles(config)
    )
```

Update `configs/agent.yaml`:

```yaml
grasp_policy:
  mode: legacy
  enabled_profiles: []
  actionability_diagnostics: false
  actionability_gate: false
```

- [ ] **Step 4: Run tests and config check**

Run:

```bash
python -m pytest tests/test_grasp_policy.py -q
python - <<'PY'
import yaml
from pathlib import Path
data = yaml.safe_load(Path("configs/agent.yaml").read_text(encoding="utf-8"))
assert data["grasp_policy"]["mode"] == "legacy"
assert data["grasp_policy"]["enabled_profiles"] == []
assert data["grasp_policy"]["actionability_diagnostics"] is False
assert data["grasp_policy"]["actionability_gate"] is False
print("agent config actionability defaults OK")
PY
```

Expected: pytest PASS and script prints `agent config actionability defaults OK`.

- [ ] **Step 5: Commit**

```bash
git add configs/agent.yaml src/grasp_policy.py tests/test_grasp_policy.py
git commit -m "Gate actionability work behind profiled flags" \
  -m "Actionability diagnostics and gating must stay opt-in so legacy evaluation remains comparable to Phase 4." \
  -m "Constraint: Default agent config remains legacy with both actionability flags disabled." \
  -m "Confidence: high" \
  -m "Scope-risk: narrow" \
  -m "Tested: python -m pytest tests/test_grasp_policy.py -q"
```

---

### Task 4: Attach Planner Diagnostics Without Reordering

**Files:**

- Modify: `src/grasp_planner.py`
- Modify: `tests/test_grasp_strategy.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_grasp_strategy.py`:

```python

def test_profiled_actionability_diagnostics_record_target_resolution_without_reordering():
    planner = GraspPlanner(
        vlm=MockVLM(['{"grip_norm": [0.5, 0.5]}']),
        env=_GeometryAwareEnv(),
        llm=None,
        grasp_policy_config={
            "mode": "profiled",
            "enabled_profiles": ["small_round_slippery"],
            "actionability_diagnostics": True,
            "actionability_gate": False,
        },
    )
    h = _hyp("lemon", visible_features="round yellow smooth waxy fruit")
    h.grasp_strategy = GraspStrategy(strategy="gentle_side", slip_risk="high")

    cands = planner.plan(h)

    assert [c.source for c in cands] == [
        "strategy_gentle_side",
        "vlm_top_grasp",
        "geometric_centroid",
    ]
    diag = getattr(cands[0], "_embosight_attempt_diagnostic")
    assert diag["target_body"] == "distr_counter_main"
    assert diag["target_body_category"] == "lemon"
    assert diag["target_resolution_source"] == "normalized_category"
    assert diag["candidate_actionability_reason"] == "not_evaluated"
    assert diag["candidate_actionability_actionable"] is True
    assert diag["candidate_actionability_hard_reject"] is False


def test_legacy_does_not_attach_actionability_diagnostics():
    planner = GraspPlanner(
        vlm=MockVLM(['{"grip_norm": [0.5, 0.5]}']),
        env=_GeometryAwareEnv(),
        llm=None,
        grasp_policy_config={
            "mode": "legacy",
            "enabled_profiles": ["small_round_slippery"],
            "actionability_diagnostics": True,
        },
    )
    h = _hyp("lemon", visible_features="round yellow smooth waxy fruit")
    h.grasp_strategy = GraspStrategy(strategy="gentle_side", slip_risk="high")

    cands = planner.plan(h)

    diag = getattr(cands[0], "_embosight_attempt_diagnostic", {})
    assert "candidate_actionability_reason" not in diag
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```bash
python -m pytest tests/test_grasp_strategy.py::test_profiled_actionability_diagnostics_record_target_resolution_without_reordering tests/test_grasp_strategy.py::test_legacy_does_not_attach_actionability_diagnostics -q
```

Expected: FAIL because planner does not attach target/actionability diagnostics.

- [ ] **Step 3: Implement planner diagnostic attachment**

In `src/grasp_planner.py`, add a helper before `_apply_candidate_source_policy(...)`:

```python
    def _attach_actionability_diagnostics(
        self,
        hyp: Hypothesis,
        cands: list[GraspCandidate],
        env,
        profile: str,
    ) -> None:
        from src.grasp_actionability import unknown_actionability
        from src.grasp_policy import actionability_diagnostics_enabled
        from src.grasp_policy import merge_candidate_attempt_diagnostic
        from src.target_resolution import resolve_target_body

        if not actionability_diagnostics_enabled(self.grasp_policy_config):
            return

        selected_strategy = (
            hyp.grasp_strategy.strategy
            if hyp.grasp_strategy is not None
            else None
        )
        resolution = resolve_target_body(
            requested_label=getattr(hyp, "label", None),
            selected_label=getattr(hyp, "label", None),
            env=env,
            allow_fallback=False,
        )
        for candidate in cands:
            actionability = unknown_actionability(
                candidate,
                selected_strategy=selected_strategy,
                target_body=resolution.target_body,
                reason="not_evaluated",
            )
            merge_candidate_attempt_diagnostic(
                candidate,
                {
                    **resolution.to_diagnostic(),
                    **actionability.to_diagnostic(),
                    "candidate_actionability_policy": "diagnostics_only",
                    "legacy_first_candidate_actionable": None,
                    "final_first_candidate_actionable": None,
                    "no_actionable_candidate": False,
                },
            )
```

Then update `_apply_candidate_source_policy(...)` so the profile is classified once and diagnostics are attached before candidate-source policy diagnostics:

```python
        profile = self._classify_profile_for_policy(hyp, cands, env)
        self._attach_actionability_diagnostics(hyp, cands, env, profile)
```

Keep the existing call to `apply_candidate_source_policy(...)` after this block.

- [ ] **Step 4: Run focused tests**

Run:

```bash
python -m pytest tests/test_grasp_strategy.py::test_profiled_actionability_diagnostics_record_target_resolution_without_reordering tests/test_grasp_strategy.py::test_legacy_does_not_attach_actionability_diagnostics -q
```

Expected: PASS.

- [ ] **Step 5: Run related planner tests**

Run:

```bash
python -m pytest tests/test_grasp_strategy.py tests/test_grasp_policy.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/grasp_planner.py tests/test_grasp_strategy.py
git commit -m "Record planner actionability diagnostics without reordering" \
  -m "Profiled diagnostics now carry target resolution and unknown actionability markers on candidates while leaving candidate order untouched." \
  -m "Constraint: No legacy behavior change; diagnostics attach only when profiled actionability diagnostics are enabled." \
  -m "Rejected: Use current EEF as a planner reachability oracle | it would repeat the is_reachable placeholder mistake." \
  -m "Confidence: medium" \
  -m "Scope-risk: narrow" \
  -m "Tested: python -m pytest tests/test_grasp_strategy.py tests/test_grasp_policy.py -q"
```

---

### Task 5: Preserve Actionability Diagnostics In Attempts And Oracle

**Files:**

- Modify: `src/action_executor.py`
- Modify: `src/eval_oracle.py`
- Modify: `eval/run_long_generalization.py`
- Modify: `tests/test_action_executor_v1.py`
- Modify: `tests/test_eval_oracle.py`
- Modify: `tests/test_long_generalization_runner.py`

- [ ] **Step 1: Write failing executor test**

Append to `tests/test_action_executor_v1.py`:

```python

def test_attempt_preserves_actionability_diagnostics():
    from src.grasp_policy import merge_candidate_attempt_diagnostic
    from src.action_executor import ActionExecutor
    from src.world_belief import DecomposedTask

    h, c = _hyp_with_candidate()
    merge_candidate_attempt_diagnostic(c, {
        "target_body": "obj_main",
        "target_body_category": "apple",
        "target_resolution_source": "normalized_category",
        "candidate_actionability_policy": "diagnostics_only",
        "candidate_actionability_actionable": True,
        "candidate_actionability_hard_reject": False,
        "candidate_actionability_reason": "not_evaluated",
        "legacy_first_candidate_actionable": None,
        "final_first_candidate_actionable": None,
        "no_actionable_candidate": False,
    })
    env = FakeEnv(descend_ok=True, lift_ok=True, obj_lifts=True)

    result = ActionExecutor().act(h, DecomposedTask(primary_target="apple"), env)

    diag = result.attempt.diagnostic
    assert diag["target_body"] == "obj_main"
    assert diag["candidate_actionability_policy"] == "diagnostics_only"
    assert diag["candidate_actionability_reason"] == "not_evaluated"
```

- [ ] **Step 2: Write failing oracle and runner tests**

Extend the existing diagnostic dict in `tests/test_eval_oracle.py::test_oracle_summary_includes_final_grasp_evidence`:

```python
                            "target_body": "obj_main",
                            "target_body_category": "lemon",
                            "target_resolution_source": "normalized_category",
                            "target_resolution_used_fallback": False,
                            "candidate_actionability_policy": "diagnostics_only",
                            "candidate_actionability_actionable": True,
                            "candidate_actionability_hard_reject": False,
                            "candidate_actionability_reason": "not_evaluated",
                            "legacy_first_candidate_actionable": True,
                            "final_first_candidate_actionable": True,
                            "no_actionable_candidate": False,
```

Add assertions:

```python
    assert data["target_body"] == "obj_main"
    assert data["target_body_category"] == "lemon"
    assert data["target_resolution_source"] == "normalized_category"
    assert data["target_resolution_used_fallback"] is False
    assert data["candidate_actionability_policy"] == "diagnostics_only"
    assert data["candidate_actionability_actionable"] is True
    assert data["candidate_actionability_hard_reject"] is False
    assert data["candidate_actionability_reason"] == "not_evaluated"
    assert data["legacy_first_candidate_actionable"] is True
    assert data["final_first_candidate_actionable"] is True
    assert data["no_actionable_candidate"] is False
```

Add to `tests/test_long_generalization_runner.py`:

```python

def test_summarize_results_counts_actionability_usage_and_failure_family():
    module = _load_module()

    summary = module.summarize_results([
        {
            "scenario_id": "random_seed_4",
            "success": False,
            "actual_object": "lemon_wedge",
            "grasp_failure_mode": "ik_unreachable",
            "candidate_actionability_policy": "pre_grasp_gate",
            "candidate_actionability_actionable": False,
            "candidate_actionability_hard_reject": True,
            "candidate_actionability_reason": "axis_gap_too_large",
            "target_resolution_source": "normalized_category",
            "no_actionable_candidate": False,
            "steps": 4,
            "time_s": 100.0,
        },
        {
            "scenario_id": "random_seed_7",
            "success": False,
            "actual_object": "juice",
            "failure_reason": "MAX_STEPS reached",
            "action_sequence": ["observe", "ask_user", "ask_user", "ask_user"],
            "steps": 12,
            "time_s": 60.0,
        },
    ])

    assert summary["failure_family_breakdown"] == {
        "planning_actionability_failure": 1,
        "target_selection_failure": 1,
    }
    assert summary["failure_mode_by_actionability_reason"]["axis_gap_too_large"]["ik_unreachable"] == 1
    assert summary["candidate_actionability_usage"] == {
        "pre_grasp_gate:axis_gap_too_large:hard_reject": 1,
    }
    assert summary["target_resolution_source_usage"] == {
        "normalized_category": 1,
    }
    assert summary["no_actionable_candidate_count"] == 0
```

- [ ] **Step 3: Run tests and confirm failure**

Run:

```bash
python -m pytest tests/test_action_executor_v1.py::test_attempt_preserves_actionability_diagnostics tests/test_eval_oracle.py::test_oracle_summary_includes_final_grasp_evidence tests/test_long_generalization_runner.py::test_summarize_results_counts_actionability_usage_and_failure_family -q
```

Expected: FAIL because oracle and long-generalization do not expose the new fields yet.

- [ ] **Step 4: Extend oracle fields**

In `src/eval_oracle.py`, add dataclass fields:

```python
    target_body: str | None = None
    target_body_category: str | None = None
    target_resolution_source: str | None = None
    target_resolution_used_fallback: bool | None = None
    candidate_actionability_policy: str | None = None
    candidate_actionability_actionable: bool | None = None
    candidate_actionability_hard_reject: bool | None = None
    candidate_actionability_reason: str | None = None
    legacy_first_candidate_actionable: bool | None = None
    final_first_candidate_actionable: bool | None = None
    no_actionable_candidate: bool | None = None
```

In `summarize_episode(...)`, pass:

```python
        target_body=_str_or_none(diagnostic.get("target_body")),
        target_body_category=_str_or_none(diagnostic.get("target_body_category")),
        target_resolution_source=_str_or_none(diagnostic.get("target_resolution_source")),
        target_resolution_used_fallback=_bool_or_none(
            diagnostic.get("target_resolution_used_fallback"),
        ),
        candidate_actionability_policy=_str_or_none(
            diagnostic.get("candidate_actionability_policy"),
        ),
        candidate_actionability_actionable=_bool_or_none(
            diagnostic.get("candidate_actionability_actionable"),
        ),
        candidate_actionability_hard_reject=_bool_or_none(
            diagnostic.get("candidate_actionability_hard_reject"),
        ),
        candidate_actionability_reason=_str_or_none(
            diagnostic.get("candidate_actionability_reason"),
        ),
        legacy_first_candidate_actionable=_bool_or_none(
            diagnostic.get("legacy_first_candidate_actionable"),
        ),
        final_first_candidate_actionable=_bool_or_none(
            diagnostic.get("final_first_candidate_actionable"),
        ),
        no_actionable_candidate=_bool_or_none(
            diagnostic.get("no_actionable_candidate"),
        ),
```

- [ ] **Step 5: Extend long-generalization parsing and summary**

In `eval/run_long_generalization.py`, append to `FINAL_GRASP_ORACLE_FIELDS`:

```python
    "target_body",
    "target_body_category",
    "target_resolution_source",
    "target_resolution_used_fallback",
    "candidate_actionability_policy",
    "candidate_actionability_actionable",
    "candidate_actionability_hard_reject",
    "candidate_actionability_reason",
    "legacy_first_candidate_actionable",
    "final_first_candidate_actionable",
    "no_actionable_candidate",
```

In `summarize_results(...)`, add counters:

```python
    failure_family_breakdown: dict[str, int] = {}
    failure_mode_by_actionability_reason: dict[str, dict[str, int]] = {}
    candidate_actionability_usage: dict[str, int] = {}
    target_resolution_source_usage: dict[str, int] = {}
    no_actionable_candidate_count = 0
```

Inside the result loop:

```python
        family = _failure_family(r)
        if family and not r.get("success"):
            failure_family_breakdown[family] = failure_family_breakdown.get(family, 0) + 1
        actionability_key = _candidate_actionability_usage_key(r)
        if actionability_key:
            candidate_actionability_usage[actionability_key] = (
                candidate_actionability_usage.get(actionability_key, 0) + 1
            )
        actionability_reason = _bucket_name(r.get("candidate_actionability_reason"))
        if actionability_reason != "unknown" and not r.get("success"):
            _add_nested_count(
                failure_mode_by_actionability_reason,
                actionability_reason,
                str(_failure_reason(r)),
            )
        target_source = _bucket_name(r.get("target_resolution_source"))
        if target_source != "unknown":
            target_resolution_source_usage[target_source] = (
                target_resolution_source_usage.get(target_source, 0) + 1
            )
        if bool(r.get("no_actionable_candidate")):
            no_actionable_candidate_count += 1
```

Add return fields:

```python
        "failure_family_breakdown": dict(
            sorted(failure_family_breakdown.items(), key=lambda x: (-x[1], x[0])),
        ),
        "failure_mode_by_actionability_reason": _sorted_nested_counts(
            failure_mode_by_actionability_reason,
        ),
        "candidate_actionability_usage": dict(
            sorted(candidate_actionability_usage.items(), key=lambda x: (-x[1], x[0])),
        ),
        "target_resolution_source_usage": dict(
            sorted(target_resolution_source_usage.items(), key=lambda x: (-x[1], x[0])),
        ),
        "no_actionable_candidate_count": no_actionable_candidate_count,
```

Add helpers near `_failure_reason(...)`:

```python
def _candidate_actionability_usage_key(result: dict[str, Any]) -> str | None:
    policy = _bucket_name(result.get("candidate_actionability_policy"))
    reason = _bucket_name(result.get("candidate_actionability_reason"))
    if policy == "unknown" and reason == "unknown":
        return None
    if bool(result.get("candidate_actionability_hard_reject")):
        state = "hard_reject"
    elif bool(result.get("candidate_actionability_actionable")):
        state = "actionable"
    else:
        state = "not_actionable"
    return f"{policy}:{reason}:{state}"


def _failure_family(result: dict[str, Any]) -> str | None:
    if result.get("success"):
        return None
    reason = _failure_reason(result)
    if reason == "clarification_loop":
        return "target_selection_failure"
    if reason == "safety_loop":
        return "safety_decision_failure"
    actionability_reason = _bucket_name(result.get("candidate_actionability_reason"))
    if actionability_reason in {
        "axis_gap_too_small",
        "axis_gap_too_large",
        "below_grasp_point",
        "target_body_unresolved",
        "no_actionable_candidate",
    }:
        return "planning_actionability_failure"
    if reason in {"ik_unreachable", "hit_z_floor"}:
        return "planning_actionability_failure"
    if reason in {"slipped_descend", "slipped_lift", "gripper_empty"}:
        return "execution_failure"
    return "planning_loop"
```

- [ ] **Step 6: Run focused tests**

Run:

```bash
python -m pytest tests/test_action_executor_v1.py::test_attempt_preserves_actionability_diagnostics tests/test_eval_oracle.py::test_oracle_summary_includes_final_grasp_evidence tests/test_long_generalization_runner.py::test_summarize_results_counts_actionability_usage_and_failure_family -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/action_executor.py src/eval_oracle.py eval/run_long_generalization.py tests/test_action_executor_v1.py tests/test_eval_oracle.py tests/test_long_generalization_runner.py
git commit -m "Carry actionability diagnostics through evaluation" \
  -m "Actionability fields now survive attempts, oracle summaries, and long-generalization aggregation so GPU runs can separate target, planning, and execution failures." \
  -m "Constraint: Diagnostic fields do not change candidate choice or execution flow." \
  -m "Confidence: high" \
  -m "Scope-risk: moderate" \
  -m "Tested: focused actionability/oracle/runner pytest selection"
```

---

### Task 6: Add Profiled Pre-Grasp Candidate Gate

**Files:**

- Modify: `src/action_executor.py`
- Modify: `tests/test_action_executor_v1.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_action_executor_v1.py`:

```python

class _PreGraspGateEnv(FakeEnv):
    def __init__(self):
        super().__init__(descend_ok=True, lift_ok=True, obj_lifts=True)
        self.pre_results = ["axis_gap_too_large", "strict_ok"]
        self.pre_grasp_candidates: list[str] = []

    def move_to_pre_grasp_diagnostic(self, candidate, height_m=0.05):
        from src.grasp_execution import evaluate_pre_grasp_handoff

        self.calls.append("pre_grasp_diag")
        self.pre_grasp_candidates.append(candidate.source)
        reason = self.pre_results.pop(0)
        if reason == "strict_ok":
            return evaluate_pre_grasp_handoff(
                move_ok=True,
                final_eef=np.array([0.5, 0.0, 0.95], dtype=np.float32),
                pre_pos=np.array([0.5, 0.0, 0.95], dtype=np.float32),
                grasp_point=np.array([0.5, 0.0, 0.9], dtype=np.float32),
                approach_dir=np.array([0.0, 0.0, -1.0], dtype=np.float32),
                finger_width_m=0.04,
                height_m=0.05,
            )
        return evaluate_pre_grasp_handoff(
            move_ok=False,
            final_eef=np.array([0.34, 0.0, 0.9], dtype=np.float32),
            pre_pos=np.array([0.45, 0.0, 0.9], dtype=np.float32),
            grasp_point=np.array([0.5, 0.0, 0.9], dtype=np.float32),
            approach_dir=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            finger_width_m=0.06,
            height_m=0.05,
        )


def test_profiled_actionability_gate_skips_hard_rejected_pre_grasp_candidate():
    from src.action_executor import ActionExecutor
    from src.world_belief import DecomposedTask, GraspCandidate, Hypothesis

    rejected = GraspCandidate(
        point_3d=np.array([0.5, 0.0, 0.9], dtype=np.float32),
        approach_dir=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        finger_width_m=0.06,
        score=0.70,
        source="strategy_gentle_side",
    )
    fallback = GraspCandidate(
        point_3d=np.array([0.5, 0.0, 0.9], dtype=np.float32),
        approach_dir=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        finger_width_m=0.04,
        score=0.75,
        source="vlm_top_grasp",
    )
    h = Hypothesis(
        object_id="o0",
        label="lemon",
        label_alternatives=[("lemon", 0.9)],
        label_entropy=0.1,
        position_3d=np.array([0.5, 0.0, 0.9], dtype=np.float32),
        position_std_m=0.02,
        grasp_candidates=[rejected, fallback],
    )
    env = _PreGraspGateEnv()
    exe = ActionExecutor(grasp_policy_config={
        "mode": "profiled",
        "enabled_profiles": ["small_round_slippery"],
        "actionability_gate": True,
    })

    result = exe.act(h, DecomposedTask(primary_target="lemon"), env)

    assert result.success is True
    assert result.attempt.candidate.source == "vlm_top_grasp"
    assert env.pre_grasp_candidates == ["strategy_gentle_side", "vlm_top_grasp"]
    diag = result.attempt.diagnostic
    assert diag["candidate_actionability_policy"] == "pre_grasp_gate"
    assert diag["candidate_actionability_reason"] == "strict_ok"
    assert diag["candidate_actionability_actionable"] is True
    assert diag["no_actionable_candidate"] is False
    assert diag["skipped_candidate_sources"] == ["strategy_gentle_side"]


def test_actionability_gate_returns_no_actionable_candidate_when_all_hard_reject():
    from src.action_executor import ActionExecutor
    from src.world_belief import DecomposedTask

    h, c = _hyp_with_candidate()
    c.source = "strategy_gentle_side"
    env = _PreGraspGateEnv()
    env.pre_results = ["axis_gap_too_large"]
    exe = ActionExecutor(grasp_policy_config={
        "mode": "profiled",
        "enabled_profiles": ["small_round_slippery"],
        "actionability_gate": True,
    })

    result = exe.act(h, DecomposedTask(primary_target="apple"), env)

    assert result.success is False
    assert result.attempt.failure_mode == "ik_unreachable"
    assert result.attempt.diagnostic["no_actionable_candidate"] is True
    assert result.attempt.diagnostic["candidate_actionability_reason"] == "axis_gap_too_large"
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```bash
python -m pytest tests/test_action_executor_v1.py::test_profiled_actionability_gate_skips_hard_rejected_pre_grasp_candidate tests/test_action_executor_v1.py::test_actionability_gate_returns_no_actionable_candidate_when_all_hard_reject -q
```

Expected: FAIL because `ActionExecutor.act()` picks one candidate and returns after the first pre-grasp failure.

- [ ] **Step 3: Refactor candidate selection into a pre-grasp gate loop**

In `src/action_executor.py`, import inside `act()` where needed:

```python
        from src.grasp_actionability import actionability_from_pre_grasp_result
        from src.grasp_policy import actionability_gate_enabled
```

After selecting candidates, compute:

```python
        selected_strategy = (
            target.grasp_strategy.strategy
            if getattr(target, "grasp_strategy", None) is not None
            else None
        )
        profile = self._candidate_attempt_diagnostic(candidate).get("grasp_profile")
        gate_enabled = actionability_gate_enabled(self.grasp_policy_config, profile)
        skipped_sources: list[str] = []
```

Replace the single-candidate pre-grasp block with a loop shaped like this:

```python
        candidates_to_try = [
            c for c in target.grasp_candidates
            if self._cand_sig(c) not in used
        ]
        last_pre_grasp_failure = None

        for candidate in candidates_to_try:
            self._merge_candidate_attempt_diagnostic(
                candidate,
                self._classify_profile_diagnostic(target, candidate, env),
            )
            selected_strategy = (
                target.grasp_strategy.strategy
                if getattr(target, "grasp_strategy", None) is not None
                else None
            )
            profile = self._candidate_attempt_diagnostic(candidate).get("grasp_profile")
            gate_enabled = actionability_gate_enabled(self.grasp_policy_config, profile)

            pre_result = self._move_to_pre_grasp_with_recovery(candidate, env)
            actionability = actionability_from_pre_grasp_result(
                candidate,
                pre_result,
                selected_strategy=selected_strategy,
                target_body=self._resolve_target_body(target, env),
            )
            self._merge_candidate_attempt_diagnostic(
                candidate,
                {
                    **actionability.to_diagnostic(),
                    "candidate_actionability_policy": (
                        "pre_grasp_gate" if gate_enabled else "diagnostics_only"
                    ),
                    "no_actionable_candidate": False,
                    "skipped_candidate_sources": list(skipped_sources),
                },
            )
            if pre_result.ok or pre_result.handoff_ok:
                break
            last_pre_grasp_failure = (candidate, pre_result, actionability)
            if gate_enabled and actionability.hard_reject:
                skipped_sources.append(str(getattr(candidate, "source", "unknown")))
                try:
                    self.release_and_retreat(env)
                except Exception:
                    pass
                continue
            return self._failed_result(
                candidate,
                "ik_unreachable",
                {
                    "stage": "pre_grasp",
                    "pre_grasp_reason": pre_result.reason,
                    **self._pre_grasp_details(pre_result),
                },
                env,
            )
        else:
            if last_pre_grasp_failure is not None:
                failed_candidate, failed_pre_result, failed_actionability = last_pre_grasp_failure
                return self._failed_result(
                    failed_candidate,
                    "ik_unreachable",
                    {
                        **failed_actionability.to_diagnostic(),
                        "candidate_actionability_policy": "pre_grasp_gate",
                        "stage": "pre_grasp",
                        "pre_grasp_reason": failed_pre_result.reason,
                        "no_actionable_candidate": True,
                        "skipped_candidate_sources": list(skipped_sources),
                        **self._pre_grasp_details(failed_pre_result),
                    },
                    env,
                )
            return self._failed_result(
                None,
                "ik_unreachable",
                {"reason": "no_candidate", "no_actionable_candidate": True},
                env,
            )
```

Extract the old diagnostic pre-grasp logic into:

```python
    def _move_to_pre_grasp_with_recovery(self, candidate, env):
        if hasattr(env, "move_to_pre_grasp_diagnostic"):
            pre_result = env.move_to_pre_grasp_diagnostic(candidate)
            if not (pre_result.ok or pre_result.handoff_ok):
                _MAX_NUDGE_ITERS = 3
                for _nudge_iter in range(_MAX_NUDGE_ITERS):
                    if not pre_result.needs_recovery:
                        break
                    recover_ok = self._recover_pre_grasp(env, candidate, pre_result)
                    if not recover_ok:
                        return pre_result
                    if hasattr(env, "evaluate_pre_grasp_at_current"):
                        pre_result = env.evaluate_pre_grasp_at_current(candidate)
                    else:
                        pre_result = env.move_to_pre_grasp_diagnostic(candidate)
                    if pre_result.ok or pre_result.handoff_ok:
                        break
            return pre_result

        class _BoolPreResult:
            def __init__(self, ok: bool):
                self.ok = ok
                self.handoff_ok = ok
                self.needs_recovery = False
                self.reason = "strict_ok" if ok else "pre_grasp_unreachable"
                self.total_error_m = 0.0
                self.lateral_error_m = 0.0
                self.axis_error_m = 0.0
                self.approach_gap_m = 0.0
                self.lateral_limit_m = 0.0

        return _BoolPreResult(bool(env.move_to_pre_grasp(candidate)))
```

Keep the rest of the existing descend/close/lift logic after a candidate has passed pre-grasp.

- [ ] **Step 4: Run focused gate tests**

Run:

```bash
python -m pytest tests/test_action_executor_v1.py::test_profiled_actionability_gate_skips_hard_rejected_pre_grasp_candidate tests/test_action_executor_v1.py::test_actionability_gate_returns_no_actionable_candidate_when_all_hard_reject -q
```

Expected: PASS.

- [ ] **Step 5: Run full action executor tests**

Run:

```bash
python -m pytest tests/test_action_executor_v1.py tests/test_action_executor_phase6.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/action_executor.py tests/test_action_executor_v1.py
git commit -m "Gate profiled candidates at pre-grasp actionability" \
  -m "The executor now treats pre-grasp diagnostics as the first authoritative actionability check. In profiled gate mode, hard-rejected candidates can be skipped before descend or close, allowing the next candidate to run." \
  -m "Constraint: Legacy mode and diagnostics-only mode still return after the first pre-grasp failure as before." \
  -m "Rejected: Planner-only IK prediction | current planner lacks enough state to predict pre-grasp feasibility without repeating the is_reachable placeholder problem." \
  -m "Confidence: medium" \
  -m "Scope-risk: moderate" \
  -m "Tested: python -m pytest tests/test_action_executor_v1.py tests/test_action_executor_phase6.py -q"
```

---

### Task 7: Make Candidate-Source Policy Respect Hard Rejects

**Files:**

- Modify: `src/grasp_policy.py`
- Modify: `src/grasp_planner.py`
- Modify: `tests/test_grasp_strategy.py`

- [ ] **Step 1: Write failing planner test**

Append to `tests/test_grasp_strategy.py`:

```python

class _HardRejectGentleSideEnv(_FakeEnv):
    def preview_candidate_actionability(self, candidate, **kwargs):
        if candidate.source == "strategy_gentle_side":
            return {
                "actionable": False,
                "hard_reject": True,
                "reason": "axis_gap_too_large",
            }
        return {
            "actionable": True,
            "hard_reject": False,
            "reason": "preview_ok",
        }


def test_profiled_actionability_gate_prevents_promoting_hard_rejected_strategy():
    planner = GraspPlanner(
        vlm=MockVLM(['{"grip_norm": [0.5, 0.5]}']),
        env=_HardRejectGentleSideEnv(),
        llm=None,
        grasp_policy_config={
            "mode": "profiled",
            "enabled_profiles": ["small_round_slippery"],
            "actionability_diagnostics": True,
            "actionability_gate": True,
        },
    )
    h = _hyp("lemon", visible_features="round yellow smooth waxy fruit")
    h.grasp_strategy = GraspStrategy(strategy="gentle_side", slip_risk="high")

    cands = planner.plan(h)

    assert [c.source for c in cands] == [
        "vlm_top_grasp",
        "strategy_gentle_side",
        "geometric_centroid",
    ]
    diag = getattr(cands[0], "_embosight_attempt_diagnostic")
    assert diag["candidate_actionability_policy"] == "profiled_preview_gate"
    assert diag["legacy_first_candidate_actionable"] is True
    assert diag["final_first_candidate_actionable"] is True
    assert diag["no_actionable_candidate"] is False
```

- [ ] **Step 2: Run test and confirm failure**

Run:

```bash
python -m pytest tests/test_grasp_strategy.py::test_profiled_actionability_gate_prevents_promoting_hard_rejected_strategy -q
```

Expected: FAIL because planner currently lets candidate-source policy promote `strategy_gentle_side`.

- [ ] **Step 3: Add preview diagnostics support**

In `src/grasp_actionability.py`, add:

```python
def actionability_from_preview(
    candidate: Any,
    preview: dict[str, Any],
    *,
    selected_strategy: str | None,
    target_body: str | None,
) -> CandidateActionability:
    return CandidateActionability(
        source=str(getattr(candidate, "source", "unknown")),
        selected_strategy=selected_strategy,
        executed_strategy=executed_strategy_name(candidate, selected_strategy),
        target_body=target_body,
        actionable=bool(preview.get("actionable", True)),
        hard_reject=bool(preview.get("hard_reject", False)),
        reason=str(preview.get("reason", "preview_unknown")),
        total_error_m=_float_or_none(preview.get("total_error_m")),
        lateral_error_m=_float_or_none(preview.get("lateral_error_m")),
        axis_error_m=_float_or_none(preview.get("axis_error_m")),
        approach_gap_m=_float_or_none(preview.get("approach_gap_m")),
        lateral_limit_m=_float_or_none(preview.get("lateral_limit_m")),
        score_modifier=_float_or_none(preview.get("score_modifier")) or 0.0,
    )
```

In `src/grasp_planner.py`, update `_attach_actionability_diagnostics(...)`:

```python
        from src.grasp_actionability import actionability_from_preview
```

Inside the candidate loop:

```python
            if hasattr(env, "preview_candidate_actionability"):
                preview = env.preview_candidate_actionability(
                    candidate,
                    target_body=resolution.target_body,
                    selected_strategy=selected_strategy,
                )
                actionability = actionability_from_preview(
                    candidate,
                    preview if isinstance(preview, dict) else {},
                    selected_strategy=selected_strategy,
                    target_body=resolution.target_body,
                )
                policy_name = "profiled_preview_gate"
            else:
                actionability = unknown_actionability(
                    candidate,
                    selected_strategy=selected_strategy,
                    target_body=resolution.target_body,
                    reason="not_evaluated",
                )
                policy_name = "diagnostics_only"
```

Record `candidate_actionability_policy` as `policy_name`.

- [ ] **Step 4: Apply ordering only when preview proves hard reject**

In `src/grasp_planner.py`, after attaching diagnostics but before candidate-source policy:

```python
        cands = self._apply_actionability_preview_gate(profile, cands)
```

Add:

```python
    def _apply_actionability_preview_gate(
        self,
        profile: str,
        cands: list[GraspCandidate],
    ) -> list[GraspCandidate]:
        from src.grasp_policy import (
            actionability_gate_enabled,
            merge_candidate_attempt_diagnostic,
        )

        if not actionability_gate_enabled(self.grasp_policy_config, profile):
            return cands

        actionable = []
        rejected = []
        for candidate in cands:
            diag = getattr(candidate, "_embosight_attempt_diagnostic", {}) or {}
            if diag.get("candidate_actionability_hard_reject") is True:
                rejected.append(candidate)
            else:
                actionable.append(candidate)

        if not actionable:
            for candidate in cands:
                merge_candidate_attempt_diagnostic(candidate, {
                    "legacy_first_candidate_actionable": False,
                    "final_first_candidate_actionable": False,
                    "no_actionable_candidate": True,
                })
            return cands

        final_order = actionable + rejected
        legacy_first_actionable = (
            (getattr(cands[0], "_embosight_attempt_diagnostic", {}) or {})
            .get("candidate_actionability_hard_reject") is not True
        ) if cands else None
        final_first_actionable = (
            (getattr(final_order[0], "_embosight_attempt_diagnostic", {}) or {})
            .get("candidate_actionability_hard_reject") is not True
        ) if final_order else None
        for candidate in final_order:
            merge_candidate_attempt_diagnostic(candidate, {
                "legacy_first_candidate_actionable": legacy_first_actionable,
                "final_first_candidate_actionable": final_first_actionable,
                "no_actionable_candidate": False,
            })
        return final_order
```

- [ ] **Step 5: Run focused test**

Run:

```bash
python -m pytest tests/test_grasp_strategy.py::test_profiled_actionability_gate_prevents_promoting_hard_rejected_strategy -q
```

Expected: PASS.

- [ ] **Step 6: Run planner policy tests**

Run:

```bash
python -m pytest tests/test_grasp_strategy.py tests/test_grasp_policy.py tests/test_grasp_actionability.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/grasp_actionability.py src/grasp_planner.py tests/test_grasp_strategy.py
git commit -m "Prevent policy promotion of hard-rejected candidates" \
  -m "When a deterministic actionability preview marks a candidate as a hard reject, profiled candidate-source policy no longer promotes it ahead of actionable alternatives." \
  -m "Constraint: Real sim still relies on executor pre-grasp gate unless a non-moving preview hook is available." \
  -m "Confidence: medium" \
  -m "Scope-risk: moderate" \
  -m "Tested: python -m pytest tests/test_grasp_strategy.py tests/test_grasp_policy.py tests/test_grasp_actionability.py -q"
```

---

### Task 8: Extend Smoke And Baseline Reports

**Files:**

- Modify: `scripts/validate_lemon_grasp_multi.sh`
- Modify: `scripts/run_grasp_baseline.sh`
- Modify: `tests/test_grasp_baseline_script.py`

- [ ] **Step 1: Write failing script tests**

Append to `tests/test_grasp_baseline_script.py`:

```python

def test_validate_lemon_script_includes_actionability_fields():
    text = Path("scripts/validate_lemon_grasp_multi.sh").read_text(encoding="utf-8")

    assert "target_resolution_source" in text
    assert "candidate_actionability_policy" in text
    assert "candidate_actionability_reason" in text
    assert "no_actionable_candidate" in text


def test_baseline_report_includes_actionability_summary_sections():
    text = Path("scripts/run_grasp_baseline.sh").read_text(encoding="utf-8")

    assert "failure_family_breakdown" in text
    assert "failure_mode_by_actionability_reason" in text
    assert "candidate_actionability_usage" in text
    assert "target_resolution_source_usage" in text
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```bash
python -m pytest tests/test_grasp_baseline_script.py -q
```

Expected: FAIL because scripts do not include the new fields yet.

- [ ] **Step 3: Update smoke CSV fields**

In `scripts/validate_lemon_grasp_multi.sh`, add CSV columns:

```bash
target_resolution_source,target_body,candidate_actionability_policy,candidate_actionability_reason,candidate_actionability_actionable,candidate_actionability_hard_reject,no_actionable_candidate
```

Add extraction variables near existing `candidate_source_policy` extraction:

```bash
    target_resolution_source="$(extract_json_scalar 'target_resolution_source' "${log_path}")"
    target_body="$(extract_json_scalar 'target_body' "${log_path}")"
    candidate_actionability_policy="$(extract_json_scalar 'candidate_actionability_policy' "${log_path}")"
    candidate_actionability_reason="$(extract_json_scalar 'candidate_actionability_reason' "${log_path}")"
    candidate_actionability_actionable="$(extract_json_scalar 'candidate_actionability_actionable' "${log_path}")"
    candidate_actionability_hard_reject="$(extract_json_scalar 'candidate_actionability_hard_reject' "${log_path}")"
    no_actionable_candidate="$(extract_json_scalar 'no_actionable_candidate' "${log_path}")"
```

Append the values to the CSV row in the same order.

Add run summary lines:

```bash
    echo "  target_resolution_source: ${target_resolution_source:-unknown}"
    echo "  target_body: ${target_body:-unknown}"
    echo "  candidate_actionability_policy: ${candidate_actionability_policy:-unknown}"
    echo "  candidate_actionability_reason: ${candidate_actionability_reason:-unknown}"
    echo "  no_actionable_candidate: ${no_actionable_candidate:-unknown}"
```

- [ ] **Step 4: Update baseline report sections**

In `scripts/run_grasp_baseline.sh`, include summary keys:

```python
    "failure_family_breakdown",
    "failure_mode_by_actionability_reason",
    "candidate_actionability_usage",
    "target_resolution_source_usage",
```

- [ ] **Step 5: Run script tests and Bash syntax**

Run:

```bash
python -m pytest tests/test_grasp_baseline_script.py -q
"C:\Program Files\Git\bin\bash.exe" -n scripts/validate_lemon_grasp_multi.sh
"C:\Program Files\Git\bin\bash.exe" -n scripts/run_grasp_baseline.sh
```

Expected: pytest PASS and both Bash syntax checks exit 0.

- [ ] **Step 6: Commit**

```bash
git add scripts/validate_lemon_grasp_multi.sh scripts/run_grasp_baseline.sh tests/test_grasp_baseline_script.py
git commit -m "Expose actionability diagnostics in run reports" \
  -m "Smoke CSVs and baseline reports now include target-resolution and actionability buckets needed to judge the next GPU runs." \
  -m "Constraint: Reporting only; no execution behavior change." \
  -m "Confidence: high" \
  -m "Scope-risk: narrow" \
  -m "Tested: python -m pytest tests/test_grasp_baseline_script.py -q; bash -n scripts"
```

---

### Task 9: Final Verification

**Files:**

- No new files.
- Verify all files touched by Tasks 1-8.

- [ ] **Step 1: Run focused test suite**

Run:

```bash
python -m pytest \
  tests/test_target_resolution.py \
  tests/test_grasp_actionability.py \
  tests/test_grasp_policy.py \
  tests/test_grasp_strategy.py \
  tests/test_action_executor_v1.py \
  tests/test_action_executor_phase6.py \
  tests/test_eval_oracle.py \
  tests/test_long_generalization_runner.py \
  tests/test_grasp_baseline_script.py \
  -q
```

Expected: PASS.

- [ ] **Step 2: Run full tests**

Run:

```bash
python -m pytest tests/ -q --no-header
```

Expected: PASS.

- [ ] **Step 3: Run ruff**

Run:

```bash
python -m ruff check \
  src/target_resolution.py \
  src/grasp_actionability.py \
  src/grasp_policy.py \
  src/grasp_planner.py \
  src/action_executor.py \
  src/eval_oracle.py \
  eval/run_long_generalization.py \
  tests/test_target_resolution.py \
  tests/test_grasp_actionability.py \
  tests/test_grasp_policy.py \
  tests/test_grasp_strategy.py \
  tests/test_action_executor_v1.py \
  tests/test_eval_oracle.py \
  tests/test_long_generalization_runner.py \
  tests/test_grasp_baseline_script.py
```

Expected: `All checks passed!`

- [ ] **Step 4: Run shell syntax checks**

Run:

```bash
"C:\Program Files\Git\bin\bash.exe" -n scripts/validate_lemon_grasp_multi.sh
"C:\Program Files\Git\bin\bash.exe" -n scripts/run_grasp_baseline.sh
```

Expected: exit 0 for both commands.

- [ ] **Step 5: Run diff check**

Run:

```bash
git diff --check
```

Expected: no whitespace errors. CRLF warnings are acceptable if they match existing repository behavior.

- [ ] **Step 6: Inspect dirty worktree boundaries**

Run:

```bash
git status --short
```

Expected:

```text
Only files intentionally changed by this plan should be staged or committed.
.omx/ must remain untracked.
docs/CRAIC2026_查新报告.md must remain unstaged.
Unrelated dirty docs from previous work must remain unstaged.
```

- [ ] **Step 7: Final commit if needed**

If Tasks 1-8 were committed individually, skip this step.

If changes remain uncommitted, commit only the files touched by this plan:

```bash
git add \
  configs/agent.yaml \
  src/target_resolution.py \
  src/grasp_actionability.py \
  src/grasp_policy.py \
  src/grasp_planner.py \
  src/action_executor.py \
  src/eval_oracle.py \
  eval/run_long_generalization.py \
  scripts/validate_lemon_grasp_multi.sh \
  scripts/run_grasp_baseline.sh \
  tests/test_target_resolution.py \
  tests/test_grasp_actionability.py \
  tests/test_grasp_policy.py \
  tests/test_grasp_strategy.py \
  tests/test_action_executor_v1.py \
  tests/test_action_executor_phase6.py \
  tests/test_eval_oracle.py \
  tests/test_long_generalization_runner.py \
  tests/test_grasp_baseline_script.py

git commit -m "Add actionability-first grasp diagnostics and gate" \
  -m "Target resolution, candidate actionability, and pre-grasp gating give the grasp pipeline a shared physical feasibility contract before further profile tuning." \
  -m "Constraint: Legacy defaults remain unchanged; profiled gate is opt-in." \
  -m "Rejected: Continue tuning small_round_slippery squeeze/depth | Phase 4 showed promoted gentle-side candidates failed at pre-grasp feasibility." \
  -m "Confidence: medium" \
  -m "Scope-risk: moderate" \
  -m "Directive: Do not broaden actionability gate beyond explicitly enabled profiles until GPU evidence is reviewed." \
  -m "Tested: focused pytest; full pytest; ruff; bash syntax; git diff --check" \
  -m "Not-tested: GPU smoke/gen50"
```

---

## GPU Validation Commands

Create a profiled diagnostics-only config on GPU:

```bash
python - <<'PY'
from pathlib import Path
import yaml

base = yaml.safe_load(Path("configs/agent.yaml").read_text()) or {}
base["grasp_policy"] = {
    "mode": "profiled",
    "enabled_profiles": ["small_round_slippery"],
    "actionability_diagnostics": True,
    "actionability_gate": False,
}
Path("/tmp/agent-profiled-actionability-diagnostics.yaml").write_text(
    yaml.safe_dump(base, sort_keys=False),
)
PY
```

Run fixed lemon diagnostics-only smoke:

```bash
AGENT_CONFIG=/tmp/agent-profiled-actionability-diagnostics.yaml \
RUN_ID=phase5-actionability-diagnostics-smoke-$(date +%Y%m%d_%H%M%S) \
bash scripts/validate_lemon_grasp_multi.sh
```

Run diagnostics-only gen50:

```bash
python -m eval.run_long_generalization \
  --count 50 \
  --parallel 4 \
  --run-id phase5-actionability-diagnostics-gen50-$(date +%Y%m%d_%H%M%S) \
  --agent-config /tmp/agent-profiled-actionability-diagnostics.yaml
```

Create a profiled gate config only after diagnostics are reviewed:

```bash
python - <<'PY'
from pathlib import Path
import yaml

base = yaml.safe_load(Path("configs/agent.yaml").read_text()) or {}
base["grasp_policy"] = {
    "mode": "profiled",
    "enabled_profiles": ["small_round_slippery"],
    "actionability_diagnostics": True,
    "actionability_gate": True,
}
Path("/tmp/agent-profiled-actionability-gate.yaml").write_text(
    yaml.safe_dump(base, sort_keys=False),
)
PY
```

Run gate smoke:

```bash
AGENT_CONFIG=/tmp/agent-profiled-actionability-gate.yaml \
RUN_ID=phase5-actionability-gate-smoke-$(date +%Y%m%d_%H%M%S) \
bash scripts/validate_lemon_grasp_multi.sh
```

Run gate gen50:

```bash
python -m eval.run_long_generalization \
  --count 50 \
  --parallel 4 \
  --run-id phase5-actionability-gate-gen50-$(date +%Y%m%d_%H%M%S) \
  --agent-config /tmp/agent-profiled-actionability-gate.yaml
```

## Success Criteria

Minimum CPU criteria:

```text
focused pytest passes
full pytest passes
ruff passes
bash syntax checks pass
git diff --check passes
legacy config defaults remain unchanged except new disabled fields
```

Minimum GPU diagnostics-only criteria:

```text
fixed lemon remains stable
false success remains 0
post_lift_verified remains true for successes
target/actionability fields appear in oracle and long summary
failure_family_breakdown separates target, safety, planning, and execution failures
```

Minimum GPU gate criteria:

```text
fixed lemon remains stable
no increase in false success
hard-rejected pre-grasp candidates are skipped before close_gripper
random_seed_4 and random_seed_13 no longer terminate solely on the promoted gentle_side pre-grasp failure
candidate-source mismatch does not reappear for small_round_slippery
```

## Execution Notes

- Do not touch `.omx/`.
- Do not stage `docs/CRAIC2026_查新报告.md`.
- Do not stage unrelated dirty docs unless the user explicitly asks.
- Do not tune depth or squeeze in this plan.
- Do not change memory retrieval in this plan.
- If a test reveals that the pre-grasp gate changes legacy behavior, stop and fix the gate condition before continuing.
