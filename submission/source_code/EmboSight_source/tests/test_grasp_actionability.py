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

    candidate = _candidate()
    result = evaluate_pre_grasp_handoff(
        move_ok=False,
        final_eef=np.array([0.34, 0.0, 0.9], dtype=np.float32),
        pre_pos=np.array([0.45, 0.0, 0.9], dtype=np.float32),
        grasp_point=np.array([0.5, 0.0, 0.9], dtype=np.float32),
        approach_dir=candidate.approach_dir,
        finger_width_m=candidate.finger_width_m,
        height_m=0.05,
    )

    actionability = actionability_from_pre_grasp_result(
        candidate,
        result,
        selected_strategy="gentle_side",
        target_body="obj_main",
    )

    assert actionability.actionable is False
    assert actionability.hard_reject is True
    assert actionability.reason == "axis_gap_too_large"
    assert actionability.stage == "pre_grasp"
    assert actionability.source == "strategy_gentle_side"
    assert actionability.executed_strategy == "gentle_side"
    assert actionability.target_body == "obj_main"


def test_actionability_from_pre_grasp_lateral_misaligned_is_recoverable():
    from src.grasp_actionability import actionability_from_pre_grasp_result

    candidate = _candidate()
    result = evaluate_pre_grasp_handoff(
        move_ok=False,
        final_eef=np.array([0.45, 0.08, 0.9], dtype=np.float32),
        pre_pos=np.array([0.45, 0.0, 0.9], dtype=np.float32),
        grasp_point=np.array([0.5, 0.0, 0.9], dtype=np.float32),
        approach_dir=candidate.approach_dir,
        finger_width_m=candidate.finger_width_m,
        height_m=0.05,
    )

    actionability = actionability_from_pre_grasp_result(
        candidate,
        result,
        selected_strategy="gentle_side",
        target_body="obj_main",
    )

    assert actionability.actionable is False
    assert actionability.hard_reject is False
    assert actionability.reason == "lateral_misaligned"


def test_unknown_actionability_is_not_a_hard_reject():
    from src.grasp_actionability import unknown_actionability

    candidate = _candidate()
    actionability = unknown_actionability(
        candidate,
        selected_strategy="gentle_side",
        target_body="obj_main",
        reason="not_evaluated",
    )

    assert actionability.actionable is True
    assert actionability.hard_reject is False
    assert actionability.reason == "not_evaluated"


def test_actionability_diagnostic_keys_are_stable():
    from src.grasp_actionability import unknown_actionability

    candidate = _candidate()
    data = unknown_actionability(
        candidate,
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
    assert data["candidate_actionability_stage"] == "planner"
    assert data["actionability_status"] == "actionable"
    assert data["actionability_reason"] == "not_evaluated"
    assert data["actionability_stage"] == "planner"
