from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_summarize_episode_extracts_failure_mode_and_target(tmp_path):
    from src.eval_oracle import summarize_episode

    episode = {
        "query": "pick up the apple",
        "snapshots": [
            {"step": 1, "target_summary": None},
            {
                "step": 2,
                "most_uncertain_axis": "label",
                "target_summary": {
                    "label": "apple",
                    "label_entropy": 0.2,
                    "position_3d": [0.4, 0.1, 0.8],
                    "position_std_m": 0.02,
                    "safety_entropy": 0.1,
                    "grasp_uncertainty": 0.2,
                },
            },
        ],
        "actions": [
            {"kind": "observe"},
            {"kind": "plan_grasp_candidates"},
            {"kind": "grasp"},
        ],
        "evidence": [
            {
                "source": "vlm_ground",
                "timestamp": 1.0,
                "raw_payload": {
                    "hypotheses": [
                        {"label": "apple", "label_alternatives": [["apple", 0.8]]}
                    ]
                },
            },
            {
                "source": "grasp_attempt",
                "timestamp": 2.0,
                "raw_payload": {
                    "success": False,
                    "attempt": {
                        "failure_mode": "hit_z_floor",
                        "candidate_source": "vlm_top_grasp",
                    },
                },
            },
        ],
        "final_result": {
            "success": False,
            "failure_reason": "MAX_STEPS reached",
        },
    }
    path = tmp_path / "episode.json"
    path.write_text(json.dumps(episode), encoding="utf-8")

    summary = summarize_episode(
        path,
        scenario_id="fixed_apple_001",
        expected_object="apple",
        actual_object="apple",
    )

    assert summary.scenario_id == "fixed_apple_001"
    assert summary.object_match is True
    assert summary.success is False
    assert summary.failure_reason == "MAX_STEPS reached"
    assert summary.action_sequence == ["observe", "plan_grasp_candidates", "grasp"]
    assert summary.vlm_labels == ["apple"]
    assert summary.selected_target_label == "apple"
    assert summary.selected_target_label_entropy == 0.2
    assert summary.selected_target_position_std_m == 0.02
    assert summary.selected_target_safety_entropy == 0.1
    assert summary.selected_target_grasp_uncertainty == 0.2
    assert summary.dominant_uncertainty_axis == "label"
    assert summary.planning_blockers == []
    assert summary.grasp_failure_mode == "hit_z_floor"
    assert summary.grasp_candidate_source == "vlm_top_grasp"
    assert summary.to_dict()["grasp_failure_mode"] == "hit_z_floor"


def test_oracle_summary_includes_final_grasp_evidence(tmp_path):
    from src.eval_oracle import summarize_episode

    episode = {
        "query": "pick up the lemon",
        "snapshots": [
            {
                "step": 4,
                "most_uncertain_axis": "grasp",
                "target_summary": {
                    "label": "lemon",
                    "label_entropy": 0.1,
                    "position_3d": [0.125, -2.857, 0.947],
                    "position_std_m": 0.02,
                    "safety_entropy": 0.1,
                    "grasp_uncertainty": 0.1,
                },
            },
        ],
        "actions": [
            {"kind": "observe"},
            {"kind": "classify_safety"},
            {"kind": "plan_grasp_candidates"},
            {"kind": "grasp"},
        ],
        "evidence": [
            {
                "source": "grasp_attempt",
                "timestamp": 2.0,
                "raw_payload": {
                    "success": True,
                    "attempt": {
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
                            "finger_width_m": 0.04,
                            "grasp_profile": "small_round_slippery",
                            "grasp_policy_mode": "profiled",
                            "grasp_policy_applied": True,
                            "grasp_policy_profile": "small_round_slippery",
                            "legacy_depth_margin_m": 0.025,
                            "legacy_squeeze_extra_steps": 4,
                            "candidate_source_policy": "prefer_selected_strategy_candidate",
                            "candidate_source_policy_applied": True,
                            "legacy_first_candidate_source": "vlm_top_grasp",
                            "final_first_candidate_source": "strategy_top_down",
                            "target_resolution_status": "resolved",
                            "target_body": "obj_main",
                            "target_body_category": "lemon",
                            "resolved_body_name": "obj_main",
                            "resolved_body_category": "lemon",
                            "target_resolution_source": "normalized_category",
                            "target_resolution_used_fallback": False,
                            "candidate_actionability_policy": "diagnostics_only",
                            "candidate_actionability_actionable": True,
                            "candidate_actionability_hard_reject": False,
                            "candidate_actionability_reason": "not_evaluated",
                            "actionability_status": "actionable",
                            "actionability_reason": "not_evaluated",
                            "actionability_stage": "planner",
                            "actionability_gate_enabled": False,
                            "actionability_gate_applied": False,
                            "actionability_skip_reason": None,
                            "legacy_first_candidate_actionable": True,
                            "final_first_candidate_actionable": True,
                            "no_actionable_candidate": False,
                            "execution_failure_stage": "pre_close_alignment",
                            "execution_failure_reason": "object_displaced_before_close",
                            "execution_failure_recoverable": True,
                            "execution_recovery_enabled": True,
                            "execution_recovery_applied": False,
                            "execution_recovery_reason": None,
                            "execution_recovery_skip_count": 0,
                            "pre_close_lateral_error_m": 0.089,
                            "pre_close_lateral_limit_m": 0.020,
                        },
                    },
                },
            },
        ],
        "final_result": {
            "success": True,
            "failure_reason": None,
        },
    }
    path = tmp_path / "episode.json"
    path.write_text(json.dumps(episode), encoding="utf-8")

    summary = summarize_episode(
        path,
        scenario_id="fixed_lemon_001",
        expected_object="lemon",
        actual_object="lemon",
    )
    data = summary.to_dict()

    assert data["post_lift_obj_pos"] == [0.134, -2.855, 1.038]
    assert data["post_lift_eef_pos"] == [0.127, -2.86, 1.055]
    assert data["post_lift_obj_delta_z"] == pytest.approx(0.091)
    assert data["selected_strategy"] == "top_down"
    assert data["executed_strategy"] == "top_down"
    assert data["depth_margin_m"] == 0.010
    assert data["squeeze_extra_steps"] == 18
    assert data["finger_width_m"] == 0.04
    assert data["grasp_profile"] == "small_round_slippery"
    assert data["grasp_policy_mode"] == "profiled"
    assert data["grasp_policy_applied"] is True
    assert data["grasp_policy_profile"] == "small_round_slippery"
    assert data["legacy_depth_margin_m"] == 0.025
    assert data["legacy_squeeze_extra_steps"] == 4
    assert data["candidate_source_policy"] == "prefer_selected_strategy_candidate"
    assert data["candidate_source_policy_applied"] is True
    assert data["legacy_first_candidate_source"] == "vlm_top_grasp"
    assert data["final_first_candidate_source"] == "strategy_top_down"
    assert data["target_resolution_status"] == "resolved"
    assert data["target_body"] == "obj_main"
    assert data["target_body_category"] == "lemon"
    assert data["resolved_body_name"] == "obj_main"
    assert data["resolved_body_category"] == "lemon"
    assert data["target_resolution_source"] == "normalized_category"
    assert data["target_resolution_used_fallback"] is False
    assert data["candidate_actionability_policy"] == "diagnostics_only"
    assert data["candidate_actionability_actionable"] is True
    assert data["candidate_actionability_hard_reject"] is False
    assert data["candidate_actionability_reason"] == "not_evaluated"
    assert data["actionability_status"] == "actionable"
    assert data["actionability_reason"] == "not_evaluated"
    assert data["actionability_stage"] == "planner"
    assert data["actionability_gate_enabled"] is False
    assert data["actionability_gate_applied"] is False
    assert data["actionability_skip_reason"] is None
    assert data["legacy_first_candidate_actionable"] is True
    assert data["final_first_candidate_actionable"] is True
    assert data["no_actionable_candidate"] is False
    assert data["execution_failure_stage"] == "pre_close_alignment"
    assert data["execution_failure_reason"] == "object_displaced_before_close"
    assert data["execution_failure_recoverable"] is True
    assert data["execution_recovery_enabled"] is True
    assert data["execution_recovery_applied"] is False
    assert data["execution_recovery_skip_count"] == 0
    assert data["pre_close_lateral_error_m"] == pytest.approx(0.089)
    assert data["attempts_count"] == 1
    assert data["post_lift_verified"] is True


def test_summarize_episode_reports_planning_blockers(tmp_path):
    from src.eval_oracle import summarize_episode

    episode = {
        "query": "pick up the blender_jug",
        "snapshots": [
            {
                "step": 3,
                "most_uncertain_axis": "label",
                "target_summary": {
                    "label": "blender_jug",
                    "label_entropy": 0.95,
                    "position_3d": [7.68, -0.98, 0.94],
                    "position_std_m": 0.02,
                    "safety_entropy": 0.2,
                    "grasp_uncertainty": None,
                },
            },
        ],
        "actions": [
            {"kind": "observe"},
            {"kind": "classify_safety"},
            {"kind": "re_observe"},
            {"kind": "ask_user"},
        ],
        "evidence": [],
        "final_result": {
            "success": False,
            "failure_reason": "MAX_STEPS reached",
        },
    }
    path = tmp_path / "episode.json"
    path.write_text(json.dumps(episode), encoding="utf-8")

    summary = summarize_episode(path)

    assert summary.selected_target_label == "blender_jug"
    assert summary.selected_target_label_entropy == 0.95
    assert summary.dominant_uncertainty_axis == "label"
    assert summary.planning_blockers == ["label_entropy>=0.80"]


def test_summarize_episode_does_not_report_low_hazard_safety_blocker(tmp_path):
    from src.eval_oracle import summarize_episode

    episode = {
        "query": "pick up the tupperware",
        "snapshots": [
            {
                "step": 3,
                "most_uncertain_axis": "safety",
                "target_summary": {
                    "label": "tupperware",
                    "label_entropy": 0.1,
                    "position_3d": [0.35, -3.19, 0.94],
                    "position_std_m": 0.02,
                    "safety_entropy": 0.52,
                    "safety_dist": {
                        "safe": 0.60,
                        "fragile": 0.38,
                        "sharp": 0.01,
                        "hot": 0.01,
                        "chemical": 0.0,
                    },
                    "grasp_uncertainty": None,
                },
            },
        ],
        "actions": [
            {"kind": "observe"},
            {"kind": "classify_safety"},
            {"kind": "ask_user"},
        ],
        "evidence": [],
        "final_result": {
            "success": False,
            "failure_reason": "MAX_STEPS reached",
        },
    }
    path = tmp_path / "episode.json"
    path.write_text(json.dumps(episode), encoding="utf-8")

    summary = summarize_episode(path)

    assert summary.planning_blockers == []
