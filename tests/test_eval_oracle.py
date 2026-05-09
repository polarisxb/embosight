from __future__ import annotations

import json
import sys
from pathlib import Path

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
