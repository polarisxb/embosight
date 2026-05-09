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
    assert summary.grasp_failure_mode == "hit_z_floor"
    assert summary.grasp_candidate_source == "vlm_top_grasp"
    assert summary.to_dict()["grasp_failure_mode"] == "hit_z_floor"
