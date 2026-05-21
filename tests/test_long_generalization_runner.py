from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


RUNNER_PATH = Path(__file__).parent.parent / "eval" / "run_long_generalization.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_long_generalization", str(RUNNER_PATH))
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generate_seed_scenarios():
    module = _load_module()

    scenarios = module.generate_seed_scenarios(seed_start=101, count=3)

    assert [s["id"] for s in scenarios] == [
        "random_seed_101", "random_seed_102", "random_seed_103",
    ]
    assert [s["seed"] for s in scenarios] == [101, 102, 103]
    assert all(s["query"] == "pick up anything" for s in scenarios)
    assert all(s["expected_object"] is None for s in scenarios)
    assert all(s["user_mode"] == "fake_from_robocasa" for s in scenarios)


def test_load_completed_results_ignores_corrupt_lines(tmp_path):
    module = _load_module()
    path = tmp_path / "results.jsonl"
    path.write_text(
        json.dumps({"scenario_id": "random_seed_101", "success": True}) + "\n"
        "not-json\n"
        + json.dumps({"scenario_id": "random_seed_102", "success": False}) + "\n",
        encoding="utf-8",
    )

    completed = module.load_completed_results(path)

    assert set(completed) == {"random_seed_101", "random_seed_102"}
    assert completed["random_seed_101"]["success"] is True
    assert completed["random_seed_102"]["success"] is False


def test_summarize_results_counts_failures_strategies_objects_and_slowest():
    module = _load_module()
    results = [
        {
            "scenario_id": "random_seed_101", "success": True, "error": None,
            "failure_reason": None, "grasp_failure_mode": None,
            "grasp_strategy": "strategy_top_down", "actual_object": "apple",
            "steps": 4, "time_s": 10.0,
        },
        {
            "scenario_id": "random_seed_102", "success": False, "error": None,
            "failure_reason": "MAX_STEPS reached", "grasp_failure_mode": None,
            "grasp_strategy": "strategy_top_down", "actual_object": "wine",
            "steps": 12, "time_s": 20.0,
        },
        {
            "scenario_id": "random_seed_103", "success": False, "error": "timeout",
            "failure_reason": "timeout", "grasp_failure_mode": None,
            "grasp_strategy": None, "actual_object": None,
            "steps": None, "time_s": 900.0,
        },
    ]

    summary = module.summarize_results(results)

    assert summary["total"] == 3
    assert summary["successes"] == 1
    assert summary["success_rate"] == 1 / 3
    assert summary["errors"] == 1
    assert summary["timeouts"] == 1
    assert summary["failure_breakdown"] == {
        "MAX_STEPS reached": 1,
        "timeout": 1,
    }
    assert summary["strategy_usage"] == {"strategy_top_down": 2}
    assert summary["object_distribution"] == {"apple": 1, "wine": 1}
    assert summary["slowest_runs"][0]["scenario_id"] == "random_seed_103"


def test_summarize_results_uses_selected_target_for_object_cross_tabs():
    module = _load_module()
    results = [
        {
            "scenario_id": "fixed_lemon_001",
            "success": True,
            "failure_reason": None,
            "grasp_failure_mode": None,
            "actual_object": "tupperware",
            "selected_target_label": "lemon",
            "grasp_profile": "small_round_slippery",
            "steps": 4,
            "time_s": 115.0,
        },
    ]

    summary = module.summarize_results(results)

    assert summary["object_distribution"] == {"lemon": 1}
    assert summary["success_rate_by_object"]["lemon"]["success_rate"] == 1.0
    assert "tupperware" not in summary["success_rate_by_object"]


def test_summarize_results_classifies_max_steps_action_loops():
    module = _load_module()
    results = [
        {
            "scenario_id": "random_seed_14",
            "success": False,
            "failure_reason": "MAX_STEPS reached",
            "grasp_failure_mode": None,
            "actual_object": "juice",
            "selected_target_label": "juice",
            "grasp_strategy": None,
            "executed_strategy": None,
            "action_sequence": [
                "observe",
                "observe",
                "ask_user",
                "ask_user",
                "ask_user",
            ],
            "steps": 13,
            "time_s": 50.0,
        },
        {
            "scenario_id": "random_seed_16",
            "success": False,
            "failure_reason": "MAX_STEPS reached",
            "grasp_failure_mode": None,
            "actual_object": "coffee_cup",
            "selected_target_label": "coffee_cup",
            "grasp_strategy": None,
            "executed_strategy": None,
            "action_sequence": [
                "observe",
                "classify_safety",
                "re_observe",
                "classify_safety",
                "classify_safety",
            ],
            "steps": 13,
            "time_s": 55.0,
        },
        {
            "scenario_id": "random_seed_23",
            "success": False,
            "failure_reason": "MAX_STEPS reached",
            "grasp_failure_mode": None,
            "actual_object": "kebab_skewer",
            "selected_target_label": "kebab_skewer",
            "grasp_strategy": None,
            "executed_strategy": None,
            "action_sequence": [
                "observe",
                "plan_grasp_candidates",
                "re_observe",
                "observe",
                "re_observe",
            ],
            "steps": 13,
            "time_s": 60.0,
        },
    ]

    summary = module.summarize_results(results)

    assert summary["failure_breakdown"] == {
        "clarification_loop": 1,
        "planning_loop": 1,
        "safety_loop": 1,
    }
    assert summary["failure_mode_by_object"]["juice"]["clarification_loop"] == 1
    assert summary["failure_mode_by_object"]["coffee_cup"]["safety_loop"] == 1
    assert summary["failure_mode_by_object"]["kebab_skewer"]["planning_loop"] == 1


def test_summarize_results_builds_diagnostic_cross_tabs():
    module = _load_module()
    summary = module.summarize_results([
        {
            "scenario_id": "random_seed_0",
            "success": False,
            "actual_object": "straw",
            "grasp_failure_mode": "slipped_descend",
            "grasp_strategy": "strategy_top_down",
            "executed_strategy": "top_down",
            "grasp_profile": "thin_flat",
            "steps": 8,
            "time_s": 10.0,
        },
        {
            "scenario_id": "random_seed_1",
            "success": False,
            "actual_object": "jug",
            "grasp_failure_mode": "ik_unreachable",
            "grasp_strategy": "vlm_top_grasp",
            "executed_strategy": "top_down",
            "grasp_profile": "handled",
            "steps": 9,
            "time_s": 12.0,
        },
        {
            "scenario_id": "random_seed_2",
            "success": True,
            "actual_object": "lemon",
            "grasp_failure_mode": "success",
            "grasp_strategy": "strategy_top_down",
            "executed_strategy": "top_down",
            "grasp_profile": "small_round_slippery",
            "steps": 4,
            "time_s": 8.0,
        },
    ])

    assert summary["failure_mode_by_object"]["straw"]["slipped_descend"] == 1
    assert summary["failure_mode_by_candidate_source"]["vlm_top_grasp"]["ik_unreachable"] == 1
    assert summary["failure_mode_by_executed_strategy"]["top_down"]["slipped_descend"] == 1
    assert summary["success_rate_by_object"]["lemon"]["success_rate"] == 1.0
    assert summary["success_rate_by_profile"]["small_round_slippery"]["success_rate"] == 1.0


def test_summarize_results_counts_grasp_policy_usage():
    module = _load_module()
    summary = module.summarize_results([
        {
            "scenario_id": "fixed_lemon_001",
            "success": True,
            "actual_object": "lemon",
            "grasp_failure_mode": "success",
            "grasp_policy_mode": "legacy",
            "grasp_policy_applied": False,
            "grasp_policy_profile": "small_round_slippery",
            "candidate_source_policy": "legacy",
            "candidate_source_policy_applied": False,
            "legacy_first_candidate_source": "vlm_top_grasp",
            "final_first_candidate_source": "vlm_top_grasp",
            "steps": 4,
            "time_s": 8.0,
        },
        {
            "scenario_id": "fixed_lime_001",
            "success": True,
            "actual_object": "lime",
            "grasp_failure_mode": "success",
            "grasp_policy_mode": "profiled",
            "grasp_policy_applied": True,
            "grasp_policy_profile": "small_round_slippery",
            "candidate_source_policy": "prefer_selected_strategy_candidate",
            "candidate_source_policy_applied": True,
            "legacy_first_candidate_source": "vlm_top_grasp",
            "final_first_candidate_source": "strategy_gentle_side",
            "steps": 4,
            "time_s": 8.5,
        },
    ])

    assert summary["grasp_policy_usage"] == {
        "legacy:small_round_slippery:not_applied": 1,
        "profiled:small_round_slippery:applied": 1,
    }
    assert summary["candidate_source_policy_usage"] == {
        "legacy:not_applied": 1,
        "prefer_selected_strategy_candidate:applied": 1,
    }
    assert summary["candidate_source_transition_usage"] == {
        "vlm_top_grasp->strategy_gentle_side": 1,
        "vlm_top_grasp->vlm_top_grasp": 1,
    }


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
    assert (
        summary["failure_mode_by_actionability_reason"]["axis_gap_too_large"]
        ["ik_unreachable"]
        == 1
    )
    assert summary["candidate_actionability_usage"] == {
        "pre_grasp_gate:axis_gap_too_large:hard_reject": 1,
    }
    assert summary["target_resolution_source_usage"] == {
        "normalized_category": 1,
    }
    assert summary["no_actionable_candidate_count"] == 0


def test_summarize_results_counts_execution_failure_recovery_fields():
    module = _load_module()

    summary = module.summarize_results([
        {
            "success": False,
            "grasp_failure_mode": "slipped_descend",
            "execution_failure_stage": "pre_close_alignment",
            "execution_failure_reason": "object_displaced_before_close",
            "execution_recovery_applied": True,
            "execution_recovery_reason": "retry_next_candidate",
            "selected_target_label": "boxed_food",
            "grasp_strategy": "vlm_top_grasp",
            "executed_strategy": "top_down",
            "grasp_profile": "wide_ungraspable",
            "time_s": 1.0,
            "steps": 4,
        },
    ])

    assert summary["execution_failure_stage_usage"] == {
        "pre_close_alignment": 1,
    }
    assert summary["failure_mode_by_execution_stage"] == {
        "pre_close_alignment": {"slipped_descend": 1},
    }
    assert summary["failure_mode_by_execution_reason"] == {
        "object_displaced_before_close": {"slipped_descend": 1},
    }
    assert summary["execution_recovery_usage"] == {
        "applied:retry_next_candidate": 1,
    }


def test_parse_run_fixed_output_extracts_oracle_and_episode_result():
    module = _load_module()
    stdout = '''
========== EPISODE RESULT ==========
scenario: random_seed_101
success : True
speech  : 我来拿apple
steps   : 4
time    : 12.3s

========== ORACLE SUMMARY ==========
{
  "success": true,
  "failure_reason": null,
  "grasp_failure_mode": "success",
  "grasp_candidate_source": "strategy_top_down",
  "action_sequence": ["observe", "classify_safety", "plan_grasp_candidates", "grasp"],
  "selected_target_label": "apple",
  "actual_object": "apple"
}
episode: logs/episodes/episode_1.json
'''

    result = module.parse_run_fixed_output(
        scenario_id="random_seed_101",
        seed=101,
        returncode=0,
        stdout=stdout,
        stderr="",
        elapsed=12.34,
    )

    assert result["scenario_id"] == "random_seed_101"
    assert result["seed"] == 101
    assert result["success"] is True
    assert result["steps"] == 4
    assert result["speech"] == "我来拿apple"
    assert result["grasp_failure_mode"] == "success"
    assert result["grasp_strategy"] == "strategy_top_down"
    assert result["action_sequence"] == [
        "observe", "classify_safety", "plan_grasp_candidates", "grasp",
    ]
    assert result["actual_object"] == "apple"
    assert result["selected_target_label"] == "apple"


def test_parse_run_fixed_output_preserves_final_grasp_oracle_fields():
    module = _load_module()
    stdout = '''
========== EPISODE RESULT ==========
scenario: random_seed_101
success : True
speech  : done
steps   : 4
time    : 12.3s

========== ORACLE SUMMARY ==========
{
  "success": true,
  "failure_reason": null,
  "grasp_failure_mode": "success",
  "grasp_candidate_source": "strategy_top_down",
  "selected_strategy": "top_down",
  "executed_strategy": "top_down",
  "post_lift_obj_pos": [0.134, -2.855, 1.038],
  "post_lift_obj_delta_z": 0.091,
  "depth_margin_m": 0.01,
  "squeeze_extra_steps": 18,
  "grasp_profile": "small_round_slippery",
  "grasp_policy_mode": "profiled",
  "grasp_policy_applied": true,
  "grasp_policy_profile": "small_round_slippery",
  "legacy_depth_margin_m": 0.025,
  "legacy_squeeze_extra_steps": 4,
  "candidate_source_policy": "prefer_selected_strategy_candidate",
  "candidate_source_policy_applied": true,
  "legacy_first_candidate_source": "vlm_top_grasp",
  "final_first_candidate_source": "strategy_top_down",
  "target_resolution_status": "resolved",
  "target_body": "obj_main",
  "target_body_category": "lemon",
  "resolved_body_name": "obj_main",
  "resolved_body_category": "lemon",
  "target_resolution_source": "normalized_category",
  "target_resolution_used_fallback": false,
  "candidate_actionability_policy": "diagnostics_only",
  "candidate_actionability_actionable": true,
  "candidate_actionability_hard_reject": false,
  "candidate_actionability_reason": "not_evaluated",
  "actionability_status": "actionable",
  "actionability_reason": "not_evaluated",
  "actionability_stage": "planner",
  "actionability_gate_enabled": false,
  "actionability_gate_applied": false,
  "actionability_skip_reason": null,
  "legacy_first_candidate_actionable": true,
  "final_first_candidate_actionable": true,
  "no_actionable_candidate": false,
  "action_sequence": ["observe", "classify_safety", "plan_grasp_candidates", "grasp"],
  "selected_target_label": "lemon",
  "actual_object": "lemon"
}
episode: logs/episodes/episode_1.json
'''

    result = module.parse_run_fixed_output(
        scenario_id="random_seed_101",
        seed=101,
        returncode=0,
        stdout=stdout,
        stderr="",
        elapsed=12.34,
    )

    assert result["selected_strategy"] == "top_down"
    assert result["executed_strategy"] == "top_down"
    assert result["post_lift_obj_pos"] == [0.134, -2.855, 1.038]
    assert result["post_lift_obj_delta_z"] == 0.091
    assert result["depth_margin_m"] == 0.01
    assert result["squeeze_extra_steps"] == 18
    assert result["grasp_profile"] == "small_round_slippery"
    assert result["grasp_policy_mode"] == "profiled"
    assert result["grasp_policy_applied"] is True
    assert result["grasp_policy_profile"] == "small_round_slippery"
    assert result["legacy_depth_margin_m"] == 0.025
    assert result["legacy_squeeze_extra_steps"] == 4
    assert result["candidate_source_policy"] == "prefer_selected_strategy_candidate"
    assert result["candidate_source_policy_applied"] is True
    assert result["legacy_first_candidate_source"] == "vlm_top_grasp"
    assert result["final_first_candidate_source"] == "strategy_top_down"
    assert result["target_resolution_status"] == "resolved"
    assert result["target_body"] == "obj_main"
    assert result["target_body_category"] == "lemon"
    assert result["resolved_body_name"] == "obj_main"
    assert result["resolved_body_category"] == "lemon"
    assert result["target_resolution_source"] == "normalized_category"
    assert result["target_resolution_used_fallback"] is False
    assert result["candidate_actionability_policy"] == "diagnostics_only"
    assert result["candidate_actionability_actionable"] is True
    assert result["candidate_actionability_hard_reject"] is False
    assert result["candidate_actionability_reason"] == "not_evaluated"
    assert result["actionability_status"] == "actionable"
    assert result["actionability_reason"] == "not_evaluated"
    assert result["actionability_stage"] == "planner"
    assert result["actionability_gate_enabled"] is False
    assert result["actionability_gate_applied"] is False
    assert result["actionability_skip_reason"] is None
    assert result["legacy_first_candidate_actionable"] is True
    assert result["final_first_candidate_actionable"] is True
    assert result["no_actionable_candidate"] is False


def test_prepare_memory_dir_writes_empty_index_and_domains(tmp_path):
    module = _load_module()
    memory_dir = tmp_path / "memory" / "random_seed_101"

    module.prepare_memory_dir(memory_dir)

    assert (memory_dir / "index.yaml").exists()
    assert (memory_dir / "grasp_experience.yaml").exists()
    assert (memory_dir / "recognition_hints.yaml").exists()
    index_text = (memory_dir / "index.yaml").read_text(encoding="utf-8")
    assert "grasp" in index_text
    assert "recognition" in index_text


def test_parse_cli_args_defaults():
    module = _load_module()

    args = module.parse_args(["--count", "5"])

    assert args.seed_start == 0
    assert args.count == 5
    assert args.parallel == 4
    assert args.timeout_s == 900
    assert args.resume is False
    assert args.run_id is not None


def test_parse_cli_args_custom():
    module = _load_module()

    args = module.parse_args([
        "--seed-start", "100",
        "--count", "50",
        "--parallel", "2",
        "--run-id", "overnight-1",
        "--resume",
        "--timeout-s", "1800",
    ])

    assert args.seed_start == 100
    assert args.count == 50
    assert args.parallel == 2
    assert args.run_id == "overnight-1"
    assert args.resume is True
    assert args.timeout_s == 1800


def test_write_scenarios_yaml(tmp_path):
    module = _load_module()

    scenarios = module.generate_seed_scenarios(seed_start=0, count=2)
    out_path = tmp_path / "scenarios.yaml"
    module.write_scenarios_yaml(scenarios, out_path)

    import yaml
    loaded = yaml.safe_load(out_path.read_text(encoding="utf-8"))
    assert len(loaded["scenarios"]) == 2
    assert loaded["scenarios"][0]["id"] == "random_seed_0"
    assert loaded["scenarios"][1]["seed"] == 1


def test_format_summary_text():
    module = _load_module()
    summary = {
        "total": 3, "completed": 3, "successes": 1,
        "success_rate": 1 / 3, "errors": 1, "timeouts": 1,
        "avg_steps": 8.0, "avg_time_s": 310.0,
        "failure_breakdown": {"timeout": 1, "MAX_STEPS reached": 1},
        "strategy_usage": {"strategy_top_down": 2},
        "grasp_policy_usage": {"profiled:small_round_slippery:applied": 1},
        "candidate_source_policy_usage": {
            "prefer_selected_strategy_candidate:applied": 1,
        },
        "candidate_source_transition_usage": {
            "vlm_top_grasp->strategy_gentle_side": 1,
        },
        "object_distribution": {"apple": 1, "wine": 1},
        "slowest_runs": [], "failed_runs": [],
    }

    text = module.format_summary_text(summary)

    assert "33.3%" in text
    assert "timeout" in text
    assert "strategy_top_down" in text
    assert "profiled:small_round_slippery:applied" in text
    assert "prefer_selected_strategy_candidate:applied" in text
    assert "vlm_top_grasp->strategy_gentle_side" in text


def test_format_summary_text_includes_diagnostic_cross_tabs():
    module = _load_module()
    summary = {
        "total": 3,
        "completed": 3,
        "successes": 1,
        "success_rate": 1 / 3,
        "errors": 0,
        "timeouts": 0,
        "avg_steps": 7.0,
        "avg_time_s": 10.0,
        "failure_breakdown": {"slipped_descend": 1},
        "strategy_usage": {"strategy_top_down": 2},
        "object_distribution": {"lemon": 1, "straw": 1},
        "failure_mode_by_object": {"straw": {"slipped_descend": 1}},
        "failure_mode_by_candidate_source": {
            "strategy_top_down": {"slipped_descend": 1},
        },
        "failure_mode_by_executed_strategy": {
            "top_down": {"slipped_descend": 1},
        },
        "success_rate_by_profile": {
            "small_round_slippery": {"successes": 1, "total": 1, "success_rate": 1.0},
        },
        "slowest_runs": [],
        "failed_runs": [],
    }

    text = module.format_summary_text(summary)

    assert "Failure Mode By Object" in text
    assert "failure_mode_by_candidate_source" in text
    assert "success_rate_by_profile" in text
