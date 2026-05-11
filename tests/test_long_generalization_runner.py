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
