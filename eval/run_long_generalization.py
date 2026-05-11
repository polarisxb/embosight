#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def generate_seed_scenarios(seed_start: int, count: int) -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    for seed in range(seed_start, seed_start + count):
        scenarios.append({
            "id": f"random_seed_{seed}",
            "env_name": "PickPlaceCounterToCabinet",
            "seed": seed,
            "query": "pick up anything",
            "expected_object": None,
            "user_mode": "fake_from_robocasa",
            "max_resets": 1,
        })
    return scenarios


def load_completed_results(results_path: Path) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    if not results_path.exists():
        return completed
    with open(results_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            scenario_id = item.get("scenario_id")
            if scenario_id:
                completed[str(scenario_id)] = item
    return completed


def append_jsonl(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    successes = sum(1 for r in results if r.get("success"))
    errors = sum(1 for r in results if r.get("error"))
    timeouts = sum(1 for r in results if r.get("failure_reason") == "timeout")
    step_values = [r["steps"] for r in results if isinstance(r.get("steps"), int)]
    time_values = [float(r["time_s"]) for r in results if r.get("time_s") is not None]

    failure_breakdown: dict[str, int] = {}
    strategy_usage: dict[str, int] = {}
    object_distribution: dict[str, int] = {}
    failed_runs: list[dict[str, Any]] = []

    for r in results:
        if not r.get("success"):
            reason = (
                r.get("grasp_failure_mode")
                or r.get("failure_reason")
                or r.get("error")
                or "unknown"
            )
            failure_breakdown[str(reason)] = failure_breakdown.get(str(reason), 0) + 1
            failed_runs.append(r)
        strategy = r.get("grasp_strategy")
        if strategy:
            strategy_usage[str(strategy)] = strategy_usage.get(str(strategy), 0) + 1
        actual_object = r.get("actual_object")
        if actual_object:
            object_distribution[str(actual_object)] = object_distribution.get(str(actual_object), 0) + 1

    slowest_runs = sorted(
        results,
        key=lambda r: float(r.get("time_s") or 0.0),
        reverse=True,
    )[:10]

    return {
        "total": total,
        "completed": total,
        "successes": successes,
        "success_rate": successes / total if total else 0.0,
        "errors": errors,
        "timeouts": timeouts,
        "avg_steps": sum(step_values) / len(step_values) if step_values else 0.0,
        "avg_time_s": sum(time_values) / len(time_values) if time_values else 0.0,
        "failure_breakdown": dict(sorted(failure_breakdown.items(), key=lambda x: (-x[1], x[0]))),
        "strategy_usage": dict(sorted(strategy_usage.items(), key=lambda x: (-x[1], x[0]))),
        "object_distribution": dict(sorted(object_distribution.items(), key=lambda x: (-x[1], x[0]))),
        "slowest_runs": slowest_runs,
        "failed_runs": failed_runs,
    }
