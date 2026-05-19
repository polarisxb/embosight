#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

FINAL_GRASP_ORACLE_FIELDS = (
    "selected_strategy",
    "executed_strategy",
    "post_lift_obj_pos",
    "post_lift_obj_delta_z",
    "post_lift_eef_pos",
    "depth_margin_m",
    "squeeze_extra_steps",
    "finger_width_m",
    "grasp_profile",
    "grasp_profile_confidence",
    "grasp_profile_reasons",
    "grasp_policy_mode",
    "grasp_policy_applied",
    "grasp_policy_profile",
    "legacy_depth_margin_m",
    "legacy_squeeze_extra_steps",
    "candidate_source_policy",
    "candidate_source_policy_applied",
    "legacy_first_candidate_source",
    "final_first_candidate_source",
    "attempts_count",
    "post_lift_verified",
)

ORACLE_DIAGNOSTIC_FIELDS = (
    "selected_target_label",
)


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


def parse_run_fixed_output(
    scenario_id: str,
    seed: int,
    returncode: int,
    stdout: str,
    stderr: str,
    elapsed: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "scenario_id": scenario_id,
        "seed": seed,
        "success": returncode == 0,
        "error": None,
        "steps": None,
        "time_s": round(elapsed, 1),
        "failure_reason": None,
        "grasp_failure_mode": None,
        "grasp_strategy": None,
        "action_sequence": [],
        "speech": None,
        "actual_object": None,
    }
    for key in ORACLE_DIAGNOSTIC_FIELDS:
        result[key] = None
    for key in FINAL_GRASP_ORACLE_FIELDS:
        result[key] = None

    try:
        oracle_start = stdout.find("ORACLE SUMMARY")
        if oracle_start >= 0:
            json_start = stdout.find("{", oracle_start)
            json_end = stdout.find("\nepisode:", json_start)
            if json_end < 0:
                json_end = stdout.rfind("}")
            if json_start >= 0 and json_end >= 0:
                oracle_json = stdout[json_start:json_end + 1]
                oracle = json.loads(oracle_json)
                result["success"] = oracle.get("success", False)
                result["failure_reason"] = oracle.get("failure_reason")
                result["grasp_failure_mode"] = oracle.get("grasp_failure_mode")
                result["grasp_strategy"] = oracle.get("grasp_candidate_source")
                result["action_sequence"] = oracle.get("action_sequence", [])
                result["actual_object"] = oracle.get("actual_object")
                for key in ORACLE_DIAGNOSTIC_FIELDS:
                    result[key] = oracle.get(key)
                for key in FINAL_GRASP_ORACLE_FIELDS:
                    result[key] = oracle.get(key)

        ep_start = stdout.find("EPISODE RESULT")
        if ep_start >= 0:
            for line in stdout[ep_start:ep_start + 500].splitlines():
                if line.startswith("steps"):
                    try:
                        result["steps"] = int(line.split(":", 1)[1].strip())
                    except (ValueError, IndexError):
                        pass
                elif line.startswith("speech"):
                    result["speech"] = line.split(":", 1)[1].strip() if ":" in line else None
                elif line.startswith("reason") and not result["failure_reason"]:
                    result["failure_reason"] = line.split(":", 1)[1].strip()
    except Exception as e:
        if returncode != 0 and not result["error"]:
            tail = (stderr or stdout)[-500:] if (stderr or stdout) else "unknown"
            result["error"] = f"parse_failed ({type(e).__name__}: {e}): {tail[-200:]}"

    if returncode != 0 and not result["failure_reason"] and not result["error"]:
        result["failure_reason"] = "subprocess_nonzero_exit"
        result["error"] = stderr[-300:] if stderr else "no stderr"

    return result


def prepare_memory_dir(memory_dir: Path) -> None:
    memory_dir.mkdir(parents=True, exist_ok=True)
    grasp_path = memory_dir / "grasp_experience.yaml"
    recognition_path = memory_dir / "recognition_hints.yaml"
    safety_path = memory_dir / "safety_knowledge.yaml"
    index_path = memory_dir / "index.yaml"

    for p in (grasp_path, recognition_path, safety_path):
        if not p.exists():
            p.write_text(yaml.dump({"entries": []}, allow_unicode=True), encoding="utf-8")
    index_path.write_text(
        yaml.dump({
            "version": 1,
            "domains": {
                "grasp": str(grasp_path),
                "recognition": str(recognition_path),
                "safety": str(safety_path),
            },
        }, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def run_one_seed_subprocess(
    scenario: dict[str, Any],
    run_dir: Path,
    scenarios_config: Path,
    gpu_id: int,
    config: str,
    agent_config: str,
    log_level: str,
    timeout_s: int,
) -> dict[str, Any]:
    scenario_id = str(scenario["id"])
    seed = int(scenario["seed"])
    per_run_log_dir = run_dir / "per_run_logs"
    per_run_log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = per_run_log_dir / f"{scenario_id}.stdout.txt"
    stderr_path = per_run_log_dir / f"{scenario_id}.stderr.txt"
    memory_dir = run_dir / "memory" / scenario_id
    prepare_memory_dir(memory_dir)

    env_vars = os.environ.copy()
    env_vars["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    run_script = os.environ.get("EMBOSIGHT_RUN_FIXED_SCRIPT", "eval/run_fixed.py")
    cmd = [
        sys.executable, run_script,
        "--scenario", scenario_id,
        "--scenarios-config", str(scenarios_config),
        "--config", config,
        "--agent-config", agent_config,
        "--log-level", log_level,
        "--allow-object-mismatch",
        "--memory-dir", str(memory_dir),
    ]

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            env=env_vars,
            timeout=timeout_s,
        )
        elapsed = time.time() - t0
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")
        result = parse_run_fixed_output(
            scenario_id=scenario_id,
            seed=seed,
            returncode=proc.returncode,
            stdout=stdout,
            stderr=stderr,
            elapsed=elapsed,
        )
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout if isinstance(e.stdout, str) else ""
        stderr = e.stderr if isinstance(e.stderr, str) else ""
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")
        result = {
            "scenario_id": scenario_id,
            "seed": seed,
            "success": False,
            "error": f"subprocess timeout ({timeout_s}s)",
            "time_s": float(timeout_s),
            "steps": None,
            "failure_reason": "timeout",
            "grasp_failure_mode": None,
            "grasp_strategy": None,
            "action_sequence": [],
            "speech": None,
            "actual_object": None,
        }
        for key in ORACLE_DIAGNOSTIC_FIELDS:
            result[key] = None
        for key in FINAL_GRASP_ORACLE_FIELDS:
            result[key] = None
    except Exception as e:
        elapsed = time.time() - t0
        result = {
            "scenario_id": scenario_id,
            "seed": seed,
            "success": False,
            "error": f"{type(e).__name__}: {e}",
            "time_s": round(elapsed, 1),
            "steps": None,
            "failure_reason": "subprocess_error",
            "grasp_failure_mode": None,
            "grasp_strategy": None,
            "action_sequence": [],
            "speech": None,
            "actual_object": None,
        }
        for key in ORACLE_DIAGNOSTIC_FIELDS:
            result[key] = None
        for key in FINAL_GRASP_ORACLE_FIELDS:
            result[key] = None

    result["stdout_path"] = str(stdout_path)
    result["stderr_path"] = str(stderr_path)
    result["memory_dir"] = str(memory_dir)
    return result


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    successes = sum(1 for r in results if r.get("success"))
    errors = sum(1 for r in results if r.get("error"))
    timeouts = sum(1 for r in results if r.get("failure_reason") == "timeout")
    step_values = [r["steps"] for r in results if isinstance(r.get("steps"), int)]
    time_values = [float(r["time_s"]) for r in results if r.get("time_s") is not None]

    failure_breakdown: dict[str, int] = {}
    strategy_usage: dict[str, int] = {}
    grasp_policy_usage: dict[str, int] = {}
    candidate_source_policy_usage: dict[str, int] = {}
    candidate_source_transition_usage: dict[str, int] = {}
    object_distribution: dict[str, int] = {}
    failed_runs: list[dict[str, Any]] = []
    failure_mode_by_object: dict[str, dict[str, int]] = {}
    failure_mode_by_candidate_source: dict[str, dict[str, int]] = {}
    failure_mode_by_executed_strategy: dict[str, dict[str, int]] = {}
    failure_mode_by_profile: dict[str, dict[str, int]] = {}
    success_by_object: dict[str, dict[str, int]] = {}
    success_by_profile: dict[str, dict[str, int]] = {}

    for r in results:
        object_name = _summary_object_name(r)
        profile = _bucket_name(r.get("grasp_profile"))
        _add_success_bucket(success_by_object, object_name, bool(r.get("success")))
        _add_success_bucket(success_by_profile, profile, bool(r.get("success")))
        if not r.get("success"):
            reason = _failure_reason(r)
            failure_breakdown[str(reason)] = failure_breakdown.get(str(reason), 0) + 1
            failed_runs.append(r)
            _add_nested_count(failure_mode_by_object, object_name, str(reason))
            _add_nested_count(
                failure_mode_by_candidate_source,
                _bucket_name(r.get("grasp_strategy")),
                str(reason),
            )
            _add_nested_count(
                failure_mode_by_executed_strategy,
                _bucket_name(r.get("executed_strategy")),
                str(reason),
            )
            _add_nested_count(failure_mode_by_profile, profile, str(reason))
        strategy = r.get("grasp_strategy")
        if strategy:
            strategy_usage[str(strategy)] = strategy_usage.get(str(strategy), 0) + 1
        policy_key = _grasp_policy_usage_key(r)
        if policy_key:
            grasp_policy_usage[policy_key] = grasp_policy_usage.get(policy_key, 0) + 1
        candidate_policy_key = _candidate_source_policy_usage_key(r)
        if candidate_policy_key:
            candidate_source_policy_usage[candidate_policy_key] = (
                candidate_source_policy_usage.get(candidate_policy_key, 0) + 1
            )
        transition_key = _candidate_source_transition_key(r)
        if transition_key:
            candidate_source_transition_usage[transition_key] = (
                candidate_source_transition_usage.get(transition_key, 0) + 1
            )
        if object_name != "unknown":
            object_distribution[object_name] = object_distribution.get(object_name, 0) + 1

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
        "grasp_policy_usage": dict(sorted(grasp_policy_usage.items(), key=lambda x: (-x[1], x[0]))),
        "candidate_source_policy_usage": dict(
            sorted(candidate_source_policy_usage.items(), key=lambda x: (-x[1], x[0])),
        ),
        "candidate_source_transition_usage": dict(
            sorted(
                candidate_source_transition_usage.items(),
                key=lambda x: (-x[1], x[0]),
            ),
        ),
        "object_distribution": dict(sorted(object_distribution.items(), key=lambda x: (-x[1], x[0]))),
        "failure_mode_by_object": _sorted_nested_counts(failure_mode_by_object),
        "failure_mode_by_candidate_source": _sorted_nested_counts(failure_mode_by_candidate_source),
        "failure_mode_by_executed_strategy": _sorted_nested_counts(failure_mode_by_executed_strategy),
        "failure_mode_by_profile": _sorted_nested_counts(failure_mode_by_profile),
        "success_rate_by_object": _success_rates(success_by_object),
        "success_rate_by_profile": _success_rates(success_by_profile),
        "slowest_runs": slowest_runs,
        "failed_runs": failed_runs,
    }


def _bucket_name(value: Any) -> str:
    text = str(value).strip() if value is not None else ""
    return text if text else "unknown"


def _summary_object_name(result: dict[str, Any]) -> str:
    return _bucket_name(
        result.get("selected_target_label")
        or result.get("target_object")
        or result.get("actual_object"),
    )


def _grasp_policy_usage_key(result: dict[str, Any]) -> str | None:
    mode = _bucket_name(result.get("grasp_policy_mode"))
    if mode == "unknown":
        return None
    profile = _bucket_name(result.get("grasp_policy_profile"))
    applied = "applied" if bool(result.get("grasp_policy_applied")) else "not_applied"
    return f"{mode}:{profile}:{applied}"


def _candidate_source_policy_usage_key(result: dict[str, Any]) -> str | None:
    policy = _bucket_name(result.get("candidate_source_policy"))
    if policy == "unknown":
        return None
    applied = (
        "applied"
        if bool(result.get("candidate_source_policy_applied"))
        else "not_applied"
    )
    return f"{policy}:{applied}"


def _candidate_source_transition_key(result: dict[str, Any]) -> str | None:
    legacy = _bucket_name(result.get("legacy_first_candidate_source"))
    final = _bucket_name(result.get("final_first_candidate_source"))
    if legacy == "unknown" and final == "unknown":
        return None
    return f"{legacy}->{final}"


def _failure_reason(result: dict[str, Any]) -> str:
    reason = _bucket_name(
        result.get("grasp_failure_mode")
        or result.get("failure_reason")
        or result.get("error"),
    )
    if reason == "MAX_STEPS reached" and result.get("action_sequence"):
        return _max_steps_loop_reason(result.get("action_sequence"))
    return reason


def _max_steps_loop_reason(action_sequence: Any) -> str:
    if not isinstance(action_sequence, list):
        return "planning_loop"

    actions = [str(action) for action in action_sequence]
    if _dominant_or_terminal_action(actions, "ask_user"):
        return "clarification_loop"
    if _dominant_or_terminal_action(actions, "classify_safety"):
        return "safety_loop"
    return "planning_loop"


def _dominant_or_terminal_action(actions: list[str], action_name: str) -> bool:
    if not actions:
        return False
    count = actions.count(action_name)
    return count >= 3 and (actions[-1] == action_name or count >= len(actions) / 2)


def _add_nested_count(
    table: dict[str, dict[str, int]],
    bucket: str,
    reason: str,
) -> None:
    slot = table.setdefault(bucket, {})
    slot[reason] = slot.get(reason, 0) + 1


def _add_success_bucket(
    table: dict[str, dict[str, int]],
    bucket: str,
    success: bool,
) -> None:
    slot = table.setdefault(bucket, {"successes": 0, "total": 0})
    slot["total"] += 1
    if success:
        slot["successes"] += 1


def _sorted_nested_counts(table: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
    sorted_table: dict[str, dict[str, int]] = {}
    for bucket in sorted(table):
        sorted_table[bucket] = dict(
            sorted(table[bucket].items(), key=lambda item: (-item[1], item[0])),
        )
    return sorted_table


def _success_rates(table: dict[str, dict[str, int]]) -> dict[str, dict[str, float | int]]:
    rates: dict[str, dict[str, float | int]] = {}
    for bucket in sorted(table):
        successes = int(table[bucket]["successes"])
        total_count = int(table[bucket]["total"])
        rates[bucket] = {
            "successes": successes,
            "total": total_count,
            "success_rate": successes / total_count if total_count else 0.0,
        }
    return rates


def write_scenarios_yaml(scenarios: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.dump({"scenarios": scenarios}, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def format_summary_text(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("LONG GENERALIZATION RUN SUMMARY")
    lines.append("=" * 60)
    lines.append(f"Total scenarios : {summary['total']}")
    lines.append(f"Completed       : {summary['completed']}")
    lines.append(f"Successes       : {summary['successes']}")
    lines.append(f"Success rate    : {summary['success_rate'] * 100:.1f}%")
    lines.append(f"Errors          : {summary['errors']}")
    lines.append(f"Timeouts        : {summary['timeouts']}")
    lines.append(f"Avg steps       : {summary['avg_steps']:.1f}")
    lines.append(f"Avg time (s)    : {summary['avg_time_s']:.1f}")
    lines.append("")
    lines.append("--- Failure Breakdown ---")
    for reason, count in summary.get("failure_breakdown", {}).items():
        lines.append(f"  {reason}: {count}")
    lines.append("")
    lines.append("--- Strategy Usage ---")
    for strategy, count in summary.get("strategy_usage", {}).items():
        lines.append(f"  {strategy}: {count}")
    lines.append("")
    lines.append("--- Grasp Policy Usage ---")
    for policy, count in summary.get("grasp_policy_usage", {}).items():
        lines.append(f"  {policy}: {count}")
    lines.append("")
    lines.append("--- Candidate Source Policy Usage ---")
    for policy, count in summary.get("candidate_source_policy_usage", {}).items():
        lines.append(f"  {policy}: {count}")
    lines.append("")
    lines.append("--- Candidate Source Transition Usage ---")
    for transition, count in summary.get("candidate_source_transition_usage", {}).items():
        lines.append(f"  {transition}: {count}")
    lines.append("")
    lines.append("--- Object Distribution ---")
    for obj, count in summary.get("object_distribution", {}).items():
        lines.append(f"  {obj}: {count}")
    _append_nested_summary(
        lines,
        "Failure Mode By Object",
        "failure_mode_by_object",
        summary.get("failure_mode_by_object", {}),
    )
    _append_nested_summary(
        lines,
        "Failure Mode By Candidate Source",
        "failure_mode_by_candidate_source",
        summary.get("failure_mode_by_candidate_source", {}),
    )
    _append_nested_summary(
        lines,
        "Failure Mode By Executed Strategy",
        "failure_mode_by_executed_strategy",
        summary.get("failure_mode_by_executed_strategy", {}),
    )
    _append_success_summary(
        lines,
        "Success Rate By Object",
        "success_rate_by_object",
        summary.get("success_rate_by_object", {}),
    )
    _append_success_summary(
        lines,
        "Success Rate By Profile",
        "success_rate_by_profile",
        summary.get("success_rate_by_profile", {}),
    )
    lines.append("=" * 60)
    return "\n".join(lines)


def _append_nested_summary(
    lines: list[str],
    title: str,
    key: str,
    table: dict[str, dict[str, int]],
) -> None:
    if not table:
        return
    lines.append("")
    lines.append(f"--- {title} ---")
    lines.append(f"{key}:")
    for bucket, counts in table.items():
        lines.append(f"  {bucket}:")
        for reason, count in counts.items():
            lines.append(f"    {reason}: {count}")


def _append_success_summary(
    lines: list[str],
    title: str,
    key: str,
    table: dict[str, dict[str, Any]],
) -> None:
    if not table:
        return
    lines.append("")
    lines.append(f"--- {title} ---")
    lines.append(f"{key}:")
    for bucket, stats in table.items():
        rate = float(stats.get("success_rate", 0.0)) * 100.0
        successes = int(stats.get("successes", 0))
        total_count = int(stats.get("total", 0))
        lines.append(f"  {bucket}: {successes}/{total_count} ({rate:.1f}%)")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Long-run pure generalization evaluation for EmboSight",
    )
    parser.add_argument("--seed-start", type=int, default=0,
                        help="First seed (default: 0)")
    parser.add_argument("--count", type=int, required=True,
                        help="Number of seeds to run")
    parser.add_argument("--parallel", type=int, default=4,
                        help="Max parallel subprocesses (default: 4)")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Run identifier (default: timestamp)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint (skip completed seeds)")
    parser.add_argument("--timeout-s", type=int, default=900,
                        help="Per-seed subprocess timeout in seconds (default: 900)")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--agent-config", default="configs/agent.yaml")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    if args.run_id is None:
        args.run_id = datetime.now().strftime("gen_%Y%m%d_%H%M%S")
    return args


logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    run_dir = Path("logs/long_generalization") / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.jsonl"
    summary_json_path = run_dir / "summary.json"
    summary_txt_path = run_dir / "summary.txt"
    scenarios_yaml_path = run_dir / "scenarios.yaml"

    scenarios = generate_seed_scenarios(args.seed_start, args.count)
    write_scenarios_yaml(scenarios, scenarios_yaml_path)

    completed = load_completed_results(results_path) if args.resume else {}
    pending = [s for s in scenarios if s["id"] not in completed]

    logger.info(
        "Run %s: %d total, %d completed, %d pending, parallel=%d, timeout=%ds",
        args.run_id, len(scenarios), len(completed), len(pending),
        args.parallel, args.timeout_s,
    )

    if not pending:
        logger.info("All seeds already completed. Generating summary only.")
    else:
        manifest = {
            "run_id": args.run_id,
            "seed_start": args.seed_start,
            "count": args.count,
            "parallel": args.parallel,
            "timeout_s": args.timeout_s,
            "started_at": datetime.now().isoformat(),
        }
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

        # Thread-safe GPU pool: each subprocess acquires a GPU before
        # running and releases it after, preventing two processes from
        # sharing the same GPU when tasks complete at different speeds.
        gpu_pool: queue.Queue[int] = queue.Queue()
        for g in range(args.parallel):
            gpu_pool.put(g)

        def _run_with_gpu(scenario: dict) -> dict:
            gpu_id = gpu_pool.get()
            try:
                return run_one_seed_subprocess(
                    scenario=scenario,
                    run_dir=run_dir,
                    scenarios_config=scenarios_yaml_path,
                    gpu_id=gpu_id,
                    config=args.config,
                    agent_config=args.agent_config,
                    log_level=args.log_level,
                    timeout_s=args.timeout_s,
                )
            finally:
                gpu_pool.put(gpu_id)

        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            futures = {}
            for scenario in pending:
                future = pool.submit(_run_with_gpu, scenario)
                futures[future] = scenario

            for future in as_completed(futures):
                scenario = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    result = {
                        "scenario_id": scenario["id"],
                        "seed": scenario["seed"],
                        "success": False,
                        "error": f"future_exception: {type(e).__name__}: {e}",
                        "time_s": 0.0,
                        "steps": None,
                        "failure_reason": "executor_error",
                        "grasp_failure_mode": None,
                        "grasp_strategy": None,
                        "action_sequence": [],
                        "speech": None,
                        "actual_object": None,
                    }
                    for key in ORACLE_DIAGNOSTIC_FIELDS:
                        result[key] = None
                    for key in FINAL_GRASP_ORACLE_FIELDS:
                        result[key] = None
                append_jsonl(results_path, result)
                completed[result["scenario_id"]] = result
                done = len(completed)
                total = len(scenarios)
                status = "OK" if result.get("success") else "FAIL"
                logger.info(
                    "[%d/%d] %s %s (%.1fs)",
                    done, total, status, result["scenario_id"],
                    float(result.get("time_s") or 0),
                )

    all_results = list(completed.values())
    summary = summarize_results(all_results)
    summary_json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    text = format_summary_text(summary)
    summary_txt_path.write_text(text, encoding="utf-8")
    print(text)

    return 0 if summary["success_rate"] >= 0.5 else 1


if __name__ == "__main__":
    sys.exit(main())
