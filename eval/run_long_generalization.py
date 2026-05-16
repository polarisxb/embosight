#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
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
    lines.append("--- Object Distribution ---")
    for obj, count in summary.get("object_distribution", {}).items():
        lines.append(f"  {obj}: {count}")
    lines.append("=" * 60)
    return "\n".join(lines)


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

        with ProcessPoolExecutor(max_workers=args.parallel) as pool:
            futures = {}
            for i, scenario in enumerate(pending):
                gpu_id = i % args.parallel
                future = pool.submit(
                    run_one_seed_subprocess,
                    scenario=scenario,
                    run_dir=run_dir,
                    scenarios_config=scenarios_yaml_path,
                    gpu_id=gpu_id,
                    config=args.config,
                    agent_config=args.agent_config,
                    log_level=args.log_level,
                    timeout_s=args.timeout_s,
                )
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
