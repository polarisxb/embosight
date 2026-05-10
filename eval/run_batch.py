#!/usr/bin/env python3
"""批量运行所有 eval 场景, 汇总统计结果。

每个场景在独立子进程中运行, 保证 GPU 内存完全隔离 (避免 CUDA OOM).

用法:
    python eval/run_batch.py                          # 跑全部场景
    python eval/run_batch.py --scenarios s1 s2 s3     # 跑指定场景
    python eval/run_batch.py --seeds 42 43 44 45 46   # 跑随机 seed
    python eval/run_batch.py --repeat 3               # 每个场景重复 3 次
    python eval/run_batch.py --parallel 4             # 4 GPU 并行

结果写入 logs/batch_results/<timestamp>.json, 终端打印汇总表。
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger(__name__)


def load_all_scenarios(config_path: str | Path) -> list[dict]:
    data = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    return data.get("scenarios", []) or []


def run_one_scenario_subprocess(
    scenario_id: str,
    log_level: str = "WARNING",
    gpu_id: int = 0,
    config: str = "configs/default.yaml",
    agent_config: str = "configs/agent.yaml",
    scenarios_config: str = "configs/eval_scenarios.yaml",
) -> dict[str, Any]:
    """在子进程中运行 run_fixed.py, 解析输出获取结果。

    子进程保证 GPU 内存完全隔离, 进程结束即释放。
    """
    env_vars = os.environ.copy()
    env_vars["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [
        sys.executable, "eval/run_fixed.py",
        "--scenario", scenario_id,
        "--scenarios-config", scenarios_config,
        "--config", config,
        "--agent-config", agent_config,
        "--log-level", log_level,
        "--allow-object-mismatch",
    ]

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            env=env_vars,
            timeout=900,  # 15 分钟超时 (CLIP 多查询需要更多时间)
        )
        elapsed = time.time() - t0
        stdout = proc.stdout
        stderr = proc.stderr
    except subprocess.TimeoutExpired:
        return {
            "scenario_id": scenario_id,
            "success": False,
            "error": "subprocess timeout (900s)",
            "time_s": 900.0,
            "steps": None,
            "failure_reason": "timeout",
            "grasp_failure_mode": None,
            "grasp_strategy": None,
            "action_sequence": [],
            "speech": None,
            "actual_object": None,
        }
    except Exception as e:
        return {
            "scenario_id": scenario_id,
            "success": False,
            "error": f"{type(e).__name__}: {e}",
            "time_s": round(time.time() - t0, 1),
            "steps": None,
            "failure_reason": "subprocess_error",
            "grasp_failure_mode": None,
            "grasp_strategy": None,
            "action_sequence": [],
            "speech": None,
            "actual_object": None,
        }

    # 解析 ORACLE SUMMARY JSON
    result: dict[str, Any] = {
        "scenario_id": scenario_id,
        "success": proc.returncode == 0,
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

    # 从 stdout 解析 oracle JSON
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

        # 从 EPISODE RESULT 解析 steps/speech
        ep_start = stdout.find("EPISODE RESULT")
        if ep_start >= 0:
            for line in stdout[ep_start:ep_start+500].splitlines():
                if line.startswith("steps"):
                    try:
                        result["steps"] = int(line.split(":")[1].strip())
                    except (ValueError, IndexError):
                        pass
                elif line.startswith("speech"):
                    result["speech"] = line.split(":", 1)[1].strip() if ":" in line else None
                elif line.startswith("reason"):
                    if not result["failure_reason"]:
                        result["failure_reason"] = line.split(":", 1)[1].strip()
    except (json.JSONDecodeError, Exception) as e:
        if proc.returncode != 0 and not result["error"]:
            # 进程失败但未能解析输出
            err_tail = (stderr or stdout)[-500:] if (stderr or stdout) else "unknown"
            result["error"] = f"parse_failed (rc={proc.returncode}): {err_tail[-200:]}"

    if proc.returncode != 0 and not result["failure_reason"] and not result["error"]:
        result["failure_reason"] = "subprocess_nonzero_exit"
        err_tail = stderr[-300:] if stderr else "no stderr"
        result["error"] = err_tail

    return result


def print_summary_table(results: list[dict]) -> None:
    """打印汇总表格。"""
    total = len(results)
    successes = sum(1 for r in results if r["success"])
    errors = sum(1 for r in results if r["error"])

    print("\n" + "=" * 80)
    print("                       BATCH EVALUATION SUMMARY")
    print("=" * 80)

    # 表头
    header = f"{'Scenario':<30} {'OK':>3} {'Steps':>5} {'Time':>7} {'Strategy':<18} {'Failure'}"
    print(header)
    print("-" * 80)

    for r in results:
        ok = "✓" if r["success"] else ("✗" if not r["error"] else "ERR")
        steps = str(r["steps"]) if r["steps"] is not None else "-"
        time_s = f"{r['time_s']}s" if r["time_s"] is not None else "-"
        strategy = r.get("grasp_strategy") or "-"
        failure = r.get("failure_reason") or r.get("error") or "-"
        if len(failure) > 30:
            failure = failure[:27] + "..."
        sid = r["scenario_id"]
        if r.get("run_idx", 0) > 0:
            sid = f"{sid} (#{r['run_idx']+1})"
        print(f"{sid:<30} {ok:>3} {steps:>5} {time_s:>7} {strategy:<18} {failure}")

    print("-" * 80)
    rate = successes / total * 100 if total else 0
    avg_time = sum(r["time_s"] for r in results if r["time_s"]) / max(total, 1)
    avg_steps = sum(r["steps"] for r in results if r["steps"]) / max(
        sum(1 for r in results if r["steps"]), 1
    )
    print(f"{'TOTAL':<30} {successes}/{total} ({rate:.0f}%)  "
          f"avg_steps={avg_steps:.1f}  avg_time={avg_time:.1f}s  errors={errors}")

    # 失败原因统计
    fail_reasons: dict[str, int] = {}
    for r in results:
        if not r["success"] and not r["error"]:
            reason = r.get("grasp_failure_mode") or r.get("failure_reason") or "unknown"
            fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
    if fail_reasons:
        print(f"\nFailure breakdown: {dict(sorted(fail_reasons.items(), key=lambda x: -x[1]))}")

    # 策略统计
    strategies: dict[str, int] = {}
    for r in results:
        s = r.get("grasp_strategy")
        if s:
            strategies[s] = strategies.get(s, 0) + 1
    if strategies:
        print(f"Strategy usage:    {dict(sorted(strategies.items(), key=lambda x: -x[1]))}")

    print("=" * 80)


def _run_parallel(tasks: list[tuple[str, int]], args) -> list[dict]:
    """多 GPU 并行: 每个 GPU 跑一个场景, 用 concurrent.futures。"""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    n_gpus = args.parallel
    results: list[dict] = [None] * len(tasks)  # type: ignore

    with ProcessPoolExecutor(max_workers=n_gpus) as executor:
        future_to_idx = {}
        for idx, (scenario_id, gpu_id) in enumerate(tasks):
            future = executor.submit(
                run_one_scenario_subprocess,
                scenario_id=scenario_id,
                log_level=args.log_level,
                gpu_id=gpu_id,
                config=args.config,
                agent_config=args.agent_config,
                scenarios_config=args.scenarios_config,
            )
            future_to_idx[future] = idx

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                results[idx] = {
                    "scenario_id": tasks[idx][0],
                    "success": False,
                    "error": f"future_error: {e}",
                    "time_s": None,
                    "steps": None,
                    "failure_reason": "parallel_error",
                    "grasp_failure_mode": None,
                    "grasp_strategy": None,
                    "action_sequence": [],
                    "speech": None,
                    "actual_object": None,
                }
            r = results[idx]
            status = "✓ SUCCESS" if r["success"] else "✗ FAILED"
            t = f"{r['time_s']}s" if r["time_s"] else "?"
            print(f"  [{idx+1}/{len(tasks)}] {r['scenario_id']} → {status} ({t})", flush=True)

    return results  # type: ignore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Batch run EmboSight eval scenarios")
    parser.add_argument("--scenarios", nargs="*", default=None,
                        help="Scenario IDs to run (default: all)")
    parser.add_argument("--scenarios-config", default="configs/eval_scenarios.yaml")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--agent-config", default="configs/agent.yaml")
    parser.add_argument("--seeds", nargs="*", type=int, default=None,
                        help="Additional random seeds to test (generates discover scenarios)")
    parser.add_argument("--repeat", type=int, default=1,
                        help="Repeat each scenario N times")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of GPUs to use in parallel (default: 1 = sequential)")
    parser.add_argument("--log-level", default="WARNING",
                        help="Log level for sub-processes (default WARNING)")
    parser.add_argument("--output-dir", default="logs/batch_results")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    all_scenarios = load_all_scenarios(args.scenarios_config)
    if args.scenarios:
        scenarios = [s for s in all_scenarios if s["id"] in args.scenarios]
        if not scenarios:
            print(f"No matching scenarios. Available: {[s['id'] for s in all_scenarios]}")
            return 1
    else:
        scenarios = all_scenarios

    # 追加 random seed 场景 (写入临时 yaml 供子进程读取)
    if args.seeds:
        for seed in args.seeds:
            scenarios.append({
                "id": f"random_seed_{seed}",
                "env_name": "PickPlaceCounterToCabinet",
                "seed": seed,
                "query": "pick up anything",
                "expected_object": None,
                "user_mode": "fake_from_robocasa",
                "max_resets": 1,
            })
        # 写入扩展后的 scenarios 到临时文件供子进程使用
        tmp_cfg = Path(args.output_dir) / "_tmp_scenarios.yaml"
        tmp_cfg.parent.mkdir(parents=True, exist_ok=True)
        tmp_cfg.write_text(
            yaml.dump({"scenarios": scenarios}, allow_unicode=True),
            encoding="utf-8",
        )
        args.scenarios_config = str(tmp_cfg)

    # 展开 repeat
    scenario_ids: list[str] = []
    for s in scenarios:
        for _ in range(args.repeat):
            scenario_ids.append(s["id"])

    total_runs = len(scenario_ids)
    n_gpus = min(args.parallel, total_runs)
    mode = f"{n_gpus} GPU parallel" if n_gpus > 1 else "sequential (1 GPU)"
    print(f"\n=== Batch eval: {len(scenarios)} scenarios × {args.repeat} repeats "
          f"= {total_runs} runs [{mode}] ===\n", flush=True)

    if n_gpus > 1:
        # 并行模式: 分配 GPU
        tasks = [(sid, i % n_gpus) for i, sid in enumerate(scenario_ids)]
        results = _run_parallel(tasks, args)
    else:
        # 串行模式: 每个场景独立子进程
        results = []
        for i, sid in enumerate(scenario_ids):
            print(f"[{i+1}/{total_runs}] Running {sid} ...", flush=True)
            result = run_one_scenario_subprocess(
                scenario_id=sid,
                log_level=args.log_level,
                gpu_id=0,
                config=args.config,
                agent_config=args.agent_config,
                scenarios_config=args.scenarios_config,
            )
            status = "✓ SUCCESS" if result["success"] else "✗ FAILED"
            t = f"{result['time_s']}s" if result["time_s"] else "?"
            print(f"    → {status} ({t})\n", flush=True)
            results.append(result)

    # 保存 JSON
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"batch_{ts}.json"
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print_summary_table(results)
    print(f"\nDetailed results: {output_path}")

    return 0 if all(r["success"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
