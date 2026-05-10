#!/usr/bin/env python3
"""批量运行所有 eval 场景, 汇总统计结果。

用法:
    python eval/run_batch.py                          # 跑全部场景
    python eval/run_batch.py --scenarios s1 s2 s3     # 跑指定场景
    python eval/run_batch.py --seeds 42 43 44 45 46   # 跑随机 seed
    python eval/run_batch.py --repeat 3               # 每个场景重复 3 次

结果写入 logs/batch_results/<timestamp>.json, 终端打印汇总表。
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
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


def run_one_scenario(
    scenario: dict,
    top_cfg: dict,
    agent_cfg: dict,
    run_idx: int = 0,
) -> dict[str, Any]:
    """运行单个场景, 返回结果 dict。不抛异常 — 异常记录到 error 字段。"""
    from eval.run_fixed import (
        build_agent,
        get_actual_object,
        latest_episode_path,
        reset_until_expected,
        rewrite_query_for_actual_object,
        set_global_seed,
    )
    from src.eval_oracle import summarize_episode

    sid = scenario["id"]
    t0 = time.time()
    result: dict[str, Any] = {
        "scenario_id": sid,
        "run_idx": run_idx,
        "success": False,
        "error": None,
        "steps": None,
        "time_s": None,
        "failure_reason": None,
        "grasp_failure_mode": None,
        "grasp_strategy": None,
        "action_sequence": [],
        "speech": None,
        "actual_object": None,
    }

    try:
        # 每个场景独立构建 env, 避免状态残留
        cfg = dict(top_cfg)
        if scenario.get("env_name"):
            cfg.setdefault("simulator", {})["env_name"] = scenario["env_name"]

        from scripts.run_agent import _build_env
        env = _build_env(cfg)

        actual_object, _ = reset_until_expected(
            env,
            expected_object=scenario.get("expected_object"),
            seed=scenario.get("seed"),
            max_resets=int(scenario.get("max_resets", 1)),
        )
        result["actual_object"] = actual_object

        query = rewrite_query_for_actual_object(
            str(scenario.get("query", "pick up anything")),
            str(scenario.get("user_mode", "fake_from_robocasa")),
            actual_object,
        )

        agent = build_agent(
            cfg, agent_cfg, env,
            str(scenario.get("user_mode", "fake_from_robocasa")),
            query,
        )
        ep_result = agent.run(query, env)

        result["success"] = ep_result.success
        result["steps"] = ep_result.n_steps
        result["time_s"] = round(ep_result.elapsed_seconds, 1)
        result["speech"] = ep_result.speech
        if not ep_result.success:
            result["failure_reason"] = ep_result.failure_reason

        # oracle summary
        episode_path = latest_episode_path(agent_cfg["logger"]["log_dir"])
        if episode_path:
            summary = summarize_episode(
                episode_path,
                scenario_id=sid,
                expected_object=scenario.get("expected_object"),
                actual_object=actual_object,
            )
            sd = summary.to_dict()
            result["grasp_failure_mode"] = sd.get("grasp_failure_mode")
            result["action_sequence"] = sd.get("action_sequence", [])
            result["grasp_strategy"] = sd.get("grasp_candidate_source")

        # 清理 env
        try:
            env.close()
        except Exception:
            pass

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        result["time_s"] = round(time.time() - t0, 1)
        logger.error("[batch] scenario %s failed:\n%s", sid, traceback.format_exc())

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
    parser.add_argument("--log-level", default="WARNING",
                        help="Log level (default WARNING to reduce noise)")
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

    # 追加 random seed 场景
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

    top_cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    agent_cfg = yaml.safe_load(Path(args.agent_config).read_text(encoding="utf-8")) or {}

    total_runs = len(scenarios) * args.repeat
    print(f"\n=== Batch eval: {len(scenarios)} scenarios × {args.repeat} repeats = {total_runs} runs ===\n")

    results: list[dict] = []
    for i, scenario in enumerate(scenarios):
        for run_idx in range(args.repeat):
            run_num = i * args.repeat + run_idx + 1
            sid = scenario["id"]
            print(f"[{run_num}/{total_runs}] Running {sid}" +
                  (f" (repeat {run_idx+1}/{args.repeat})" if args.repeat > 1 else "") +
                  " ...", flush=True)

            t0 = time.time()
            result = run_one_scenario(scenario, top_cfg, agent_cfg, run_idx)
            elapsed = time.time() - t0

            status = "✓ SUCCESS" if result["success"] else "✗ FAILED"
            print(f"    → {status} ({elapsed:.1f}s)\n", flush=True)
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
