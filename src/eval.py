"""零样本评估脚本

评估 EmboSight 在 Seen / Unseen 任务上的性能。

评估指标:
    - 任务分解准确率
    - 视角规划成功率
    - 视障描述完整度
    - 视障友好度评分（5 维度）
    - 端到端任务成功率

使用:
    python -m src.eval --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ============================================================
# 任务定义
# ============================================================

SEEN_QUERIES: list[str] = [
    "我的药瓶在哪里？",
    "桌上有什么？",
    "帮我拿水杯",
    "看冰箱里有什么",
    "判断台面是否安全",
]

UNSEEN_QUERIES: list[str] = [
    "把那个有点凉的东西挪到我手边",
    "我怀疑桌上那个东西过期了",
    "中间不要有挡道的",
    "找一下我刚才放下的那个东西",
    "给我描述一下我面前的厨房",
    "什么是热的",
    "哪里能放下我手里的东西",
    "有没有锋利的东西在附近",
    "早餐在哪里",
    "周围有没有移动的东西",
]


@dataclass
class EvalResult:
    """单次评估结果"""

    query: str
    is_seen: bool
    decomp_ok: bool = False
    coverage_rate: float = 0.0
    blind_friendly_score: float = 0.0
    end2end_success: bool = False
    n_viewpoints: int = 0
    speech_output: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "is_seen": self.is_seen,
            "decomp_ok": self.decomp_ok,
            "coverage_rate": self.coverage_rate,
            "blind_friendly_score": self.blind_friendly_score,
            "end2end_success": self.end2end_success,
            "n_viewpoints": self.n_viewpoints,
            "speech_output": self.speech_output,
        }


# ============================================================
# 评估主流程
# ============================================================

def evaluate_query(pipeline, env, query: str, is_seen: bool) -> EvalResult:
    """评估单个查询

    TODO: 实现具体评估逻辑
    """
    result = EvalResult(query=query, is_seen=is_seen)

    try:
        env.reset()
        out = pipeline.run(query, env)

        result.decomp_ok = len(out.get("subtasks", [])) > 0
        result.n_viewpoints = len(out.get("observations", []))
        result.speech_output = out.get("speech", "")

        # TODO: 实现以下评估指标
        # 1. coverage_rate: 子任务被回答的比例
        # 2. blind_friendly_score: 五维度加权评分
        # 3. end2end_success: 人工/自动审核
        result.coverage_rate = 0.0  # 占位
        result.blind_friendly_score = 0.0  # 占位
        result.end2end_success = False  # 占位

    except Exception as e:
        logger.error(f"评估失败 ({query}): {e}")

    return result


def aggregate_results(results: list[EvalResult]) -> dict[str, Any]:
    """汇总评估结果"""
    seen = [r for r in results if r.is_seen]
    unseen = [r for r in results if not r.is_seen]

    def avg(lst: list[float]) -> float:
        return sum(lst) / len(lst) if lst else 0.0

    summary = {
        "n_seen": len(seen),
        "n_unseen": len(unseen),
        "seen": {
            "decomp_acc": avg([float(r.decomp_ok) for r in seen]),
            "coverage_rate": avg([r.coverage_rate for r in seen]),
            "blind_friendly_score": avg([r.blind_friendly_score for r in seen]),
            "end2end_success": avg([float(r.end2end_success) for r in seen]),
            "avg_viewpoints": avg([float(r.n_viewpoints) for r in seen]),
        },
        "unseen": {
            "decomp_acc": avg([float(r.decomp_ok) for r in unseen]),
            "coverage_rate": avg([r.coverage_rate for r in unseen]),
            "blind_friendly_score": avg([r.blind_friendly_score for r in unseen]),
            "end2end_success": avg([float(r.end2end_success) for r in unseen]),
            "avg_viewpoints": avg([float(r.n_viewpoints) for r in unseen]),
        },
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="EmboSight 零样本评估")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output", type=str, default="results/eval_results.json")
    parser.add_argument("--seen-only", action="store_true")
    parser.add_argument("--unseen-only", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # 延迟导入避免 import 时副作用
    from .env_wrapper import EnvWrapper
    from .pipeline import EmboSightPipeline

    pipeline = EmboSightPipeline(args.config)
    env = EnvWrapper()

    queries: list[tuple[str, bool]] = []
    if not args.unseen_only:
        queries += [(q, True) for q in SEEN_QUERIES]
    if not args.seen_only:
        queries += [(q, False) for q in UNSEEN_QUERIES]

    results: list[EvalResult] = []
    for i, (query, is_seen) in enumerate(queries, 1):
        logger.info(f"[{i}/{len(queries)}] {'Seen' if is_seen else 'Unseen'}: {query}")
        result = evaluate_query(pipeline, env, query, is_seen)
        results.append(result)

    summary = aggregate_results(results)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "results": [r.to_dict() for r in results],
                "summary": summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    logger.info(f"评估完成，结果保存至 {output_path}")
    logger.info(f"汇总: {json.dumps(summary, ensure_ascii=False, indent=2)}")


if __name__ == "__main__":
    main()