"""SafetyClassifier v1: LLM 直出 safety_dist + entropy, 不依赖关键词表 / YAML。

老 SafetyGate / SafetyDecision 已删除 (Phase 15)。
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from pathlib import Path

from src.world_belief import Evidence, Hypothesis

logger = logging.getLogger(__name__)


_DEFAULT_CLASSIFY_PROMPT = "prompts/safety/classify.txt"


def _shannon_safety(probs: dict[str, float]) -> float:
    total = sum(p for p in probs.values() if p > 0)
    if total <= 0:
        return 0.0
    h = 0.0
    for p in probs.values():
        if p > 0:
            q = p / total
            h -= q * math.log(q)
    return h


class SafetyClassifier:
    """LLM 输出 safety_dist + entropy, 不依赖关键词表 / YAML 规则。"""

    def __init__(self, llm, prompt_path: str = _DEFAULT_CLASSIFY_PROMPT):
        self.llm = llm
        p = Path(prompt_path)
        self._template = p.read_text(encoding="utf-8") if p.exists() else None

    def classify(self, hyp: Hypothesis) -> Evidence:
        prompt = self._build_prompt(hyp)
        try:
            raw = self.llm.generate(prompt, system="")
        except Exception as e:
            return Evidence(
                source="llm_safety", timestamp=time.time(),
                raw_payload={"dist": {}, "entropy": 0.0,
                             "reasoning": f"llm_failed: {e}"},
            )

        data = self._extract_json(raw)
        if data is None:
            return Evidence(
                source="llm_safety", timestamp=time.time(),
                raw_payload={"dist": {}, "entropy": 0.0,
                             "reasoning": "parse_failed",
                             "raw": raw[:500]},
            )

        dist = {str(k): float(v) for k, v in data.get("dist", {}).items()}
        total = sum(dist.values())
        if total > 0:
            dist = {k: v / total for k, v in dist.items()}
        entropy = _shannon_safety(dist)
        return Evidence(
            source="llm_safety", timestamp=time.time(),
            raw_payload={
                "dist": dist,
                "entropy": entropy,
                "reasoning": str(data.get("reasoning", "")),
            },
        )

    def _build_prompt(self, hyp: Hypothesis) -> str:
        if self._template is None:
            return f"Classify safety of {hyp.label}, return JSON dist."
        alts_top3 = hyp.label_alternatives[:3]
        alts_text = ", ".join(f"{lbl}({p:.2f})" for lbl, p in alts_top3)
        features = "; ".join(
            f"{vp}: ..." for vp in hyp.observed_in_views
        ) or "(无)"
        pose_text = (
            "upright" if (hyp.pose_estimate is None or hyp.pose_estimate.upright)
            else "side"
        )
        return (
            self._template
            .replace("{label}", hyp.label)
            .replace("{alternatives_top3}", alts_text)
            .replace("{features}", features)
            .replace("{pose_summary}", pose_text)
        )

    @staticmethod
    def _extract_json(raw: str):
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            return None
