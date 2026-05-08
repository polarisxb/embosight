"""QueryAwareGrounder: query-aware VLM grounding, 直出 Hypothesis。

替代老 vlm_grounding.py + scene_describer.py:
- prompt 注入 user_query (根因①)
- 输出 alternatives 概率分布, 直出 entropy
- 温度缩放纠正 VLM 概率过自信 (F3)
- 删除 Level 0-4 + alias + semantic_pairs + GT cross-check 全套规则 (根因②)

设计参考: §6.3 / §4.1
"""
from __future__ import annotations

import json
import logging
import math
import re
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.vlm_cache import VLMCache
from src.world_belief import (
    Constraint,
    Evidence,
    Hypothesis,
    WorldBelief,
)

logger = logging.getLogger(__name__)


_DEFAULT_GROUND_PROMPT = "prompts/perception/query_aware_ground.txt"


def _shannon(probs: list[float]) -> float:
    total = sum(p for p in probs if p > 0)
    if total <= 0:
        return 0.0
    h = 0.0
    for p in probs:
        if p > 0:
            q = p / total
            h -= q * math.log(q)
    return h


def _temperature_scale(probs: list[tuple[str, float]],
                       tau: float) -> list[tuple[str, float]]:
    """温度缩放 (F3): p_i' = p_i^(1/τ) / Σ p_j^(1/τ)。τ=1.0 即归一化但不重塑。"""
    if tau == 1.0:
        total = sum(p for _, p in probs) or 1.0
        return [(lbl, p / total) for lbl, p in probs]
    inv = 1.0 / tau
    raised = [(lbl, p ** inv if p > 0 else 0.0) for lbl, p in probs]
    s = sum(p for _, p in raised) or 1.0
    return [(lbl, p / s) for lbl, p in raised]


class QueryAwareGrounder:

    def __init__(
        self,
        vlm,
        llm,
        cache: VLMCache,
        ground_prompt_path: str = _DEFAULT_GROUND_PROMPT,
        zoom_prompt_path: str = "prompts/perception/zoom_disambiguate.txt",
        parallax_prompt_path: str = "prompts/perception/parallax_localize.txt",
        verify_prompt_path: str = "prompts/perception/verify_grasp.txt",
        label_temperature: float = 1.5,
    ):
        self.vlm = vlm
        self.llm = llm
        self.cache = cache
        self.label_temperature = label_temperature
        self._ground_template = self._load(ground_prompt_path)
        # zoom/parallax/verify prompts: Phase 12 用; 此处仅记路径
        self._zoom_path = zoom_prompt_path
        self._parallax_path = parallax_prompt_path
        self._verify_path = verify_prompt_path
        self._next_obj_id = 0

    @staticmethod
    def _load(path: str) -> Optional[str]:
        p = Path(path)
        return p.read_text(encoding="utf-8") if p.exists() else None

    # ──────────────────────────────────────
    # 主入口: observe
    # ──────────────────────────────────────

    def observe(self, viewpoint, env, belief: WorldBelief) -> Evidence:
        """拍 viewpoint, query-aware VLM, 返回 Evidence (含 hypotheses[])。

        失败时返回 source='vlm_failed' Evidence (Edge 9.8)。
        """
        obs = env.observe(viewpoint)
        image_path = getattr(obs, "image_path", str(obs))
        # 图像尺寸
        img_w, img_h = 256, 256
        try:
            from PIL import Image
            with Image.open(image_path) as im:
                img_w, img_h = im.size
        except Exception:
            pass

        primary = belief.decomposed.primary_target if belief.decomposed else ""
        constraints = belief.decomposed.constraints if belief.decomposed else []
        prompt = self._build_query_aware_prompt(primary, constraints, img_w, img_h)

        cached = self.cache.get(image_path, prompt)
        if cached is not None:
            raw = cached
        else:
            try:
                raw = self.vlm.describe(image_path, prompt=prompt)
                self.cache.put(image_path, prompt, raw)
            except Exception as e:
                logger.warning(f"[perception] VLM call failed: {e}")
                return Evidence(
                    source="vlm_failed", timestamp=time.time(),
                    raw_payload={
                        "error": str(e),
                        "viewpoint": getattr(viewpoint, "name", str(viewpoint)),
                    },
                )

        hyps = self._parse_to_hypotheses(raw, viewpoint, env)
        return Evidence(
            source="vlm_ground", timestamp=time.time(),
            raw_payload={
                "viewpoint": getattr(viewpoint, "name", str(viewpoint)),
                "hypotheses": [self._hyp_to_dict(h) for h in hyps],
                "image_path": image_path,
                "raw_vlm_text": raw[:1000],
            },
        )

    # ──────────────────────────────────────
    # Prompt build
    # ──────────────────────────────────────

    def _build_query_aware_prompt(
        self,
        primary_target: str,
        constraints: list[Constraint],
        img_w: int = 256,
        img_h: int = 256,
    ) -> str:
        if self._ground_template is None:
            return f"List all objects. User wants {primary_target}."
        constraints_text = "\n".join(
            f"- {c.kind}: {c.target_label or c.text or ''} ({c.reason})"
            for c in constraints
        ) or "(无)"
        return (
            self._ground_template
            .replace("{primary_target}", primary_target or "<unknown>")
            .replace("{constraints}", constraints_text)
            .replace("{img_w}", str(img_w))
            .replace("{img_h}", str(img_h))
        )

    # ──────────────────────────────────────
    # Parse: VLM JSON → Hypothesis
    # ──────────────────────────────────────

    def _parse_to_hypotheses(self, raw: str, viewpoint, env) -> list[Hypothesis]:
        """把 VLM JSON 解析成 Hypothesis 列表, 含温度缩放 + 熵计算 (F3)。"""
        data = self._extract_json(raw)
        if not data:
            return []
        objects = data.get("objects", [])
        hyps: list[Hypothesis] = []
        for obj in objects:
            try:
                bbox = tuple(int(x) for x in obj.get("bbox_2d", [0, 0, 0, 0]))
                label = str(obj.get("label", "unknown"))
                alts_raw = obj.get("alternatives", [[label, 1.0]])
                alts = [(str(lbl), float(p)) for lbl, p in alts_raw]
                # 温度缩放 + 归一化
                alts_scaled = _temperature_scale(alts, self.label_temperature)
                entropy = _shannon([p for _, p in alts_scaled])

                # position 投影 (粗略: 取 bbox 中心 + 估深度; 真实投影在 Phase 12)
                pos_3d, pos_std = self._estimate_position(bbox, viewpoint, env)

                vp_name = getattr(viewpoint, "name", str(viewpoint)) if viewpoint else "v0"
                h = Hypothesis(
                    object_id=f"obj_{self._next_obj_id}",
                    label=label,
                    label_alternatives=sorted(alts_scaled, key=lambda x: x[1], reverse=True),
                    label_entropy=entropy,
                    position_3d=pos_3d,
                    position_std_m=pos_std,
                    bbox_per_view={vp_name: bbox},
                    observed_in_views=[vp_name],
                )
                hyps.append(h)
                self._next_obj_id += 1
            except (ValueError, TypeError, KeyError) as e:
                logger.warning(f"[perception] skip malformed object: {e}; obj={obj}")
        return hyps

    @staticmethod
    def _extract_json(raw: str) -> Optional[dict]:
        # 容忍 markdown fence; 用 greedy 抓最外层 {...}
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _estimate_position(
        bbox: tuple[int, int, int, int], viewpoint, env,
    ) -> tuple[np.ndarray, float]:
        """粗略 position 估计: 单视角先用 prior。

        真实多视角投影在 Phase 12 通过 src/projection.py 实现。
        """
        return np.array([0.0, 0.0, 0.9], dtype=np.float32), 0.10

    @staticmethod
    def _hyp_to_dict(h: Hypothesis) -> dict[str, Any]:
        return {
            "object_id": h.object_id,
            "label": h.label,
            "label_alternatives": h.label_alternatives,
            "label_entropy": h.label_entropy,
            "position_3d": h.position_3d.tolist(),
            "position_std_m": h.position_std_m,
            "bbox_per_view": {k: list(v) for k, v in h.bbox_per_view.items()},
        }

    # ──────────────────────────────────────
    # re_observe / verify_grasp - Phase 12 实现
    # ──────────────────────────────────────

    def re_observe(self, target: Hypothesis, strategy: str, env, belief: WorldBelief) -> Evidence:
        raise NotImplementedError("re_observe implemented in Phase 12")

    def verify_grasp(self, target: Hypothesis, env) -> tuple[bool, float]:
        raise NotImplementedError("verify_grasp implemented in Phase 12")
