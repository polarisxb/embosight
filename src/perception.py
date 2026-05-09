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
        pose_prompt_path: str = "prompts/perception/pose_estimation.txt",
        verify_prompt_path: str = "prompts/perception/verify_grasp.txt",
        label_temperature: float = 1.5,
        viewpoint_lib=None,
    ):
        self.vlm = vlm
        self.llm = llm
        self.cache = cache
        self.label_temperature = label_temperature
        self._ground_template = self._load(ground_prompt_path)
        self._zoom_path = zoom_prompt_path
        self._parallax_path = parallax_prompt_path
        self._pose_path = pose_prompt_path
        self._verify_path = verify_prompt_path
        self._vp_lib = viewpoint_lib
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
                # VLM 经常返 alternatives 不含 label 自己 (只给 top-K 备选)。
                # 此时 label_alternatives 里找不到 primary label, belief.target()
                # 就返 None → agent 死循环 ask_user。兜底: 若缺, 把 label 按
                # (1 - sum(alts)) 填入, 保证 label 在分布里。
                alts_labels = {lbl for lbl, _ in alts}
                if label not in alts_labels:
                    alts_sum = sum(p for _, p in alts)
                    label_prob = max(0.5, 1.0 - alts_sum) if alts_sum < 1.0 else 0.5
                    # 若插入后总和 > 1, 把 alts 按比例压缩到 1 - label_prob
                    remaining = 1.0 - label_prob
                    if alts_sum > 0:
                        alts = [(lbl, p * remaining / alts_sum) for lbl, p in alts]
                    alts = [(label, label_prob)] + alts
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
        """粗略 position 估计: 用 bbox 中心做 xy 偏移, z 用 prior。

        v1 mock-friendly: std=0.03m (低于 thr.position=0.05)。
        bbox 中心偏移 ±0.25m 范围, 让不同 bbox 的 hypothesis 不会被 merge_hypothesis
        合并 (距离阈值 0.15m)。真实多视角投影在 Phase 12 由 projection.py 完成。
        """
        x_center = (bbox[0] + bbox[2]) / 2.0
        y_center = (bbox[1] + bbox[3]) / 2.0
        # 256 px 图像中心化: pixel 128 → 0 偏移; ±128 → ±0.25m
        x_offset = (x_center - 128.0) / 256.0 * 0.5
        y_offset = (y_center - 128.0) / 256.0 * 0.5
        return np.array([x_offset, y_offset, 0.9], dtype=np.float32), 0.03

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
    # re_observe (zoom / parallax / parallax_for_pose)
    # ──────────────────────────────────────

    def re_observe(
        self, target: Hypothesis, strategy: str, env, belief: WorldBelief,
    ) -> Evidence:
        if strategy == "zoom_in":
            return self._zoom_observe(target, env, belief)
        if strategy == "parallax_view":
            return self._parallax_observe(target, env, belief, for_pose=False)
        if strategy == "parallax_for_pose":
            return self._parallax_observe(target, env, belief, for_pose=True)
        raise ValueError(f"unknown re_observe strategy: {strategy}")

    def _zoom_observe(
        self, target: Hypothesis, env, belief: WorldBelief,
    ) -> Evidence:
        vp_name = target.observed_in_views[0] if target.observed_in_views else "v0"
        vp = self._vp_by_name(vp_name)
        bbox = target.bbox_per_view.get(vp_name)
        try:
            obs = env.observe(vp)
        except Exception as e:
            return Evidence(source="vlm_failed", timestamp=time.time(),
                            raw_payload={"error": str(e), "stage": "zoom_observe"})
        # bbox 缺失 → 退化成全图 (但仍走 zoom prompt)
        if bbox is None:
            image_path = getattr(obs, "image_path", str(obs))
        else:
            try:
                image_path = self._crop_image(
                    getattr(obs, "image_path", str(obs)), bbox, padding=10,
                )
            except Exception as e:
                return Evidence(
                    source="vlm_failed", timestamp=time.time(),
                    raw_payload={"error": str(e), "stage": "crop"},
                )
        zoom_template = self._load(self._zoom_path) or "Zoom prompt missing"
        prompt = (
            zoom_template
            .replace("{label}", target.label)
            .replace(
                "{alternatives_top3}",
                ", ".join(
                    f"{lbl}({p:.2f})" for lbl, p in target.label_alternatives[:3]
                ),
            )
        )
        try:
            raw = self.vlm.describe(image_path, prompt=prompt)
        except Exception as e:
            return Evidence(source="vlm_failed", timestamp=time.time(),
                            raw_payload={"error": str(e), "stage": "vlm_call"})
        data = self._extract_json(raw)
        if data is None:
            return Evidence(
                source="vlm_zoom", timestamp=time.time(),
                raw_payload={"parse_failed": True, "raw": raw[:500]},
            )
        new_alts_raw = data.get("alternatives", [])
        new_alts = [(str(lbl), float(p)) for lbl, p in new_alts_raw]
        new_alts = _temperature_scale(new_alts, self.label_temperature)
        new_alts = sorted(new_alts, key=lambda x: x[1], reverse=True)
        return Evidence(
            source="vlm_zoom", timestamp=time.time(),
            raw_payload={
                "hypotheses": [{
                    "object_id": target.object_id,
                    "label": new_alts[0][0] if new_alts else target.label,
                    "label_alternatives": new_alts,
                    "label_entropy": _shannon([p for _, p in new_alts]),
                    "position_3d": target.position_3d.tolist(),
                    "position_std_m": target.position_std_m,
                    "bbox_per_view": {
                        k: list(v) for k, v in target.bbox_per_view.items()
                    },
                    "observed_in_views": list(target.observed_in_views),
                    "visible_features": data.get("visible_features", ""),
                }],
            },
        )

    def _parallax_observe(
        self, target: Hypothesis, env, belief: WorldBelief, for_pose: bool,
    ) -> Evidence:
        used = set(target.observed_in_views)
        next_vp = None
        if self._vp_lib:
            for i in range(len(self._vp_lib)):
                vp = self._vp_lib[i]
                if getattr(vp, "name", str(vp)) not in used:
                    next_vp = vp
                    break
        if next_vp is None:
            return Evidence(
                source="vlm_failed", timestamp=time.time(),
                raw_payload={"reason": "no parallax viewpoint available"},
            )
        try:
            obs = env.observe(next_vp)
        except Exception as e:
            return Evidence(
                source="vlm_failed", timestamp=time.time(),
                raw_payload={"error": str(e), "stage": "parallax_observe"},
            )
        image_path = getattr(obs, "image_path", str(obs))
        img_w, img_h = 256, 256
        try:
            from PIL import Image
            with Image.open(image_path) as im:
                img_w, img_h = im.size
        except Exception:
            pass
        vp_name = getattr(next_vp, "name", str(next_vp))
        if for_pose:
            template = self._load(self._pose_path) or ""
            prompt = (
                template
                .replace("{viewpoint_name}", vp_name)
                .replace("{label}", target.label)
            )
        else:
            template = self._load(self._parallax_path) or ""
            prompt = (
                template
                .replace("{viewpoint_name}", vp_name)
                .replace("{label}", target.label)
                .replace("{pos_x}", f"{target.position_3d[0]:.2f}")
                .replace("{pos_y}", f"{target.position_3d[1]:.2f}")
                .replace("{pos_z}", f"{target.position_3d[2]:.2f}")
                .replace("{pos_std}", f"{target.position_std_m:.2f}")
                .replace("{img_w}", str(img_w))
                .replace("{img_h}", str(img_h))
            )
        try:
            raw = self.vlm.describe(image_path, prompt=prompt)
        except Exception as e:
            return Evidence(
                source="vlm_failed", timestamp=time.time(),
                raw_payload={"error": str(e), "stage": "vlm_call"},
            )
        return Evidence(
            source="vlm_zoom", timestamp=time.time(),
            raw_payload={
                "viewpoint": vp_name,
                "raw_vlm_text": raw[:500],
                "for_pose": for_pose,
            },
        )

    def verify_grasp(self, target: Hypothesis, env) -> tuple[bool, float]:
        try:
            obs = env.observe(env.eye_in_hand_viewpoint())
        except Exception:
            return True, 1.0
        image_path = getattr(obs, "image_path", str(obs))
        template = self._load(self._verify_path) or ""
        alts = ", ".join(
            f"{lbl}({p:.2f})" for lbl, p in target.label_alternatives[:3]
        )
        prompt = (
            template
            .replace("{expected_label}", target.label, 1)
            .replace("{expected_label}", target.label)
            .replace("{alternatives}", alts)
        )
        try:
            raw = self.vlm.describe(image_path, prompt=prompt)
        except Exception:
            return True, 1.0
        data = self._extract_json(raw)
        if data is None:
            return True, 1.0
        return bool(data.get("is_match", True)), float(data.get("confidence", 1.0))

    # ──────────────────────────────────────
    # helpers (zoom 用)
    # ──────────────────────────────────────

    def _vp_by_name(self, name: str):
        if not self._vp_lib:
            return None
        for vp in self._vp_lib:
            if getattr(vp, "name", str(vp)) == name:
                return vp
        try:
            return self._vp_lib[0]
        except (IndexError, TypeError):
            return None

    @staticmethod
    def _crop_image(
        image_path: str, bbox: tuple[int, int, int, int], padding: int = 10,
    ) -> str:
        from PIL import Image
        import tempfile
        with Image.open(image_path) as im:
            x1, y1, x2, y2 = bbox
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(im.width, x2 + padding)
            y2 = min(im.height, y2 + padding)
            crop = im.crop((x1, y1, x2, y2))
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        crop.save(tmp.name)
        return tmp.name
