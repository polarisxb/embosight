"""GraspPlanner: 生成 GraspCandidate, 含三种策略 + 可达性过滤。

策略:
- geometric_centroid: 物体中心 + 顶抓 approach=-z
- axis_aligned_side : pose.upright=False 时, 沿物体短轴侧抓
- vlm_top_grasp     : VLM 看 eye_in_hand 图建议 grip 点

设计参考: §6.5
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np

from src.world_belief import GraspAttempt, GraspCandidate, Hypothesis

logger = logging.getLogger(__name__)


_DEFAULT_PROMPT = "prompts/grasp/suggest_top_grasp.txt"


class GraspPlanner:

    def __init__(self, vlm, env, prompt_path: str = _DEFAULT_PROMPT):
        self.vlm = vlm
        self.env = env
        p = Path(prompt_path)
        self._template = p.read_text(encoding="utf-8") if p.exists() else None

    # ──────────────────────────────────────
    # plan / regenerate_after_failure
    # ──────────────────────────────────────

    def plan(self, hyp: Hypothesis, env=None) -> list[GraspCandidate]:
        env = env or self.env
        cands: list[GraspCandidate] = []

        # 1. geometric_centroid
        cands.append(GraspCandidate(
            point_3d=hyp.position_3d.copy(),
            approach_dir=np.array([0, 0, -1.0]),
            finger_width_m=0.04,
            score=0.7,
            source="geometric_centroid",
        ))

        # 2. axis_aligned_side (pose 横放时)
        if hyp.pose_estimate is not None and not hyp.pose_estimate.upright:
            cands.append(GraspCandidate(
                point_3d=hyp.position_3d.copy(),
                approach_dir=np.array([1.0, 0, 0]),
                finger_width_m=0.04,
                score=0.65,
                source="axis_aligned_side",
            ))

        # 3. vlm_top_grasp (eye_in_hand)
        try:
            v = self._vlm_grasp(hyp, env)
            if v is not None:
                cands.append(v)
        except Exception as e:
            logger.debug(f"[grasp_planner] vlm_top_grasp skipped: {e}")

        # 可达性过滤
        cands = [
            c for c in cands
            if env.is_reachable(c.point_3d, c.approach_dir)
        ]
        cands.sort(key=lambda c: c.score, reverse=True)
        return cands

    def regenerate_after_failure(
        self, hyp: Hypothesis, last_attempt: GraspAttempt,
    ) -> list[GraspCandidate]:
        new_cands = self.plan(hyp)
        # hit_z_floor 时, 强烈倾向侧抓
        if last_attempt.failure_mode == "hit_z_floor":
            new_cands = [c for c in new_cands
                         if c.source != "geometric_centroid"]
            if not any(c.source == "axis_aligned_side" for c in new_cands):
                new_cands.append(GraspCandidate(
                    point_3d=hyp.position_3d.copy(),
                    approach_dir=np.array([1.0, 0, 0]),
                    finger_width_m=0.04, score=0.55,
                    source="axis_aligned_side",
                ))
        # 排除已用过的 (相同 source + 几何位姿)
        used = {self._cand_sig(a.candidate) for a in hyp.grasp_attempts}
        return [c for c in new_cands if self._cand_sig(c) not in used]

    @staticmethod
    def _cand_sig(c: GraspCandidate) -> tuple:
        return (
            c.source,
            round(float(c.point_3d[0]), 3),
            round(float(c.point_3d[1]), 3),
            round(float(c.point_3d[2]), 3),
            round(float(c.approach_dir[0]), 2),
            round(float(c.approach_dir[1]), 2),
            round(float(c.approach_dir[2]), 2),
        )

    # ──────────────────────────────────────
    # vlm_top_grasp
    # ──────────────────────────────────────

    def _vlm_grasp(self, hyp: Hypothesis, env) -> Optional[GraspCandidate]:
        if self._template is None:
            return None
        try:
            obs = env.observe(env.eye_in_hand_viewpoint())
        except Exception:
            return None
        pose_text = "upright" if (hyp.pose_estimate is None or hyp.pose_estimate.upright) else "side"
        prompt = (
            self._template
            .replace("{label}", hyp.label)
            .replace("{pose}", pose_text)
        )
        raw = self.vlm.describe(getattr(obs, "image_path", "/dev/null"), prompt=prompt)
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return None
        try:
            json.loads(m.group())
        except json.JSONDecodeError:
            return None
        # VLM 给的 grip_norm [x, y] 暂时直接用作 score 加权; 真实 3D 投影 Phase 12
        return GraspCandidate(
            point_3d=hyp.position_3d.copy(),
            approach_dir=np.array([0, 0, -1.0]),
            finger_width_m=0.04, score=0.75,
            source="vlm_top_grasp",
        )
