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

from src.world_belief import GraspAttempt, GraspCandidate, GraspStrategy, Hypothesis

logger = logging.getLogger(__name__)


_DEFAULT_PROMPT = "prompts/grasp/suggest_top_grasp.txt"


_DEFAULT_STRATEGY_PROMPT = "prompts/grasp/select_strategy.txt"


class GraspPlanner:

    def __init__(self, vlm, env, llm=None,
                 prompt_path: str = _DEFAULT_PROMPT,
                 strategy_prompt_path: str = _DEFAULT_STRATEGY_PROMPT):
        self.vlm = vlm
        self.env = env
        self.llm = llm
        p = Path(prompt_path)
        self._template = p.read_text(encoding="utf-8") if p.exists() else None
        sp = Path(strategy_prompt_path)
        self._strategy_template = sp.read_text(encoding="utf-8") if sp.exists() else None

    # ──────────────────────────────────────
    # plan / regenerate_after_failure
    # ──────────────────────────────────────

    # ──────────────────────────────────────
    # LLM 策略选择
    # ──────────────────────────────────────

    def select_strategy(self, hyp: Hypothesis, memory_advice: str = "") -> GraspStrategy:
        """让 LLM 根据物体外观 + 安全属性选择抓取策略。"""
        if not self.llm or not self._strategy_template:
            return GraspStrategy(strategy="top_down", reasoning="no LLM",
                                 speech=f"我来拿{hyp.label}")

        safety_text = ", ".join(
            f"{k}={v:.2f}" for k, v in sorted(
                hyp.safety_dist.items(), key=lambda x: x[1], reverse=True,
            )
        ) if hyp.safety_dist else "unknown"
        pose_text = (
            "upright" if (hyp.pose_estimate is None or hyp.pose_estimate.upright)
            else "side/tilted"
        )
        prompt = (
            self._strategy_template
            .replace("{label}", hyp.label)
            .replace("{visible_features}", hyp.visible_features or "(no description)")
            .replace("{safety_dist}", safety_text)
            .replace("{pose}", pose_text)
            .replace("{past_experience}", memory_advice or "No prior experience with this object.")
        )

        try:
            raw = self.llm.generate(prompt, system="")
            data = self._extract_json(raw)
            if data and "strategy" in data:
                strat = str(data["strategy"]).lower()
                valid = {"top_down", "gentle_side", "handle_grasp", "scoop_under", "refuse"}
                if strat not in valid:
                    strat = "top_down"
                return GraspStrategy(
                    strategy=strat,
                    approach_axis=str(data.get("approach_axis", "z")),
                    reasoning=str(data.get("reasoning", "")),
                    speech=str(data.get("speech", f"我来拿{hyp.label}")),
                )
        except Exception as e:
            logger.warning("[grasp_planner] strategy selection failed: %s", e)

        return GraspStrategy(strategy="top_down", reasoning="fallback",
                             speech=f"我来拿{hyp.label}")

    # ──────────────────────────────────────
    # 策略驱动的候选生成
    # ──────────────────────────────────────

    _STRATEGY_PARAMS: dict[str, dict] = {
        "top_down":      {"approach_dir": [0, 0, -1.0], "finger_width": 0.04, "score": 0.75, "depth_margin": 0.015},
        "gentle_side":   {"approach_dir": [1, 0,  0.0], "finger_width": 0.06, "score": 0.70, "depth_margin": 0.010},
        "handle_grasp":  {"approach_dir": [1, 0,  0.0], "finger_width": 0.03, "score": 0.70, "depth_margin": 0.015},
        "scoop_under":   {"approach_dir": [0, 0, -0.3], "finger_width": 0.08, "score": 0.65, "depth_margin": 0.020},
    }

    def plan(self, hyp: Hypothesis, env=None) -> list[GraspCandidate]:
        env = env or self.env
        cands: list[GraspCandidate] = []

        # 如果有 LLM 选定的策略, 优先用策略生成候选
        strategy = hyp.grasp_strategy
        if strategy and strategy.strategy != "refuse":
            params = self._STRATEGY_PARAMS.get(
                strategy.strategy, self._STRATEGY_PARAMS["top_down"],
            )
            cands.append(GraspCandidate(
                point_3d=hyp.position_3d.copy(),
                approach_dir=np.array(params["approach_dir"]),
                finger_width_m=params["finger_width"],
                score=params["score"],
                source=f"strategy_{strategy.strategy}",
            ))
            logger.info(
                "[grasp_planner] strategy=%s → approach=%s width=%.2fm",
                strategy.strategy, params["approach_dir"], params["finger_width"],
            )

        # 兜底: geometric_centroid (总是加, 分数低于策略候选)
        cands.append(GraspCandidate(
            point_3d=hyp.position_3d.copy(),
            approach_dir=np.array([0, 0, -1.0]),
            finger_width_m=0.04,
            score=0.50,
            source="geometric_centroid",
        ))

        # axis_aligned_side (pose 横放时)
        if hyp.pose_estimate is not None and not hyp.pose_estimate.upright:
            cands.append(GraspCandidate(
                point_3d=hyp.position_3d.copy(),
                approach_dir=np.array([1.0, 0, 0]),
                finger_width_m=0.04,
                score=0.45,
                source="axis_aligned_side",
            ))

        # vlm_top_grasp (eye_in_hand)
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

    @staticmethod
    def _extract_json(raw: str) -> Optional[dict]:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            return None
