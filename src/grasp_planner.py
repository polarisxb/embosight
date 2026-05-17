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

_FAIL_BAN_THRESHOLD = 3  # ≥ N 次失败的策略从 LLM 可选列表中移除

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

    @staticmethod
    def _parse_banned_strategies(memory_advice: str) -> set[str]:
        """从 memory_advice 中提取失败次数 ≥ _FAIL_BAN_THRESHOLD 的策略名。

        识别格式: "avoid <strategy> (<reason> x<N>)"
        """
        banned: set[str] = set()
        for m in re.finditer(r"avoid\s+(\w+)\s+\([^)]*x(\d+)", memory_advice):
            strat, count = m.group(1), int(m.group(2))
            if count >= _FAIL_BAN_THRESHOLD:
                banned.add(strat)
        return banned

    def select_strategy(self, hyp: Hypothesis, memory_advice: str = "") -> GraspStrategy:
        """让 LLM 根据物体外观 + 安全属性选择抓取策略。

        失败 ≥ _FAIL_BAN_THRESHOLD 次的策略会从可选列表中物理删除。
        """
        all_strategies = {"tilted_grasp", "top_down", "gentle_side", "handle_grasp", "scoop_under", "refuse"}
        banned = self._parse_banned_strategies(memory_advice)
        available = all_strategies - banned
        if banned:
            logger.info("[grasp_planner] banned strategies (fail≥%d): %s",
                        _FAIL_BAN_THRESHOLD, banned)

        if not self.llm or not self._strategy_template:
            # 无 LLM 时, 从未禁止的策略中按优先级选择
            for fallback in ["top_down", "tilted_grasp", "handle_grasp", "gentle_side"]:
                if fallback in available:
                    return GraspStrategy(strategy=fallback, reasoning="no LLM",
                                         speech=f"我来拿{hyp.label}")
            return GraspStrategy(strategy="top_down", reasoning="no LLM (all banned)",
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

        # 从 prompt 模板中物理删除已禁止的策略行
        template = self._strategy_template
        if banned:
            lines = template.split("\n")
            lines = [ln for ln in lines
                     if not any(f"- {b}:" in ln for b in banned)]
            template = "\n".join(lines)

        prompt = (
            template
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
                # 后置检查: LLM 仍选了被禁/无效策略 → 覆盖
                if strat not in available or strat not in all_strategies:
                    old = strat
                    for fb in ["top_down", "tilted_grasp", "handle_grasp", "gentle_side"]:
                        if fb in available:
                            strat = fb
                            break
                    logger.warning(
                        "[grasp_planner] LLM chose banned/invalid '%s', "
                        "overriding to '%s'", old, strat,
                    )
                return GraspStrategy(
                    strategy=strat,
                    approach_axis=str(data.get("approach_axis", "z")),
                    reasoning=str(data.get("reasoning", "")),
                    speech=str(data.get("speech", f"我来拿{hyp.label}")),
                )
        except Exception as e:
            logger.warning("[grasp_planner] strategy selection failed: %s", e)

        # fallback: 从未禁止的策略中选
        for fb in ["top_down", "tilted_grasp", "handle_grasp", "gentle_side"]:
            if fb in available:
                return GraspStrategy(strategy=fb, reasoning="fallback",
                                     speech=f"我来拿{hyp.label}")
        return GraspStrategy(strategy="top_down", reasoning="fallback (all banned)",
                             speech=f"我来拿{hyp.label}")

    # ──────────────────────────────────────
    # 策略驱动的候选生成
    # ──────────────────────────────────────

    _STRATEGY_PARAMS: dict[str, dict] = {
        "tilted_grasp":  {"approach_dir": "tilted", "finger_width": 0.04, "score": 0.80, "depth_margin": 0.015},
        "top_down":      {"approach_dir": [0, 0, -1.0], "finger_width": 0.04, "score": 0.75, "depth_margin": 0.015},
        "gentle_side":   {"approach_dir": [1, 0,  0.0], "finger_width": 0.06, "score": 0.70, "depth_margin": 0.010},
        "handle_grasp":  {"approach_dir": [1, 0,  0.0], "finger_width": 0.03, "score": 0.70, "depth_margin": 0.015},
        "scoop_under":   {"approach_dir": [0, 0, -0.3], "finger_width": 0.08, "score": 0.65, "depth_margin": 0.020},
    }

    _TILT_ANGLE_DEG = 35  # 倾斜俯冲角 (从垂直方向计)

    _SIDE_TILT_Z = -0.47  # 侧抓下倾分量 (归一化后≈25°)

    def _tilted_approach_dir(self, obj_pos: np.ndarray, env) -> np.ndarray:
        """计算 35° 斜俯冲方向: 主要从上方, 略微侧移。

        与 _side_approach_dir 的区别:
        - side: 主要水平 (从侧面), 略微下倾
        - tilted: 主要垂直 (从上方), 略微侧移

        35° from vertical → sin(35°)≈0.574 水平, cos(35°)≈0.819 垂直
        工作空间充裕 (仍在物体正上方附近), 手指与竖直物体成 45° → 有摩擦面。

        Returns:
            3D unit vector in world frame.
        """
        sin_a = np.sin(np.deg2rad(self._TILT_ANGLE_DEG))
        cos_a = np.cos(np.deg2rad(self._TILT_ANGLE_DEG))
        try:
            base_pos = env.get_base_pose()[0]
        except Exception:
            base_pos = env.get_eef_pos()
        delta_xy = obj_pos[:2] - base_pos[:2]
        d = float(np.linalg.norm(delta_xy))
        if d < 0.01:
            raw = np.array([sin_a, 0.0, -cos_a], dtype=np.float32)
        else:
            raw = np.array(
                [delta_xy[0] / d * sin_a, delta_xy[1] / d * sin_a, -cos_a],
                dtype=np.float32,
            )
        return raw / np.linalg.norm(raw)

    def _side_approach_dir(self, obj_pos: np.ndarray, env) -> np.ndarray:
        """计算从机器人指向物体的侧报接近方向, 带 ~25° 下倾。

        Returns:
            3D unit vector in world frame.
            Fallback [1, 0, -0.47] (normalized) if positions are too close.
        """
        try:
            base_pos = env.get_base_pose()[0]  # (3,)
        except Exception:
            base_pos = env.get_eef_pos()  # fallback
        delta_xy = obj_pos[:2] - base_pos[:2]
        d = float(np.linalg.norm(delta_xy))
        if d < 0.01:
            raw = np.array([1.0, 0.0, self._SIDE_TILT_Z], dtype=np.float32)
        else:
            raw = np.array([delta_xy[0] / d, delta_xy[1] / d, self._SIDE_TILT_Z],
                           dtype=np.float32)
        return raw / np.linalg.norm(raw)

    def plan(self, hyp: Hypothesis, env=None) -> list[GraspCandidate]:
        env = env or self.env
        cands: list[GraspCandidate] = []

        # 如果有 LLM 选定的策略, 优先用策略生成候选
        strategy = hyp.grasp_strategy
        if strategy and strategy.strategy != "refuse":
            params = self._STRATEGY_PARAMS.get(
                strategy.strategy, self._STRATEGY_PARAMS["top_down"],
            )
            # 接近方向计算
            raw_ad = params["approach_dir"]
            if raw_ad == "tilted":
                ad = self._tilted_approach_dir(hyp.position_3d, env)
            elif isinstance(raw_ad, (list, np.ndarray)):
                raw_ad = np.array(raw_ad, dtype=np.float32)
                has_xy = max(abs(raw_ad[0]), abs(raw_ad[1])) > 0.1
                if has_xy:  # 有水平分量 → 动态计算侧抓方向
                    ad = self._side_approach_dir(hyp.position_3d, env)
                else:
                    ad = raw_ad
            else:
                ad = self._tilted_approach_dir(hyp.position_3d, env)

            # 策略抓点偏移: 细长直立物体的质心处最窄, 需偏移
            grasp_pt = hyp.position_3d.copy()
            is_upright = (hyp.pose_estimate is None or hyp.pose_estimate.upright)
            if is_upright and strategy.strategy == "handle_grasp":
                grasp_pt[2] += 0.03   # 上移 3cm → 手柄中段
            elif is_upright and strategy.strategy == "top_down":
                grasp_pt[2] -= 0.015  # 下移 1.5cm → 碗端/宽端
            elif is_upright and strategy.strategy == "tilted_grasp":
                grasp_pt[2] -= 0.02   # 下移 2cm → 手柄上段 (避开碗沿)

            cands.append(GraspCandidate(
                point_3d=grasp_pt,
                approach_dir=ad,
                finger_width_m=params["finger_width"],
                score=params["score"],
                source=f"strategy_{strategy.strategy}",
            ))
            logger.info(
                "[grasp_planner] strategy=%s → approach=%s width=%.2fm",
                strategy.strategy, ad, params["finger_width"],
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
            ad_side = self._side_approach_dir(hyp.position_3d, env)
            cands.append(GraspCandidate(
                point_3d=hyp.position_3d.copy(),
                approach_dir=ad_side,
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
                ad_side = self._side_approach_dir(hyp.position_3d, self.env)
                new_cands.append(GraspCandidate(
                    point_3d=hyp.position_3d.copy(),
                    approach_dir=ad_side,
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
