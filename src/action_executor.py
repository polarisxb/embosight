"""ActionExecutor v1 — Hypothesis-based 抓取执行 + 结构化 failure_mode。

设计参考: §5.3 / §6.5 / Edge 9.6
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# v1 ActionResult (基于 Hypothesis)
# ============================================================

@dataclass
class GraspActionResult:
    """ActionExecutor.act 返回。"""
    success: bool
    attempt: Any                       # GraspAttempt (避免循环 import)
    new_observations: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "attempt": {
                "failure_mode": self.attempt.failure_mode,
                "diagnostic": self.attempt.diagnostic,
                "candidate_source": self.attempt.candidate.source,
            },
        }


# ============================================================
# ActionExecutor v1
# ============================================================

class ActionExecutor:
    """v1: act(target Hypothesis) → GraspActionResult, 结构化失败模式。

    与老 execute() 区别:
    - 直接接 Hypothesis (不需要 ActionPlan/grounding 中介)
    - 失败模式结构化: ik_unreachable / hit_z_floor / slipped / verify_mismatch
    - verify_grasp 占位由 perception.verify_grasp 注入 (Phase 12)
    """

    def __init__(self, scene_describer=None) -> None:
        # scene_describer 参数保留接口兼容; v1 不使用 (verify_grasp 走 perception)
        self.describer = scene_describer

    def act(self, target, decomposed, env) -> GraspActionResult:
        """v1 主接口: 抓取 target Hypothesis, 失败结构化回写。"""
        from src.world_belief import GraspAttempt

        used = {self._cand_sig(a.candidate) for a in target.grasp_attempts}
        candidate = next(
            (c for c in target.grasp_candidates
             if self._cand_sig(c) not in used),
            None,
        )
        if candidate is None:
            return self._failed_result(
                None, "ik_unreachable",
                {"reason": "no_candidate"}, env,
            )

        # 记录物体初始 z（用于 lift 后验证物体是否跟随）
        obj_z_before = self._get_obj_z(target, env)

        # 1. pre-grasp
        if not env.move_to_pre_grasp(candidate):
            return self._failed_result(
                candidate, "ik_unreachable",
                {"stage": "pre_grasp"}, env,
            )

        # 2. descend (策略感知的 depth margin)
        z_target = float(candidate.point_3d[2])
        margin_m = 0.015  # 默认
        if hasattr(target, "grasp_strategy") and target.grasp_strategy:
            from src.grasp_planner import GraspPlanner
            params = GraspPlanner._STRATEGY_PARAMS.get(
                target.grasp_strategy.strategy, {},
            )
            margin_m = params.get("depth_margin", 0.015)
        descend_ok, z_actual = env.descend(
            candidate.point_3d, target_label=getattr(target, "label", None),
            margin_m=margin_m,
        )
        if not descend_ok:
            # ── z-stall recovery: 在当前位置尝试夹取 (不再上抬, 上抬会远离物体) ──
            z_retry = float(z_actual)
            logger.info(
                "[act] hit_z_floor at z=%.3f, retrying grasp at current z",
                z_actual,
            )
            env.close_gripper(target_label=getattr(target, "label", None))
            lift_ok, final_z = env.lift()
            if lift_ok:
                logger.info("[act] z-stall recovery succeeded, lifted to z=%.3f", final_z)
                z_actual = z_retry
            else:
                return self._failed_result(
                    candidate, "hit_z_floor",
                    {"z_target": z_target, "z_actual": float(z_actual),
                     "z_retry": z_retry, "stage": "descend_retry_failed"},
                    env,
                )
        else:
            # 3. close gripper (正常路径)
            env.close_gripper(target_label=getattr(target, "label", None))

            # 4. lift
            lift_ok, final_z = env.lift()
            if not lift_ok:
                return self._failed_result(
                    candidate, "slipped",
                    {"z_target": z_target, "z_actual": float(z_actual),
                     "final_z": float(final_z), "stage": "lift"},
                    env,
                )

        # 5. post-lift 物体跟随验证 (防止"夹住后滑落"的假阳性)
        obj_z_after = self._get_obj_z(target, env)
        if obj_z_before is not None and obj_z_after is not None:
            obj_dz = obj_z_after - obj_z_before
            if obj_dz < 0.02:
                logger.warning(
                    "[act] object NOT lifted: z_before=%.3f z_after=%.3f Δ=%.3f",
                    obj_z_before, obj_z_after, obj_dz,
                )
                return self._failed_result(
                    candidate, "slipped",
                    {"z_target": z_target, "z_actual": float(z_actual),
                     "final_z": float(final_z),
                     "obj_z_before": obj_z_before, "obj_z_after": obj_z_after,
                     "stage": "post_lift_verify"},
                    env,
                )
            logger.info(
                "[act] post-lift verified: obj Δz=%.3f (%.3f→%.3f)",
                obj_dz, obj_z_before, obj_z_after,
            )

        eef = env.get_eef_pos()
        attempt = GraspAttempt(
            timestamp=time.time(),
            candidate=candidate,
            failure_mode="success",
            end_effector_pose_reached=tuple(np.asarray(eef).tolist())
            + (0.0, 0.0, 0.0),
            diagnostic={"z_target": z_target, "z_actual": float(z_actual),
                        "final_z": float(final_z), "stage": "complete"},
        )
        return GraspActionResult(success=True, attempt=attempt)

    def verify_grasp(self, target, env) -> tuple[bool, float]:
        """post-grasp 语义验证占位; 真实实现走 perception.verify_grasp。"""
        return True, 1.0

    def release_and_retreat(self, env, retreat_height_m: float = 0.10) -> None:
        """F6: verify_mismatch / 异常退出时, 先松开夹爪再撤回。"""
        env.open_gripper()
        try:
            current = env.get_eef_pos()
            target = (
                np.asarray(current, dtype=np.float32)
                + np.array([0.0, 0.0, retreat_height_m], dtype=np.float32)
            )
            env.move_arm_to(target, threshold_m=0.02)
        except Exception as e:
            logger.warning(f"[release_and_retreat] retreat failed: {e}")

    def _failed_result(self, candidate, mode: str, diag: dict, env) -> GraspActionResult:
        from src.world_belief import GraspAttempt, GraspCandidate
        try:
            self.release_and_retreat(env)
        except Exception:
            pass
        if candidate is None:
            candidate = GraspCandidate(
                point_3d=np.zeros(3, dtype=np.float32),
                approach_dir=np.zeros(3, dtype=np.float32),
                finger_width_m=0.04, score=0.0,
                source="geometric_centroid",
            )
        attempt = GraspAttempt(
            timestamp=time.time(),
            candidate=candidate,
            failure_mode=mode,  # type: ignore[arg-type]
            end_effector_pose_reached=(0.0,) * 6,
            diagnostic=diag,
        )
        return GraspActionResult(success=False, attempt=attempt)

    @staticmethod
    def _get_obj_z(target, env) -> float | None:
        """获取目标物体当前 z 坐标 (通过 sim body position)"""
        try:
            label = getattr(target, "label", None)
            if not label:
                return None
            type_map = env._get_obj_type_map()
            body_name = next(
                (b for b, c in type_map.items() if c == label), None
            )
            if body_name is None:
                # fallback: obj_main
                body_name = "obj_main"
            pos = env._get_body_pos(body_name)
            return float(pos[2]) if pos is not None else None
        except Exception as e:
            logger.debug(f"[_get_obj_z] failed: {e}")
            return None

    @staticmethod
    def _cand_sig(c) -> tuple:
        return (
            round(float(c.point_3d[0]), 3),
            round(float(c.point_3d[1]), 3),
            round(float(c.point_3d[2]), 3),
            round(float(c.approach_dir[0]), 2),
            round(float(c.approach_dir[1]), 2),
            round(float(c.approach_dir[2]), 2),
        )
