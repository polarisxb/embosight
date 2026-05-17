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
    - 失败模式结构化: ik_unreachable / hit_z_floor / slipped_lift /
      gripper_empty / verify_mismatch
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

        # Phase 4: explicit base navigation (decouples nav from arm control).
        # Best-effort: if env doesn't implement navigate_base_to (legacy mock)
        # OR if it returns False/raises, fall through to legacy move_to_pre_grasp
        # which still has drive_base=True for base approach (Phase 3 fallback).
        if hasattr(env, "navigate_base_to"):
            try:
                env.navigate_base_to(
                    target_xy=candidate.point_3d[:2],
                    offset_m=0.45,
                )
            except Exception as e:
                logger.debug(
                    f"[act] navigate_base_to failed: {e}, falling through"
                )

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

        # 解析候选 approach_dir (默认 top_down)
        approach_dir = np.asarray(
            getattr(candidate, "approach_dir", [0.0, 0.0, -1.0]),
            dtype=np.float32,
        )
        ad_norm = float(np.linalg.norm(approach_dir))
        if ad_norm < 1e-6:
            approach_dir = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        else:
            approach_dir = approach_dir / ad_norm
        is_top_down = (
            approach_dir[2] < -0.9
            and abs(approach_dir[0]) < 0.1
            and abs(approach_dir[1]) < 0.1
        )

        descend_ok, z_actual = env.approach(
            candidate.point_3d,
            approach_dir=approach_dir,
            target_label=getattr(target, "label", None),
            margin_m=margin_m,
        )
        if not descend_ok:
            # z-stall recovery: 底盘前进让手臂到达更低 z
            # 适用于任何有显著垂直分量的 approach (top_down / tilted)
            z_target_eff = z_target + margin_m * float(approach_dir[2])
            gap = float(z_actual) - z_target_eff
            has_downward = float(approach_dir[2]) < -0.5
            if has_downward and gap > 0.01:
                # ── 工作空间恢复: z-stall 说明手臂在当前底盘位置到达极限 ──
                logger.info(
                    "[act] z-stall gap=%.3fm, repositioning base closer",
                    gap,
                )
                obj_xy = candidate.point_3d[:2].astype(np.float32)
                base_pos, _ = env.get_base_pose()
                direction = obj_xy - base_pos[:2]
                step = min(0.08, float(np.linalg.norm(direction)) * 0.3)
                if float(np.linalg.norm(direction)) > 0.01:
                    nudge = direction / np.linalg.norm(direction) * step
                    nudge_target = env.get_eef_pos().copy()
                    nudge_target[0] += nudge[0]
                    nudge_target[1] += nudge[1]
                    env.move_arm_to(nudge_target, threshold_m=0.03, max_steps=300)
                # 底盘靠近后重新下降 (强制垂直, 倾斜路径已证明不可达)
                _vert = np.array([0.0, 0.0, -1.0], dtype=np.float32)
                descend_ok2, z_actual = env.approach(
                    candidate.point_3d,
                    approach_dir=_vert,
                    target_label=getattr(target, "label", None),
                    margin_m=margin_m,
                )
                if descend_ok2:
                    logger.info(
                        "[act] base reposition succeeded, descend reached z=%.3f",
                        z_actual,
                    )
                    grasp_ok = env.close_gripper(
                        target_label=getattr(target, "label", None)
                    )
                    # Phase 6.2: early micro-lift slip detection
                    failed = self._verify_grasp_via_micro_lift(
                        env, target, candidate, grasp_ok,
                        stage="lift_after_reposition",
                    )
                    if failed is not None:
                        return failed
                    lift_ok, final_z = env.lift(approach_dir=_vert)
                    if not lift_ok:
                        return self._failed_result(
                            candidate,
                            "slipped_lift" if grasp_ok else "gripper_empty",
                            {"z_target": z_target, "z_actual": float(z_actual),
                             "final_z": float(final_z), "stage": "lift_after_reposition",
                             "grasp_confirmed": bool(grasp_ok)},
                            env,
                        )
                else:
                    # 底盘靠近后仍然 stall → 在当前位置尝试夹取
                    logger.info(
                        "[act] reposition didn't help (z=%.3f), trying grasp at current z",
                        z_actual,
                    )
                    grasp_ok = env.close_gripper(
                        target_label=getattr(target, "label", None)
                    )
                    # Phase 6.2: early micro-lift slip detection
                    failed = self._verify_grasp_via_micro_lift(
                        env, target, candidate, grasp_ok,
                        stage="descend_reposition_failed",
                    )
                    if failed is not None:
                        return failed
                    lift_ok, final_z = env.lift(approach_dir=_vert)
                    if not lift_ok:
                        return self._failed_result(
                            candidate,
                            "hit_z_floor" if not grasp_ok else "slipped_lift",
                            {"z_target": z_target, "z_actual": float(z_actual),
                             "stage": "descend_reposition_failed",
                             "grasp_confirmed": bool(grasp_ok)},
                            env,
                        )
            else:
                # 1) top_down gap≤1cm 已经足够近, 或
                # 2) 侧抓 approach 失败 (workspace) → 直接在当前位置夹取试试
                reason = (
                    "z-stall close_enough" if is_top_down
                    else "side approach incomplete"
                )
                logger.info(
                    "[act] %s (gap=%.3fm), grasping at current pose",
                    reason, gap,
                )
                grasp_ok = env.close_gripper(
                    target_label=getattr(target, "label", None)
                )
                # Phase 6.2: early micro-lift slip detection
                failed = self._verify_grasp_via_micro_lift(
                    env, target, candidate, grasp_ok,
                    stage="approach_incomplete",
                )
                if failed is not None:
                    return failed
                lift_ok, final_z = env.lift(approach_dir=approach_dir)
                if not lift_ok:
                    if not grasp_ok:
                        # 没夹住任何东西 → IK/接近问题
                        mode = "hit_z_floor" if is_top_down else "ik_unreachable"
                    else:
                        # 已 grasp_ok 但 lift 失败 → 夹住但拎不起来
                        mode = "slipped_lift"
                    return self._failed_result(
                        candidate, mode,
                        {"z_target": z_target, "z_actual": float(z_actual),
                         "stage": "approach_incomplete",
                         "is_top_down": bool(is_top_down),
                         "grasp_confirmed": bool(grasp_ok)},
                        env,
                    )
        else:
            # 3. close gripper (正常路径)
            grasp_ok = env.close_gripper(
                target_label=getattr(target, "label", None)
            )

            # 3.5 Phase 6.2: early micro-lift slip detection
            failed = self._verify_grasp_via_micro_lift(
                env, target, candidate, grasp_ok, stage="lift",
            )
            if failed is not None:
                return failed

            # 4. lift
            lift_ok, final_z = env.lift(approach_dir=approach_dir)
            if not lift_ok:
                return self._failed_result(
                    candidate,
                    "slipped_lift" if grasp_ok else "gripper_empty",
                    {"z_target": z_target, "z_actual": float(z_actual),
                     "final_z": float(final_z), "stage": "lift",
                     "grasp_confirmed": bool(grasp_ok)},
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
                    candidate, "slipped_lift",
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

    def _verify_grasp_via_micro_lift(
        self,
        env,
        target,
        candidate,
        grasp_ok: bool,
        stage: str,
    ) -> "GraspActionResult | None":
        """Phase 6.2: 关爪后早期 slip 检测 (设计 docs/09 §5).

        若 grasp 不稳 (obj 不跟随 2cm micro-lift), 立即 return slipped_lift,
        省下后续 ~20s 完整 lift 浪费.

        统一覆盖 act() 的 4 个 close+lift 分支:
          - lift_after_reposition (base 靠近后 grasp)
          - descend_reposition_failed (靠近无效, 强行 grasp)
          - approach_incomplete (z-stall / side 失败时 grasp)
          - lift (正常路径)

        Returns:
            failed GraspActionResult 若 micro-lift 检出 slip
            None 若 (a) grasp_ok=False (没夹住就跳过 verify)
                   (b) env 没有该 API (backward compat)
                   (c) target body 解析失败 (defer to post-lift Δz)
                   (d) micro-lift 通过 (object follows)
                   (e) micro-lift 抛异常 (保守 continue, 不 block)
        """
        if not grasp_ok:
            return None
        if not hasattr(env, "verify_grasp_by_micro_lift"):
            return None
        try:
            target_body = self._resolve_target_body(target, env)
            if not target_body:
                return None
            follows = env.verify_grasp_by_micro_lift(
                target_body, lift_m=0.02, threshold=0.5,
            )
            if not follows:
                return self._failed_result(
                    candidate, "slipped_lift",
                    {
                        "stage": "micro_lift_verify",
                        "branch": stage,
                        "reason": "object_not_following",
                        "threshold": 0.5,
                        "lift_m": 0.02,
                    },
                    env,
                )
            return None
        except Exception as e:
            logger.debug(
                f"[act] micro_lift error: {e}, continuing to full lift"
            )
            return None

    @staticmethod
    def _resolve_target_body(target, env) -> str | None:
        """获取 Hypothesis.label 对应的 sim body name.

        通过 env._get_obj_type_map() 反查 (body_name -> category) 的字典.
        与 _get_obj_z 用的解析逻辑保持一致.
        """
        try:
            label = getattr(target, "label", None)
            if not label:
                return None
            if not hasattr(env, "_get_obj_type_map"):
                return None
            type_map = env._get_obj_type_map()
            for body, cat in type_map.items():
                if cat == label:
                    return body
            return None
        except Exception:
            return None

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
