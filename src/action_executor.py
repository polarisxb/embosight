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

        # Parse approach_dir up front for both offset selection and
        # later descend stages.
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

        # Phase 4 + 9c: explicit base navigation with safe minimum offset.
        #
        # Post-navigation GPU probes showed offsets below 0.55m place PandaOmron
        # in an arm-control-degenerate state: all OSC pulses collapse to tiny
        # common drift and z motion is lost. Offset 0.55m restores meaningful
        # x/z response while 0.65m matches the reset baseline.
        target_z = float(candidate.point_3d[2])
        offset_m = 0.55
        logger.info(
            "[act] safe navigate offset=0.55m (target_z=%.3f, top_down=%s)",
            target_z,
            is_top_down,
        )

        # Best-effort: if env doesn't implement navigate_base_to (legacy mock)
        # OR if it returns False/raises, fall through to legacy move_to_pre_grasp
        # which still has drive_base=True for base approach (Phase 3 fallback).
        if hasattr(env, "navigate_base_to"):
            try:
                env.navigate_base_to(
                    target_xy=candidate.point_3d[:2],
                    offset_m=offset_m,
                )
            except Exception as e:
                logger.debug(
                    f"[act] navigate_base_to failed: {e}, falling through"
                )

        # 1. pre-grasp (diagnostic-aware with iterative nudge recovery)
        if hasattr(env, "move_to_pre_grasp_diagnostic"):
            pre_result = env.move_to_pre_grasp_diagnostic(candidate)
            if not (pre_result.ok or pre_result.handoff_ok):
                # Iterative nudge recovery: each nudge reduces lateral by
                # ~60-70 %, so 3 iterations bring 64 mm → ~5 mm (< 20 mm).
                _MAX_NUDGE_ITERS = 3
                for _nudge_iter in range(_MAX_NUDGE_ITERS):
                    if not pre_result.needs_recovery:
                        break
                    recover_ok = self._recover_pre_grasp(
                        env, candidate, pre_result,
                    )
                    if not recover_ok:
                        return self._failed_result(
                            candidate, "ik_unreachable",
                            {
                                "stage": "pre_grasp",
                                "pre_grasp_reason": "base_recovery_failed",
                                "original_reason": pre_result.reason,
                                **self._pre_grasp_details(pre_result),
                            },
                            env,
                        )
                    # After base nudge, evaluate from current EEF WITHOUT
                    # re-running move_arm_to.  Re-running would activate
                    # orientation control (ori_err ~0.9 rad after base
                    # rotation) which drifts position from ~24 mm back to
                    # ~62 mm — undoing the nudge (GPU run c2e4ab1).
                    if hasattr(env, "evaluate_pre_grasp_at_current"):
                        pre_result = env.evaluate_pre_grasp_at_current(
                            candidate,
                        )
                    else:
                        pre_result = env.move_to_pre_grasp_diagnostic(
                            candidate,
                        )
                    if pre_result.ok or pre_result.handoff_ok:
                        break
                if not (pre_result.ok or pre_result.handoff_ok):
                    return self._failed_result(
                        candidate, "ik_unreachable",
                        {
                            "stage": "pre_grasp",
                            "pre_grasp_reason": pre_result.reason,
                            **self._pre_grasp_details(pre_result),
                        },
                        env,
                    )
        elif not env.move_to_pre_grasp(candidate):
            return self._failed_result(
                candidate, "ik_unreachable",
                {"stage": "pre_grasp"}, env,
            )

        # 2. descend (策略感知的 depth margin + LLM 推理的 slip_risk 调整)
        z_target = float(candidate.point_3d[2])
        squeeze_extra_steps = 0
        if hasattr(target, "grasp_strategy") and target.grasp_strategy:
            # 优先用 LLM 推理的 depth_margin_m (随 slip_risk 动态),
            # 后退到 _STRATEGY_PARAMS 默认值.
            margin_m = float(
                getattr(target.grasp_strategy, "depth_margin_m", 0.0) or 0.0
            )
            if margin_m <= 0.0:
                from src.grasp_planner import GraspPlanner
                params = GraspPlanner._STRATEGY_PARAMS.get(
                    target.grasp_strategy.strategy, {},
                )
                margin_m = float(params.get("depth_margin", 0.015))
            squeeze_extra_steps = int(
                getattr(target.grasp_strategy, "squeeze_extra_steps", 0) or 0
            )
        else:
            margin_m = 0.015  # 默认

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
                # 优先用真实 base XY (绕开 anchor (10,10) 限制).
                # get_base_pose() 返回 mount anchor 而非真实 mobile base 位置,
                # 用它算 direction 会给出错误方向 (指向 anchor 而非物体).
                base_xy = None
                if hasattr(env, "_read_real_base_xy"):
                    base_xy = env._read_real_base_xy()
                if base_xy is None:
                    base_pos, _ = env.get_base_pose()
                    base_xy = base_pos[:2]
                base_xy = np.asarray(base_xy, dtype=np.float32)
                direction = obj_xy - base_xy
                dir_norm = float(np.linalg.norm(direction))
                step = min(0.08, dir_norm * 0.3)
                if dir_norm > 0.01:
                    nudge_xy = (direction / dir_norm * step).astype(
                        np.float32,
                    )
                    # 用 nudge_base_world_xy 平移底盘 (刚体), EEF 跟随.
                    # 之前错误地用 move_arm_to(drive_base=False) 只移动手臂
                    # → 手臂更伸展 → z 可达反而更差.
                    if hasattr(env, "nudge_base_world_xy"):
                        env.nudge_base_world_xy(nudge_xy)
                    else:
                        nudge_target = env.get_eef_pos().copy()
                        nudge_target[0] += float(nudge_xy[0])
                        nudge_target[1] += float(nudge_xy[1])
                        env.move_arm_to(
                            nudge_target, threshold_m=0.03, max_steps=300,
                        )
                # ── 底盘靠近后: 先横向对齐, 再垂直下降 ──
                # nudge_base_world_xy 移动底盘时 EEF 跟随 (关节不变),
                # 导致 EEF 横向偏移 ~4-5cm. 若直接 descend, 需同时
                # 完成横向 49mm + 纵向 25mm 的复合运动 → IK regression
                # → 手臂摆动可能撞飞柠檬. 解决: 分两步, 先横向归位.
                self._refresh_candidate_xy_from_live_object(
                    target, candidate, env, stage="post_z_stall_nudge",
                )
                eef_after_nudge = env.get_eef_pos()
                realign_target = np.array([
                    candidate.point_3d[0],
                    candidate.point_3d[1],
                    float(eef_after_nudge[2]),  # 保持当前安全高度
                ], dtype=np.float32)
                lateral_offset = float(np.linalg.norm(
                    realign_target[:2] - eef_after_nudge[:2]
                ))
                if lateral_offset > 0.005:  # > 5mm 才需要横向归位
                    logger.info(
                        "[act] post-nudge lateral re-align: "
                        "offset=%.3fm, target_xy=(%.3f, %.3f) z=%.3f",
                        lateral_offset,
                        realign_target[0], realign_target[1],
                        realign_target[2],
                    )
                    env.open_gripper()  # 确保夹爪全开, 防止归位时推物体
                    env.move_arm_to(
                        realign_target, threshold_m=0.01, max_steps=300,
                    )
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
                    self._log_pre_grasp_alignment(target, env)
                    alignment_failed = self._abort_if_pre_close_misaligned(
                        target, candidate, env, stage="pre_close_alignment",
                    )
                    if alignment_failed is not None:
                        return alignment_failed
                    grasp_ok = env.close_gripper(
                        target_label=getattr(target, "label", None),
                        squeeze_extra_steps=squeeze_extra_steps,
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
                    self._log_pre_grasp_alignment(target, env)
                    alignment_failed = self._abort_if_pre_close_misaligned(
                        target, candidate, env, stage="pre_close_alignment",
                    )
                    if alignment_failed is not None:
                        return alignment_failed
                    grasp_ok = env.close_gripper(
                        target_label=getattr(target, "label", None),
                        squeeze_extra_steps=squeeze_extra_steps,
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
                alignment_failed = self._abort_if_pre_close_misaligned(
                    target, candidate, env, stage="pre_close_alignment",
                )
                if alignment_failed is not None:
                    return alignment_failed
                grasp_ok = env.close_gripper(
                    target_label=getattr(target, "label", None),
                    squeeze_extra_steps=squeeze_extra_steps,
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
            alignment_failed = self._abort_if_pre_close_misaligned(
                target, candidate, env, stage="pre_close_alignment",
            )
            if alignment_failed is not None:
                return alignment_failed
            grasp_ok = env.close_gripper(
                target_label=getattr(target, "label", None),
                squeeze_extra_steps=squeeze_extra_steps,
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
        obj_pos_after = self._get_obj_pos(target, env)
        obj_z_after = (
            float(obj_pos_after[2])
            if obj_pos_after is not None
            else self._get_obj_z(target, env)
        )
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

        eef = np.asarray(env.get_eef_pos(), dtype=np.float32)
        diagnostic = {
            "z_target": z_target,
            "z_actual": float(z_actual),
            "final_z": float(final_z),
            "stage": "complete",
            "post_lift_eef_pos": eef[:3].tolist(),
        }
        if obj_pos_after is not None:
            diagnostic["post_lift_obj_pos"] = obj_pos_after[:3].tolist()
        if obj_z_before is not None:
            diagnostic["obj_z_before"] = obj_z_before
        if obj_z_after is not None:
            diagnostic["obj_z_after"] = obj_z_after
        attempt = GraspAttempt(
            timestamp=time.time(),
            candidate=candidate,
            failure_mode="success",
            end_effector_pose_reached=tuple(eef.tolist()) + (0.0, 0.0, 0.0),
            diagnostic=diagnostic,
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

    @staticmethod
    def _pre_grasp_details(result) -> dict:
        """Serialize key diagnostic fields from a PreGraspResult."""
        return {
            "total_error_m": float(getattr(result, "total_error_m", 0.0)),
            "lateral_error_m": float(getattr(result, "lateral_error_m", 0.0)),
            "axis_error_m": float(getattr(result, "axis_error_m", 0.0)),
            "approach_gap_m": float(getattr(result, "approach_gap_m", 0.0)),
            "lateral_limit_m": float(getattr(result, "lateral_limit_m", 0.0)),
        }

    _MAX_LATERAL_NUDGE_M = 0.10

    def _recover_pre_grasp(self, env, candidate, prior_result) -> bool:
        """Bounded recovery for pre-grasp lateral misalignment.

        Preference order:
        1. env.recover_pre_grasp(candidate, prior_result) — caller hook.
        2. env.nudge_base_world_xy(residual) — pure-translation base nudge
           by the EEF residual (pre_pos - final_eef). Bypasses both arm
           OSC saturation and base controller friction; the rigidly
           attached EEF lands at pre_pos without recomputing the approach
           direction.
        3. env.navigate_base_to(target_xy, offset_m=0.65) — last-resort
           re-navigate (legacy mocks without nudge primitive).

        Why nudge instead of navigate (GPU run 696da4e):
            navigate_base_to(virtual_target) recomputes the base→target
            direction; even a 6cm virtual-target shift rotates the
            approach angle by ~6° and teleports the base to a new yaw
            with the arm still extended → IK regression (lateral
            0.063→0.218m). drive_base=True velocity commands fall below
            OmronMobileBase friction threshold (~0.05 vs 0.25 needed) so
            the base doesn't actually move. nudge_base_world_xy bypasses
            both: pure world-frame translation, yaw preserved, EEF
            translates rigidly with the base.

        Returns False if no recovery primitive is available or the call raises.
        """
        try:
            if hasattr(env, "recover_pre_grasp"):
                ok = env.recover_pre_grasp(candidate, prior_result)
                return True if ok is None else bool(ok)

            if hasattr(env, "nudge_base_world_xy"):
                final_eef = getattr(prior_result, "final_eef", None)
                pre_pos = getattr(prior_result, "pre_pos", None)
                if final_eef is not None and pre_pos is not None:
                    pre_xy = np.asarray(pre_pos, dtype=np.float32)[:2]
                    eef_xy = np.asarray(final_eef, dtype=np.float32)[:2]
                    residual = pre_xy - eef_xy
                    r_norm = float(np.linalg.norm(residual))
                    if r_norm > 1e-3:
                        if r_norm > self._MAX_LATERAL_NUDGE_M:
                            residual = residual * (
                                self._MAX_LATERAL_NUDGE_M / r_norm
                            )
                        logger.info(
                            "[act] base nudge: residual=(%.3f,%.3f) "
                            "|Δ|=%.3fm",
                            float(residual[0]), float(residual[1]),
                            float(np.linalg.norm(residual)),
                        )
                        env.nudge_base_world_xy(residual)
                        return True

            if hasattr(env, "navigate_base_to"):
                logger.info(
                    "[act] pre-grasp recovery fallback: re-navigate offset=0.65m"
                )
                env.navigate_base_to(
                    target_xy=candidate.point_3d[:2], offset_m=0.65,
                )
                return True

            return False
        except Exception as e:
            logger.debug(f"[act] pre-grasp recovery failed: {e}")
            return False

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
    def _log_pre_grasp_alignment(target, env) -> None:
        """Diagnostic: log EEF vs object XY alignment before close_gripper.

        GPU logs will show whether the object was displaced by arm motion
        during base nudge / IK recovery.
        """
        try:
            eef = env.get_eef_pos()
            body = ActionExecutor._resolve_target_body(target, env)
            if body is None:
                return
            obj_pos = env._get_body_pos(body)
            if obj_pos is None:
                return
            lateral = float(np.linalg.norm(eef[:2] - obj_pos[:2]))
            logger.info(
                "[pre_grasp_align] eef=(%.3f,%.3f,%.3f) obj=(%.3f,%.3f,%.3f) "
                "lateral=%.4fm z_diff=%.4fm",
                eef[0], eef[1], eef[2],
                obj_pos[0], obj_pos[1], obj_pos[2],
                lateral, eef[2] - obj_pos[2],
            )
        except Exception as e:
            logger.debug(f"[pre_grasp_align] failed: {e}")

    def _abort_if_pre_close_misaligned(
        self,
        target,
        candidate,
        env,
        stage: str,
    ) -> "GraspActionResult | None":
        """Abort before closing if the object has moved away from the candidate.

        Contact-aware descent and base recovery can push round objects sideways.
        Closing on the stale candidate wastes an attempt and records the wrong
        failure as a lift slip. Use the live sim body position as the source of
        truth just before gripper closure.
        """
        try:
            body = self._resolve_target_body(target, env)
            if body is None or not hasattr(env, "_get_body_pos"):
                return None
            obj_pos = env._get_body_pos(body)
            if obj_pos is None:
                return None
            obj_pos = np.asarray(obj_pos, dtype=np.float32)
            if obj_pos.shape != (3,) or not np.all(np.isfinite(obj_pos)):
                return None
            eef = np.asarray(env.get_eef_pos(), dtype=np.float32)
            if eef.shape[0] < 3:
                return None

            finger_width = float(
                getattr(candidate, "finger_width_m", 0.04) or 0.04
            )
            lateral_limit = min(max(finger_width * 0.5, 0.015), 0.045)
            lateral = float(np.linalg.norm(eef[:2] - obj_pos[:2]))
            z_diff = float(eef[2] - obj_pos[2])
            if lateral <= lateral_limit:
                return None

            try:
                target.position_3d = obj_pos.copy()
                if getattr(target, "pose_estimate", None) is not None:
                    target.pose_estimate.position = obj_pos.copy()
                current_std = float(getattr(target, "position_std_m", 0.02))
                target.position_std_m = min(current_std, 0.02)
            except Exception:
                pass

            candidate_xy = np.asarray(
                getattr(candidate, "point_3d", np.zeros(3)),
                dtype=np.float32,
            )[:2]
            logger.warning(
                "[pre_close_align] abort: eef=(%.3f,%.3f,%.3f) "
                "obj=(%.3f,%.3f,%.3f) candidate_xy=(%.3f,%.3f) "
                "lateral=%.4fm > limit=%.4fm",
                eef[0], eef[1], eef[2],
                obj_pos[0], obj_pos[1], obj_pos[2],
                candidate_xy[0], candidate_xy[1],
                lateral, lateral_limit,
            )
            return self._failed_result(
                candidate,
                "slipped_descend",
                {
                    "stage": stage,
                    "reason": "object_displaced_before_close",
                    "target_body": body,
                    "lateral_error_m": lateral,
                    "lateral_limit_m": lateral_limit,
                    "z_diff_m": z_diff,
                    "eef_pos": eef[:3].tolist(),
                    "obj_pos": obj_pos.tolist(),
                    "candidate_xy": candidate_xy.tolist(),
                },
                env,
            )
        except Exception as e:
            logger.debug(f"[pre_close_align] skipped after error: {e}")
            return None

    def _refresh_candidate_xy_from_live_object(
        self,
        target,
        candidate,
        env,
        stage: str,
        threshold_m: float = 0.025,
    ) -> dict | None:
        """Update stale candidate XY from live simulator body position.

        Base nudges and contact-aware descents can move round objects before
        the next re-align step. Candidate z may encode a wrist/grasp height,
        so only XY is refreshed here; the hypothesis keeps the full live
        object pose for reporting and replanning.
        """
        try:
            obj_pos = self._get_obj_pos(target, env)
            if obj_pos is None:
                return None
            point = np.asarray(candidate.point_3d, dtype=np.float32).copy()
            drift = float(np.linalg.norm(obj_pos[:2] - point[:2]))
            if drift <= threshold_m:
                return {
                    "stage": stage,
                    "refreshed": False,
                    "drift_m": drift,
                    "obj_pos": obj_pos.tolist(),
                    "candidate_point": point.tolist(),
                }

            old_point = point.copy()
            point[:2] = obj_pos[:2]
            candidate.point_3d = point
            try:
                target.position_3d = obj_pos.copy()
                if getattr(target, "pose_estimate", None) is not None:
                    target.pose_estimate.position = obj_pos.copy()
                current_std = float(getattr(target, "position_std_m", 0.02))
                target.position_std_m = min(current_std, 0.02)
            except Exception:
                pass
            logger.info(
                "[live_obj_refresh] stage=%s drift=%.3fm "
                "candidate_xy=(%.3f,%.3f)->live_xy=(%.3f,%.3f)",
                stage,
                drift,
                old_point[0],
                old_point[1],
                obj_pos[0],
                obj_pos[1],
            )
            return {
                "stage": stage,
                "refreshed": True,
                "drift_m": drift,
                "old_candidate_point": old_point.tolist(),
                "new_candidate_point": point.tolist(),
                "obj_pos": obj_pos.tolist(),
            }
        except Exception as e:
            logger.debug(f"[live_obj_refresh] skipped after error: {e}")
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
    def _get_obj_pos(target, env) -> np.ndarray | None:
        """Return current target body world position from the simulator."""
        try:
            if not hasattr(env, "_get_body_pos"):
                return None
            body_name = ActionExecutor._resolve_target_body(target, env)
            if body_name is None:
                body_name = "obj_main"
            pos = env._get_body_pos(body_name)
            if pos is None:
                return None
            pos = np.asarray(pos, dtype=np.float32)
            if pos.shape[0] < 3 or not np.all(np.isfinite(pos[:3])):
                return None
            return pos[:3]
        except Exception as e:
            logger.debug(f"[_get_obj_pos] failed: {e}")
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
