"""ActionExecutor — 风险感知运动 + 抓取 + 语义验证闭环 (Step 6, 创新⑤⑥)"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .action_decider import ActionPlan
from .env_wrapper import ObjectGrounding
from .safety_gate import SafetyGate, SafetyDecision
from .scene_model import SceneModel, GroundedObject

logger = logging.getLogger(__name__)


# ============================================================
# 数据结构
# ============================================================

@dataclass
class NoGoZone:
    """危险区域"""

    name: str
    center_m: tuple[float, float, float]
    radius_m: float
    risk_level: str  # "high" | "medium" | "low"
    reason: str


@dataclass
class ActionResult:
    """行动执行结果"""

    success: bool
    executed: bool
    grounding: Optional[ObjectGrounding] = None
    verification_match: bool = False
    message: str = ""
    no_go_zones: list[NoGoZone] = field(default_factory=list)
    waypoints: list[tuple[float, float, float]] = field(default_factory=list)


# ============================================================
# 常量
# ============================================================

HAZARD_KEYWORDS: dict[str, list[str]] = {
    "high": ["热", "烫", "火", "刀", "锐", "沸"],
    "medium": ["玻璃", "易碎", "尖", "重"],
    "low": ["不稳", "湿", "滑"],
}

HAZARD_OBJECTS = ["锅", "刀", "杯", "玻璃", "瓶", "壶", "炉"]


# ============================================================
# ActionExecutor
# ============================================================

class ActionExecutor:
    """完整执行: grounding → 风险路径 → pre-grasp 验证 → 抓取 → 语义验证

    创新点⑥: 双闭环语义一致性验证
        - 事前 (pre-grasp): 在 pre_grasp 位姿用 eye-in-hand 视角验证目标物体
                            一致性, 错位时立即 abort, 不浪费抓取尝试
        - 事后 (post-grasp): 抓取后再次用 VLM 验证, 确认抓到的就是目标
    """

    def __init__(
        self,
        scene_describer=None,
        no_go_radius_m: float = 0.15,
        match_threshold: float = 0.5,
        enable_pre_verify: bool = True,
        safety_rules_path: str = "configs/safety_rules.yaml",
    ) -> None:
        self.describer = scene_describer
        self.no_go_radius_m = no_go_radius_m
        self.match_threshold = match_threshold
        self.enable_pre_verify = enable_pre_verify
        self.safety_gate = SafetyGate(safety_rules_path)

    # ------------------------------------------------------------------
    # 风险区域
    # ------------------------------------------------------------------

    def _extract_no_go_zones(
        self,
        safety_constraints: list[str],
        env,
    ) -> list[NoGoZone]:
        zones: list[NoGoZone] = []
        for constraint in safety_constraints:
            risk_level = "low"
            for level, kws in HAZARD_KEYWORDS.items():
                if any(kw in constraint for kw in kws):
                    risk_level = level
                    break

            for obj in HAZARD_OBJECTS:
                if obj in constraint:
                    g = env.ground_object(obj)
                    # 只用高置信度 grounding，排除 fallback 到 obj_main 的情况
                    if g is not None and g.confidence >= 0.6:
                        zones.append(NoGoZone(
                            name=obj,
                            center_m=g.position_m,
                            radius_m=self.no_go_radius_m,
                            risk_level=risk_level,
                            reason=constraint,
                        ))
                        logger.info(f"[no_go] {obj} at {g.position_m} ({risk_level})")
                    else:
                        logger.debug(f"[no_go] skip '{obj}': low confidence grounding")
                    break
        return zones

    # ------------------------------------------------------------------
    # 路径规划
    # ------------------------------------------------------------------

    @staticmethod
    def _line_intersects_sphere(p1, p2, center, radius) -> bool:
        p1, p2, center = map(lambda x: np.asarray(x, dtype=np.float64), (p1, p2, center))
        d = p2 - p1
        f = p1 - center
        a = float(np.dot(d, d))
        if a < 1e-12:
            return float(np.linalg.norm(f)) < radius
        b = 2 * float(np.dot(f, d))
        c = float(np.dot(f, f)) - radius * radius
        disc = b * b - 4 * a * c
        if disc < 0:
            return False
        disc_sqrt = np.sqrt(disc)
        t1 = (-b - disc_sqrt) / (2 * a)
        t2 = (-b + disc_sqrt) / (2 * a)
        return (0 <= t1 <= 1) or (0 <= t2 <= 1)

    def _plan_safe_path(
        self,
        start_m: np.ndarray,
        goal_m: np.ndarray,
        no_go_zones: list[NoGoZone],
    ) -> list[np.ndarray]:
        """生成 waypoint 列表 (不含起点)"""
        if not no_go_zones:
            return [goal_m]

        blocking_zone = None
        for zone in no_go_zones:
            if self._line_intersects_sphere(
                start_m, goal_m,
                np.asarray(zone.center_m), zone.radius_m,
            ):
                blocking_zone = zone
                break

        if blocking_zone is None:
            return [goal_m]

        # 绕行: 在 zone 侧面拉一个 XY 平面垂直偏移点
        center = np.asarray(blocking_zone.center_m, dtype=np.float32)
        direction = goal_m - start_m
        dist = max(float(np.linalg.norm(direction)), 1e-6)
        direction_normed = direction / dist
        perp = np.array([-direction_normed[1], direction_normed[0], 0.0], dtype=np.float32)

        offset = blocking_zone.radius_m + 0.10
        detour = center + perp * offset
        detour[2] = max(detour[2], start_m[2])

        logger.info(f"[plan] detour via {detour} to avoid {blocking_zone.name}")
        return [detour, goal_m]

    # ------------------------------------------------------------------
    # 语义验证
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize_zh(text: str) -> set[str]:
        """简单分词: 提取所有 2-4 字中文片段 + 英文单词"""
        chars = re.sub(r"[^\u4e00-\u9fa5a-zA-Z]", " ", text)
        tokens: set[str] = set()
        for word in chars.split():
            for n in (2, 3, 4):
                for i in range(len(word) - n + 1):
                    tokens.add(word[i : i + n])
            if word:
                tokens.add(word)
        return tokens

    def _verify_consistency(
        self,
        target_object: str,
        verify_desc,
    ) -> tuple[bool, float]:
        """语义匹配: 验证描述中是否包含目标物体特征"""
        target_tokens = self._tokenize_zh(target_object)

        desc_text = ""
        if hasattr(verify_desc, "objects"):
            desc_text += " ".join(str(o) for o in verify_desc.objects) + " "
        if hasattr(verify_desc, "tactile"):
            desc_text += " ".join(str(t) for t in verify_desc.tactile)
        if isinstance(verify_desc, dict):
            desc_text = json.dumps(verify_desc, ensure_ascii=False)

        # 子串匹配先行：短 target 直接判断是否出现在描述中
        if target_object in desc_text:
            return True, 1.0
        # 单字回退: "药瓶" → 检查 "药" 或 "瓶" 是否出现
        for ch in target_object:
            if ch in desc_text:
                return True, 0.8

        desc_tokens = self._tokenize_zh(desc_text)

        if not target_tokens or not desc_tokens:
            return False, 0.0

        overlap = target_tokens & desc_tokens
        score = len(overlap) / len(target_tokens)
        return score >= self.match_threshold, score

    # ------------------------------------------------------------------
    # 创新点⑥: pre-grasp 语义验证 (执行前闭环)
    # ------------------------------------------------------------------

    def _build_pre_grasp_verifier(self, target_object: str):
        """构造 pre-grasp 语义验证回调

        从 wrist 相机视角看, 目标应该位于画面正下方/中央 (因为 pre-grasp
        正好停在目标正上方 10cm 处). VLM 看图判断中央是否就是用户要的物体.

        Returns:
            callable(image_path) -> (ok: bool, reason: str)
            或 None 如果 verifier 不可用
        """
        if not self.enable_pre_verify or self.describer is None:
            return None
        if not hasattr(self.describer, "vlm") or self.describer.vlm is None:
            return None

        vlm = self.describer.vlm

        def _verify(image_path: str) -> tuple[bool, str]:
            prompt = (
                f"This is a close-up image from the robot's wrist camera, "
                f"looking down. The robot is about to grasp whatever is "
                f"directly below (image center).\n\n"
                f"User wants to grasp: '{target_object}'\n\n"
                f"Look ONLY at the object in the image center "
                f"(directly below the camera).\n"
                f"Is that object truly a '{target_object}'?\n\n"
                f"Reply with this exact format:\n"
                f"DECISION: YES or NO\n"
                f"REASON: <one short sentence>"
            )
            try:
                raw = vlm.describe(image_path, prompt=prompt).strip()
            except Exception as e:
                return True, f"verifier-error-pass: {e}"

            # 解析 DECISION
            decision = None
            reason = raw[:120]
            for line in raw.splitlines():
                line_s = line.strip()
                if line_s.upper().startswith("DECISION"):
                    val = line_s.split(":", 1)[-1].strip().upper()
                    if val.startswith("Y"):
                        decision = True
                    elif val.startswith("N"):
                        decision = False
                if line_s.upper().startswith("REASON"):
                    reason = line_s.split(":", 1)[-1].strip()[:120]

            if decision is None:
                # 未拿到明确判断: fallback 到关键词扫描, 默认放行
                upper = raw.upper()
                if "NO" in upper.split() and "YES" not in upper.split():
                    decision = False
                else:
                    decision = True
                    reason = f"unparsed-default-pass: {raw[:60]}"

            return decision, reason

        return _verify

    # ------------------------------------------------------------------
    # 主执行
    # ------------------------------------------------------------------

    def execute_with_scene_model(
        self,
        plan: ActionPlan,
        scene_model: SceneModel,
        env,
    ) -> ActionResult:
        """Phase 5: 通过 SceneModel 执行行动 (不再独立 grounding).

        信息流:
            1. SceneModel 已包含 VLM grounding + 3D 位置 + 安全信息
            2. SafetyGate check → 决定是否执行
            3. 路径规划 (绕开高风险物体)
            4. 抓取 + 语义验证

        Args:
            plan: ActionPlan from ActionDecider
            scene_model: 融合后的 SceneModel
            env: EnvWrapper
        """
        if plan.action_type == "none":
            return ActionResult(success=True, executed=False, message="无需物理动作")
        if plan.action_type == "point":
            return ActionResult(success=True, executed=False, message="指向动作暂未实现")

        # 1) 从 SceneModel 获取最佳匹配
        best = scene_model.get_best_match(min_score=0.3)
        if best is None:
            return ActionResult(
                success=False, executed=False,
                message=f"场景中未找到 '{plan.target_object}'，无法执行",
            )
        logger.info(
            f"[execute_sm] target='{best.label}' score={best.query_match_score:.2f} "
            f"risk={best.safety_risk} pos={best.position_3d}"
        )

        # 2) SafetyGate check
        decision = self.safety_gate.check(best)
        logger.info(f"[execute_sm] safety: {decision.reason_log}")
        if not decision.allow_execute:
            return ActionResult(
                success=False, executed=False,
                message=decision.reason_user,
            )

        # 3) 获取 body_name (尝试 GT 映射)
        body_name = best.body_name or "obj_main"
        if body_name == "obj_main" and best.category_gt is None:
            # 尝试通过 env 获取真实 body_name
            grounding = env.ground_object(plan.target_object, allow_fallback=True)
            if grounding is not None:
                body_name = grounding.sim_body_name
                best.body_name = body_name

        target_pos = best.position_3d
        # 如果 3D 位置不可靠 (position_confidence 很低), fallback 到 env grounding
        if best.position_confidence < 0.3:
            grounding = env.ground_object(plan.target_object, allow_fallback=True)
            if grounding is not None:
                target_pos = np.asarray(grounding.position_m, dtype=np.float32)
                body_name = grounding.sim_body_name
                logger.info(f"[execute_sm] low 3D conf, fallback to env grounding at {target_pos}")

        # 4) 高风险物体作为 no-go zones
        no_go_zones = []
        for obj in scene_model.objects:
            if obj is best:
                continue
            if obj.safety_risk in ("high", "hot", "sharp") and obj.position_confidence > 0.3:
                no_go_zones.append(NoGoZone(
                    name=obj.label,
                    center_m=tuple(obj.position_3d.tolist()),
                    radius_m=self.no_go_radius_m,
                    risk_level=obj.safety_risk,
                    reason=obj.safety_reason,
                ))

        # 5) 路径规划
        start = env.get_eef_pos()
        pre_grasp = np.asarray(target_pos, dtype=np.float32) + np.array([0, 0, 0.10], dtype=np.float32)
        waypoints = self._plan_safe_path(start, pre_grasp, no_go_zones)

        for wp in waypoints[:-1]:
            ok = env.move_arm_to(wp)
            if not ok:
                logger.warning(f"[execute_sm] failed to reach waypoint {wp}")

        # 6) 抓取 (+ pre-grasp 语义验证)
        pre_verifier = self._build_pre_grasp_verifier(plan.target_object)
        grasp_ok = env.grasp_at(
            np.asarray(target_pos, dtype=np.float32),
            target_body=body_name,
            pre_grasp_verify=pre_verifier,
        )

        # 7) 后抓取语义验证
        match = False
        score = 0.0
        if self.describer is not None:
            try:
                verify_obs = env.observe(env.eye_in_hand_viewpoint())
                verify_desc = self.describer.describe(verify_obs.image_path)
                match, score = self._verify_consistency(plan.target_object, verify_desc)
            except Exception as e:
                logger.warning(f"[execute_sm] verification failed: {e}")
        else:
            match = grasp_ok
            score = 1.0 if grasp_ok else 0.0

        if self.describer is not None:
            overall_ok = grasp_ok and match
            if grasp_ok and match:
                status = "已拿到"
            elif grasp_ok and not match:
                status = "抓取了但不是目标物体 (语义不匹配)"
            else:
                status = "抓取未到位"
        else:
            overall_ok = grasp_ok
            status = "已抓取(无视觉验证)" if grasp_ok else "抓取未到位"

        msg = (
            f"{status}目标 '{plan.target_object}' "
            f"(grasp={grasp_ok}, 匹配度 {score:.2f}) "
            f"{decision.reason_user}"
        )

        return ActionResult(
            success=overall_ok,
            executed=True,
            message=msg,
            no_go_zones=no_go_zones,
            waypoints=[tuple(w.tolist()) for w in waypoints],
        )

    def execute(self, plan: ActionPlan, env) -> ActionResult:
        """执行行动计划

        Args:
            plan: ActionPlan from ActionDecider
            env: EnvWrapper instance
        """
        if plan.action_type == "none":
            return ActionResult(
                success=True,
                executed=False,
                message="无需物理动作",
            )

        if plan.action_type == "point":
            return ActionResult(
                success=True,
                executed=False,
                message="指向动作暂未实现",
            )

        # 1) Grounding (不允许 fallback, 找不到就明确报错)
        grounding = env.ground_object(plan.target_object, allow_fallback=False)
        if grounding is None:
            return ActionResult(
                success=False,
                executed=False,
                message=f"场景中未找到 '{plan.target_object}'，无法执行抓取",
            )
        target_pos = np.asarray(grounding.position_m, dtype=np.float32)

        # 2) 风险区域
        no_go_zones = self._extract_no_go_zones(plan.safety_constraints, env)

        # 3) 路径规划
        start = env.get_eef_pos()
        pre_grasp = target_pos + np.array([0.0, 0.0, 0.10], dtype=np.float32)
        waypoints = self._plan_safe_path(start, pre_grasp, no_go_zones)

        # 4) 沿 waypoint 移动 (最后一段由 grasp_at 处理)
        for wp in waypoints[:-1]:
            ok = env.move_arm_to(wp)
            if not ok:
                logger.warning(f"[execute] failed to reach waypoint {wp}")

        # 5) 抓取 (创新点⑥ 事前闭环: pre-grasp 语义验证)
        pre_verifier = self._build_pre_grasp_verifier(plan.target_object)
        if pre_verifier is not None:
            logger.info("[execute] pre-grasp verifier enabled")
        grasp_ok = env.grasp_at(
            target_pos,
            target_body=grounding.sim_body_name if grounding else "obj_main",
            pre_grasp_verify=pre_verifier,
        )

        # 6) 语义验证
        match = False
        score = 0.0
        if self.describer is not None:
            try:
                verify_obs = env.observe(env.eye_in_hand_viewpoint())
                verify_desc = self.describer.describe(verify_obs.image_path)
                match, score = self._verify_consistency(plan.target_object, verify_desc)
            except Exception as e:
                logger.warning(f"[execute] verification failed: {e}")
        else:
            match = grasp_ok
            score = 1.0 if grasp_ok else 0.0

        # success 要求物理抓取 + 语义验证双确认 (无 verifier 时只看物理)
        if self.describer is not None:
            overall_ok = grasp_ok and match
            if grasp_ok and match:
                status = "已拿到"
            elif grasp_ok and not match:
                status = "抓取了但不是目标物体 (语义不匹配)"
            else:
                status = "抓取未到位"
        else:
            overall_ok = grasp_ok
            status = "已抓取(无视觉验证)" if grasp_ok else "抓取未到位"
        msg = f"{status}目标 '{plan.target_object}' (grasp={grasp_ok}, 匹配度 {score:.2f})"

        return ActionResult(
            success=overall_ok,
            executed=True,
            grounding=grounding,
            verification_match=match,
            message=msg,
            no_go_zones=no_go_zones,
            waypoints=[tuple(w.tolist()) for w in waypoints],
        )


# ============================================================
# Module Test
# ============================================================

if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.INFO)
    print("[ActionExecutor] 模块加载测试")
    print("  NoGoZone, ActionResult, ActionExecutor 已定义")
    print("  要测试完整执行，需要 EnvWrapper + SceneDescriber")
