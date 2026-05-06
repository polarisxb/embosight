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
    """完整执行: grounding → 风险路径 → 抓取 → 语义验证"""

    def __init__(
        self,
        scene_describer=None,
        no_go_radius_m: float = 0.15,
        match_threshold: float = 0.5,
    ) -> None:
        self.describer = scene_describer
        self.no_go_radius_m = no_go_radius_m
        self.match_threshold = match_threshold

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
    # 主执行
    # ------------------------------------------------------------------

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

        # 1) Grounding
        grounding = env.ground_object(plan.target_object)
        if grounding is None:
            return ActionResult(
                success=False,
                executed=False,
                message=f"无法定位目标物体: {plan.target_object}",
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

        # 5) 抓取
        grasp_ok = env.grasp_at(target_pos)

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

        # grasp 物理成功即为成功，VLM 验证是补充信息
        overall_ok = grasp_ok
        if overall_ok:
            status = "已拿到" if match else "已抓取(视觉待确认)"
        else:
            status = "抓取未到位"
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
