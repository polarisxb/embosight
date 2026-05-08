"""EmboSight Agent v1 - 信念状态数据结构。

纯数据层: 无 IO, 无 LLM/VLM 调用。所有状态修改通过显式方法。

设计参考: docs/superpowers/specs/2026-05-08-emboSight-belief-driven-agent-design.md §4
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# 4.1 Pose / GraspCandidate / GraspAttempt
# ============================================================

@dataclass
class Pose:
    """物体姿态估计 (6D)。"""
    position: np.ndarray              # (3,) world coord
    rotation_quat: np.ndarray         # (4,) (x, y, z, w)
    upright: bool = True              # 横/竖 (粗略, 由 VLM 判)


@dataclass
class GraspCandidate:
    """单个候选抓点。"""
    point_3d: np.ndarray              # 抓点世界坐标
    approach_dir: np.ndarray          # 接近方向 (单位向量, 指向物体)
    finger_width_m: float             # 张开宽度估计
    score: float                      # 0-1: 综合几何 + 姿态 + 可达性
    source: Literal[
        "vlm_top_grasp", "geometric_centroid",
        "axis_aligned_side", "user_corrected",
    ] = "geometric_centroid"


@dataclass
class GraspAttempt:
    """已经试过的抓取记录。"""
    timestamp: float
    candidate: GraspCandidate
    failure_mode: Literal[
        "success",
        "hit_z_floor",                # OSC 卡 z, 没下到目标深度
        "ik_unreachable",             # 工作空间外
        "collision",                  # 撞到其他物体
        "slipped",                    # 关爪后物体掉了
        "verify_mismatch",            # post-grasp VLM 说抓错了
        "timeout",                    # OSC 步数耗尽
    ]
    end_effector_pose_reached: tuple[float, ...]  # (x, y, z, roll, pitch, yaw)
    diagnostic: dict[str, Any] = field(default_factory=dict)


# ============================================================
# 4.1 Hypothesis (4 轴结构化不确定性)
# ============================================================

@dataclass
class Hypothesis:
    """场景中一个候选物体, 带 4 轴结构化不确定性。"""
    object_id: str
    
    # ──── 1. 类别轴 ────
    label: str
    label_alternatives: list[tuple[str, float]]   # [("label", prob), ...]
    label_entropy: float                          # H(alternatives), 越大越不确定
    
    # ──── 2. 位置轴 ────
    position_3d: np.ndarray                       # (3,) world coord
    position_std_m: float                         # 多视角投影 std (m)
    bbox_per_view: dict[str, tuple[int, int, int, int]] = field(default_factory=dict)
    
    # ──── 3. 风险轴 ────
    # 开放 key dict, 默认 v1 类: safe / fragile / sharp / hot / chemical
    # 后续可加 weight / wet 等
    safety_dist: dict[str, float] = field(default_factory=dict)
    safety_entropy: float = 1.0                   # 初始最大熵 (未分类)
    
    # ──── 4. 抓取轴 ────
    pose_estimate: Optional[Pose] = None
    pose_uncertainty: float = 1.0
    grasp_candidates: list[GraspCandidate] = field(default_factory=list)
    grasp_attempts: list[GraspAttempt] = field(default_factory=list)
    
    # ──── 元信息 ────
    observed_in_views: list[str] = field(default_factory=list)
    times_re_observed: int = 0
    last_action_failed: Optional[str] = None
    
    # ──────────────────────────────────────
    # 派生属性
    # ──────────────────────────────────────
    
    @property
    def grasp_feasibility(self) -> float:
        """剩余可用候选中分数最高者。失败过的不重复试。"""
        used = {self._cand_key(a.candidate) for a in self.grasp_attempts}
        unused = [c for c in self.grasp_candidates
                  if self._cand_key(c) not in used]
        return max((c.score for c in unused), default=0.0)
    
    @property
    def grasp_uncertainty(self) -> Optional[float]:
        """grasp 不确定度。
        
        返回 None 表示"尚未规划"——既无 candidates 又无 attempts。这种状态下 grasp 轴
        不参与 most_uncertain_axis 排序, 也不阻止 is_confident_to_act 的非 grasp 轴
        confident 判定; 避免 episode 初期 4 轴默认 1.0 时 grasp 占 max 而过早 plan_grasp。
        
        一旦 plan 过 (即使空 candidates) 或有过 attempt: 失败 ≥2 次强制 1.0; 否则 1-feasibility。
        """
        if not self.grasp_candidates and not self.grasp_attempts:
            return None
        n_fail = sum(
            1 for a in self.grasp_attempts if a.failure_mode != "success"
        )
        if n_fail >= 2:
            return 1.0
        return 1.0 - self.grasp_feasibility
    
    @staticmethod
    def _cand_key(c: GraspCandidate) -> tuple:
        return (
            round(float(c.point_3d[0]), 3),
            round(float(c.point_3d[1]), 3),
            round(float(c.point_3d[2]), 3),
            round(float(c.approach_dir[0]), 2),
            round(float(c.approach_dir[1]), 2),
            round(float(c.approach_dir[2]), 2),
        )
    
    def overall_uncertainty(self) -> float:
        """各轴 max, 决定是否进 is_confident_to_act。归一化到 [0, 1]。
        
        grasp_uncertainty=None 时不参与 max。
        """
        norm_pos = min(1.0, self.position_std_m / 0.30)
        axes = [self.label_entropy, norm_pos, self.safety_entropy]
        gu = self.grasp_uncertainty
        if gu is not None:
            axes.append(gu)
        return max(axes)
