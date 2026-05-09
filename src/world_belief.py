"""EmboSight Agent v1 - 信念状态数据结构。

纯数据层: 无 IO, 无 LLM/VLM 调用。所有状态修改通过显式方法。

设计参考: docs/superpowers/specs/2026-05-08-emboSight-belief-driven-agent-design.md §4
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _shannon(probs: list[float]) -> float:
    """Shannon entropy in nats; 输入概率不必归一, 内部归一。"""
    total = sum(p for p in probs if p > 0)
    if total <= 0:
        return 0.0
    h = 0.0
    for p in probs:
        if p > 0:
            q = p / total
            h -= q * math.log(q)
    return h


def _label_key(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


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
        """剩余可用候选中分数最高者。物理失败过的不重复试; verify_mismatch
        是身份识别错而非物理失败, 同 candidate 物理上仍可用, 不计入 used。
        """
        used = {
            self._cand_key(a.candidate) for a in self.grasp_attempts
            if a.failure_mode != "verify_mismatch"
        }
        unused = [c for c in self.grasp_candidates
                  if self._cand_key(c) not in used]
        return max((c.score for c in unused), default=0.0)
    
    @property
    def grasp_uncertainty(self) -> Optional[float]:
        """grasp 不确定度。
        
        返回 None 表示"尚未规划"——既无 candidates 又无 attempts。这种状态下 grasp 轴
        不参与 most_uncertain_axis 排序, 也不阻止 is_confident_to_act 的非 grasp 轴
        confident 判定; 避免 episode 初期 4 轴默认 1.0 时 grasp 占 max 而过早 plan_grasp。
        
        一旦 plan 过 (即使空 candidates) 或有过 attempt: 物理失败 ≥2 次强制 1.0; 否则
        1-feasibility。verify_mismatch 不算物理失败 (物体认错而非抓不住)。
        """
        if not self.grasp_candidates and not self.grasp_attempts:
            return None
        n_fail = sum(
            1 for a in self.grasp_attempts
            if a.failure_mode not in {"success", "verify_mismatch"}
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


# ============================================================
# 4.6 DecomposedTask / Constraint
# ============================================================

@dataclass
class Constraint:
    kind: Literal["avoid", "prefer_view", "max_force", "user_hint"]
    target_label: Optional[str] = None
    text: Optional[str] = None
    reason: str = ""


@dataclass
class DecomposedTask:
    primary_target: str
    constraints: list[Constraint] = field(default_factory=list)
    raw_query: str = ""


# ============================================================
# 4.3 Evidence / Action / BeliefSnapshot
# ============================================================

EvidenceSource = Literal[
    "vlm_ground", "vlm_zoom", "vlm_verify",
    "llm_safety", "llm_decompose", "user_answer",
    "grasp_attempt", "depth_projection", "vlm_failed",
]


@dataclass
class Evidence:
    """一次工具调用的原始结果, 用于审计 + replay。"""
    source: EvidenceSource
    timestamp: float
    raw_payload: dict[str, Any]
    consumed_by: list[str] = field(default_factory=list)


ActionKind = Literal[
    "observe", "re_observe", "classify_safety",
    "plan_grasp_candidates", "grasp", "ask_user", "give_up",
]


@dataclass
class Action:
    kind: ActionKind
    target_hypothesis: Optional["Hypothesis"] = None
    viewpoint: Optional[Any] = None    # Viewpoint 类在 active_planner, 此处不直接 import
    strategy: Optional[Literal["zoom_in", "parallax_view", "parallax_for_pose"]] = None
    question: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BeliefSnapshot:
    """某时刻 belief 的浅拷贝, 用于 EpisodeLogger。"""
    step: int
    timestamp: float
    n_hypotheses: int
    target_summary: Optional[dict[str, Any]]
    most_uncertain_axis: str
    overall_uncertainty: float
    n_evidence: int
    open_questions_count: int


# ============================================================
# 4.5 EpisodeResult
# ============================================================

@dataclass
class EpisodeResult:
    success: bool
    target: Optional[Hypothesis]
    speech: str
    belief_trace: list[BeliefSnapshot]
    action_history: list[Action]
    n_steps: int
    elapsed_seconds: float
    failure_reason: Optional[str] = None


# ============================================================
# 4.4 WorldBelief
# ============================================================

@dataclass
class WorldBelief:
    """主信念状态, 贯穿整个 episode。"""
    user_query: str
    decomposed: Optional[DecomposedTask] = None
    
    hypotheses: list[Hypothesis] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    action_history: list[Action] = field(default_factory=list)
    user_constraints: list[str] = field(default_factory=list)
    
    # ──── 默认阈值 (可被 _dynamic_thresholds 覆盖) ────
    # label/safety 按 nat-base entropy 设定:
    #   label=0.80 对应 top-1 ~0.60 (VLM 对 sim 图像典型输出)
    #   safety=0.50 对应 safety_dist 4 类分布的中等确信
    # 旧值 0.30 要求 top-1 > 0.94, 对 VLM 在 sim 上不现实
    DEFAULT_THRESHOLDS = {
        "label": 0.80, "position": 0.10, "safety": 0.50, "grasp": 0.40,
    }
    HIGH_RISK_THRESHOLDS = {
        "label": 0.50, "position": 0.05, "safety": 0.30, "grasp": 0.25,
    }
    AMBIGUITY_PROB_GAP = 0.20      # top1/top2 概率差 < 此值 → 模糊
    LABEL_CONFIDENT_PROB_MIN = 0.55
    LABEL_CONFIDENT_PROB_GAP = 0.30
    
    # ──── 查询接口 ──────────────────────────
    
    def target(self) -> Optional[Hypothesis]:
        """返回最匹配 user_query 的 hypothesis。
        
        - 如无 decomposed 或无 hypotheses, None
        - 取 label_alternatives 里 primary_target 的概率最大者
        - 副分: 当前 label 文本含 primary_target 也算 0.5
        - top1 与 top2 的概率差 < 0.2 → 视为模糊, 返回 None (Edge case 9.12)
        """
        if not self.hypotheses or not self.decomposed:
            return None
        target_word = self.decomposed.primary_target.lower()
        target_key = _label_key(target_word)
        scored: list[tuple[float, Hypothesis]] = []
        for h in self.hypotheses:
            label_matches = (
                target_word in h.label.lower()
                or target_key in _label_key(h.label)
            )
            prob = 0.0
            if h.label_alternatives:
                top_label, top_prob = max(h.label_alternatives, key=lambda item: item[1])
                top_matches = (
                    target_word in top_label.lower()
                    or target_key in _label_key(top_label)
                )
                if top_matches:
                    prob = top_prob
            if label_matches:
                alt_prob = next(
                    (p for lbl, p in h.label_alternatives
                     if target_word in lbl.lower() or target_key in _label_key(lbl)),
                    0.0,
                )
                prob = max(prob, alt_prob, 0.5)
            if prob > 0:
                prob = max(prob, 0.5)
                scored.append((prob, h))
        if not scored:
            return None
        scored.sort(key=lambda t: t[0], reverse=True)
        if len(scored) >= 2 and (scored[0][0] - scored[1][0]) < self.AMBIGUITY_PROB_GAP:
            return None
        return scored[0][1]
    
    def is_confident_to_act(
        self,
        label_thr: Optional[float] = None,
        pos_thr_m: Optional[float] = None,
        safety_thr: Optional[float] = None,
        grasp_thr: Optional[float] = None,
    ) -> bool:
        """所有轴都低于阈值才能动手。grasp=None 视为不 confident。"""
        h = self.target()
        if h is None:
            return False
        thr = self._dynamic_thresholds(h, label_thr, pos_thr_m, safety_thr, grasp_thr)
        gu = h.grasp_uncertainty if h.grasp_uncertainty is not None else 1.0
        return (
            self.is_label_confident(h, thr["label"])
            and h.position_std_m < thr["position"]
            and self.is_safety_confident(h, thr["safety"])
            and gu                < thr["grasp"]
        )

    def is_label_confident(self, h: Hypothesis, label_thr: Optional[float] = None) -> bool:
        thr = self.DEFAULT_THRESHOLDS["label"] if label_thr is None else label_thr
        if h.label_entropy < thr:
            return True
        if thr < self.DEFAULT_THRESHOLDS["label"]:
            return False
        if not self.decomposed or not h.label_alternatives:
            return False
        target_key = _label_key(self.decomposed.primary_target)
        ranked = sorted(h.label_alternatives, key=lambda item: item[1], reverse=True)
        top_label, top_prob = ranked[0]
        second_prob = ranked[1][1] if len(ranked) > 1 else 0.0
        return (
            target_key in _label_key(top_label)
            and top_prob >= self.LABEL_CONFIDENT_PROB_MIN
            and (top_prob - second_prob) >= self.LABEL_CONFIDENT_PROB_GAP
        )

    def is_safety_confident(self, h: Hypothesis, safety_thr: Optional[float] = None) -> bool:
        thr = self.DEFAULT_THRESHOLDS["safety"] if safety_thr is None else safety_thr
        if h.safety_entropy < thr:
            return True
        if not h.safety_dist:
            return False
        hazard_prob = (
            h.safety_dist.get("sharp", 0.0)
            + h.safety_dist.get("hot", 0.0)
            + h.safety_dist.get("chemical", 0.0)
        )
        return hazard_prob < 0.05
    
    def most_uncertain_axis(self) -> Literal["label", "position", "safety", "grasp"]:
        """最不确定的轴; grasp_uncertainty=None 时跳过。"""
        h = self.target()
        if h is None:
            return "label"
        norm_pos = min(1.0, h.position_std_m / 0.30)
        scores: dict[str, float] = {
            "label":    h.label_entropy,
            "position": norm_pos,
            "safety":   h.safety_entropy,
        }
        if h.grasp_uncertainty is not None:
            scores["grasp"] = h.grasp_uncertainty
        return max(scores, key=scores.get)        # type: ignore[return-value]
    
    def used_views(self) -> set[str]:
        return {
            getattr(a.viewpoint, "name", str(a.viewpoint))
            for a in self.action_history
            if a.kind == "observe" and a.viewpoint is not None
        }
    
    # ──── 内部 ───────────────────────────────
    
    def _dynamic_thresholds(
        self, h: Hypothesis,
        label_thr, pos_thr_m, safety_thr, grasp_thr,
    ) -> dict[str, float]:
        """高风险物体严格, safe 物体宽松。
        
        high_risk = h.safety_dist 中 sharp/hot/chemical 之和 > 0.5
        """
        risk_score = (
            h.safety_dist.get("sharp", 0.0)
            + h.safety_dist.get("hot", 0.0)
            + h.safety_dist.get("chemical", 0.0)
        )
        base = self.HIGH_RISK_THRESHOLDS if risk_score > 0.5 else self.DEFAULT_THRESHOLDS
        return {
            "label":    label_thr   if label_thr   is not None else base["label"],
            "position": pos_thr_m   if pos_thr_m   is not None else base["position"],
            "safety":   safety_thr  if safety_thr  is not None else base["safety"],
            "grasp":    grasp_thr   if grasp_thr   is not None else base["grasp"],
        }
    
    # ──── 状态修改 ──────────────────────────
    
    MERGE_DISTANCE_M = 0.15           # TODO(v1.1): 实测调
    MERGE_LABEL_INTERSECTION_MIN = 0.30  # TODO(v1.1): 实测调
    PRUNE_MIN_STEPS = 3
    PRUNE_PHANTOM_ENTROPY = 0.95  # 只剪极高熵 (VLM 在 sim 上典型 0.7-0.8)
    
    def add_hypothesis(self, h: Hypothesis) -> None:
        self.hypotheses.append(h)
    
    def merge_hypothesis(self, existing: Hypothesis, new_data: Hypothesis) -> bool:
        """尝试把 new_data 合并进 existing。返回 True 表示合并成功。
        
        条件:
        - position 距离 < MERGE_DISTANCE_M
        - label_alternatives 概率交集 ≥ MERGE_LABEL_INTERSECTION_MIN
        """
        if existing not in self.hypotheses:
            return False
        dist = float(np.linalg.norm(existing.position_3d - new_data.position_3d))
        if dist >= self.MERGE_DISTANCE_M:
            return False
        # label_alternatives 概率交集
        e_dict = dict(existing.label_alternatives)
        intersect = sum(min(p, e_dict.get(lbl, 0.0))
                        for lbl, p in new_data.label_alternatives)
        if intersect < self.MERGE_LABEL_INTERSECTION_MIN:
            return False
        # 合并: 位置加权平均, label 取概率高的
        n_old = len(existing.observed_in_views) or 1
        n_new = len(new_data.observed_in_views) or 1
        existing.position_3d = (
            existing.position_3d * n_old + new_data.position_3d * n_new
        ) / (n_old + n_new)
        existing.observed_in_views.extend(
            v for v in new_data.observed_in_views
            if v not in existing.observed_in_views
        )
        existing.bbox_per_view.update(new_data.bbox_per_view)
        # label_alternatives: 概率平均后归一化
        merged_alts: dict[str, float] = dict(existing.label_alternatives)
        for lbl, p in new_data.label_alternatives:
            merged_alts[lbl] = (merged_alts.get(lbl, 0.0) + p) / 2
        total = sum(merged_alts.values()) or 1.0
        existing.label_alternatives = sorted(
            ((lbl, p / total) for lbl, p in merged_alts.items()),
            key=lambda x: x[1], reverse=True,
        )
        existing.label = existing.label_alternatives[0][0]
        # entropy 重算
        existing.label_entropy = _shannon([p for _, p in existing.label_alternatives])
        # 多视角 std (粗略: 简单更新)
        existing.position_std_m = max(existing.position_std_m, dist / 2)
        return True
    
    def prune_phantom_hypotheses(self) -> int:
        """删除疑似幻觉 hypothesis (1 视角 + 高熵 + 步数>3)。返回删除数。"""
        if len(self.action_history) <= self.PRUNE_MIN_STEPS:
            return 0
        before = len(self.hypotheses)
        self.hypotheses = [
            h for h in self.hypotheses
            if not (
                len(h.observed_in_views) <= 1
                and h.label_entropy > self.PRUNE_PHANTOM_ENTROPY
                and not h.grasp_attempts
            )
        ]
        return before - len(self.hypotheses)
    
    def snapshot(self, step: int) -> BeliefSnapshot:
        import time as _t
        h = self.target()
        target_summary = None
        if h is not None:
            target_summary = {
                "label": h.label,
                "label_entropy": h.label_entropy,
                "position_3d": h.position_3d.tolist(),
                "position_std_m": h.position_std_m,
                "safety_dist": h.safety_dist,
                "safety_entropy": h.safety_entropy,
                "grasp_uncertainty": h.grasp_uncertainty,
            }
        h_for_axis = self.target()
        if h_for_axis is not None:
            ovr = h_for_axis.overall_uncertainty()
        else:
            ovr = 1.0
        return BeliefSnapshot(
            step=step,
            timestamp=_t.time(),
            n_hypotheses=len(self.hypotheses),
            target_summary=target_summary,
            most_uncertain_axis=self.most_uncertain_axis(),
            overall_uncertainty=ovr,
            n_evidence=len(self.evidence),
            open_questions_count=len(self.open_questions),
        )
    
    def consume_user_answer(self, question: str, answer: str, llm) -> None:
        """v1 简化版: 把 (question, answer) 追加到 user_constraints。
        
        TODO(v1.1): LLM 解析答案 → boost/demote/constraint/unhelpful (设计稿 §6.1 + Edge 9.4/9.10)。
        - 当前 v1: 答案仅作字符串保留, 下一轮 NBV/perception prompt 通过把 user_constraints
          merge 进 decomposed.constraints 让 LLM "看见"。
        - v1.1 升级: 加载 prompts/agent/user_answer_parse.txt, 输出 {boost: id, demote: id,
          new_constraint: {...}, unhelpful: bool}, 直接修改 hypothesis.label_alternatives 概率。
        - llm 参数当前不使用, 保留接口签名以避免 v1.1 升级时破坏调用方。
        """
        _ = llm  # v1 未使用; v1.1 接入 LLM 解析时启用
        self.user_constraints.append(f"Q: {question} | A: {answer}")
        if not self.decomposed:
            return
        target_key = _label_key(self.decomposed.primary_target)
        if target_key not in _label_key(answer):
            return
        h = self.target()
        if h is None:
            return
        if any(a.failure_mode == "verify_mismatch" for a in h.grasp_attempts):
            return
        labels = [lbl for lbl, _ in h.label_alternatives if _label_key(lbl) != target_key]
        h.label_alternatives = [(self.decomposed.primary_target, 0.90)]
        if labels:
            share = 0.10 / len(labels)
            h.label_alternatives.extend((lbl, share) for lbl in labels)
        h.label = self.decomposed.primary_target
        h.label_entropy = _shannon([p for _, p in h.label_alternatives])
    
    def compose_clarification(self) -> str:
        """构造给用户的澄清问题。"""
        h = self.target()
        if h is None:
            return f"我没看清您要的{self.decomposed.primary_target if self.decomposed else 'something'}, 您能描述一下它附近还有什么吗?"
        alts = [lbl for lbl, _ in h.label_alternatives[:2]]
        return f"我看到一个像{alts[0]}的东西, 也可能是{alts[1] if len(alts) > 1 else '别的'}, 您要的是哪个?"
