"""WorldBelief / Hypothesis / Evidence / Action 单元测试。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np

from src.world_belief import (
    Hypothesis, GraspCandidate, GraspAttempt,
    Action, Evidence,
    DecomposedTask,
    WorldBelief,
)


def _basic_hyp(label="apple", label_e=0.2, pos_std=0.04, safe_e=0.2,
               candidates=None, attempts=None, alternatives=None):
    return Hypothesis(
        object_id=f"obj_{label}",
        label=label,
        label_alternatives=alternatives or [(label, 0.8), ("other", 0.2)],
        label_entropy=label_e,
        position_3d=np.array([0.5, 0.0, 0.9]),
        position_std_m=pos_std,
        safety_entropy=safe_e,
        grasp_candidates=candidates or [],
        grasp_attempts=attempts or [],
    )


class TestHypothesisBasics:
    def test_minimal_construct(self):
        """构造一个最小 Hypothesis。"""
        h = Hypothesis(
            object_id="obj_0",
            label="apple",
            label_alternatives=[("apple", 0.8), ("pear", 0.2)],
            label_entropy=0.50,
            position_3d=np.array([0.5, 0.0, 0.9]),
            position_std_m=0.05,
        )
        assert h.label == "apple"
        assert h.label_entropy == 0.50
        assert h.position_std_m == 0.05
        assert h.safety_entropy == 1.0  # 默认最大熵
        assert h.grasp_candidates == []
        assert h.grasp_attempts == []


class TestGraspUncertainty:
    def _make(self, candidates=None, attempts=None):
        return Hypothesis(
            object_id="o0", label="x",
            label_alternatives=[("x", 1.0)], label_entropy=0.0,
            position_3d=np.zeros(3), position_std_m=0.0,
            grasp_candidates=candidates or [],
            grasp_attempts=attempts or [],
        )
    
    def test_no_candidates_no_attempts_returns_none(self):
        """未规划时 grasp_uncertainty 必须是 None (F2)。"""
        h = self._make()
        assert h.grasp_uncertainty is None
    
    def test_with_candidate_no_attempt_returns_one_minus_score(self):
        """有候选无尝试 → 1 - 最高 score。"""
        c = GraspCandidate(
            point_3d=np.array([0.5, 0, 0.9]),
            approach_dir=np.array([0, 0, -1]),
            finger_width_m=0.04, score=0.8,
        )
        h = self._make(candidates=[c])
        assert h.grasp_uncertainty == pytest.approx(0.2)
    
    def test_two_failures_force_one(self):
        """连续 ≥2 次非 success 强制 1.0 (触发 ask_user)。"""
        c = GraspCandidate(
            point_3d=np.array([0.5, 0, 0.9]),
            approach_dir=np.array([0, 0, -1]),
            finger_width_m=0.04, score=0.9,
        )
        a1 = GraspAttempt(timestamp=1.0, candidate=c, failure_mode="hit_z_floor",
                          end_effector_pose_reached=(0,0,0,0,0,0))
        a2 = GraspAttempt(timestamp=2.0, candidate=c, failure_mode="ik_unreachable",
                          end_effector_pose_reached=(0,0,0,0,0,0))
        h = self._make(candidates=[c], attempts=[a1, a2])
        assert h.grasp_uncertainty == 1.0
    
    def test_used_candidate_excluded_from_feasibility(self):
        """失败过的候选不重复试。"""
        c1 = GraspCandidate(point_3d=np.array([0.5,0,0.9]),
                            approach_dir=np.array([0,0,-1]),
                            finger_width_m=0.04, score=0.9)
        c2 = GraspCandidate(point_3d=np.array([0.6,0,0.9]),
                            approach_dir=np.array([0,0,-1]),
                            finger_width_m=0.04, score=0.6)
        a1 = GraspAttempt(timestamp=1.0, candidate=c1, failure_mode="hit_z_floor",
                          end_effector_pose_reached=(0,0,0,0,0,0))
        h = self._make(candidates=[c1, c2], attempts=[a1])
        # c1 已用过, 只剩 c2 score=0.6 → uncertainty = 0.4
        assert h.grasp_uncertainty == pytest.approx(0.4)


class TestOverallUncertainty:
    def _make(self, label_e=0.0, pos_std=0.0, safe_e=0.0, grasp_unc=None,
              candidates=None, attempts=None):
        return Hypothesis(
            object_id="o0", label="x",
            label_alternatives=[("x", 1.0)], label_entropy=label_e,
            position_3d=np.zeros(3), position_std_m=pos_std,
            safety_entropy=safe_e,
            grasp_candidates=candidates or [],
            grasp_attempts=attempts or [],
        )
    
    def test_grasp_none_skipped(self):
        """grasp=None 时 overall 仅看 label/pos/safety (F2)。"""
        h = self._make(label_e=0.4, pos_std=0.0, safe_e=0.2)
        assert h.overall_uncertainty() == pytest.approx(0.4)
    
    def test_position_normalized(self):
        """position_std_m / 0.30 归一化到 [0,1]。"""
        h = self._make(label_e=0.0, pos_std=0.15, safe_e=0.0)
        # 0.15 / 0.30 = 0.5
        assert h.overall_uncertainty() == pytest.approx(0.5)


class TestActionEvidence:
    def test_action_default_metadata_dict(self):
        a = Action(kind="observe")
        assert a.kind == "observe"
        assert a.metadata == {}
    
    def test_evidence_default_consumed_by_empty(self):
        ev = Evidence(source="vlm_ground", timestamp=1.0, raw_payload={"x": 1})
        assert ev.consumed_by == []
    
    def test_decomposed_task_constraints_empty(self):
        dt = DecomposedTask(primary_target="apple", raw_query="拿苹果")
        assert dt.constraints == []
        assert dt.primary_target == "apple"


class TestWorldBeliefTarget:
    def test_empty_belief_target_is_none(self):
        b = WorldBelief(user_query="拿苹果")
        b.decomposed = DecomposedTask(primary_target="apple")
        assert b.target() is None
    
    def test_no_decomposed_target_is_none(self):
        b = WorldBelief(user_query="anything")
        b.hypotheses = [_basic_hyp()]
        assert b.target() is None
    
    def test_label_match_returns_hyp(self):
        b = WorldBelief(user_query="拿苹果")
        b.decomposed = DecomposedTask(primary_target="apple")
        h = _basic_hyp(label="apple",
                       alternatives=[("apple", 0.9), ("kiwi", 0.1)])
        b.hypotheses = [h]
        assert b.target() is h
    
    def test_top1_top2_close_returns_none(self):
        """top1 概率与 top2 差 < 0.2 → 模糊 (9.12)。"""
        b = WorldBelief(user_query="拿苹果")
        b.decomposed = DecomposedTask(primary_target="apple")
        h1 = _basic_hyp(label="apple_1",
                        alternatives=[("apple", 0.4), ("pear", 0.3)])
        h2 = _basic_hyp(label="apple_2",
                        alternatives=[("apple", 0.5), ("pear", 0.2)])
        b.hypotheses = [h1, h2]
        # top1=h2 (0.5), top2=h1 (0.4), diff=0.1 < 0.2 → None
        assert b.target() is None


class TestIsConfidentToAct:
    def test_no_target_not_confident(self):
        b = WorldBelief(user_query="拿苹果")
        b.decomposed = DecomposedTask(primary_target="apple")
        assert b.is_confident_to_act() is False
    
    def test_grasp_none_not_confident(self):
        """grasp_uncertainty=None 视为不 confident (F2)。"""
        b = WorldBelief(user_query="拿苹果")
        b.decomposed = DecomposedTask(primary_target="apple")
        # 其他 3 轴全 confident
        h = _basic_hyp(label="apple", label_e=0.1, pos_std=0.02, safe_e=0.1,
                       alternatives=[("apple", 0.9)])
        b.hypotheses = [h]
        assert b.is_confident_to_act() is False
    
    def test_all_axes_confident_returns_true(self):
        c = GraspCandidate(point_3d=np.array([0.5,0,0.9]),
                           approach_dir=np.array([0,0,-1]),
                           finger_width_m=0.04, score=0.9)
        b = WorldBelief(user_query="拿苹果")
        b.decomposed = DecomposedTask(primary_target="apple")
        h = _basic_hyp(label="apple", label_e=0.1, pos_std=0.02, safe_e=0.1,
                       alternatives=[("apple", 0.9)], candidates=[c])
        b.hypotheses = [h]
        assert b.is_confident_to_act() is True


class TestMostUncertainAxis:
    def test_no_target_returns_label(self):
        b = WorldBelief(user_query="拿苹果")
        b.decomposed = DecomposedTask(primary_target="apple")
        assert b.most_uncertain_axis() == "label"
    
    def test_grasp_none_skipped(self):
        """grasp=None 时不参与最大轴选择 (F2)。"""
        b = WorldBelief(user_query="拿苹果")
        b.decomposed = DecomposedTask(primary_target="apple")
        h = _basic_hyp(label="apple", label_e=0.5, pos_std=0.02, safe_e=0.1,
                       alternatives=[("apple", 0.9)])
        b.hypotheses = [h]
        # 4 轴: label=0.5, pos=0.067, safe=0.1, grasp=None → label 最大
        assert b.most_uncertain_axis() == "label"
    
    def test_safety_max(self):
        b = WorldBelief(user_query="拿苹果")
        b.decomposed = DecomposedTask(primary_target="apple")
        h = _basic_hyp(label="apple", label_e=0.1, pos_std=0.02, safe_e=0.8,
                       alternatives=[("apple", 0.9)])
        b.hypotheses = [h]
        assert b.most_uncertain_axis() == "safety"


class TestMerge:
    def test_close_distance_overlap_label_merges(self):
        """距离 < 0.15m + 概率交集 > 0.30 → 合并。"""
        b = WorldBelief(user_query="x")
        b.decomposed = DecomposedTask(primary_target="apple")
        h1 = _basic_hyp(label="apple",
                        alternatives=[("apple", 0.7), ("pear", 0.3)])
        h1.position_3d = np.array([0.50, 0.0, 0.9])
        h1.observed_in_views = ["v1"]
        b.hypotheses = [h1]
        h2 = _basic_hyp(label="apple",
                        alternatives=[("apple", 0.8), ("kiwi", 0.2)])
        h2.position_3d = np.array([0.52, 0.01, 0.91])    # 距离 ~0.022m
        h2.observed_in_views = ["v2"]
        merged = b.merge_hypothesis(h1, h2)
        assert merged is True
        assert len(b.hypotheses) == 1
        assert "v2" in b.hypotheses[0].observed_in_views
    
    def test_far_distance_does_not_merge(self):
        b = WorldBelief(user_query="x")
        b.decomposed = DecomposedTask(primary_target="apple")
        h1 = _basic_hyp(label="apple")
        h1.position_3d = np.array([0.5, 0, 0.9])
        b.hypotheses = [h1]
        h2 = _basic_hyp(label="apple")
        h2.position_3d = np.array([0.8, 0, 0.9])         # 距离 0.3m > 0.15
        merged = b.merge_hypothesis(h1, h2)
        assert merged is False
        assert len(b.hypotheses) == 1                    # 还没 add 进去
    
    def test_low_label_intersection_does_not_merge(self):
        b = WorldBelief(user_query="x")
        b.decomposed = DecomposedTask(primary_target="apple")
        h1 = _basic_hyp(label="apple",
                        alternatives=[("apple", 0.9), ("pear", 0.1)])
        h2 = _basic_hyp(label="bottle",
                        alternatives=[("bottle", 0.9), ("can", 0.1)])
        h2.position_3d = h1.position_3d + np.array([0.02, 0, 0])
        b.hypotheses = [h1]
        # 概率交集: 仅 (apple, 0.9)/(apple, 0.0) = 0; bottle 0.9/0.0 = 0; → 0
        merged = b.merge_hypothesis(h1, h2)
        assert merged is False


class TestPrune:
    def test_phantom_pruned(self):
        """1 视角 + entropy>0.7 + 步数>3 → 删。"""
        b = WorldBelief(user_query="x")
        h_ghost = _basic_hyp(label="ghost", label_e=0.85,
                             alternatives=[("ghost", 0.4), ("blob", 0.4)])
        h_ghost.observed_in_views = ["v1"]
        b.hypotheses = [h_ghost]
        # 模拟 4 步
        for _ in range(4):
            b.action_history.append(Action(kind="observe"))
        n = b.prune_phantom_hypotheses()
        assert n == 1
        assert b.hypotheses == []
    
    def test_multi_view_not_pruned(self):
        b = WorldBelief(user_query="x")
        h = _basic_hyp(label="apple", label_e=0.85)
        h.observed_in_views = ["v1", "v2"]
        b.hypotheses = [h]
        for _ in range(4):
            b.action_history.append(Action(kind="observe"))
        n = b.prune_phantom_hypotheses()
        assert n == 0


class TestSnapshot:
    def test_snapshot_basic(self):
        b = WorldBelief(user_query="拿苹果")
        b.decomposed = DecomposedTask(primary_target="apple")
        h = _basic_hyp(label="apple", label_e=0.4,
                       alternatives=[("apple", 0.9)])
        b.hypotheses = [h]
        snap = b.snapshot(step=2)
        assert snap.step == 2
        assert snap.n_hypotheses == 1
        assert snap.most_uncertain_axis == "label"
        assert snap.target_summary is not None
        assert snap.target_summary["label"] == "apple"


class TestEdgeCases:
    def test_high_risk_tightens_thresholds(self):
        """sharp+hot+chemical > 0.5 → high_risk (label thr 0.30 → 0.15)。"""
        b = WorldBelief(user_query="拿削皮器")
        b.decomposed = DecomposedTask(primary_target="peeler")
        c = GraspCandidate(point_3d=np.array([0.5,0,0.9]),
                           approach_dir=np.array([0,0,-1]),
                           finger_width_m=0.04, score=0.9)
        h = _basic_hyp(label="peeler",
                       alternatives=[("peeler", 0.85)])
        h.label_entropy = 0.60    # 普通 < 0.80 confident, high-risk > 0.50 不 confident
        h.position_std_m = 0.02
        h.safety_dist = {"sharp": 0.7, "safe": 0.3}
        h.safety_entropy = 0.10
        h.grasp_candidates = [c]
        b.hypotheses = [h]
        # high_risk 阈值收紧 label=0.50, label_entropy=0.60 > 0.50 → 不 confident
        assert b.is_confident_to_act() is False
    
    def test_merge_distance_boundary(self):
        """距离 0.149 vs 0.151。"""
        b = WorldBelief(user_query="x")
        b.decomposed = DecomposedTask(primary_target="apple")
        h1 = _basic_hyp(label="apple")
        h1.position_3d = np.array([0.5, 0.0, 0.9])
        b.hypotheses = [h1]
        # 0.149 → merge ok
        h_close = _basic_hyp(label="apple")
        h_close.position_3d = np.array([0.5 + 0.149, 0, 0.9])
        assert b.merge_hypothesis(h1, h_close) is True
        # reset
        h1.position_3d = np.array([0.5, 0.0, 0.9])
        # 0.151 → no merge
        h_far = _basic_hyp(label="apple")
        h_far.position_3d = np.array([0.5 + 0.151, 0, 0.9])
        assert b.merge_hypothesis(h1, h_far) is False
    
    def test_prune_recent_steps_kept(self):
        """步数 ≤ PRUNE_MIN_STEPS → 不 prune (即使是幻觉)。"""
        b = WorldBelief(user_query="x")
        h = _basic_hyp(label="ghost", label_e=0.85)
        h.observed_in_views = ["v1"]
        b.hypotheses = [h]
        # 仅 2 步
        b.action_history.append(Action(kind="observe"))
        b.action_history.append(Action(kind="observe"))
        n = b.prune_phantom_hypotheses()
        assert n == 0
    
    def test_consume_user_answer_appends_constraint(self):
        b = WorldBelief(user_query="x")
        b.consume_user_answer("您要的是哪个?", "圆形的", llm=None)
        assert len(b.user_constraints) == 1
        assert "圆形的" in b.user_constraints[0]
