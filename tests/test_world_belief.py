"""WorldBelief / Hypothesis / Evidence / Action 单元测试。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np

from src.world_belief import (
    Hypothesis, Pose, GraspCandidate, GraspAttempt,
    Action, Evidence, BeliefSnapshot, EpisodeResult,
    DecomposedTask, Constraint,
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
