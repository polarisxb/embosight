"""GraspPlanner 单元测试。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json

import numpy as np

from tests._mocks import MockVLM


def _hyp(label="apple", upright=True, pos_std=0.02):
    from src.world_belief import Hypothesis, Pose
    h = Hypothesis(
        object_id="o0", label=label,
        label_alternatives=[(label, 0.9)], label_entropy=0.1,
        position_3d=np.array([0.5, 0.0, 0.9]),
        position_std_m=pos_std,
    )
    h.pose_estimate = Pose(
        position=np.array([0.5, 0, 0.9]),
        rotation_quat=np.array([0, 0, 0, 1]),
        upright=upright,
    )
    return h


class FakeEnv:
    def __init__(self, reachable_fn=None):
        self._reachable_fn = reachable_fn or (lambda p, d: True)

    def is_reachable(self, point_3d, approach_dir):
        return self._reachable_fn(point_3d, approach_dir)

    def observe(self, vp):
        class Obs:
            image_path = "/dev/null"
        return Obs()

    def eye_in_hand_viewpoint(self):
        class V:
            name = "eye_in_hand"
        return V()


class TestGraspPlannerPlan:
    def test_geometric_centroid_always_first(self):
        from src.grasp_planner import GraspPlanner
        gp = GraspPlanner(vlm=MockVLM([]), env=FakeEnv())
        cands = gp.plan(_hyp(), env=FakeEnv())
        assert len(cands) >= 1
        assert any(c.source == "geometric_centroid" for c in cands)

    def test_axis_aligned_side_when_horizontal(self):
        """横放物体 → 加 axis_aligned_side 候选。"""
        from src.grasp_planner import GraspPlanner
        gp = GraspPlanner(vlm=MockVLM([]), env=FakeEnv())
        cands = gp.plan(_hyp(upright=False), env=FakeEnv())
        assert any(c.source == "axis_aligned_side" for c in cands)

    def test_unreachable_filtered(self):
        from src.grasp_planner import GraspPlanner
        env = FakeEnv(reachable_fn=lambda p, d: False)
        gp = GraspPlanner(vlm=MockVLM([]), env=env)
        cands = gp.plan(_hyp(), env=env)
        assert cands == []

    def test_vlm_top_grasp_used_if_available(self):
        from src.grasp_planner import GraspPlanner
        vlm = MockVLM(responses=[json.dumps({
            "grip_norm": [0.5, 0.5], "finger_align": "x",
        })])
        gp = GraspPlanner(vlm=vlm, env=FakeEnv())
        cands = gp.plan(_hyp(), env=FakeEnv())
        assert any(c.source == "vlm_top_grasp" for c in cands)

    def test_sorted_by_score_desc(self):
        from src.grasp_planner import GraspPlanner
        gp = GraspPlanner(vlm=MockVLM([]), env=FakeEnv())
        cands = gp.plan(_hyp(upright=False), env=FakeEnv())
        scores = [c.score for c in cands]
        assert scores == sorted(scores, reverse=True)


class TestRegenerateAfterFailure:
    def test_excludes_failed_candidate(self):
        from src.grasp_planner import GraspPlanner
        from src.world_belief import GraspAttempt, GraspCandidate
        gp = GraspPlanner(vlm=MockVLM([]), env=FakeEnv())
        h = _hyp()
        c1 = GraspCandidate(point_3d=h.position_3d.copy(),
                            approach_dir=np.array([0, 0, -1]),
                            finger_width_m=0.04, score=0.9,
                            source="geometric_centroid")
        h.grasp_candidates = [c1]
        attempt = GraspAttempt(timestamp=1.0, candidate=c1,
                               failure_mode="hit_z_floor",
                               end_effector_pose_reached=(0,) * 6)
        h.grasp_attempts = [attempt]
        new_cands = gp.regenerate_after_failure(h, attempt)
        for c in new_cands:
            same_pt = tuple(c.point_3d) == tuple(c1.point_3d)
            same_dir = tuple(c.approach_dir) == tuple(c1.approach_dir)
            assert not (same_pt and same_dir and c.source == c1.source)

    def test_horizontal_pose_after_z_floor_failure(self):
        """hit_z_floor 失败 + pose 转横 → 强制 axis_aligned_side。"""
        from src.grasp_planner import GraspPlanner
        from src.world_belief import GraspAttempt, GraspCandidate
        gp = GraspPlanner(vlm=MockVLM([]), env=FakeEnv())
        h = _hyp(upright=False)
        c1 = GraspCandidate(point_3d=h.position_3d,
                            approach_dir=np.array([0, 0, -1]),
                            finger_width_m=0.04, score=0.9,
                            source="geometric_centroid")
        h.grasp_candidates = [c1]
        attempt = GraspAttempt(timestamp=1.0, candidate=c1,
                               failure_mode="hit_z_floor",
                               end_effector_pose_reached=(0,) * 6)
        h.grasp_attempts = [attempt]
        new_cands = gp.regenerate_after_failure(h, attempt)
        assert any(c.source == "axis_aligned_side" for c in new_cands)


class TestEdgeNoPose:
    def test_no_pose_falls_back_to_centroid_only(self):
        from src.grasp_planner import GraspPlanner
        gp = GraspPlanner(vlm=MockVLM([]), env=FakeEnv())
        h = _hyp()
        h.pose_estimate = None
        cands = gp.plan(h, env=FakeEnv())
        assert all(c.source != "axis_aligned_side" for c in cands)
        assert any(c.source == "geometric_centroid" for c in cands)
