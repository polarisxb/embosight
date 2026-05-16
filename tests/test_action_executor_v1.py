"""ActionExecutor v1 (Hypothesis-based) 单元测试。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np


def _hyp_with_candidate(score=0.9):
    from src.world_belief import GraspCandidate, Hypothesis
    c = GraspCandidate(point_3d=np.array([0.5, 0, 0.9]),
                       approach_dir=np.array([0, 0, -1]),
                       finger_width_m=0.04, score=score,
                       source="geometric_centroid")
    h = Hypothesis(
        object_id="o0", label="apple",
        label_alternatives=[("apple", 0.9)], label_entropy=0.1,
        position_3d=np.array([0.5, 0, 0.9]), position_std_m=0.02,
        grasp_candidates=[c],
    )
    return h, c


class FakeEnv:
    def __init__(self, descend_ok=True, ik_ok=True, lift_ok=True,
                 final_z=0.05, obj_lifts=True):
        self.descend_ok = descend_ok
        self.ik_ok = ik_ok
        self.lift_ok = lift_ok
        self.final_z = final_z
        self.obj_lifts = obj_lifts
        self._gripper_open = True
        self._lifted = False
        self.calls: list[str] = []

    def move_to_pre_grasp(self, candidate) -> bool:
        self.calls.append("move_to_pre_grasp")
        return self.ik_ok

    def descend(self, point_3d, target_label=None, **kwargs):
        self.calls.append("descend")
        if self.descend_ok:
            return True, point_3d[2]
        return False, point_3d[2] + 0.03   # 卡住

    def approach(self, point_3d, approach_dir, target_label=None, **kwargs):
        ad = np.asarray(approach_dir, dtype=np.float32)
        self.calls.append(
            f"approach[{ad[0]:+.0f},{ad[1]:+.0f},{ad[2]:+.0f}]"
        )
        if self.descend_ok:
            return True, float(point_3d[2])
        return False, float(point_3d[2]) + 0.03

    def close_gripper(self, target_label=None) -> bool:
        self.calls.append("close")
        self._gripper_open = False
        return True

    def open_gripper(self) -> bool:
        self.calls.append("open")
        self._gripper_open = True
        return True

    def lift(self, height_m: float = 0.10, **kwargs) -> tuple[bool, float]:
        self.calls.append("lift")
        if self.lift_ok and self.obj_lifts:
            self._lifted = True
        return self.lift_ok, self.final_z

    def get_eef_pos(self):
        return np.array([0.5, 0, 0.95])

    def get_base_pose(self):
        return np.array([0.0, 0.0, 0.0]), np.eye(3, dtype=np.float32)

    def _get_obj_type_map(self):
        return {"obj_main": "apple"}

    def _get_body_pos(self, body_name):
        z = 0.98 if self._lifted else 0.9
        return np.array([0.5, 0, z])

    def move_arm_to(self, pos, **kw):
        self.calls.append("move")
        return True


class TestAct:
    def test_success_path(self):
        from src.action_executor import ActionExecutor
        from src.world_belief import DecomposedTask
        env = FakeEnv()
        exe = ActionExecutor(scene_describer=None)
        h, _ = _hyp_with_candidate()
        result = exe.act(h, DecomposedTask(primary_target="apple"), env)
        assert result.success is True
        assert result.attempt.failure_mode == "success"

    def test_ik_unreachable_classified(self):
        from src.action_executor import ActionExecutor
        from src.world_belief import DecomposedTask
        env = FakeEnv(ik_ok=False)
        exe = ActionExecutor(scene_describer=None)
        h, _ = _hyp_with_candidate()
        result = exe.act(h, DecomposedTask(primary_target="apple"), env)
        assert result.attempt.failure_mode == "ik_unreachable"
        assert result.success is False

    def test_hit_z_floor_recovery_obj_not_lifted(self):
        """descend fails → base reposition → re-descend fails → grasp at current z →
        lift arm ok but object stays → post-lift verify catches slipped."""
        from src.action_executor import ActionExecutor
        from src.world_belief import DecomposedTask
        env = FakeEnv(descend_ok=False, lift_ok=True, obj_lifts=False)
        exe = ActionExecutor(scene_describer=None)
        h, _ = _hyp_with_candidate()
        result = exe.act(h, DecomposedTask(primary_target="apple"), env)
        assert result.success is False
        assert result.attempt.failure_mode == "slipped"

    def test_hit_z_floor_recovery_obj_lifted(self):
        """descend fails → base reposition → re-descend fails → grasp at current z →
        arm lifts AND object follows → success."""
        from src.action_executor import ActionExecutor
        from src.world_belief import DecomposedTask
        env = FakeEnv(descend_ok=False, lift_ok=True, obj_lifts=True)
        exe = ActionExecutor(scene_describer=None)
        h, _ = _hyp_with_candidate()
        result = exe.act(h, DecomposedTask(primary_target="apple"), env)
        assert result.success is True
        assert result.attempt.failure_mode == "success"

    def test_side_approach_passes_correct_dir(self):
        """When candidate has approach_dir=[1,0,0], act() should call env.approach
        with that direction (not the default [0,0,-1])."""
        from src.action_executor import ActionExecutor
        from src.world_belief import (
            DecomposedTask, GraspCandidate, Hypothesis,
        )

        env = FakeEnv()
        exe = ActionExecutor(scene_describer=None)
        c = GraspCandidate(
            point_3d=np.array([0.5, 0, 0.9]),
            approach_dir=np.array([1.0, 0.0, 0.0]),
            finger_width_m=0.03, score=0.9,
            source="side_test",
        )
        h = Hypothesis(
            object_id="o0", label="apple",
            label_alternatives=[("apple", 0.9)], label_entropy=0.1,
            position_3d=np.array([0.5, 0, 0.9]), position_std_m=0.02,
            grasp_candidates=[c],
        )
        exe.act(h, DecomposedTask(primary_target="apple"), env)
        side_calls = [c for c in env.calls if c.startswith("approach[+1")]
        assert len(side_calls) >= 1, (
            f"Expected at least one side approach call, got: {env.calls}"
        )

    def test_top_down_uses_approach_with_z_minus_1(self):
        """Default top_down candidate should call approach with [0,0,-1]."""
        from src.action_executor import ActionExecutor
        from src.world_belief import DecomposedTask

        env = FakeEnv()
        exe = ActionExecutor(scene_describer=None)
        h, _ = _hyp_with_candidate()
        exe.act(h, DecomposedTask(primary_target="apple"), env)
        # Look for any approach call with z = -1
        td_calls = [c for c in env.calls if "approach[" in c and ",-1]" in c]
        assert len(td_calls) >= 1, (
            f"Expected approach[...,-1] call, got: {env.calls}"
        )

    def test_hit_z_floor_recovery_fails(self):
        """descend fails AND lift after reposition also fails → hit_z_floor."""
        from src.action_executor import ActionExecutor
        from src.world_belief import DecomposedTask
        env = FakeEnv(descend_ok=False, lift_ok=False, final_z=0.0)
        exe = ActionExecutor(scene_describer=None)
        h, _ = _hyp_with_candidate()
        result = exe.act(h, DecomposedTask(primary_target="apple"), env)
        assert result.attempt.failure_mode == "hit_z_floor"
        assert result.success is False

    def test_slipped_classified(self):
        from src.action_executor import ActionExecutor
        from src.world_belief import DecomposedTask
        env = FakeEnv(lift_ok=False, final_z=0.0)
        exe = ActionExecutor(scene_describer=None)
        h, _ = _hyp_with_candidate()
        result = exe.act(h, DecomposedTask(primary_target="apple"), env)
        assert result.attempt.failure_mode == "slipped"

    def test_no_candidates_returns_failure(self):
        from src.action_executor import ActionExecutor
        from src.world_belief import DecomposedTask, Hypothesis
        env = FakeEnv()
        exe = ActionExecutor(scene_describer=None)
        h = Hypothesis(
            object_id="o0", label="x",
            label_alternatives=[("x", 1.0)], label_entropy=0.0,
            position_3d=np.zeros(3), position_std_m=0.0,
        )
        result = exe.act(h, DecomposedTask(primary_target="x"), env)
        assert result.success is False
        assert result.attempt.failure_mode == "ik_unreachable"


class TestVerifyGrasp:
    def test_verify_returns_bool_and_conf(self):
        from src.action_executor import ActionExecutor
        exe = ActionExecutor(scene_describer=None)
        h, _ = _hyp_with_candidate()
        env = FakeEnv()
        ok, conf = exe.verify_grasp(h, env)
        assert isinstance(ok, bool)
        assert 0.0 <= conf <= 1.0


class TestReleaseAndRetreat:
    def test_release_and_retreat_opens_then_lifts(self):
        """F6: 撤回必须 open + 提升, 否则后续 observe 被夹爪挡。"""
        from src.action_executor import ActionExecutor
        env = FakeEnv()
        exe = ActionExecutor(scene_describer=None)
        exe.release_and_retreat(env, retreat_height_m=0.10)
        assert "open" in env.calls
        assert any(c.startswith("move") for c in env.calls)
        # open 必须在 move 前 (先放再走)
        assert env.calls.index("open") < env.calls.index("move")


class TestDiagnostic:
    def test_diagnostic_contains_z_target_on_success(self):
        from src.action_executor import ActionExecutor
        from src.world_belief import DecomposedTask
        exe = ActionExecutor(scene_describer=None)
        env = FakeEnv()
        h, _ = _hyp_with_candidate()
        result = exe.act(h, DecomposedTask(primary_target="apple"), env)
        assert "z_target" in result.attempt.diagnostic
        assert "final_z" in result.attempt.diagnostic


class TestStructure:
    def test_to_dict_serializable(self):
        from src.action_executor import ActionExecutor
        from src.world_belief import DecomposedTask
        exe = ActionExecutor(scene_describer=None)
        env = FakeEnv()
        h, _ = _hyp_with_candidate()
        result = exe.act(h, DecomposedTask(primary_target="apple"), env)
        d = result.to_dict()
        assert d["success"] is True
        assert d["attempt"]["failure_mode"] == "success"
        assert d["attempt"]["candidate_source"] == "geometric_centroid"

    def test_used_candidates_excluded(self):
        """已 attempt 过的 candidate 不再选 → no candidate → ik_unreachable。"""
        from src.action_executor import ActionExecutor
        from src.world_belief import DecomposedTask, GraspAttempt
        exe = ActionExecutor(scene_describer=None)
        env = FakeEnv()
        h, c = _hyp_with_candidate()
        h.grasp_attempts = [GraspAttempt(
            timestamp=0.0, candidate=c, failure_mode="hit_z_floor",
            end_effector_pose_reached=(0.0,) * 6,
        )]
        result = exe.act(h, DecomposedTask(primary_target="apple"), env)
        assert result.success is False
