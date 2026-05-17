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
                 final_z=0.05, obj_lifts=True, grasp_ok=True):
        self.descend_ok = descend_ok
        self.ik_ok = ik_ok
        self.lift_ok = lift_ok
        self.final_z = final_z
        self.obj_lifts = obj_lifts
        self.grasp_ok = grasp_ok
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
        return self.grasp_ok

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
        assert result.attempt.failure_mode == "slipped_lift"

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
        """descend fails AND lift after reposition also fails + gripper empty → hit_z_floor."""
        from src.action_executor import ActionExecutor
        from src.world_belief import DecomposedTask
        env = FakeEnv(descend_ok=False, lift_ok=False, final_z=0.0, grasp_ok=False)
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
        assert result.attempt.failure_mode == "slipped_lift"

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


# ============================================================
# Phase 4: navigate_base_to integration tests
# ============================================================

class _NavCapturingEnv(FakeEnv):
    """FakeEnv extension that records navigate_base_to calls."""
    def __init__(self, navigate_return=True, navigate_raises=False, **kwargs):
        super().__init__(**kwargs)
        self.navigate_calls: list[dict] = []
        self._navigate_return = navigate_return
        self._navigate_raises = navigate_raises

    def navigate_base_to(self, target_xy, offset_m: float = 0.45) -> bool:
        self.navigate_calls.append({
            "target_xy": tuple(np.asarray(target_xy).tolist()),
            "offset_m": float(offset_m),
        })
        self.calls.append("navigate")
        if self._navigate_raises:
            raise RuntimeError("simulated navigate failure")
        return self._navigate_return


class TestPhase4NavigateIntegration:
    def test_act_calls_navigate_before_pre_grasp(self):
        """navigate_base_to must be invoked before move_to_pre_grasp.

        This is the core Phase 4 contract: explicit nav decoupled from arm OSC.
        """
        from src.action_executor import ActionExecutor
        from src.world_belief import DecomposedTask
        env = _NavCapturingEnv()
        exe = ActionExecutor(scene_describer=None)
        h, c = _hyp_with_candidate()

        result = exe.act(h, DecomposedTask(primary_target="apple"), env)

        assert "navigate" in env.calls, "navigate_base_to must be called"
        assert "move_to_pre_grasp" in env.calls, "pre_grasp must still run"
        # Ordering: navigate strictly before pre_grasp
        assert env.calls.index("navigate") < env.calls.index("move_to_pre_grasp")
        # Navigate target == candidate.point_3d[:2]
        assert len(env.navigate_calls) == 1
        expected_xy = tuple(c.point_3d[:2].tolist())
        actual_xy = env.navigate_calls[0]["target_xy"]
        assert np.allclose(actual_xy, expected_xy, atol=1e-5)
        # Defaults match design doc
        assert env.navigate_calls[0]["offset_m"] == 0.45
        # Pipeline completes successfully (FakeEnv all-green)
        assert result.success is True

    def test_act_falls_through_when_navigate_raises(self):
        """If navigate raises, act() must NOT crash — fall through to legacy
        pre_grasp path (which still has drive_base=True internal fallback)."""
        from src.action_executor import ActionExecutor
        from src.world_belief import DecomposedTask
        env = _NavCapturingEnv(navigate_raises=True)
        exe = ActionExecutor(scene_describer=None)
        h, _ = _hyp_with_candidate()

        # Must not raise; pre_grasp still runs.
        result = exe.act(h, DecomposedTask(primary_target="apple"), env)
        assert "navigate" in env.calls
        assert "move_to_pre_grasp" in env.calls
        # End-to-end still succeeds because FakeEnv's pre_grasp is all-green
        assert result.success is True

    def test_act_works_without_navigate_base_to_method(self):
        """Legacy mocks without navigate_base_to must still work.

        The hasattr() guard in act() lets pre-Phase-4 test fixtures continue
        to pass without any modification (backward compatibility)."""
        from src.action_executor import ActionExecutor
        from src.world_belief import DecomposedTask
        # Plain FakeEnv has no navigate_base_to attribute
        env = FakeEnv()
        assert not hasattr(env, "navigate_base_to")
        exe = ActionExecutor(scene_describer=None)
        h, _ = _hyp_with_candidate()

        result = exe.act(h, DecomposedTask(primary_target="apple"), env)
        # Pipeline runs end-to-end as before Phase 4
        assert result.success is True
        assert "move_to_pre_grasp" in env.calls


# ============================================================
# Phase 8a: torso lift integration tests
# ============================================================

class _TorsoCapturingEnv(_NavCapturingEnv):
    """FakeEnv extension that records set_torso_height + get_torso_height."""

    def __init__(self, current_torso: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self._torso_h = float(current_torso)
        self.torso_calls: list[float] = []

    def get_torso_height(self):
        return self._torso_h

    def set_torso_height(self, height_m: float) -> bool:
        self.torso_calls.append(float(height_m))
        self.calls.append("set_torso")
        self._torso_h = float(height_m)
        return True


class TestPhase8aTorsoLift:
    def test_top_down_lowers_torso_before_pre_grasp(self):
        """For top_down approach with low target z, act() should lower torso
        before pre-grasp so arm can reach down to grasp z."""
        from src.action_executor import ActionExecutor
        from src.world_belief import DecomposedTask
        # Target z=0.91 → required drop = 0.97 - (0.91 - 0.02) = 0.08m
        env = _TorsoCapturingEnv(current_torso=0.0)
        exe = ActionExecutor(scene_describer=None)
        h, c = _hyp_with_candidate()
        c.point_3d = np.array([0.5, 0.0, 0.91])  # low counter object
        h.position_3d = c.point_3d
        h.grasp_candidates = [c]

        exe.act(h, DecomposedTask(primary_target="apple"), env)

        assert env.torso_calls, "set_torso_height must be called for top_down"
        # set_torso should fire BEFORE pre_grasp
        assert env.calls.index("set_torso") < env.calls.index("move_to_pre_grasp")
        # Required drop = 0.97 - (0.91 - 0.02) = 0.08, so target = 0 - 0.08
        np.testing.assert_allclose(env.torso_calls[0], -0.08, atol=1e-6)

    def test_high_target_does_not_lower_torso(self):
        """If target z is high enough that arm can already reach (≥0.97 - 0.02
        = 0.95), no torso adjustment is needed."""
        from src.action_executor import ActionExecutor
        from src.world_belief import DecomposedTask
        env = _TorsoCapturingEnv(current_torso=0.0)
        exe = ActionExecutor(scene_describer=None)
        h, c = _hyp_with_candidate()
        c.point_3d = np.array([0.5, 0.0, 1.20])  # high cabinet object
        h.position_3d = c.point_3d
        h.grasp_candidates = [c]

        exe.act(h, DecomposedTask(primary_target="apple"), env)

        # required_drop = max(0, 0.97 - 1.18) = 0 → no torso call
        assert env.torso_calls == [], (
            f"high target should not trigger torso adjust, got {env.torso_calls}"
        )

    def test_side_approach_does_not_lower_torso(self):
        """Side grasps don't suffer the vertical-reach bottleneck;
        torso adjustment is gated to top_down only."""
        from src.action_executor import ActionExecutor
        from src.world_belief import (
            DecomposedTask, GraspCandidate, Hypothesis,
        )
        env = _TorsoCapturingEnv(current_torso=0.0)
        exe = ActionExecutor(scene_describer=None)
        c = GraspCandidate(
            point_3d=np.array([0.5, 0.0, 0.91]),
            approach_dir=np.array([1.0, 0.0, 0.0]),  # side
            finger_width_m=0.04, score=0.9,
            source="side_test",
        )
        h = Hypothesis(
            object_id="o0", label="apple",
            label_alternatives=[("apple", 0.9)], label_entropy=0.1,
            position_3d=c.point_3d, position_std_m=0.02,
            grasp_candidates=[c],
        )

        exe.act(h, DecomposedTask(primary_target="apple"), env)

        assert env.torso_calls == [], (
            f"side approach must not adjust torso, got {env.torso_calls}"
        )

    def test_act_works_without_set_torso_height(self):
        """Backward compat: legacy mocks lacking set_torso_height still work."""
        from src.action_executor import ActionExecutor
        from src.world_belief import DecomposedTask
        env = _NavCapturingEnv()  # has navigate, lacks set_torso_height
        assert not hasattr(env, "set_torso_height")
        exe = ActionExecutor(scene_describer=None)
        h, c = _hyp_with_candidate()
        c.point_3d = np.array([0.5, 0.0, 0.91])
        h.position_3d = c.point_3d
        h.grasp_candidates = [c]

        result = exe.act(h, DecomposedTask(primary_target="apple"), env)
        assert result.success is True
        assert "move_to_pre_grasp" in env.calls
