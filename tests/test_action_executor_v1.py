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

    def close_gripper(self, target_label=None, squeeze_extra_steps: int = 0) -> bool:
        self.calls.append("close")
        self.last_squeeze_extra = int(squeeze_extra_steps)
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


class _ApproachMarginRecorder(FakeEnv):
    """FakeEnv that records margin_m passed to approach()."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_margin_m: float | None = None

    def approach(self, point_3d, approach_dir, target_label=None,
                 margin_m: float = 0.015, **kwargs):
        self.last_margin_m = float(margin_m)
        return super().approach(
            point_3d, approach_dir, target_label=target_label,
            margin_m=margin_m, **kwargs,
        )


class TestAdaptiveForceParams:
    """Step 0+1: ActionExecutor must propagate GraspStrategy.squeeze_extra_steps
    to env.close_gripper and GraspStrategy.depth_margin_m to env.approach.

    Without this, LLM-derived adaptive grip force has no effect on actual
    grasp execution, which is the lemon slipped_lift root cause #3
    (decision layer not connected to control layer).
    """

    def test_squeeze_extra_steps_passed_through_to_close_gripper(self):
        from src.action_executor import ActionExecutor
        from src.world_belief import DecomposedTask, GraspStrategy
        env = _ApproachMarginRecorder()
        exe = ActionExecutor(scene_describer=None)
        h, _ = _hyp_with_candidate()
        h.grasp_strategy = GraspStrategy(
            strategy="top_down",
            mass_g=100.0,
            slip_risk="high",
            squeeze_extra_steps=18,
            depth_margin_m=0.025,
        )
        exe.act(h, DecomposedTask(primary_target="apple"), env)
        assert env.last_squeeze_extra == 18, (
            f"strategy.squeeze_extra_steps=18 must reach env.close_gripper, "
            f"got {env.last_squeeze_extra}"
        )

    def test_depth_margin_m_passed_through_to_approach(self):
        from src.action_executor import ActionExecutor
        from src.world_belief import DecomposedTask, GraspStrategy
        env = _ApproachMarginRecorder()
        exe = ActionExecutor(scene_describer=None)
        h, _ = _hyp_with_candidate()
        h.grasp_strategy = GraspStrategy(
            strategy="top_down",
            mass_g=100.0,
            slip_risk="high",
            squeeze_extra_steps=18,
            depth_margin_m=0.025,
        )
        exe.act(h, DecomposedTask(primary_target="apple"), env)
        assert env.last_margin_m is not None
        assert abs(env.last_margin_m - 0.025) < 1e-6, (
            f"strategy.depth_margin_m=0.025 must reach env.approach, "
            f"got {env.last_margin_m}"
        )

    def test_no_strategy_falls_back_to_default_margin(self):
        from src.action_executor import ActionExecutor
        from src.world_belief import DecomposedTask
        env = _ApproachMarginRecorder()
        exe = ActionExecutor(scene_describer=None)
        h, _ = _hyp_with_candidate()
        # No grasp_strategy set
        exe.act(h, DecomposedTask(primary_target="apple"), env)
        assert env.last_margin_m == 0.015  # legacy default
        assert env.last_squeeze_extra == 0  # no boost without strategy


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
        # Phase 9c safe offset: default fixture is top_down z=0.9 (<0.95)
        # but post-navigation probes showed offsets below 0.55 lock arm OSC.
        assert env.navigate_calls[0]["offset_m"] == 0.55
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
# Phase 8b: adaptive navigate offset tests
# ============================================================

class TestPhase8bAdaptiveOffset:
    """Run 8 baseline showed PandaMobile torso is UP-only ([0, 0.34]), but
    Run 9c post-navigation probes showed close offsets below 0.55m lock arm
    OSC, so grasp navigation must preserve a safe minimum offset."""

    def test_low_top_down_target_uses_safe_offset(self):
        """top_down + target z<0.95 → offset 0.55m (preserves arm control)."""
        from src.action_executor import ActionExecutor
        from src.world_belief import DecomposedTask
        env = _NavCapturingEnv()
        exe = ActionExecutor(scene_describer=None)
        h, c = _hyp_with_candidate()
        c.point_3d = np.array([0.5, 0.0, 0.91])  # counter-height object
        h.position_3d = c.point_3d
        h.grasp_candidates = [c]

        exe.act(h, DecomposedTask(primary_target="apple"), env)

        assert env.navigate_calls, "navigate_base_to must be called"
        assert env.navigate_calls[0]["offset_m"] == 0.55, (
            f"low top_down target should use offset 0.55, "
            f"got {env.navigate_calls[0]['offset_m']}"
        )

    def test_high_top_down_target_uses_safe_offset(self):
        """top_down + target z>=0.95 → offset 0.55m (minimum safe control distance)."""
        from src.action_executor import ActionExecutor
        from src.world_belief import DecomposedTask
        env = _NavCapturingEnv()
        exe = ActionExecutor(scene_describer=None)
        h, c = _hyp_with_candidate()
        c.point_3d = np.array([0.5, 0.0, 1.20])  # high cabinet object
        h.position_3d = c.point_3d
        h.grasp_candidates = [c]

        exe.act(h, DecomposedTask(primary_target="apple"), env)

        assert env.navigate_calls[0]["offset_m"] == 0.55

    def test_side_approach_uses_standard_offset_regardless_of_z(self):
        """Side / tilted approach is not vertical-reach bottlenecked, so
        the close-offset rule applies only to top_down."""
        from src.action_executor import ActionExecutor
        from src.world_belief import (
            DecomposedTask, GraspCandidate, Hypothesis,
        )
        env = _NavCapturingEnv()
        exe = ActionExecutor(scene_describer=None)
        c = GraspCandidate(
            point_3d=np.array([0.5, 0.0, 0.91]),  # low z, but side approach
            approach_dir=np.array([1.0, 0.0, 0.0]),
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

        assert env.navigate_calls[0]["offset_m"] == 0.55, (
            "side approach should use safe 0.55 even with low z"
        )

    def test_threshold_boundary_z_0p95_is_safe_offset(self):
        """Boundary check: z == 0.95 still preserves the minimum safe offset."""
        from src.action_executor import ActionExecutor
        from src.world_belief import DecomposedTask
        env = _NavCapturingEnv()
        exe = ActionExecutor(scene_describer=None)
        h, c = _hyp_with_candidate()
        c.point_3d = np.array([0.5, 0.0, 0.95])
        h.position_3d = c.point_3d
        h.grasp_candidates = [c]

        exe.act(h, DecomposedTask(primary_target="apple"), env)

        assert env.navigate_calls[0]["offset_m"] == 0.55


# ============================================================
# Phase 9d: Diagnostic pre-grasp handoff + recovery
# ============================================================

class _DiagnosticPreGraspEnv(FakeEnv):
    """Mock env exposing move_to_pre_grasp_diagnostic and recovery hooks.

    Two-call sequence: first call returns `first_*` config; second call
    (after recovery) returns `second_*` config.
    """

    def __init__(
        self,
        *,
        first_handoff_ok: bool,
        first_needs_recovery: bool = False,
        first_reason: str = "safe_handoff",
        second_handoff_ok: bool = False,
        second_needs_recovery: bool = False,
        second_reason: str = "lateral_misaligned",
        recovery_raises: bool = False,
    ):
        super().__init__()
        self._first_handoff_ok = first_handoff_ok
        self._first_needs_recovery = first_needs_recovery
        self._first_reason = first_reason
        self._second_handoff_ok = second_handoff_ok
        self._second_needs_recovery = second_needs_recovery
        self._second_reason = second_reason
        self._recovery_raises = recovery_raises
        self.pre_grasp_diag_calls = 0
        self.recovery_calls = 0

    def move_to_pre_grasp_diagnostic(self, candidate, height_m: float = 0.05):
        from types import SimpleNamespace
        self.pre_grasp_diag_calls += 1
        self.calls.append("pre_grasp_diag")
        if self.pre_grasp_diag_calls == 1:
            handoff = self._first_handoff_ok
            needs = self._first_needs_recovery
            reason = self._first_reason
        else:
            handoff = self._second_handoff_ok
            needs = self._second_needs_recovery
            reason = self._second_reason
        return SimpleNamespace(
            ok=False,
            handoff_ok=handoff,
            needs_recovery=needs,
            reason=reason,
            total_error_m=0.07,
            lateral_error_m=0.06,
            axis_error_m=0.01,
            approach_gap_m=0.04,
            lateral_limit_m=0.02,
            min_approach_gap_m=0.01,
            max_approach_gap_m=0.08,
            move_ok=False,
        )

    def recover_pre_grasp(self, candidate, prior_result):
        self.calls.append("recover_pre_grasp")
        self.recovery_calls += 1
        if self._recovery_raises:
            raise RuntimeError("recover_pre_grasp failed")
        return True


class TestDiagnosticPreGrasp:
    def test_safe_handoff_proceeds_to_approach(self):
        from src.action_executor import ActionExecutor
        from src.world_belief import DecomposedTask
        env = _DiagnosticPreGraspEnv(first_handoff_ok=True)
        exe = ActionExecutor(scene_describer=None)
        h, _ = _hyp_with_candidate()

        result = exe.act(h, DecomposedTask(primary_target="apple"), env)

        assert result.success is True
        assert env.pre_grasp_diag_calls == 1
        assert env.recovery_calls == 0
        assert any(c.startswith("approach[") for c in env.calls)

    def test_lateral_misalignment_triggers_recovery_and_succeeds(self):
        from src.action_executor import ActionExecutor
        from src.world_belief import DecomposedTask
        env = _DiagnosticPreGraspEnv(
            first_handoff_ok=False,
            first_needs_recovery=True,
            first_reason="lateral_misaligned",
            second_handoff_ok=True,
            second_reason="safe_handoff",
        )
        exe = ActionExecutor(scene_describer=None)
        h, _ = _hyp_with_candidate()

        result = exe.act(h, DecomposedTask(primary_target="apple"), env)

        assert result.success is True
        assert env.pre_grasp_diag_calls == 2
        assert env.recovery_calls == 1
        assert any(c.startswith("approach[") for c in env.calls)

    def test_lateral_misalignment_recovery_failure_reports_specific_reason(self):
        from src.action_executor import ActionExecutor
        from src.world_belief import DecomposedTask
        env = _DiagnosticPreGraspEnv(
            first_handoff_ok=False,
            first_needs_recovery=True,
            first_reason="lateral_misaligned",
            second_handoff_ok=False,
            second_needs_recovery=False,
            second_reason="lateral_misaligned",
        )
        exe = ActionExecutor(scene_describer=None)
        h, _ = _hyp_with_candidate()

        result = exe.act(h, DecomposedTask(primary_target="apple"), env)

        assert result.success is False
        assert result.attempt.failure_mode == "ik_unreachable"
        assert result.attempt.diagnostic.get("pre_grasp_reason") == "lateral_misaligned"
        assert result.attempt.diagnostic.get("stage") == "pre_grasp"
        assert not any(c.startswith("approach[") for c in env.calls)

    def test_recovery_exception_reports_base_recovery_failed(self):
        from src.action_executor import ActionExecutor
        from src.world_belief import DecomposedTask
        env = _DiagnosticPreGraspEnv(
            first_handoff_ok=False,
            first_needs_recovery=True,
            first_reason="lateral_misaligned",
            recovery_raises=True,
        )
        exe = ActionExecutor(scene_describer=None)
        h, _ = _hyp_with_candidate()

        result = exe.act(h, DecomposedTask(primary_target="apple"), env)

        assert result.success is False
        assert result.attempt.failure_mode == "ik_unreachable"
        assert result.attempt.diagnostic.get("pre_grasp_reason") == "base_recovery_failed"

    def test_no_recovery_for_axis_gap_too_small(self):
        from src.action_executor import ActionExecutor
        from src.world_belief import DecomposedTask
        env = _DiagnosticPreGraspEnv(
            first_handoff_ok=False,
            first_needs_recovery=False,
            first_reason="axis_gap_too_small",
        )
        exe = ActionExecutor(scene_describer=None)
        h, _ = _hyp_with_candidate()

        result = exe.act(h, DecomposedTask(primary_target="apple"), env)

        assert result.success is False
        assert result.attempt.failure_mode == "ik_unreachable"
        assert result.attempt.diagnostic.get("pre_grasp_reason") == "axis_gap_too_small"
        assert env.pre_grasp_diag_calls == 1
        assert env.recovery_calls == 0

    def test_legacy_bool_env_still_works(self):
        """Env without diagnostic API falls back to bool move_to_pre_grasp."""
        from src.action_executor import ActionExecutor
        from src.world_belief import DecomposedTask
        env = FakeEnv(ik_ok=True)
        exe = ActionExecutor(scene_describer=None)
        h, _ = _hyp_with_candidate()

        result = exe.act(h, DecomposedTask(primary_target="apple"), env)

        assert result.success is True
        assert "move_to_pre_grasp" in env.calls


class _BaseNudgeRecoveryEnv(FakeEnv):
    """Mock env that has nudge_base_world_xy but NOT recover_pre_grasp.

    Returns lateral_misaligned on first pre-grasp call, then safe_handoff
    on second; records the nudge_base_world_xy delta so we can verify the
    executor passed the EEF residual.
    """

    def __init__(self, *, residual_world: tuple[float, float]):
        super().__init__()
        self._residual = np.asarray(residual_world, dtype=np.float32)
        self.pre_grasp_diag_calls = 0
        self.nudge_calls: list[dict] = []
        self.navigate_calls: list[dict] = []

    def navigate_base_to(self, target_xy, offset_m: float):
        self.navigate_calls.append({
            "target_xy": np.asarray(target_xy, dtype=np.float32).copy(),
            "offset_m": float(offset_m),
        })

    def nudge_base_world_xy(self, delta_xy):
        self.nudge_calls.append({
            "delta_xy": np.asarray(delta_xy, dtype=np.float32).copy(),
        })

    def move_to_pre_grasp_diagnostic(self, candidate, height_m: float = 0.05):
        from types import SimpleNamespace
        self.pre_grasp_diag_calls += 1
        self.calls.append("pre_grasp_diag")
        grasp_point = np.asarray(candidate.point_3d, dtype=np.float32)
        pre_pos = grasp_point + np.array(
            [0.0, 0.0, 0.05], dtype=np.float32,
        )
        if self.pre_grasp_diag_calls == 1:
            # final_eef = pre_pos - residual (so pre_pos - final_eef = residual)
            final_eef = pre_pos - np.array(
                [self._residual[0], self._residual[1], 0.0],
                dtype=np.float32,
            )
            return SimpleNamespace(
                ok=False,
                handoff_ok=False,
                needs_recovery=True,
                reason="lateral_misaligned",
                final_eef=final_eef,
                pre_pos=pre_pos,
                total_error_m=float(np.linalg.norm(self._residual)),
                lateral_error_m=float(np.linalg.norm(self._residual)),
                axis_error_m=0.0,
                approach_gap_m=0.05,
                lateral_limit_m=0.02,
                min_approach_gap_m=0.01,
                max_approach_gap_m=0.08,
                move_ok=False,
            )
        # Second call: pretend the nudge cancelled the residual.
        return SimpleNamespace(
            ok=False,
            handoff_ok=True,
            needs_recovery=False,
            reason="safe_handoff",
            final_eef=pre_pos,
            pre_pos=pre_pos,
            total_error_m=0.005,
            lateral_error_m=0.005,
            axis_error_m=0.0,
            approach_gap_m=0.05,
            lateral_limit_m=0.02,
            min_approach_gap_m=0.01,
            max_approach_gap_m=0.08,
            move_ok=False,
        )


class TestBaseNudgeRecovery:
    def test_recovery_nudges_base_by_eef_residual(self):
        """Recovery should call nudge_base_world_xy with delta = pre_pos - final_eef.

        This is the world-frame XY residual; translating the base by this
        amount rigidly moves the EEF to pre_pos, bypassing both arm OSC
        saturation and base controller friction. Unlike navigate_base_to,
        no direction recompute happens — yaw and approach line preserved.
        """
        from src.action_executor import ActionExecutor
        from src.world_belief import DecomposedTask
        residual = (0.01, 0.06)  # 1cm +x, 6cm +y in world
        env = _BaseNudgeRecoveryEnv(residual_world=residual)
        exe = ActionExecutor(scene_describer=None)
        h, _ = _hyp_with_candidate()

        result = exe.act(h, DecomposedTask(primary_target="apple"), env)

        assert result.success is True
        assert len(env.nudge_calls) == 1
        delta = env.nudge_calls[0]["delta_xy"]
        # delta should be exactly the residual (no clipping for 6cm).
        assert np.allclose(delta, np.asarray(residual, dtype=np.float32), atol=1e-5), (
            f"expected nudge delta {residual}, got {delta}"
        )

    def test_recovery_nudge_clipped_at_max(self):
        """Residual larger than _MAX_LATERAL_NUDGE_M gets clipped, direction preserved."""
        from src.action_executor import ActionExecutor, ActionExecutor as _AE
        from src.world_belief import DecomposedTask
        # Pure +y residual of 0.25m → clipped to MAX (0.10m).
        residual = (0.0, 0.25)
        env = _BaseNudgeRecoveryEnv(residual_world=residual)
        exe = ActionExecutor(scene_describer=None)
        h, _ = _hyp_with_candidate()

        exe.act(h, DecomposedTask(primary_target="apple"), env)

        assert len(env.nudge_calls) == 1
        delta = env.nudge_calls[0]["delta_xy"]
        assert abs(float(delta[0])) < 1e-5
        assert abs(float(delta[1]) - _AE._MAX_LATERAL_NUDGE_M) < 1e-5, (
            f"expected clipped delta_y={_AE._MAX_LATERAL_NUDGE_M}, got {delta[1]}"
        )

    def test_recovery_falls_through_to_navigate_when_no_nudge_primitive(self):
        """Env without nudge_base_world_xy falls back to navigate_base_to."""
        from src.action_executor import ActionExecutor
        from src.world_belief import DecomposedTask

        class _NoNudgeEnv(FakeEnv):
            def __init__(self):
                super().__init__()
                self.pre_grasp_diag_calls = 0
                self.navigate_calls: list[dict] = []

            def navigate_base_to(self, target_xy, offset_m: float):
                self.navigate_calls.append({
                    "target_xy": np.asarray(target_xy, dtype=np.float32).copy(),
                    "offset_m": float(offset_m),
                })

            def move_to_pre_grasp_diagnostic(self, candidate, height_m=0.05):
                from types import SimpleNamespace
                self.pre_grasp_diag_calls += 1
                self.calls.append("pre_grasp_diag")
                if self.pre_grasp_diag_calls == 1:
                    return SimpleNamespace(
                        ok=False, handoff_ok=False, needs_recovery=True,
                        reason="lateral_misaligned",
                        total_error_m=0.06, lateral_error_m=0.06,
                        axis_error_m=0.0, approach_gap_m=0.05,
                        lateral_limit_m=0.02, min_approach_gap_m=0.01,
                        max_approach_gap_m=0.08, move_ok=False,
                    )
                return SimpleNamespace(
                    ok=False, handoff_ok=True, needs_recovery=False,
                    reason="safe_handoff",
                    total_error_m=0.005, lateral_error_m=0.005,
                    axis_error_m=0.0, approach_gap_m=0.05,
                    lateral_limit_m=0.02, min_approach_gap_m=0.01,
                    max_approach_gap_m=0.08, move_ok=False,
                )

        env = _NoNudgeEnv()
        exe = ActionExecutor(scene_describer=None)
        h, c = _hyp_with_candidate()
        result = exe.act(h, DecomposedTask(primary_target="apple"), env)

        assert result.success is True
        # Two navigate calls: act-entry (0.55) and recovery fallback (0.65).
        assert len(env.navigate_calls) == 2
        assert env.navigate_calls[1]["offset_m"] == 0.65

    def test_recover_pre_grasp_hook_overrides_nudge(self):
        """If env defines recover_pre_grasp, it takes precedence over nudge."""
        from src.action_executor import ActionExecutor
        from src.world_belief import DecomposedTask
        env = _DiagnosticPreGraspEnv(
            first_handoff_ok=False,
            first_needs_recovery=True,
            first_reason="lateral_misaligned",
            second_handoff_ok=True,
            second_reason="safe_handoff",
        )
        exe = ActionExecutor(scene_describer=None)
        h, _ = _hyp_with_candidate()

        result = exe.act(h, DecomposedTask(primary_target="apple"), env)

        assert result.success is True
        assert env.recovery_calls == 1  # caller hook used

    def test_recovery_uses_evaluate_pre_grasp_at_current_when_available(self):
        """After nudge, env.evaluate_pre_grasp_at_current is preferred over
        re-running move_to_pre_grasp_diagnostic.

        Regression for GPU run c2e4ab1: re-running move_to_pre_grasp_diagnostic
        triggers a second move_arm_to with orientation control that physically
        drifts the EEF from ~24mm back to ~62mm, undoing the nudge. The
        diagnostic shortcut evaluates geometry from the current (post-nudge)
        EEF without commanding any arm motion.
        """
        from types import SimpleNamespace
        from src.action_executor import ActionExecutor
        from src.world_belief import DecomposedTask

        class _NudgeAndShortcutEnv(FakeEnv):
            def __init__(self):
                super().__init__()
                self.diag_calls = 0
                self.eval_calls = 0
                self.nudge_calls: list[dict] = []

            def navigate_base_to(self, target_xy, offset_m: float):
                pass

            def nudge_base_world_xy(self, delta_xy):
                self.nudge_calls.append({
                    "delta_xy": np.asarray(delta_xy, dtype=np.float32).copy(),
                })

            def move_to_pre_grasp_diagnostic(
                self, candidate, height_m: float = 0.05,
            ):
                self.diag_calls += 1
                self.calls.append("pre_grasp_diag")
                grasp_point = np.asarray(
                    candidate.point_3d, dtype=np.float32,
                )
                pre_pos = grasp_point + np.array(
                    [0.0, 0.0, 0.05], dtype=np.float32,
                )
                # Always lateral_misaligned on initial diag call.
                final_eef = pre_pos - np.array(
                    [0.06, 0.0, 0.0], dtype=np.float32,
                )
                return SimpleNamespace(
                    ok=False, handoff_ok=False, needs_recovery=True,
                    reason="lateral_misaligned",
                    final_eef=final_eef, pre_pos=pre_pos,
                    total_error_m=0.06, lateral_error_m=0.06,
                    axis_error_m=0.0, approach_gap_m=0.05,
                    lateral_limit_m=0.02,
                    min_approach_gap_m=0.01, max_approach_gap_m=0.08,
                    move_ok=False,
                )

            def evaluate_pre_grasp_at_current(
                self, candidate, height_m: float = 0.05,
            ):
                self.eval_calls += 1
                self.calls.append("pre_grasp_eval")
                grasp_point = np.asarray(
                    candidate.point_3d, dtype=np.float32,
                )
                pre_pos = grasp_point + np.array(
                    [0.0, 0.0, 0.05], dtype=np.float32,
                )
                # Post-nudge: lateral within limit → safe_handoff.
                return SimpleNamespace(
                    ok=False, handoff_ok=True, needs_recovery=False,
                    reason="safe_handoff",
                    final_eef=pre_pos, pre_pos=pre_pos,
                    total_error_m=0.005, lateral_error_m=0.005,
                    axis_error_m=0.0, approach_gap_m=0.05,
                    lateral_limit_m=0.02,
                    min_approach_gap_m=0.01, max_approach_gap_m=0.08,
                    move_ok=False,
                )

        env = _NudgeAndShortcutEnv()
        exe = ActionExecutor(scene_describer=None)
        h, _ = _hyp_with_candidate()
        result = exe.act(h, DecomposedTask(primary_target="apple"), env)

        assert result.success is True
        # Initial strict move ran exactly once.
        assert env.diag_calls == 1
        # Recovery used the no-motion evaluator instead of re-running diag.
        assert env.eval_calls == 1
        # Exactly one nudge to recover.
        assert len(env.nudge_calls) == 1

    def test_iterative_nudge_progressively_reduces_lateral(self):
        """Multiple nudges reduce lateral residual until safe_handoff.

        Real-world nudge accuracy is ~60-70 % per iteration due to obs sync
        and OSC settling. Initial 64mm lateral may need 2-3 nudges to drop
        below 20mm lateral_limit. Verify the executor iterates instead of
        bailing out after the first attempt.
        """
        from types import SimpleNamespace
        from src.action_executor import ActionExecutor
        from src.world_belief import DecomposedTask

        class _IterativeNudgeEnv(FakeEnv):
            # Lateral residuals across iterations: 64 → 24 → 9 → 3 mm.
            _LATERALS_M = [0.064, 0.024, 0.009, 0.003]

            def __init__(self):
                super().__init__()
                self.eval_iter = 0
                self.diag_calls = 0
                self.nudge_calls: list[dict] = []

            def navigate_base_to(self, target_xy, offset_m: float):
                pass

            def nudge_base_world_xy(self, delta_xy):
                self.nudge_calls.append({
                    "delta_xy": np.asarray(delta_xy, dtype=np.float32).copy(),
                })

            def _build_result(self, lateral_m: float, candidate):
                grasp_point = np.asarray(
                    candidate.point_3d, dtype=np.float32,
                )
                pre_pos = grasp_point + np.array(
                    [0.0, 0.0, 0.05], dtype=np.float32,
                )
                # Place EEF lateral_m off in +x.
                final_eef = pre_pos - np.array(
                    [lateral_m, 0.0, 0.0], dtype=np.float32,
                )
                lateral_limit = 0.02
                handoff_ok = lateral_m <= lateral_limit
                return SimpleNamespace(
                    ok=False,
                    handoff_ok=handoff_ok,
                    needs_recovery=not handoff_ok,
                    reason=(
                        "safe_handoff" if handoff_ok
                        else "lateral_misaligned"
                    ),
                    final_eef=final_eef, pre_pos=pre_pos,
                    total_error_m=lateral_m,
                    lateral_error_m=lateral_m,
                    axis_error_m=0.0, approach_gap_m=0.05,
                    lateral_limit_m=lateral_limit,
                    min_approach_gap_m=0.01, max_approach_gap_m=0.08,
                    move_ok=False,
                )

            def move_to_pre_grasp_diagnostic(
                self, candidate, height_m: float = 0.05,
            ):
                self.diag_calls += 1
                self.calls.append("pre_grasp_diag")
                lateral = self._LATERALS_M[self.eval_iter]
                self.eval_iter += 1
                return self._build_result(lateral, candidate)

            def evaluate_pre_grasp_at_current(
                self, candidate, height_m: float = 0.05,
            ):
                self.calls.append("pre_grasp_eval")
                lateral = self._LATERALS_M[
                    min(self.eval_iter, len(self._LATERALS_M) - 1)
                ]
                self.eval_iter += 1
                return self._build_result(lateral, candidate)

        env = _IterativeNudgeEnv()
        exe = ActionExecutor(scene_describer=None)
        h, _ = _hyp_with_candidate()
        result = exe.act(h, DecomposedTask(primary_target="apple"), env)

        assert result.success is True
        # 64 → 24 → 9 mm: needs 2 nudges (9mm is below limit).
        assert len(env.nudge_calls) == 2
        # Initial diag (1) + 2 evals after each nudge.
        assert env.diag_calls == 1
        assert env.eval_iter == 3


class _ZStallNudgeEnv(FakeEnv):
    """Simulates z-stall during descent → verifies that z-stall recovery
    uses nudge_base_world_xy instead of move_arm_to(drive_base=False).

    First approach call returns descend_ok=False (z-stall), second succeeds
    (after base nudge puts arm in better workspace region).
    """
    def __init__(self):
        super().__init__(descend_ok=False, lift_ok=True, obj_lifts=True)
        self.nudge_xy_calls: list[np.ndarray] = []
        self._approach_count = 0

    def approach(self, point_3d, approach_dir, target_label=None, **kwargs):
        self._approach_count += 1
        if self._approach_count <= 1:
            return False, float(point_3d[2]) + 0.03  # z-stall
        return True, float(point_3d[2])  # after nudge, descent works

    def nudge_base_world_xy(self, delta_xy):
        self.nudge_xy_calls.append(np.asarray(delta_xy).copy())
        return True

    def get_base_pose(self):
        # Base at 0.5m from object on x-axis
        return np.array([1.0, 0.0, 0.0]), np.eye(3, dtype=np.float32)


class TestZStallRecoveryUsesBaseNudge:
    def test_z_stall_recovery_uses_nudge_base_world_xy(self):
        """z-stall recovery must use nudge_base_world_xy to actually move the
        base closer, NOT move_arm_to(drive_base=False) which only extends the
        arm and makes the workspace worse."""
        from src.action_executor import ActionExecutor
        from src.world_belief import DecomposedTask
        exe = ActionExecutor(scene_describer=None)
        env = _ZStallNudgeEnv()
        h, _ = _hyp_with_candidate()
        result = exe.act(h, DecomposedTask(primary_target="apple"), env)

        assert len(env.nudge_xy_calls) >= 1, (
            "z-stall recovery should use nudge_base_world_xy, "
            f"but no nudge calls recorded. calls={env.calls}"
        )
        # nudge direction should be toward the object (positive x direction
        # because object is at x=0.5 and base is at x=1.0 ... actually
        # object is at x=0.5, base at x=1.0 → direction is negative)
        nudge = env.nudge_xy_calls[0]
        assert nudge[0] < 0, (
            f"nudge should be toward object (negative x), got {nudge}"
        )
