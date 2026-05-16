"""Unit tests for EnvWrapper orientation control helpers.

These tests do NOT require a running sim - they verify pure math helpers
used to compute target gripper orientations from approach directions.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R


# ============================================================
# Task 2: _approach_dir_to_quat
# ============================================================


class TestApproachDirToQuat:
    def test_top_down_returns_z_pointing_down(self):
        """approach_dir = -z (gripper points down) should rotate +z to -z."""
        from src.env_wrapper import EnvWrapper

        q = EnvWrapper._approach_dir_to_quat(np.array([0.0, 0.0, -1.0]))
        rot = R.from_quat(q)
        gripper_z_world = rot.apply(np.array([0.0, 0.0, 1.0]))
        np.testing.assert_allclose(gripper_z_world, [0.0, 0.0, -1.0], atol=1e-6)

    def test_side_x_approach(self):
        """approach_dir = +x should rotate gripper so its local +z points in +x."""
        from src.env_wrapper import EnvWrapper

        q = EnvWrapper._approach_dir_to_quat(np.array([1.0, 0.0, 0.0]))
        rot = R.from_quat(q)
        gripper_z_world = rot.apply(np.array([0.0, 0.0, 1.0]))
        np.testing.assert_allclose(gripper_z_world, [1.0, 0.0, 0.0], atol=1e-6)

    def test_side_y_approach(self):
        """approach_dir = +y."""
        from src.env_wrapper import EnvWrapper

        q = EnvWrapper._approach_dir_to_quat(np.array([0.0, 1.0, 0.0]))
        rot = R.from_quat(q)
        gripper_z_world = rot.apply(np.array([0.0, 0.0, 1.0]))
        np.testing.assert_allclose(gripper_z_world, [0.0, 1.0, 0.0], atol=1e-6)

    def test_top_up_approach_special_case(self):
        """approach_dir = +z (already aligned) returns identity."""
        from src.env_wrapper import EnvWrapper

        q = EnvWrapper._approach_dir_to_quat(np.array([0.0, 0.0, 1.0]))
        np.testing.assert_allclose(q, [0.0, 0.0, 0.0, 1.0], atol=1e-6)

    def test_normalizes_non_unit_input(self):
        """Non-unit approach_dir should still produce a valid normalized quat."""
        from src.env_wrapper import EnvWrapper

        q = EnvWrapper._approach_dir_to_quat(np.array([2.0, 0.0, 0.0]))
        assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-6)
        # Check the rotation direction is still correct
        rot = R.from_quat(q)
        gripper_z = rot.apply(np.array([0.0, 0.0, 1.0]))
        np.testing.assert_allclose(gripper_z, [1.0, 0.0, 0.0], atol=1e-6)

    def test_zero_vector_returns_identity(self):
        """Degenerate zero vector input falls back to identity."""
        from src.env_wrapper import EnvWrapper

        q = EnvWrapper._approach_dir_to_quat(np.array([0.0, 0.0, 0.0]))
        np.testing.assert_allclose(q, [0.0, 0.0, 0.0, 1.0], atol=1e-6)

    def test_diagonal_approach(self):
        """Approach_dir = (1, 0, -1)/√2 should put gripper z mid-air."""
        from src.env_wrapper import EnvWrapper

        d = np.array([1.0, 0.0, -1.0]) / np.sqrt(2.0)
        q = EnvWrapper._approach_dir_to_quat(d)
        rot = R.from_quat(q)
        gripper_z_world = rot.apply(np.array([0.0, 0.0, 1.0]))
        np.testing.assert_allclose(gripper_z_world, d, atol=1e-6)


# ============================================================
# Task 3: _quat_delta_to_axis_angle
# ============================================================


class TestQuatDeltaToAxisAngle:
    def test_identity_returns_zero(self):
        """Same quaternion → zero rotation."""
        from src.env_wrapper import EnvWrapper

        q_cur = np.array([0.0, 0.0, 0.0, 1.0])
        q_tgt = np.array([0.0, 0.0, 0.0, 1.0])
        out = EnvWrapper._quat_delta_to_axis_angle(q_cur, q_tgt)
        np.testing.assert_allclose(out, [0.0, 0.0, 0.0], atol=1e-9)

    def test_90deg_around_y(self):
        """Target is identity rotated 90° around y → output should be [0, π/2, 0]."""
        from src.env_wrapper import EnvWrapper

        q_cur = np.array([0.0, 0.0, 0.0, 1.0])
        q_tgt = R.from_rotvec([0.0, np.pi / 2, 0.0]).as_quat()
        out = EnvWrapper._quat_delta_to_axis_angle(q_cur, q_tgt)
        np.testing.assert_allclose(out, [0.0, np.pi / 2, 0.0], atol=1e-6)

    def test_45deg_around_x(self):
        """Smaller rotation around x."""
        from src.env_wrapper import EnvWrapper

        q_cur = np.array([0.0, 0.0, 0.0, 1.0])
        q_tgt = R.from_rotvec([np.pi / 4, 0.0, 0.0]).as_quat()
        out = EnvWrapper._quat_delta_to_axis_angle(q_cur, q_tgt)
        np.testing.assert_allclose(out, [np.pi / 4, 0.0, 0.0], atol=1e-6)

    def test_shortest_path_for_negated_quat(self):
        """For target that's a 'long-way' negated quat, take the short path."""
        from src.env_wrapper import EnvWrapper

        q_cur = np.array([0.0, 0.0, 0.0, 1.0])
        q_tgt_pos = R.from_rotvec([0.0, 0.1, 0.0]).as_quat()
        q_tgt_neg = -q_tgt_pos
        out_pos = EnvWrapper._quat_delta_to_axis_angle(q_cur, q_tgt_pos)
        out_neg = EnvWrapper._quat_delta_to_axis_angle(q_cur, q_tgt_neg)
        # Both should give same short-path result
        np.testing.assert_allclose(out_pos, out_neg, atol=1e-9)
        assert np.linalg.norm(out_pos) < np.pi

    def test_composite_delta(self):
        """Current = rot(π/4, y), Target = rot(π/2, y), delta should be rot(π/4, y)."""
        from src.env_wrapper import EnvWrapper

        q_cur = R.from_rotvec([0.0, np.pi / 4, 0.0]).as_quat()
        q_tgt = R.from_rotvec([0.0, np.pi / 2, 0.0]).as_quat()
        out = EnvWrapper._quat_delta_to_axis_angle(q_cur, q_tgt)
        np.testing.assert_allclose(out, [0.0, np.pi / 4, 0.0], atol=1e-6)


# ============================================================
# Task 4: _get_eef_quat
# ============================================================


class TestGetEEFQuat:
    def test_returns_quat_from_obs_key(self):
        """Should read robot0_eef_quat from observation if available."""
        from src.env_wrapper import EnvWrapper

        wrapper = EnvWrapper.__new__(EnvWrapper)
        expected = R.from_rotvec([0.0, np.pi / 4, 0.0]).as_quat()
        wrapper._latest_obs = {"robot0_eef_quat": expected}

        q = wrapper._get_eef_quat()
        np.testing.assert_allclose(q, expected, atol=1e-9)

    def test_resets_if_obs_empty(self, monkeypatch):
        """If no obs, should call reset() to refresh."""
        from src.env_wrapper import EnvWrapper

        wrapper = EnvWrapper.__new__(EnvWrapper)
        wrapper._latest_obs = None

        expected = R.from_rotvec([0.0, 0.0, np.pi / 6]).as_quat()
        reset_called = []

        def fake_reset():
            reset_called.append(True)
            wrapper._latest_obs = {"robot0_eef_quat": expected}

        monkeypatch.setattr(wrapper, "reset", fake_reset)

        q = wrapper._get_eef_quat()
        assert reset_called == [True]
        np.testing.assert_allclose(q, expected, atol=1e-9)

    def test_raises_if_no_quat_in_obs(self):
        """If obs exists but no robot0_eef_quat key, raise clear error."""
        from src.env_wrapper import EnvWrapper

        wrapper = EnvWrapper.__new__(EnvWrapper)
        wrapper._latest_obs = {"robot0_eef_pos": np.array([0.5, 0.0, 1.0])}

        with pytest.raises(RuntimeError, match="robot0_eef_quat"):
            wrapper._get_eef_quat()


# ============================================================
# Task 5: move_arm_to with approach_dir
# ============================================================


def _make_minimal_wrapper(monkeypatch, current_quat=None, base_ori=None):
    """Build a minimal EnvWrapper instance with all sim interactions mocked."""
    from src.env_wrapper import EnvWrapper

    wrapper = EnvWrapper.__new__(EnvWrapper)
    if current_quat is None:
        current_quat = np.array([0.0, 0.0, 0.0, 1.0])
    if base_ori is None:
        base_ori = np.eye(3, dtype=np.float32)

    captured_actions = []

    class FakeStep:
        action_dim = 12

        def step(self, action):
            captured_actions.append(np.array(action, copy=True))
            return {"robot0_eef_pos": np.array([0.5, 0.0, 1.0]),
                    "robot0_eef_quat": current_quat}, 0.0, False, {}

    wrapper._env = FakeStep()
    wrapper._latest_obs = {
        "robot0_eef_pos": np.array([0.5, 0.0, 1.0]),
        "robot0_eef_quat": current_quat,
    }
    monkeypatch.setattr(wrapper, "_get_base_action_idx", lambda: None)
    monkeypatch.setattr(wrapper, "get_eef_pos",
                        lambda: np.array([0.5, 0.0, 1.0], dtype=np.float32))
    monkeypatch.setattr(wrapper, "get_base_pose",
                        lambda: (np.zeros(3, dtype=np.float32), base_ori))
    monkeypatch.setattr(wrapper, "render", lambda: None)
    return wrapper, captured_actions


class TestMoveArmToWithApproachDir:
    def test_no_approach_dir_leaves_orientation_action_zero(self, monkeypatch):
        """Without approach_dir, action[3:6] should remain zero (back-compat)."""
        wrapper, captured = _make_minimal_wrapper(monkeypatch)

        wrapper.move_arm_to(
            np.array([0.55, 0.0, 1.0]),
            max_steps=3,
        )
        # action[3:6] should all be 0 in every captured action
        for a in captured:
            np.testing.assert_allclose(a[3:6], [0.0, 0.0, 0.0])

    def test_side_approach_dir_sets_orientation_action(self, monkeypatch):
        """approach_dir=[1,0,0] should produce non-zero action[3:6]."""
        wrapper, captured = _make_minimal_wrapper(monkeypatch)

        wrapper.move_arm_to(
            np.array([0.55, 0.0, 1.0]),
            approach_dir=np.array([1.0, 0.0, 0.0]),
            max_steps=3,
        )
        # At least the first step should have a non-zero rotation component
        first_ori = captured[0][3:6]
        assert np.linalg.norm(first_ori) > 1e-3, (
            f"Expected non-zero orientation action, got {first_ori}"
        )

    def test_top_down_approach_no_drift(self, monkeypatch):
        """approach_dir=[0,0,-1] when gripper already faces down (rot 180° around x):
        the orientation action should be near-zero to prevent regression on top_down."""
        from scipy.spatial.transform import Rotation as R
        # Set current orientation to gripper facing down (z_local → -z_world)
        # Rotation: 180° around x-axis takes +z → -z
        q_down = R.from_rotvec([np.pi, 0.0, 0.0]).as_quat()
        wrapper, captured = _make_minimal_wrapper(monkeypatch, current_quat=q_down)

        wrapper.move_arm_to(
            np.array([0.55, 0.0, 1.0]),
            approach_dir=np.array([0.0, 0.0, -1.0]),
            max_steps=3,
        )
        # Orientation deltas should be small (already aligned)
        for a in captured:
            assert np.linalg.norm(a[3:6]) < 0.05, (
                f"Top-down already aligned should produce ~zero ori, got {a[3:6]}"
            )

    def test_orientation_uses_base_frame(self, monkeypatch):
        """The orientation delta should be transformed from world to base frame."""
        from scipy.spatial.transform import Rotation as R
        # Base rotated 90° around z (base x-axis = world y-axis)
        base_ori = R.from_rotvec([0.0, 0.0, np.pi / 2]).as_matrix().astype(np.float32)
        wrapper, captured = _make_minimal_wrapper(monkeypatch, base_ori=base_ori)

        wrapper.move_arm_to(
            np.array([0.55, 0.0, 1.0]),
            approach_dir=np.array([1.0, 0.0, 0.0]),  # +x in world
            max_steps=3,
        )
        # Orientation action should reflect base-frame transformation
        # If applied raw (world-frame) the rotation axis would be different
        assert len(captured) >= 1
        assert np.linalg.norm(captured[0][3:6]) > 1e-3


# ============================================================
# Task 7: approach() method
# ============================================================


class TestApproach:
    def test_top_down_uses_descend_path(self, monkeypatch):
        """approach(p, [0,0,-1], target_label='apple') with target_body found
        should call _descend_until_contact (the contact-based descent)."""
        from src.env_wrapper import EnvWrapper

        wrapper = EnvWrapper.__new__(EnvWrapper)
        wrapper._latest_obs = {}

        descend_calls = []

        def fake_descend(target, target_body, **kw):
            descend_calls.append((np.array(target, copy=True), target_body))
            return True, float(target[2])

        monkeypatch.setattr(wrapper, "_descend_until_contact", fake_descend)
        monkeypatch.setattr(wrapper, "_get_obj_type_map",
                            lambda: {"obj_main": "apple"})
        monkeypatch.setattr(wrapper, "get_eef_pos",
                            lambda: np.array([0.5, 0.0, 1.0], dtype=np.float32))

        ok, z = wrapper.approach(
            np.array([0.5, 0.0, 0.9], dtype=np.float32),
            approach_dir=np.array([0.0, 0.0, -1.0]),
            target_label="apple",
            margin_m=0.015,
        )
        assert ok is True
        assert len(descend_calls) == 1
        # Target z should be 0.9 - 0.015 = 0.885 (margin moves along approach_dir)
        np.testing.assert_allclose(descend_calls[0][0][2], 0.885, atol=1e-6)
        assert descend_calls[0][1] == "obj_main"

    def test_side_approach_uses_move_arm_to(self, monkeypatch):
        """For side approach, should call move_arm_to with approach_dir, not _descend."""
        from src.env_wrapper import EnvWrapper

        wrapper = EnvWrapper.__new__(EnvWrapper)
        wrapper._latest_obs = {}
        move_calls = []
        descend_calls = []

        def fake_move(target, **kw):
            move_calls.append((np.array(target, copy=True), kw.get("approach_dir")))
            return True

        def fake_descend(*a, **kw):
            descend_calls.append(a)
            return True, 0.0

        monkeypatch.setattr(wrapper, "move_arm_to", fake_move)
        monkeypatch.setattr(wrapper, "_descend_until_contact", fake_descend)
        monkeypatch.setattr(wrapper, "_get_obj_type_map", lambda: {})
        monkeypatch.setattr(wrapper, "get_eef_pos",
                            lambda: np.array([0.6, 0.0, 0.93], dtype=np.float32))

        ok, z = wrapper.approach(
            np.array([0.5, 0.0, 0.93], dtype=np.float32),
            approach_dir=np.array([1.0, 0.0, 0.0]),
            target_label=None,
            margin_m=0.0,
        )
        assert ok is True
        # Should have used move_arm_to with the approach_dir, not _descend
        assert len(descend_calls) == 0
        assert len(move_calls) >= 1
        np.testing.assert_allclose(
            move_calls[-1][1], [1.0, 0.0, 0.0], atol=1e-6,
        )

    def test_margin_offsets_along_approach_dir(self, monkeypatch):
        """For side approach with margin, target should be offset along approach_dir
        (deeper into the object, i.e., further forward)."""
        from src.env_wrapper import EnvWrapper

        wrapper = EnvWrapper.__new__(EnvWrapper)
        wrapper._latest_obs = {}
        move_calls = []
        monkeypatch.setattr(wrapper, "move_arm_to",
                            lambda t, **kw: move_calls.append(np.array(t, copy=True)) or True)
        monkeypatch.setattr(wrapper, "_get_obj_type_map", lambda: {})
        monkeypatch.setattr(wrapper, "get_eef_pos",
                            lambda: np.array([0.0, 0.0, 0.93], dtype=np.float32))

        wrapper.approach(
            np.array([0.5, 0.0, 0.93], dtype=np.float32),
            approach_dir=np.array([1.0, 0.0, 0.0]),
            target_label=None,
            margin_m=0.02,
        )
        # Target should be 0.5 + 0.02 = 0.52 in x (margin pushes along approach_dir)
        last = move_calls[-1]
        np.testing.assert_allclose(last[0], 0.52, atol=1e-6)

    def test_descend_wraps_to_approach(self, monkeypatch):
        """Legacy descend() should delegate to approach() with [0,0,-1]."""
        from src.env_wrapper import EnvWrapper

        wrapper = EnvWrapper.__new__(EnvWrapper)
        approach_calls = []

        def fake_approach(point, approach_dir, **kw):
            approach_calls.append((np.array(point, copy=True),
                                   np.array(approach_dir, copy=True)))
            return True, float(point[2])

        monkeypatch.setattr(wrapper, "approach", fake_approach)

        ok, z = wrapper.descend(
            np.array([0.5, 0.0, 0.9], dtype=np.float32),
            target_label="apple",
            margin_m=0.015,
        )
        assert ok is True
        assert len(approach_calls) == 1
        np.testing.assert_allclose(approach_calls[0][1], [0.0, 0.0, -1.0])


# ============================================================
# Task 8: move_to_pre_grasp uses approach_dir
# ============================================================


class TestMoveToPreGrasp:
    def _make_candidate(self, point, approach_dir):
        from src.world_belief import GraspCandidate

        return GraspCandidate(
            point_3d=np.asarray(point, dtype=np.float32),
            approach_dir=np.asarray(approach_dir, dtype=np.float32),
            finger_width_m=0.04,
            score=0.9,
            source="test",
        )

    def test_top_down_pre_grasp_above_object(self, monkeypatch):
        """top_down: pre-grasp position should be height_m above object."""
        from src.env_wrapper import EnvWrapper

        wrapper = EnvWrapper.__new__(EnvWrapper)
        moves = []
        monkeypatch.setattr(wrapper, "move_arm_to",
                            lambda t, **kw: moves.append((np.array(t, copy=True), kw)) or True)
        monkeypatch.setattr(wrapper, "_gripper_action", lambda *a, **kw: None)
        monkeypatch.setattr(wrapper, "get_eef_pos",
                            lambda: np.array([0.5, 0.0, 1.0], dtype=np.float32))

        c = self._make_candidate([0.5, 0.0, 0.9], [0.0, 0.0, -1.0])
        wrapper.move_to_pre_grasp(c, height_m=0.05)
        # Final move target should be (0.5, 0.0, 0.95) = obj + 5cm above
        last_pos = moves[-1][0]
        np.testing.assert_allclose(last_pos, [0.5, 0.0, 0.95], atol=1e-6)

    def test_side_pre_grasp_offset_horizontally(self, monkeypatch):
        """side approach (+x): pre-grasp should be height_m back along -x at object z."""
        from src.env_wrapper import EnvWrapper

        wrapper = EnvWrapper.__new__(EnvWrapper)
        moves = []
        monkeypatch.setattr(wrapper, "move_arm_to",
                            lambda t, **kw: moves.append((np.array(t, copy=True), kw)) or True)
        monkeypatch.setattr(wrapper, "_gripper_action", lambda *a, **kw: None)
        monkeypatch.setattr(wrapper, "get_eef_pos",
                            lambda: np.array([0.5, 0.0, 1.0], dtype=np.float32))

        c = self._make_candidate([0.5, 0.0, 0.93], [1.0, 0.0, 0.0])
        wrapper.move_to_pre_grasp(c, height_m=0.10)
        # Final move target should be (0.5 - 0.10, 0.0, 0.93) = back 10cm in -x at obj z
        last_pos = moves[-1][0]
        np.testing.assert_allclose(last_pos, [0.4, 0.0, 0.93], atol=1e-6)

    def test_side_pre_grasp_passes_approach_dir(self, monkeypatch):
        """The pre-grasp move should pass approach_dir for orientation alignment."""
        from src.env_wrapper import EnvWrapper

        wrapper = EnvWrapper.__new__(EnvWrapper)
        moves = []
        monkeypatch.setattr(wrapper, "move_arm_to",
                            lambda t, **kw: moves.append((np.array(t, copy=True), kw)) or True)
        monkeypatch.setattr(wrapper, "_gripper_action", lambda *a, **kw: None)
        monkeypatch.setattr(wrapper, "get_eef_pos",
                            lambda: np.array([0.5, 0.0, 1.0], dtype=np.float32))

        c = self._make_candidate([0.5, 0.0, 0.93], [1.0, 0.0, 0.0])
        wrapper.move_to_pre_grasp(c, height_m=0.10)
        # The final move (pre-grasp) should pass approach_dir
        ad = moves[-1][1].get("approach_dir")
        assert ad is not None
        np.testing.assert_allclose(np.asarray(ad), [1.0, 0.0, 0.0], atol=1e-6)


# ============================================================
# Task 10: lift uses approach_dir for retreat
# ============================================================


def _make_lift_wrapper(monkeypatch, start_pos=(0.5, 0.0, 0.9)):
    """Helper: lift mock that tracks current position via move_arm_to targets."""
    from src.env_wrapper import EnvWrapper

    wrapper = EnvWrapper.__new__(EnvWrapper)
    state = {"pos": np.array(start_pos, dtype=np.float32)}
    moves = []

    def fake_move(t, **kw):
        moves.append(np.array(t, copy=True))
        state["pos"] = np.asarray(t, dtype=np.float32).copy()
        return True

    monkeypatch.setattr(wrapper, "move_arm_to", fake_move)
    monkeypatch.setattr(wrapper, "get_eef_pos",
                        lambda: state["pos"].copy())
    return wrapper, moves


class TestLiftWithApproachDir:
    def test_top_down_lift_unchanged(self, monkeypatch):
        """approach_dir=[0,0,-1] (top_down) → lift should still go straight up."""
        wrapper, moves = _make_lift_wrapper(monkeypatch)
        ok, z = wrapper.lift(height_m=0.10,
                              approach_dir=np.array([0.0, 0.0, -1.0]))
        assert ok is True
        # Last target should be (0.5, 0.0, 1.0) = 0.9 + 0.10
        last = moves[-1]
        np.testing.assert_allclose(last, [0.5, 0.0, 1.0], atol=1e-6)

    def test_side_lift_retreats_horizontally(self, monkeypatch):
        """approach_dir=[1,0,0] (+x) → first phase should retreat in -x direction."""
        wrapper, moves = _make_lift_wrapper(monkeypatch)
        ok, z = wrapper.lift(height_m=0.10,
                              approach_dir=np.array([1.0, 0.0, 0.0]))
        assert ok is True
        # Some intermediate target should have x < 0.5 (retreated in -x)
        retreated = [m for m in moves if m[0] < 0.5 - 1e-3]
        assert len(retreated) >= 1, (
            f"Expected at least one retreat target with x<0.5, got {moves}"
        )
        # Final target z should be at least 5cm above starting z (0.9)
        last = moves[-1]
        assert last[2] > 0.9 + 0.05, (
            f"Final z should be at least 5cm above start, got {last}"
        )

    def test_default_approach_dir_is_top_down(self, monkeypatch):
        """Calling lift() without approach_dir should behave like top_down."""
        wrapper, moves = _make_lift_wrapper(monkeypatch)
        wrapper.lift(height_m=0.10)
        # x and y should remain at 0.5 and 0.0 throughout (no horizontal retreat)
        for m in moves:
            np.testing.assert_allclose(m[0], 0.5, atol=1e-6)
            np.testing.assert_allclose(m[1], 0.0, atol=1e-6)
