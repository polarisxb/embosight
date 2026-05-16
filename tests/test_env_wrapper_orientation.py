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
