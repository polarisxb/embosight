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
