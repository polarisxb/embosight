"""Tests for real-base pose reading and arm OSC frame handling.

The arm OSC position action is world/controller-fixed, while the base velocity
action uses the mobile base frame.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.env_wrapper import EnvWrapper  # noqa: E402


# ----------------------------------------------------------------------
# Stub sim with body_xmat support
# ----------------------------------------------------------------------


def _rot_z(yaw_rad: float) -> np.ndarray:
    c, s = float(np.cos(yaw_rad)), float(np.sin(yaw_rad))
    return np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def _build_sim_with_xmat(
    real_base_xy: tuple[float, float],
    real_base_yaw: float,
    anchor_xy: tuple[float, float] = (10.0, 10.0),
    anchor_yaw: float = -np.pi,
    include_xmat: bool = True,
):
    """Build a stub sim that exposes both body_xpos AND body_xmat.

    Mirrors the RoboCasa PandaMobile layout where:
    - 'robot0_base' is the anchor (xpos=(10,10,0), yaw=-180°)
    - 'mobilebase0_base' is the real mobile base (xpos/yaw set by qpos)
    """
    body_names = ["robot0_base", "mobilebase0_base"]
    body_xpos = [
        np.array([anchor_xy[0], anchor_xy[1], 0.0], dtype=np.float32),
        np.array([real_base_xy[0], real_base_xy[1], 0.0], dtype=np.float32),
    ]
    body_xmat_list = [
        _rot_z(anchor_yaw),
        _rot_z(real_base_yaw),
    ]

    class _Model:
        njnt = 0
        jnt_type = np.zeros(0, dtype=np.int32)
        jnt_qposadr = np.zeros(0, dtype=np.int32)
        jnt_axis = np.zeros((0, 3), dtype=np.float64)

        @staticmethod
        def body_name2id(name: str) -> int:
            if name not in body_names:
                raise ValueError(name)
            return body_names.index(name)

    class _Data:
        qpos = np.zeros(8, dtype=np.float64)
        qvel = np.zeros(8, dtype=np.float64)

        def __init__(self):
            self.body_xpos = [x.copy() for x in body_xpos]
            if include_xmat:
                # mujoco stores xmat as flat (9,) per body; we keep (3,3)
                # since our reader does .reshape(3,3) which works on both.
                self.body_xmat = [m.copy() for m in body_xmat_list]
            # else: no body_xmat attribute → fallback path

    class _Sim:
        def __init__(self):
            self.model = _Model()
            self.data = _Data()

        def forward(self):
            pass

    return _Sim()


class _RealBaseStubEnv(EnvWrapper):
    """EnvWrapper subclass that bypasses __init__ for unit testing."""

    def __init__(
        self,
        real_base_xy: tuple[float, float] = (0.751, -3.053),
        real_base_yaw: float = -2.513,  # -144° (Run 5 GPU log)
        anchor_xy: tuple[float, float] = (10.0, 10.0),
        anchor_yaw: float = -np.pi,
        include_xmat: bool = True,
    ) -> None:
        sim = _build_sim_with_xmat(
            real_base_xy=real_base_xy,
            real_base_yaw=real_base_yaw,
            anchor_xy=anchor_xy,
            anchor_yaw=anchor_yaw,
            include_xmat=include_xmat,
        )

        anchor_xmat = _rot_z(anchor_yaw)

        class _Robot:
            idn = 0
            base_pos = np.array(
                [anchor_xy[0], anchor_xy[1], 0.0], dtype=np.float32
            )
            base_ori = anchor_xmat

        class _Backend:
            pass

        backend = _Backend()
        backend.sim = sim
        backend.robots = [_Robot()]
        self._env = backend


# ======================================================================
# Tests
# ======================================================================


def test_read_real_base_pose_returns_real_xpos_and_xmat() -> None:
    """In Run 5 layout (anchor=(10,10,-180°), real=(0.751,-3.053,-144°)),
    _read_real_base_pose must return the REAL pose, not the anchor."""
    env = _RealBaseStubEnv(
        real_base_xy=(0.751, -3.053),
        real_base_yaw=np.deg2rad(-144.0),
    )
    pose = env._read_real_base_pose()
    assert pose is not None
    xpos, xmat = pose
    np.testing.assert_allclose(xpos, [0.751, -3.053, 0.0], atol=1e-4)
    expected_xmat = _rot_z(np.deg2rad(-144.0))
    np.testing.assert_allclose(xmat, expected_xmat, atol=1e-4)


def test_read_real_base_pose_distinguishes_from_anchor_pose() -> None:
    """Anchor ori (yaw=-180°) and real ori (yaw=-144°) must differ by 36°."""
    env = _RealBaseStubEnv(
        real_base_yaw=np.deg2rad(-144.0),
        anchor_yaw=np.deg2rad(-180.0),
    )
    real_pose = env._read_real_base_pose()
    _, anchor_ori = env.get_base_pose()
    assert real_pose is not None
    real_xmat = real_pose[1]
    # ori should differ — extract yaw from each
    real_yaw = np.arctan2(real_xmat[1, 0], real_xmat[0, 0])
    anchor_yaw = np.arctan2(anchor_ori[1, 0], anchor_ori[0, 0])
    delta_deg = np.rad2deg(
        np.arctan2(np.sin(real_yaw - anchor_yaw), np.cos(real_yaw - anchor_yaw))
    )
    assert abs(abs(delta_deg) - 36.0) < 1.0, (
        f"Expected 36° offset between real (-144°) and anchor (-180°), "
        f"got {delta_deg:.2f}°"
    )


def test_read_real_base_pose_falls_back_when_xmat_missing() -> None:
    """When sim.data has no body_xmat (mock / old robosuite), fall back to
    identity ori (caller will then prefer anchor via get_base_pose)."""
    env = _RealBaseStubEnv(include_xmat=False)
    pose = env._read_real_base_pose()
    assert pose is not None  # xpos still readable
    xpos, xmat = pose
    np.testing.assert_allclose(xpos, [0.751, -3.053, 0.0], atol=1e-4)
    np.testing.assert_allclose(xmat, np.eye(3), atol=1e-6)


def test_read_real_base_pose_returns_none_when_no_sim() -> None:
    """No sim attribute → return None."""
    env = _RealBaseStubEnv()
    env._env = type("X", (), {"robots": [type("R", (), {"idn": 0})()]})()
    # no `sim` attribute
    assert env._read_real_base_pose() is None


def test_arm_position_action_stays_world_frame_after_base_rotation() -> None:
    env = _RealBaseStubEnv(
        real_base_xy=(0.425, -2.861),
        real_base_yaw=np.deg2rad(177.8),
        anchor_yaw=np.deg2rad(-180.0),
    )
    captured: list[np.ndarray] = []
    current = np.array([0.2199, -2.8164, 1.6043], dtype=np.float32)

    class _Backend:
        action_dim = 12

        def __init__(self, base_backend):
            self.sim = base_backend.sim
            self.robots = base_backend.robots

        def step(self, action):
            captured.append(np.asarray(action, dtype=np.float32).copy())
            return {"robot0_eef_pos": current.copy()}, 0.0, False, {}

    env._env = _Backend(env._env)
    env.config = type("C", (), {"has_renderer": False})()
    env._base_idx_cache = None
    env._latest_obs = {"robot0_eef_pos": current.copy()}
    target = np.array([0.1248, -2.8573, 1.4500], dtype=np.float32)

    env.move_arm_to(target, max_steps=1)

    assert captured
    assert captured[0][0] < 0.0
    assert captured[0][2] < 0.0
