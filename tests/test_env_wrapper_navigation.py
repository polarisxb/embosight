"""Unit tests for EnvWrapper.navigate_base_to (Phase 2).

These tests build minimal _NavStubEnv fixtures that mimic robosuite's
sim.model / sim.data surface without spinning up RoboCasa. Joint axis and
qpos layout match the Phase 1 GPU probe results (see docs/07 D.2).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.env_wrapper import EnvWrapper  # noqa: E402


# ----------------------------------------------------------------------
# Minimal sim shim mirroring RoboCasa kitchen mobilebase layout
# (forward slide qpos[0], side slide qpos[1], yaw hinge qpos[2],
#  torso slide qpos[3] -- intentionally with same axis as yaw)
# ----------------------------------------------------------------------

# joint types matching mujoco: 2 = slide, 3 = hinge
_JT_SLIDE = 2
_JT_HINGE = 3


def _build_stub_sim(
    base_xy: tuple[float, float],
    include_joints: bool = True,
    include_torso: bool = True,
    extra_bodies: dict | None = None,
):
    """Build a fake sim object with model / data attributes.

    extra_bodies maps body_name -> xpos[3] (always identity xmat for simplicity).
    """
    # Joints (mirrors Phase 1 probe)
    joints: list[tuple[str, int, int, tuple[float, float, float]]] = []
    if include_joints:
        joints.append(("mobilebase0_joint_mobile_forward", _JT_SLIDE, 0,
                       (1.0, 0.0, 0.0)))
        joints.append(("mobilebase0_joint_mobile_side", _JT_SLIDE, 1,
                       (0.0, 1.0, 0.0)))
        joints.append(("mobilebase0_joint_mobile_yaw", _JT_HINGE, 2,
                       (0.0, 0.0, 1.0)))
        if include_torso:
            joints.append(("mobilebase0_joint_torso_height", _JT_SLIDE, 3,
                           (0.0, 0.0, 1.0)))  # same axis as yaw, but slide

    # Bodies
    body_xpos_map: dict[str, np.ndarray] = {
        "mobilebase0_base": np.array(
            [base_xy[0], base_xy[1], 0.0], dtype=np.float32
        ),
        "robot0_base": np.array([10.0, 10.0, 0.0], dtype=np.float32),
    }
    if extra_bodies:
        for k, v in extra_bodies.items():
            body_xpos_map[k] = np.asarray(v, dtype=np.float32)

    body_names = list(body_xpos_map.keys())

    class _Model:
        njnt = len(joints)
        jnt_type = np.array([j[1] for j in joints], dtype=np.int32) if joints \
            else np.zeros(0, dtype=np.int32)
        jnt_qposadr = np.array([j[2] for j in joints], dtype=np.int32) if joints \
            else np.zeros(0, dtype=np.int32)
        jnt_axis = np.array([j[3] for j in joints], dtype=np.float64) if joints \
            else np.zeros((0, 3), dtype=np.float64)

        @staticmethod
        def joint_id2name(jid: int) -> str:
            return joints[jid][0] if 0 <= jid < len(joints) else ""

        @staticmethod
        def body_name2id(name: str) -> int:
            if name not in body_names:
                raise ValueError(name)
            return body_names.index(name)

    # qpos / qvel large enough to cover all joint addrs
    qpos_size = max((j[2] for j in joints), default=-1) + 8

    class _Data:
        def __init__(self):
            self.qpos = np.zeros(qpos_size, dtype=np.float64)
            self.qvel = np.zeros(qpos_size, dtype=np.float64)
            # body_xpos accessed by [bid] indexing
            self.body_xpos = [body_xpos_map[n].copy() for n in body_names]

        # Reflect qpos -> body_xpos when sim.forward() runs
        def _refresh_base_body_from_qpos(self):
            forward = float(self.qpos[0]) if qpos_size > 0 else 0.0
            side = float(self.qpos[1]) if qpos_size > 1 else 0.0
            # Update only mobilebase0_base (skip anchor)
            for i, name in enumerate(body_names):
                if name == "mobilebase0_base":
                    self.body_xpos[i] = np.array(
                        [forward, side, 0.0], dtype=np.float32
                    )

    class _Sim:
        def __init__(self):
            self.model = _Model()
            self.data = _Data()

        def forward(self):
            self.data._refresh_base_body_from_qpos()

    sim = _Sim()
    # Seed data.body_xpos to match initial base_xy (already done in body_xpos_map)
    return sim


class _NavStubEnv(EnvWrapper):
    """Minimal EnvWrapper subclass for navigate_base_to tests.

    Bypasses __init__ to avoid robosuite. Provides fake sim with adjustable
    base_xy and mobilebase joint layout.
    """

    def __init__(
        self,
        base_xy: tuple[float, float] = (0.0, 0.0),
        include_joints: bool = True,
        include_torso: bool = True,
        extra_bodies: dict | None = None,
    ) -> None:
        sim = _build_stub_sim(
            base_xy=base_xy,
            include_joints=include_joints,
            include_torso=include_torso,
            extra_bodies=extra_bodies,
        )

        class _Robot:
            idn = 0

        class _Backend:
            pass

        backend = _Backend()
        backend.sim = sim
        backend.robots = [_Robot()]
        self._env = backend


# ======================================================================
# Tests
# ======================================================================


def test_read_real_base_xy_prefers_mobilebase_over_anchor() -> None:
    """_read_real_base_xy must return the real mobilebase position,
    NOT the (10,10,0) anchor at robot0_base."""
    env = _NavStubEnv(base_xy=(0.775, -2.882))  # matches GPU baseline
    xy = env._read_real_base_xy()
    assert xy is not None
    np.testing.assert_allclose(xy, [0.775, -2.882], atol=1e-5)


def test_get_mobilebase_joint_addrs_distinguishes_torso_from_yaw() -> None:
    """Phase 2's joint detection must NOT mistake the torso slide
    (axis=(0,0,1), type=slide) for the yaw hinge."""
    env = _NavStubEnv(base_xy=(0.0, 0.0), include_torso=True)
    addrs = env._get_mobilebase_joint_addrs()
    assert addrs is not None
    x_addr, y_addr, yaw_addr = addrs
    assert x_addr == 0    # forward (slide x)
    assert y_addr == 1    # side (slide y)
    assert yaw_addr == 2  # yaw (hinge z), NOT 3 (torso slide z)


def test_navigate_no_op_when_already_close() -> None:
    """If base already within offset_m + 0.10, do nothing and return True."""
    # Place base at 0.4m from origin target; offset=0.45, threshold=0.55 -> no-op
    env = _NavStubEnv(base_xy=(0.4, 0.0))
    qpos_before = env._env.sim.data.qpos.copy()
    ok = env.navigate_base_to((0.0, 0.0), offset_m=0.45)
    assert ok is True
    # qpos unchanged
    np.testing.assert_array_equal(env._env.sim.data.qpos, qpos_before)


def test_navigate_teleports_when_far() -> None:
    """If base far from target, teleport to within offset_m + 0.15."""
    env = _NavStubEnv(base_xy=(5.0, 5.0))
    ok = env.navigate_base_to((0.0, 0.0), offset_m=0.45)
    assert ok is True
    new_xy = env._read_real_base_xy()
    assert new_xy is not None
    new_dist = float(np.linalg.norm(new_xy))
    assert new_dist <= 0.45 + 0.15, f"new_dist {new_dist} exceeded tolerance"


def test_navigate_sets_yaw_to_face_target() -> None:
    """After teleport, base yaw should point toward target so arm faces it."""
    env = _NavStubEnv(base_xy=(5.0, 5.0))
    env.navigate_base_to((0.0, 0.0), offset_m=0.45)
    # qpos[2] = yaw. Target at world (0,0) from base coming from (+,+) -> dir = -1,-1
    # arctan2(-1, -1) = -3pi/4
    yaw = float(env._env.sim.data.qpos[2])
    expected = float(np.arctan2(-1.0, -1.0))
    assert abs(yaw - expected) < 0.1, f"yaw {yaw} expected ~{expected}"


def test_navigate_returns_false_when_joints_missing() -> None:
    """When no mobilebase joints found, return False so caller can fall through."""
    env = _NavStubEnv(base_xy=(5.0, 5.0), include_joints=False)
    ok = env.navigate_base_to((0.0, 0.0), offset_m=0.45)
    assert ok is False


def test_navigate_caches_joint_addrs() -> None:
    """_get_mobilebase_joint_addrs caches: second call must not re-scan."""
    env = _NavStubEnv(base_xy=(0.0, 0.0))
    first = env._get_mobilebase_joint_addrs()
    # Mutate model.njnt; cache should still return first result
    env._env.sim.model.njnt = 0
    second = env._get_mobilebase_joint_addrs()
    assert first == second


def test_navigate_handles_baseline_scenario_correctly() -> None:
    """Sanity: GPU baseline numbers map to a valid teleport.

    From docs/07 D.1:
      target tupperware xy = (0.346, -3.194)
      real base xy        = (0.775, -2.882)
      horiz dist          = 0.529 m (just at arm reach edge)
    With offset_m=0.45, navigate should teleport base to within ~0.45m
    of target (well inside arm reach 0.65m).
    """
    env = _NavStubEnv(base_xy=(0.775, -2.882))
    ok = env.navigate_base_to((0.346, -3.194), offset_m=0.45)
    assert ok is True
    new_xy = env._read_real_base_xy()
    new_dist = float(np.linalg.norm(new_xy - np.array([0.346, -3.194])))
    assert new_dist <= 0.45 + 0.15, (
        f"baseline scenario: new_dist {new_dist:.3f}m exceeded tolerance"
    )
