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
    anchor_xy: tuple[float, float] = (0.0, 0.0),
    anchor_yaw: float = 0.0,
):
    """Build a fake sim object with model / data attributes.

    Models the mujoco chain:
        anchor body (robot0_base, xpos=anchor_xy, xmat=rot_z(anchor_yaw))
          └─ slide_forward joint (qpos[0], axis=(1,0,0) in anchor frame)
          └─ slide_side joint    (qpos[1], axis=(0,1,0) in anchor frame)
          └─ hinge_yaw joint     (qpos[2], axis=(0,0,1) in anchor frame)
              └─ mobilebase0_base body (real position derived from anchor + qpos)

    sim.forward() recomputes mobilebase0_base xpos as:
        world_xy = anchor_xy + rot_z(anchor_yaw) @ (qpos[0], qpos[1])
    so the world<->qpos transform stays consistent with mujoco semantics.

    Initial qpos is back-computed so that mobilebase0_base.xpos matches base_xy.
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
        "robot0_base": np.array(
            [anchor_xy[0], anchor_xy[1], 0.0], dtype=np.float32
        ),
    }
    if extra_bodies:
        for k, v in extra_bodies.items():
            body_xpos_map[k] = np.asarray(v, dtype=np.float32)

    body_names = list(body_xpos_map.keys())

    # Anchor 2x2 rotation (R = rot_z(anchor_yaw))
    c, s = float(np.cos(anchor_yaw)), float(np.sin(anchor_yaw))
    R2 = np.array([[c, -s], [s, c]], dtype=np.float64)
    anchor_xy_np = np.asarray(anchor_xy, dtype=np.float64)

    # Back-compute initial qpos so that sim.forward() leaves base_xy unchanged.
    # base_xy_world = anchor + R @ (qpos[0], qpos[1]) -> qpos = R.T @ (base_xy - anchor)
    init_local = R2.T @ (np.asarray(base_xy, dtype=np.float64) - anchor_xy_np)

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
            # Seed qpos so that initial body_xpos matches base_xy
            if qpos_size > 0 and include_joints:
                self.qpos[0] = float(init_local[0])
                self.qpos[1] = float(init_local[1])
            self.qvel = np.zeros(qpos_size, dtype=np.float64)
            self.body_xpos = [body_xpos_map[n].copy() for n in body_names]

        # mujoco-style forward: recompute world_xy = anchor + R @ qpos_local
        def _refresh_base_body_from_qpos(self):
            if qpos_size < 2 or not include_joints:
                return
            local = np.array(
                [float(self.qpos[0]), float(self.qpos[1])], dtype=np.float64
            )
            world = anchor_xy_np + R2 @ local
            for i, name in enumerate(body_names):
                if name == "mobilebase0_base":
                    self.body_xpos[i] = np.array(
                        [float(world[0]), float(world[1]), 0.0],
                        dtype=np.float32,
                    )

    class _Sim:
        def __init__(self):
            self.model = _Model()
            self.data = _Data()

        def forward(self):
            self.data._refresh_base_body_from_qpos()

    sim = _Sim()
    return sim


class _NavStubEnv(EnvWrapper):
    """Minimal EnvWrapper subclass for navigate_base_to tests.

    Bypasses __init__ to avoid robosuite. Provides fake sim with adjustable
    base_xy, mobilebase joint layout, and (critically) an anchor frame
    so the world<->qpos conversion is exercised correctly.

    By default anchor is (0,0) with identity rotation -> qpos == world XY,
    matching simple-case tests. Pass `anchor_xy` and `anchor_yaw` to mimic
    the real RoboCasa PandaMobile setup (anchor at (10,10), yaw=pi).
    """

    def __init__(
        self,
        base_xy: tuple[float, float] = (0.0, 0.0),
        include_joints: bool = True,
        include_torso: bool = True,
        extra_bodies: dict | None = None,
        anchor_xy: tuple[float, float] = (0.0, 0.0),
        anchor_yaw: float = 0.0,
    ) -> None:
        sim = _build_stub_sim(
            base_xy=base_xy,
            include_joints=include_joints,
            include_torso=include_torso,
            extra_bodies=extra_bodies,
            anchor_xy=anchor_xy,
            anchor_yaw=anchor_yaw,
        )

        # Build anchor xmat from yaw (rot_z)
        c, s = float(np.cos(anchor_yaw)), float(np.sin(anchor_yaw))
        anchor_xmat = np.array(
            [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

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


def test_navigate_baseline_borderline_dist_must_teleport() -> None:
    """Phase 5 regression: dist=0.529m (baseline tupperware) MUST teleport,
    not be masked as no-op.

    Pre-fix: threshold = offset_m + 0.10 = 0.55m, 0.529 ≤ 0.55 → no-op.
    Post-fix: threshold = abs(dist - offset_m) ≤ 0.05, |0.529-0.45|=0.079
              > 0.05 → teleport.
    """
    env = _NavStubEnv(base_xy=(0.775, -2.882))
    qpos_before = env._env.sim.data.qpos.copy()
    ok = env.navigate_base_to((0.346, -3.194), offset_m=0.45)
    assert ok is True
    # qpos MUST have changed (teleport happened)
    qpos_after = env._env.sim.data.qpos
    assert not np.allclose(qpos_before, qpos_after), (
        "Expected teleport for borderline dist 0.529m, but qpos unchanged"
    )
    # And the resulting base position should be within ±0.05m of offset_m
    new_xy = env._read_real_base_xy()
    new_dist = float(np.linalg.norm(new_xy - np.array([0.346, -3.194])))
    assert abs(new_dist - 0.45) <= 0.05, (
        f"Expected dist ~0.45m after teleport, got {new_dist:.3f}m"
    )


def test_navigate_no_op_in_optimal_band() -> None:
    """When dist already in [offset_m - 0.05, offset_m + 0.05], skip teleport."""
    # Place base at exactly offset_m (0.45m) from target
    env = _NavStubEnv(base_xy=(0.45, 0.0))
    qpos_before = env._env.sim.data.qpos.copy()
    ok = env.navigate_base_to((0.0, 0.0), offset_m=0.45)
    assert ok is True
    np.testing.assert_array_equal(env._env.sim.data.qpos, qpos_before)


def test_navigate_no_op_too_close_protection() -> None:
    """When dist <= 0.10m, skip teleport to avoid collision."""
    env = _NavStubEnv(base_xy=(0.05, 0.0))  # dist 0.05m to origin
    qpos_before = env._env.sim.data.qpos.copy()
    ok = env.navigate_base_to((0.0, 0.0), offset_m=0.45)
    assert ok is True
    np.testing.assert_array_equal(env._env.sim.data.qpos, qpos_before)


def test_navigate_with_anchor_at_10_10_yaw_pi_real_robocasa_layout() -> None:
    """REGRESSION: GPU Phase 5 run 2 showed base teleported to (9.67, 13.05)
    instead of (0.71, -2.93). Root cause: qpos is anchor-LOCAL, not world.

    Real RoboCasa PandaMobile layout (from Phase 1 probe, docs/07 D.2):
        anchor robot0_base   xpos=(10, 10, 0)
        anchor robot0_base   xmat=rot_z(180°)  -> base_ori
        mobilebase0_base     xpos=(0.775, -2.882, 0)  -> real position

    With this anchor, teleport target world=(0.711, -2.929) must produce:
        delta_world = (0.711-10, -2.929-10) = (-9.289, -12.929)
        R2 = rot_z(180°)[:2,:2] = [[-1,0],[0,-1]]
        qpos_local = R2.T @ delta_world = (9.289, 12.929)

    This test runs the actual navigate_base_to with that exact anchor and
    verifies the resulting world-space base position is correct.
    """
    env = _NavStubEnv(
        base_xy=(0.775, -2.882),
        anchor_xy=(10.0, 10.0),
        anchor_yaw=float(np.pi),
    )
    # Sanity: initial body xpos matches base_xy
    initial_xy = env._read_real_base_xy()
    assert initial_xy is not None
    np.testing.assert_allclose(initial_xy, [0.775, -2.882], atol=1e-4)

    ok = env.navigate_base_to((0.346, -3.194), offset_m=0.45)
    assert ok is True

    # After teleport, real (world) base position should be ~0.45m from target
    new_xy = env._read_real_base_xy()
    assert new_xy is not None
    new_dist = float(np.linalg.norm(new_xy - np.array([0.346, -3.194])))
    assert abs(new_dist - 0.45) <= 0.05, (
        f"Expected dist ~0.45m after teleport, got {new_dist:.3f}m, "
        f"new_xy={tuple(new_xy.tolist())}"
    )
    # CRITICAL: must NOT have flown to anchor area (9.x, 1x.x)
    assert abs(float(new_xy[0])) < 5.0 and abs(float(new_xy[1])) < 5.0, (
        f"Base flew to anchor area: new_xy={tuple(new_xy.tolist())}. "
        "world<->qpos transform is broken."
    )


# ======================================================================
# Phase 8a: torso joint API tests
# ======================================================================


def test_get_torso_joint_info_finds_slide_z() -> None:
    """_get_torso_joint_info should locate the slide+(0,0,1) joint and
    NOT confuse it with yaw hinge or other joints."""
    env = _NavStubEnv(base_xy=(0.0, 0.0), include_torso=True)
    info = env._get_torso_joint_info()
    assert info is not None
    addr, lo, hi = info
    assert addr == 3  # torso is qpos[3] in our stub layout
    # Stub doesn't model jnt_range -> falls back to default [-1, 1]
    assert lo == -1.0
    assert hi == 1.0


def test_get_torso_joint_info_returns_none_when_absent() -> None:
    """When the torso joint is missing, _get_torso_joint_info returns None."""
    env = _NavStubEnv(base_xy=(0.0, 0.0), include_torso=False)
    assert env._get_torso_joint_info() is None


def test_get_torso_height_reads_qpos() -> None:
    """get_torso_height returns the current sim.data.qpos[addr]."""
    env = _NavStubEnv(base_xy=(0.0, 0.0), include_torso=True)
    # Pre-seed torso qpos[3] to 0.123
    env._env.sim.data.qpos[3] = 0.123
    h = env.get_torso_height()
    assert h is not None
    assert abs(h - 0.123) < 1e-9


def test_set_torso_height_writes_qpos_and_zeros_qvel() -> None:
    """set_torso_height teleports torso joint, zeros qvel, calls forward."""
    env = _NavStubEnv(base_xy=(0.0, 0.0), include_torso=True)
    env._env.sim.data.qvel[3] = 1.5  # nonzero velocity
    ok = env.set_torso_height(0.05)
    assert ok is True
    assert abs(float(env._env.sim.data.qpos[3]) - 0.05) < 1e-9
    assert abs(float(env._env.sim.data.qvel[3])) < 1e-9


def test_set_torso_height_clamps_to_range() -> None:
    """Out-of-range requests get silently clamped to [lo, hi]."""
    env = _NavStubEnv(base_xy=(0.0, 0.0), include_torso=True)
    # stub range falls back to [-1, 1]; request 5.0 should clamp to 1.0
    env.set_torso_height(5.0)
    assert abs(float(env._env.sim.data.qpos[3]) - 1.0) < 1e-9
    env.set_torso_height(-5.0)
    assert abs(float(env._env.sim.data.qpos[3]) - (-1.0)) < 1e-9


def test_set_torso_height_returns_false_when_joint_missing() -> None:
    """Without a torso joint, set_torso_height fails gracefully."""
    env = _NavStubEnv(base_xy=(0.0, 0.0), include_torso=False)
    ok = env.set_torso_height(0.10)
    assert ok is False
