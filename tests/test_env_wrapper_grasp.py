from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.env_wrapper import EnvConfig, EnvWrapper  # noqa: E402
from src.world_belief import GraspCandidate  # noqa: E402


class PregraspFailEnv(EnvWrapper):
    def __init__(self) -> None:
        self.move_calls = 0
        self.descend_calls = 0
        self.close_calls = 0
        self.gripper_calls = 0

    def _get_body_pos(self, body_name: str):
        return np.array([1.0, 2.0, 0.5], dtype=np.float32)

    def _compute_grasp_pose(self, body_name: str, fallback_pos: np.ndarray) -> np.ndarray:
        return np.array([1.0, 2.0, 0.6], dtype=np.float32)

    def _gripper_action(self, gripper_value: float, n_steps: int = 10) -> None:
        self.gripper_calls += 1

    def get_eef_pos(self) -> np.ndarray:
        return np.array([1.0, 2.0, 0.7], dtype=np.float32)

    def move_arm_to(self, target_pos_m, max_steps: int = 800, threshold_m: float = 0.02) -> bool:
        self.move_calls += 1
        return False

    def _descend_until_contact(
        self,
        target_pos: np.ndarray,
        target_body: str,
        step_z: float = 0.01,
        max_steps: int = 25,
    ):
        self.descend_calls += 1
        return False, float(target_pos[2])

    def _close_gripper_until_grasp(
        self, target_body: str, max_steps: int = 30, min_close_steps: int = 6
    ) -> bool:
        self.close_calls += 1
        return False


def test_grasp_at_aborts_when_pre_grasp_move_fails() -> None:
    env = PregraspFailEnv()

    ok = env.grasp_at((1.0, 2.0, 0.5), target_body="obj_main")

    assert ok is False
    assert env.move_calls == 1
    assert env.descend_calls == 0
    assert env.close_calls == 0


class PregraspThresholdEnv(EnvWrapper):
    def __init__(self) -> None:
        self.move_calls = []
        self.gripper_calls = 0

    def get_eef_pos(self) -> np.ndarray:
        return np.array([0.0, 0.0, 0.9], dtype=np.float32)

    def move_arm_to(self, target_pos_m, max_steps: int = 800,
                    threshold_m: float = 0.02, **kwargs) -> bool:
        self.move_calls.append((np.asarray(target_pos_m), max_steps, threshold_m))
        return True

    def _gripper_action(self, gripper_value: float, n_steps: int = 10) -> None:
        self.gripper_calls += 1


def test_move_to_pre_grasp_accepts_six_cm_boundary() -> None:
    env = PregraspThresholdEnv()
    candidate = GraspCandidate(
        point_3d=np.array([0.5, 0.2, 0.9], dtype=np.float32),
        approach_dir=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        finger_width_m=0.04,
        score=0.8,
    )

    assert env.move_to_pre_grasp(candidate) is True
    assert env.move_calls[-1][2] >= 0.06


class _IsReachableEnv(EnvWrapper):
    """Minimal stub exposing only get_base_pose for is_reachable tests."""
    def __init__(self, base_xy=(0.0, 0.0)) -> None:
        self._base_xy = np.asarray(base_xy, dtype=np.float32)

    def get_base_pose(self):
        pos = np.array(
            [float(self._base_xy[0]), float(self._base_xy[1]), 0.0],
            dtype=np.float32,
        )
        return pos, np.eye(3, dtype=np.float32)


def test_is_reachable_true_for_close_point() -> None:
    env = _IsReachableEnv(base_xy=(0.0, 0.0))
    p = np.array([0.4, 0.3, 0.9], dtype=np.float32)  # horiz=0.5m
    assert env.is_reachable(p, np.array([0.0, 0.0, -1.0], dtype=np.float32)) is True


def test_is_reachable_false_beyond_radius() -> None:
    env = _IsReachableEnv(base_xy=(0.0, 0.0))
    # horiz=1.0m > 0.75m threshold
    p = np.array([0.8, 0.6, 0.9], dtype=np.float32)
    assert env.is_reachable(p, np.array([0.0, 0.0, -1.0], dtype=np.float32)) is False


def test_is_reachable_uses_horizontal_only() -> None:
    """Z separation should NOT affect reachability — arm has vertical reach."""
    env = _IsReachableEnv(base_xy=(0.0, 0.0))
    # horiz=0.5m, large z gap
    p = np.array([0.5, 0.0, 2.0], dtype=np.float32)
    assert env.is_reachable(p, np.array([0.0, 0.0, -1.0], dtype=np.float32)) is True


def test_is_reachable_falls_back_to_true_on_error() -> None:
    """Geometry errors should NOT block grasping — preserve old behavior."""
    class _BadEnv(EnvWrapper):
        def __init__(self) -> None:
            pass
        def get_base_pose(self):
            raise RuntimeError("simulated failure")

    env = _BadEnv()
    p = np.array([10.0, 10.0, 0.9], dtype=np.float32)  # would normally be False
    assert env.is_reachable(p, np.array([0.0, 0.0, -1.0], dtype=np.float32)) is True


class _LiftCallRecorder(EnvWrapper):
    """Records all move_arm_to calls during lift to verify gripper_hold."""
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self._eef_z = 0.6  # tracks simulated rise

    def get_eef_pos(self) -> np.ndarray:
        return np.array([0.5, 0.0, self._eef_z], dtype=np.float32)

    def move_arm_to(self, target_pos_m, **kwargs) -> bool:
        self.calls.append({
            "target": np.asarray(target_pos_m).copy(),
            "gripper_hold": kwargs.get("gripper_hold", 0.0),
        })
        # Simulate following the target z so lift's success check passes
        self._eef_z = float(np.asarray(target_pos_m)[2])
        return True


def test_lift_passes_gripper_hold_to_all_move_calls() -> None:
    """Regression: every move_arm_to call inside lift() must hold gripper.

    The slipped_lift Δz=0 root cause was move_arm_to clearing the gripper
    action to 0 mid-lift, releasing the object. This test enforces the
    contract that lift() always passes gripper_hold=1.0.
    """
    env = _LiftCallRecorder()
    ok, _final_z = env.lift(height_m=0.10, approach_dir=None)
    assert ok is True
    assert len(env.calls) > 0, "lift should issue at least one move_arm_to call"
    for i, c in enumerate(env.calls):
        assert c["gripper_hold"] == 1.0, (
            f"call {i} target={c['target']} did not hold gripper "
            f"(gripper_hold={c['gripper_hold']})"
        )


# ============================================================
# E2E mock: action vectors actually sent to env.step during lift
# ============================================================

class _StepActionRecorder(EnvWrapper):
    """Captures every action vector that move_arm_to sends to env.step.

    Bypasses __init__ to avoid robosuite. Provides only the surface that
    move_arm_to + lift need:
      - _env.action_dim, _env.step (records action)
      - get_eef_pos (advances toward target each step)
      - _get_gripper_idx, _get_base_action_idx
      - get_base_pose, render, _latest_obs
    """
    GRIPPER_IDX = 7  # 6 arm pose + 1 (no base in this mock)
    ACTION_DIM = 8

    def __init__(self) -> None:
        self.actions_logged: list[np.ndarray] = []
        self._eef = np.array([0.5, 0.0, 0.6], dtype=np.float32)
        self._latest_obs = {"_": True}
        self._gripper_idx_cache = self.GRIPPER_IDX
        outer = self

        class _MockBackend:
            action_dim = outer.ACTION_DIM

            def step(self, action):
                a = np.asarray(action, dtype=np.float32).copy()
                outer.actions_logged.append(a)
                # Advance eef toward target so move_arm_to converges.
                # action[0:3] is the base-frame delta in env_wrapper.py.
                outer._eef = outer._eef + a[0:3] * 0.5
                return {}, 0.0, False, {}

        self._env = _MockBackend()

    def get_eef_pos(self) -> np.ndarray:
        return self._eef.copy()

    def get_base_pose(self):
        return np.zeros(3, dtype=np.float32), np.eye(3, dtype=np.float64)

    def _get_base_action_idx(self):
        return None  # this mock has no base

    def render(self) -> None:
        pass

    def reset(self):
        return {}


def test_move_arm_to_writes_gripper_hold_to_action() -> None:
    """move_arm_to(gripper_hold=1.0) must put 1.0 at gripper_idx every step."""
    env = _StepActionRecorder()
    target = np.array([0.6, 0.0, 0.65], dtype=np.float32)
    env.move_arm_to(target, threshold_m=0.001, max_steps=20, gripper_hold=1.0)

    assert len(env.actions_logged) > 0
    for i, a in enumerate(env.actions_logged):
        assert a[env.GRIPPER_IDX] == 1.0, (
            f"step {i} action[gripper]={a[env.GRIPPER_IDX]} (expected 1.0); "
            f"object would slip"
        )


def test_move_arm_to_default_leaves_gripper_neutral() -> None:
    """Legacy contract: default gripper_hold=0.0 must NOT touch gripper_idx."""
    env = _StepActionRecorder()
    target = np.array([0.6, 0.0, 0.65], dtype=np.float32)
    env.move_arm_to(target, threshold_m=0.001, max_steps=20)  # no gripper_hold

    assert len(env.actions_logged) > 0
    for a in env.actions_logged:
        assert a[env.GRIPPER_IDX] == 0.0


class _MockSim:
    """Tiny MuJoCo-like sim shim for get_base_pose tests."""
    def __init__(self, bodies: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
        # bodies: name -> (xpos[3], xmat[3,3])
        self._names = list(bodies.keys())
        self._xpos = {n: bodies[n][0] for n in bodies}
        self._xmat = {n: bodies[n][1] for n in bodies}
        outer = self

        class _Model:
            def body_name2id(self, name):
                if name not in outer._names:
                    raise ValueError(name)
                return outer._names.index(name)

        class _Data:
            @property
            def body_xpos(self):
                return [outer._xpos[n] for n in outer._names]

            @property
            def body_xmat(self):
                return [outer._xmat[n].reshape(9) for n in outer._names]

        self.model = _Model()
        self.data = _Data()


class _BasePoseEnv(EnvWrapper):
    """Stub env wrapping a _MockSim so get_base_pose can be tested without robosuite."""
    def __init__(self, bodies, fallback_base=(10.0, 10.0, 0.0)) -> None:
        outer_sim = _MockSim(bodies)

        class _Robot:
            idn = 0
            base_pos = np.asarray(fallback_base, dtype=np.float32)
            base_ori = np.eye(3, dtype=np.float32)

        class _Backend:
            sim = outer_sim
            robots = [_Robot()]

        self._env = _Backend()


def test_get_base_pose_prefers_mobilebase_body() -> None:
    """Real mobile base xpos should override the (10,10,0) anchor."""
    real_pos = np.array([0.78, -2.88, 0.0], dtype=np.float32)
    real_mat = np.eye(3, dtype=np.float32)
    fake_pos = np.array([10.0, 10.0, 0.0], dtype=np.float32)
    env = _BasePoseEnv({
        "robot0_base": (fake_pos, np.eye(3, dtype=np.float32)),
        "mobilebase0_base": (real_pos, real_mat),
    })
    pos, mat = env.get_base_pose()
    np.testing.assert_allclose(pos, real_pos, atol=1e-5)
    np.testing.assert_allclose(mat, real_mat, atol=1e-5)


def test_get_base_pose_skips_fake_robot_base() -> None:
    """If only robot0_base exists and it's at (10,10,0), skip → fallback."""
    fake_pos = np.array([10.0, 10.0, 0.0], dtype=np.float32)
    env = _BasePoseEnv(
        {"robot0_base": (fake_pos, np.eye(3, dtype=np.float32))},
        fallback_base=(0.5, 0.5, 0.0),
    )
    pos, _mat = env.get_base_pose()
    # Must not return the fake (10,10,0) — must use robot.base_pos fallback
    np.testing.assert_allclose(pos, [0.5, 0.5, 0.0], atol=1e-5)


def test_get_base_pose_uses_robot_base_when_no_mobilebase() -> None:
    """Static (non-mobile) robot has only robot0_base at real position."""
    real_pos = np.array([0.0, 0.0, 0.9], dtype=np.float32)
    env = _BasePoseEnv(
        {"robot0_base": (real_pos, np.eye(3, dtype=np.float32))},
    )
    pos, _mat = env.get_base_pose()
    np.testing.assert_allclose(pos, real_pos, atol=1e-5)


def test_is_reachable_works_after_base_pose_fix() -> None:
    """E2E: with realistic mobile base at (0.78, -2.88), an object at
    (0.35, -3.19) — 0.53m horiz away — must be reachable."""
    real_pos = np.array([0.78, -2.88, 0.0], dtype=np.float32)
    env = _BasePoseEnv({
        "robot0_base": (np.array([10.0, 10.0, 0.0], dtype=np.float32),
                        np.eye(3, dtype=np.float32)),
        "mobilebase0_base": (real_pos, np.eye(3, dtype=np.float32)),
    })
    obj_pos = np.array([0.35, -3.19, 0.94], dtype=np.float32)
    assert env.is_reachable(obj_pos, np.array([0, 0, -1.0])) is True


def test_lift_e2e_keeps_gripper_at_one_throughout() -> None:
    """E2E: from first env.step in lift() to last, gripper stays at 1.0.

    This is the strongest defense against the slipped_lift bug — even if
    someone refactors move_arm_to or lift in the future, this test catches
    any sequence of env.step calls that lets the gripper go neutral.
    """
    env = _StepActionRecorder()
    ok, _z = env.lift(height_m=0.10, approach_dir=None)
    assert ok is True

    assert len(env.actions_logged) > 0, (
        "lift should issue at least one env.step call"
    )
    for i, a in enumerate(env.actions_logged):
        assert a[env.GRIPPER_IDX] == 1.0, (
            f"env.step #{i} during lift had gripper={a[env.GRIPPER_IDX]}; "
            f"this would release the object mid-lift"
        )


def test_reset_applies_seed_before_backend_reset(monkeypatch, tmp_path) -> None:
    events = []

    class FakeBackend:
        def seed(self, seed: int) -> None:
            events.append(("seed", seed))

        def reset(self):
            events.append(("reset", None))
            return {}

    def make(**kwargs):
        events.append(("make", kwargs["seed"]))
        return FakeBackend()

    monkeypatch.setitem(sys.modules, "robocasa", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "robosuite", SimpleNamespace(make=make))
    env = EnvWrapper(EnvConfig(output_dir=str(tmp_path)))

    env.seed(42)
    env.reset()

    assert events == [
        ("make", 42),
        ("seed", 42),
        ("reset", None),
    ]
