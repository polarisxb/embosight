from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.env_wrapper import EnvConfig, EnvWrapper  # noqa: E402
from src.grasp_execution import PRE_GRASP_LATERAL_MISALIGNED, PRE_GRASP_SAFE_HANDOFF, PRE_GRASP_STRICT_OK  # noqa: E402
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
        self.final_eef = None  # if set, get_eef_pos returns this after move
        self._moved = False

    def get_eef_pos(self) -> np.ndarray:
        if self._moved and self.final_eef is not None:
            return self.final_eef.copy()
        return np.array([0.0, 0.0, 0.9], dtype=np.float32)

    def move_arm_to(self, target_pos_m, max_steps: int = 800,
                    threshold_m: float = 0.02, **kwargs) -> bool:
        self.move_calls.append((np.asarray(target_pos_m), max_steps, threshold_m))
        self._moved = True
        # If force_move_failure is set, simulate stall/max_steps even when EEF is close
        if getattr(self, "force_move_failure", False):
            return False
        # If final_eef is set, return based on actual distance
        if self.final_eef is not None:
            dist = float(np.linalg.norm(self.final_eef[:3] - np.asarray(target_pos_m)[:3]))
            return dist <= threshold_m
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


def test_move_to_pre_grasp_diagnostic_reports_lateral_misalignment() -> None:
    """Large XY offset → lateral_misaligned, needs_recovery."""
    env = PregraspThresholdEnv()
    # EEF ends up 6cm off in X from pre-grasp target
    env.final_eef = np.array([0.56, 0.2, 0.95], dtype=np.float32)
    candidate = GraspCandidate(
        point_3d=np.array([0.5, 0.2, 0.9], dtype=np.float32),
        approach_dir=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        finger_width_m=0.04,
        score=0.8,
    )

    result = env.move_to_pre_grasp_diagnostic(candidate)

    assert result.ok is False
    assert result.handoff_ok is False
    assert result.needs_recovery is True
    assert result.reason == PRE_GRASP_LATERAL_MISALIGNED


def test_move_to_pre_grasp_diagnostic_allows_small_lateral_handoff() -> None:
    """Small XY offset within limit + move_arm_to stall → safe_handoff.

    Realistic GPU pattern: move_arm_to returns False due to stall detection
    even though EEF is actually close to pre_pos.
    """
    env = PregraspThresholdEnv()
    env.force_move_failure = True
    # EEF 1cm off in X — within lateral limit (0.02 for finger_width 0.04)
    env.final_eef = np.array([0.51, 0.2, 0.95], dtype=np.float32)
    candidate = GraspCandidate(
        point_3d=np.array([0.5, 0.2, 0.9], dtype=np.float32),
        approach_dir=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        finger_width_m=0.04,
        score=0.8,
    )

    result = env.move_to_pre_grasp_diagnostic(candidate)

    assert result.ok is False  # forced move failure
    assert result.handoff_ok is True
    assert result.reason == PRE_GRASP_SAFE_HANDOFF


def test_move_to_pre_grasp_bool_accepts_safe_diagnostic_handoff() -> None:
    """Bool wrapper returns True when diagnostic says handoff_ok."""
    env = PregraspThresholdEnv()
    env.force_move_failure = True
    env.final_eef = np.array([0.51, 0.2, 0.95], dtype=np.float32)
    candidate = GraspCandidate(
        point_3d=np.array([0.5, 0.2, 0.9], dtype=np.float32),
        approach_dir=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        finger_width_m=0.04,
        score=0.8,
    )

    assert env.move_to_pre_grasp(candidate) is True
    # Strict threshold restored to 0.06 for top-down
    assert env.move_calls[-1][2] == 0.06


def test_move_to_pre_grasp_strict_threshold_is_06_for_top_down() -> None:
    """Strict move threshold for top-down is 0.06 (not the old 0.08)."""
    env = PregraspThresholdEnv()
    candidate = GraspCandidate(
        point_3d=np.array([0.5, 0.2, 0.9], dtype=np.float32),
        approach_dir=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        finger_width_m=0.04,
        score=0.8,
    )

    env.move_to_pre_grasp(candidate)
    assert env.move_calls[-1][2] == 0.06


class _BaseAwareEnv(PregraspThresholdEnv):
    """PregraspThresholdEnv variant with controllable real base xy.

    Used to verify the legacy base_approach gating against the actual
    base-to-target distance.
    """

    def __init__(self, real_base_xy: np.ndarray) -> None:
        super().__init__()
        self._real_base_xy = np.asarray(real_base_xy, dtype=np.float64)

    def _read_real_base_xy(self):
        return self._real_base_xy.copy()


def test_pre_grasp_skips_legacy_base_approach_at_recovery_offset() -> None:
    """Regression: base at recovery offset (0.65m) must not trigger
    legacy base_approach. GPU log showed this caused base to be pushed
    to target.x - 0.4 (wrong direction), corrupting the next pre-grasp.
    """
    target_xy = np.array([0.125, -2.857], dtype=np.float64)
    base_xy = np.array([0.714, -2.872], dtype=np.float64)
    # Distance ≈ 0.589 (act-entry navigate at offset 0.55).
    # After recovery navigate at offset 0.65, distance becomes ≈ 0.657m.
    recovery_base_xy = np.array([0.78, -2.87], dtype=np.float64)
    assert 0.60 < float(np.linalg.norm(recovery_base_xy - target_xy)) < 0.70

    env = _BaseAwareEnv(real_base_xy=recovery_base_xy)
    candidate = GraspCandidate(
        point_3d=np.array([target_xy[0], target_xy[1], 0.932], dtype=np.float32),
        approach_dir=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        finger_width_m=0.04,
        score=0.8,
    )

    env.move_to_pre_grasp_diagnostic(candidate)

    # Only the strict pre_pos move should be issued; no drive_base=True
    # legacy base_approach toward target.x - 0.4.
    legacy_targets = [
        call[0] for call in env.move_calls
        if abs(float(call[0][0]) - (target_xy[0] - 0.4)) < 1e-3
    ]
    assert legacy_targets == [], (
        f"legacy base_approach must be skipped when base is at recovery "
        f"offset, but got moves to {legacy_targets}"
    )


def test_pre_grasp_runs_legacy_base_approach_when_base_truly_far() -> None:
    """When base is genuinely far (e.g. legacy mock with no Phase 4 nav),
    legacy base_approach must still trigger to drive the base closer.
    """
    target_xy = np.array([0.5, 0.2], dtype=np.float64)
    far_base_xy = np.array([2.0, 0.2], dtype=np.float64)
    assert float(np.linalg.norm(far_base_xy - target_xy)) > 1.0

    env = _BaseAwareEnv(real_base_xy=far_base_xy)
    candidate = GraspCandidate(
        point_3d=np.array([target_xy[0], target_xy[1], 0.9], dtype=np.float32),
        approach_dir=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        finger_width_m=0.04,
        score=0.8,
    )

    env.move_to_pre_grasp_diagnostic(candidate)

    legacy_targets = [
        call[0] for call in env.move_calls
        if abs(float(call[0][0]) - (target_xy[0] - 0.4)) < 1e-3
    ]
    assert len(legacy_targets) >= 1, (
        f"legacy base_approach must run when base is far; calls: {env.move_calls}"
    )


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


# ============================================================
# Phase 3: drive_base opt-in flag tests
# ============================================================

class _StepActionRecorderWithBase(EnvWrapper):
    """Like _StepActionRecorder but exposes a base controller (idx=7,8).

    Used to verify the drive_base opt-in: default arm-only must not write
    to action[base_idx], opt-in must write a non-zero base velocity.
    """
    BASE_IDX = 7  # 6 arm + base[7..8] (forward/side) + 1 gripper = 10
    GRIPPER_IDX = 9
    ACTION_DIM = 10

    def __init__(self) -> None:
        self.actions_logged: list[np.ndarray] = []
        self._eef = np.array([0.5, 0.0, 0.6], dtype=np.float32)
        self._latest_obs = {"_": True}
        self._gripper_idx_cache = self.GRIPPER_IDX
        self._base_idx_cache = self.BASE_IDX
        outer = self

        class _MockBackend:
            action_dim = outer.ACTION_DIM

            def step(self, action):
                a = np.asarray(action, dtype=np.float32).copy()
                outer.actions_logged.append(a)
                outer._eef = outer._eef + a[0:3] * 0.5
                return {}, 0.0, False, {}

        self._env = _MockBackend()

    def get_eef_pos(self) -> np.ndarray:
        return self._eef.copy()

    def get_base_pose(self):
        return np.zeros(3, dtype=np.float32), np.eye(3, dtype=np.float64)

    def render(self) -> None:
        pass

    def reset(self):
        return {}


def test_move_arm_to_default_does_not_drive_base() -> None:
    """drive_base=False (default) must leave base[7], base[8] at 0 every step."""
    env = _StepActionRecorderWithBase()
    # Target far enough (dist=1m > 0.05m threshold) to expose base driving
    target = np.array([1.5, 0.0, 0.6], dtype=np.float32)
    env.move_arm_to(target, threshold_m=0.1, max_steps=10)

    assert len(env.actions_logged) > 0
    for i, a in enumerate(env.actions_logged):
        assert a[env.BASE_IDX] == 0.0, (
            f"step {i}: action[base_fwd]={a[env.BASE_IDX]} (expected 0 "
            f"with drive_base=False)"
        )
        assert a[env.BASE_IDX + 1] == 0.0, (
            f"step {i}: action[base_side]={a[env.BASE_IDX + 1]}"
        )


def test_move_arm_to_drive_base_true_writes_base_action() -> None:
    """drive_base=True (opt-in) must restore legacy mixed control:
    base[7]/base[8] should receive non-zero values when dist > 0.05m."""
    env = _StepActionRecorderWithBase()
    target = np.array([1.5, 0.0, 0.6], dtype=np.float32)
    env.move_arm_to(target, threshold_m=0.1, max_steps=10, drive_base=True)

    assert len(env.actions_logged) > 0
    base_fwds = [a[env.BASE_IDX] for a in env.actions_logged]
    assert any(abs(v) > 1e-6 for v in base_fwds), (
        f"drive_base=True did not write any base action: {base_fwds}"
    )


class _PreGraspMoveSpyEnv(EnvWrapper):
    """Captures move_arm_to kwargs to verify move_to_pre_grasp's internal
    base approach uses drive_base=True (Phase 3 fallback contract)."""

    def __init__(self) -> None:
        self.move_calls: list[dict] = []
        self._eef = np.array([0.0, 0.0, 0.6], dtype=np.float32)

    def get_eef_pos(self) -> np.ndarray:
        return self._eef.copy()

    def move_arm_to(self, target_pos_m, **kwargs) -> bool:
        # Record kwargs verbatim so we can assert drive_base is passed
        self.move_calls.append({"target": np.asarray(target_pos_m).copy(),
                                **kwargs})
        return True

    def _gripper_action(self, *_args, **_kwargs) -> None:
        pass


def test_move_to_pre_grasp_passes_drive_base_true_for_base_approach() -> None:
    """move_to_pre_grasp's internal moves must opt-in to drive_base=True.

    Both the legacy base approach (skipped when Phase 4 navigate already
    positioned the base) AND the final precision move now use drive_base=True.
    The precision move uses base driving so the velocity controller can make
    small self-damping corrections to overcome the arm mount offset (~6cm
    lateral) that arm-only OSC cannot resolve. The base velocity scales with
    distance and deactivates at <5cm, so this is safe for fine positioning.

    GPU run fcdd0ca confirmed arm-only fine pre-grasp plateaus at ~6cm
    lateral and never converges, so drive_base=True is now mandatory.
    """
    env = _PreGraspMoveSpyEnv()
    candidate = GraspCandidate(
        point_3d=np.array([0.5, 0.2, 0.9], dtype=np.float32),
        approach_dir=np.array([1.0, 0.0, 0.0], dtype=np.float32),  # side grasp
        finger_width_m=0.04,
        score=0.8,
    )

    env.move_to_pre_grasp(candidate)

    # First call should be the base approach (eef-height z); must opt-in to base.
    assert len(env.move_calls) >= 1
    first = env.move_calls[0]
    assert first.get("drive_base") is True, (
        f"base approach must pass drive_base=True, got kwargs={first}"
    )
    # The final precision move ALSO must request base driving so the velocity
    # controller can compensate the arm mount offset during convergence.
    last = env.move_calls[-1]
    assert last.get("drive_base") is True, (
        f"final precision move must pass drive_base=True to overcome "
        f"the arm mount offset, got kwargs={last}"
    )


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
