"""Phase 7 unit tests: move_arm_to stall detection.

Verifies that move_arm_to detects OSC stall (EEF position not changing
despite repeated env.step calls) and exits early instead of running to
max_steps.

Background: Phase 5 GPU run showed move_arm_to running 800 steps (50s+)
with dist unchanged. Old stall detection (check_interval=120,
stall_limit=6) needed 720 steps to fire, missing the early break window.

Phase 7 tightens to check_interval=40, stall_limit=3 → break at ~120 steps.

See: docs/09 §7 (forthcoming).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.env_wrapper import EnvWrapper  # noqa: E402


class _StallEnv(EnvWrapper):
    """Mock env that simulates OSC stall: EEF never moves regardless of action.

    Used to test that move_arm_to detects stall and returns early.
    """

    def __init__(self) -> None:
        self.actions_logged: list[np.ndarray] = []
        self._eef = np.array([0.5, 0.0, 0.6], dtype=np.float32)
        self._latest_obs = {"_": True}
        self._gripper_idx_cache = 7
        outer = self

        class _Backend:
            action_dim = 8

            def step(self, action):
                outer.actions_logged.append(
                    np.asarray(action, dtype=np.float32).copy()
                )
                # EEF unchanged - simulates OSC stall (e.g. arm at IK
                # singularity or joint-limit-locked).
                return {}, 0.0, False, {}

        self._env = _Backend()

    def get_eef_pos(self) -> np.ndarray:
        return self._eef.copy()

    def get_base_pose(self):
        return np.zeros(3, dtype=np.float32), np.eye(3, dtype=np.float64)

    def _get_base_action_idx(self):
        return None  # no base in mock

    def render(self) -> None:
        pass

    def reset(self):
        return {}


class _SlowProgressEnv(EnvWrapper):
    """Mock env where EEF makes very slow progress (1mm per step).

    Used to verify stall detection doesn't false-positive on slow but
    real convergence. With check_interval=40, progress per window
    should be ~40mm — well above 1mm threshold.
    """

    def __init__(self, mm_per_step: float = 0.001) -> None:
        self.actions_logged: list[np.ndarray] = []
        self._eef = np.array([0.5, 0.0, 0.6], dtype=np.float32)
        self._latest_obs = {"_": True}
        self._gripper_idx_cache = 7
        self._mm_per_step = mm_per_step
        outer = self

        class _Backend:
            action_dim = 8

            def step(self, action):
                outer.actions_logged.append(
                    np.asarray(action, dtype=np.float32).copy()
                )
                # Slow progress: move EEF mm_per_step toward action direction
                a = np.asarray(action, dtype=np.float32)
                if np.linalg.norm(a[0:3]) > 1e-6:
                    direction = a[0:3] / np.linalg.norm(a[0:3])
                    outer._eef = outer._eef + direction * outer._mm_per_step
                return {}, 0.0, False, {}

        self._env = _Backend()

    def get_eef_pos(self) -> np.ndarray:
        return self._eef.copy()

    def get_base_pose(self):
        return np.zeros(3, dtype=np.float32), np.eye(3, dtype=np.float64)

    def _get_base_action_idx(self):
        return None

    def render(self) -> None:
        pass

    def reset(self):
        return {}


# ======================================================================
# Tests
# ======================================================================


def test_move_arm_to_breaks_early_when_eef_completely_stalled():
    """Phase 7 regression: 800-step OSC stall must terminate in ~120 steps.

    Old detection (check_interval=120, stall_limit=6) needed 720 steps.
    New: check_interval=40, stall_limit=3 → first break possible at
    step 40*3 = 120.
    """
    env = _StallEnv()
    target = np.array([1.0, 0.0, 0.6], dtype=np.float32)  # dist = 0.5m
    ok = env.move_arm_to(target, threshold_m=0.005, max_steps=800)
    assert ok is False  # stalled, never reached target
    # Should break much earlier than 800 max_steps.
    n_steps = len(env.actions_logged)
    assert n_steps < 200, (
        f"Phase 7 stall detection failed: ran {n_steps} steps "
        f"before breaking (expected <200, was 720 in old impl)"
    )
    # Should not break too early either - need at least a few check intervals.
    assert n_steps >= 80, (
        f"Phase 7 stall detection broke too early: only {n_steps} steps. "
        f"Need at least 80 (= 2 check intervals) to confirm stall."
    )


def test_move_arm_to_does_not_false_positive_on_slow_progress():
    """Phase 7: slow but real convergence (1mm/step) must NOT trigger stall.

    1mm/step × 40 step interval = 40mm progress per check window.
    Progress threshold (Phase 7) is 1mm per window. 40mm >> 1mm → no stall.
    """
    env = _SlowProgressEnv(mm_per_step=0.001)  # 1mm/step
    target = np.array([0.6, 0.0, 0.6], dtype=np.float32)  # dist = 0.1m
    # Should converge fully: 0.1m / 0.001 = 100 steps just enough.
    # Use generous max_steps to let it finish.
    ok = env.move_arm_to(target, threshold_m=0.005, max_steps=300)
    # Either converges OR runs to max_steps (slow), but must NOT stall-break
    # early with a non-decreasing dist trajectory.
    n_steps = len(env.actions_logged)
    if not ok:
        # Should have run nearly to max_steps if not converged
        # (stall detection should be slow-progress-friendly).
        assert n_steps >= 95, (
            f"slow progress mis-classified as stall after {n_steps} steps"
        )


def test_move_arm_to_stall_log_contains_recent_dists():
    """Phase 7: stall log should expose dist trajectory for diagnosis.

    Verify that the warning log includes 'recent_dists=[...]' for GPU
    debugging (we can tell apart true-stall vs slow-progress in log).
    """
    import logging
    env = _StallEnv()
    target = np.array([1.0, 0.0, 0.6], dtype=np.float32)

    # Capture stall warning
    captured = []

    class _Capture(logging.Handler):
        def emit(self, record):
            captured.append(record.getMessage())

    logger_name = "src.env_wrapper"
    log = logging.getLogger(logger_name)
    handler = _Capture()
    log.addHandler(handler)
    log.setLevel(logging.WARNING)
    try:
        env.move_arm_to(target, threshold_m=0.005, max_steps=800)
    finally:
        log.removeHandler(handler)

    stall_msgs = [m for m in captured if "stalled at step" in m]
    assert len(stall_msgs) == 1, (
        f"expected exactly 1 stall warning, got {len(stall_msgs)}: {captured}"
    )
    msg = stall_msgs[0]
    assert "recent_dists=" in msg, (
        f"stall log missing recent_dists trajectory: {msg}"
    )


def test_move_arm_to_converges_normally_when_not_stalled():
    """Phase 7 sanity: when EEF actually moves toward target, must converge.

    Uses _StepActionRecorder which advances eef by 50% of action delta per
    step (i.e. fast convergence).
    """
    # Reuse the existing fast-convergence mock from test_env_wrapper_grasp
    from tests.test_env_wrapper_grasp import _StepActionRecorder
    env = _StepActionRecorder()
    target = np.array([0.6, 0.0, 0.65], dtype=np.float32)
    ok = env.move_arm_to(target, threshold_m=0.01, max_steps=50)
    assert ok is True, "fast convergence should reach target"
    # Should converge in a few steps (not the whole 50)
    assert len(env.actions_logged) < 30


# ======================================================================
# Phase 7 step 3: IK-unreachable regression detection
# ======================================================================


class _IKBoundaryEnv(EnvWrapper):
    """Mock env that simulates IK boundary: arm converges to `best_dist`,
    then drifts away monotonically.

    Mimics Run 6 GPU trajectory:
        Steps 1..approach_steps: dist decreases toward target by ~1cm/step
        Steps approach_steps+1..: dist increases by `drift_per_step` each step

    This models the OSC at joint limits: the closest reachable point
    becomes the "best_dist", further commands drift the arm away.
    """

    def __init__(
        self,
        approach_steps: int = 15,
        target_xyz: tuple[float, float, float] = (1.0, 0.0, 0.6),
        drift_per_step: float = 0.0003,  # 0.3mm/step
        approach_step_size: float = 0.01,  # 1cm/step approach
    ) -> None:
        self.actions_logged: list[np.ndarray] = []
        self._eef = np.array([0.5, 0.0, 0.6], dtype=np.float32)
        self._target_xyz = np.asarray(target_xyz, dtype=np.float32)
        self._latest_obs = {"_": True}
        self._gripper_idx_cache = 7
        self._step_counter = 0
        self._approach_steps = approach_steps
        self._approach_step_size = approach_step_size
        self._drift = drift_per_step
        outer = self

        class _Backend:
            action_dim = 8

            def step(self, action):
                outer.actions_logged.append(
                    np.asarray(action, dtype=np.float32).copy()
                )
                outer._step_counter += 1
                direction_to_target = outer._target_xyz - outer._eef
                n = float(np.linalg.norm(direction_to_target))
                if n < 1e-6:
                    return {}, 0.0, False, {}
                if outer._step_counter <= outer._approach_steps:
                    # Approach: move toward target by approach_step_size
                    outer._eef = outer._eef + (
                        direction_to_target / n * outer._approach_step_size
                    ).astype(np.float32)
                else:
                    # Drift: move AWAY from target
                    outer._eef = outer._eef - (
                        direction_to_target / n * outer._drift
                    ).astype(np.float32)
                return {}, 0.0, False, {}

        self._env = _Backend()

    def get_eef_pos(self) -> np.ndarray:
        return self._eef.copy()

    def get_base_pose(self):
        return np.zeros(3, dtype=np.float32), np.eye(3, dtype=np.float64)

    def _get_base_action_idx(self):
        return None

    def render(self) -> None:
        pass

    def reset(self):
        return {}


def test_move_arm_to_detects_ik_unreachable_regression_breaks_fast():
    """Phase 7 step 3: when arm reaches its IK boundary and drifts back,
    move_arm_to must break QUICKLY (regression > 5mm from best), NOT
    wait for 3 consecutive stall windows (~160+ steps).

    Run 6 GPU log: arm reached best ~0.38m, drifted back to 0.41m by
    step 720 — 720 steps wasted. With regression detection, break
    fires shortly after gate (40 steps) once 5mm regress accumulates.
    """
    env = _IKBoundaryEnv(
        approach_steps=15,
        target_xyz=(1.0, 0.0, 0.6),  # init eef at (0.5, 0, 0.6), dist=0.5m
        drift_per_step=0.0003,  # 0.3mm/step → 5mm drift in ~17 steps
    )
    target = np.array([1.0, 0.0, 0.6], dtype=np.float32)
    ok = env.move_arm_to(target, threshold_m=0.02, max_steps=800)
    assert ok is False, "IK-unreachable target should not converge"
    n_steps = len(env.actions_logged)
    # Regression check first fires at step > check_interval (40); needs
    # 5mm regress accumulated from best at step 15. Expected ~41-50.
    assert 40 < n_steps < 200, (
        f"IK regression detection failed: ran {n_steps} steps "
        f"(expected 40<n<200; Run 6 baseline was 720)"
    )


def test_move_arm_to_regression_log_emitted():
    """Phase 7 step 3: regression detection emits a warning containing
    'IK-unreachable regression' so GPU logs can distinguish it from
    plain stall (dist not changing) failures."""
    import logging
    env = _IKBoundaryEnv()  # uses approach_steps=15, drift=0.0003
    target = np.array([1.0, 0.0, 0.6], dtype=np.float32)
    captured: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record):
            captured.append(record.getMessage())

    log = logging.getLogger("src.env_wrapper")
    handler = _Capture()
    log.addHandler(handler)
    log.setLevel(logging.WARNING)
    try:
        env.move_arm_to(target, threshold_m=0.02, max_steps=800)
    finally:
        log.removeHandler(handler)

    regress_msgs = [
        m for m in captured if "IK-unreachable regression" in m
    ]
    assert len(regress_msgs) == 1, (
        f"expected 1 IK regression warning, got {len(regress_msgs)}: {captured}"
    )
    assert "best=" in regress_msgs[0]
    assert "regress=" in regress_msgs[0]


def test_move_arm_to_regression_does_not_fire_when_stuck_from_start():
    """Phase 7 step 3 guard: when arm never makes meaningful progress
    from init_dist, regression detection must NOT fire (gate 3 of 4:
    init_dist - best_dist must exceed min_progress_for_regress=1cm).

    Stall detection should fire instead, which is the correct semantic
    for a fully-stuck arm (different from IK-unreachable boundary).
    """
    import logging
    env = _StallEnv()  # EEF never moves
    target = np.array([1.0, 0.0, 0.6], dtype=np.float32)
    captured: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record):
            captured.append(record.getMessage())

    log = logging.getLogger("src.env_wrapper")
    handler = _Capture()
    log.addHandler(handler)
    log.setLevel(logging.WARNING)
    try:
        env.move_arm_to(target, threshold_m=0.005, max_steps=800)
    finally:
        log.removeHandler(handler)

    regress_msgs = [m for m in captured if "IK-unreachable regression" in m]
    stall_msgs = [m for m in captured if "stalled at step" in m]
    assert len(regress_msgs) == 0, (
        f"regression must NOT fire when arm is fully stuck: {captured}"
    )
    assert len(stall_msgs) == 1, (
        f"stall detection should fire instead, got: {captured}"
    )
