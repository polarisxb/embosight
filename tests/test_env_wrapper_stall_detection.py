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
