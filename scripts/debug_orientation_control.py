"""Standalone debug script: probe robosuite OSC_POSE orientation action convention.

This script issues pure rotation actions (action[3:6]) and logs how the gripper
orientation changes. Use to determine:
  (a) Format: axis-angle (rotation vector) vs Euler vs quaternion delta?
  (b) Frame: base / world / EEF?
  (c) Effective gain: action magnitude → radians per step?

REQUIREMENT: must run on the GPU server where robosuite + MuJoCo sim works.
Do NOT run on a CPU-only machine — env reset will fail.

Usage:
    python scripts/debug_orientation_control.py 2>&1 | tee /tmp/osc_debug.log

Findings (filled in after running):
    [ ] Format     : ???
    [ ] Frame      : ???
    [ ] Sample gain: action[4]=0.1 → about ??? rad/step around base-y
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
import numpy as np
from scipy.spatial.transform import Rotation as R

import yaml
from src.env_wrapper import EnvConfig, EnvWrapper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("debug_osc")


def get_gripper_quat_world(env) -> np.ndarray:
    """Return current gripper orientation as quaternion (xyzw) in world frame.

    Uses the standard robosuite observation key. If not available, falls back
    to site lookup via sim.data.
    """
    obs = env._latest_obs or {}
    q = obs.get("robot0_eef_quat")
    if q is not None:
        return np.asarray(q, dtype=np.float64)
    # Fallback: try site lookup
    try:
        sim = env._env.sim
        # Common site names for Panda gripper
        for name in (
            "gripper0_right_grip_site",
            "robot0_right_grip_site",
            "gripper0_grip_site",
            "robot0_grip_site",
        ):
            try:
                site_id = sim.model.site_name2id(name)
                xmat = sim.data.site_xmat[site_id].reshape(3, 3)
                return R.from_matrix(xmat).as_quat().astype(np.float64)
            except Exception:
                continue
    except Exception as e:
        logger.error(f"Cannot read gripper quat: {e}")
    raise RuntimeError("No way to read gripper orientation from sim")


def issue_action_and_step(env, action: np.ndarray, n_steps: int = 30):
    """Issue the same action for n_steps and return the final obs."""
    for _ in range(n_steps):
        obs, _, _, _ = env._env.step(action)
        env._latest_obs = obs
        try:
            env.render()
        except Exception:
            pass
    return obs


def describe_rotation_delta(q_before: np.ndarray, q_after: np.ndarray) -> str:
    """Compute axis-angle delta between two world-frame quaternions."""
    r_before = R.from_quat(q_before)
    r_after = R.from_quat(q_after)
    r_delta = r_after * r_before.inv()  # world-frame delta rotation
    rotvec_world = r_delta.as_rotvec()
    angle = float(np.linalg.norm(rotvec_world))
    if angle < 1e-6:
        return f"Δ ≈ 0 (angle={angle:.6f} rad)"
    axis_world = rotvec_world / angle
    return (
        f"Δangle={angle:.4f} rad ({np.degrees(angle):.2f}°), "
        f"axis_world=[{axis_world[0]:+.3f}, {axis_world[1]:+.3f}, {axis_world[2]:+.3f}]"
    )


def probe_axis(env, axis_idx: int, magnitude: float = 0.1, n_steps: int = 30):
    """Issue pure rotation on action[3+axis_idx] and report what happened."""
    action_dim = env._env.action_dim
    action = np.zeros(action_dim, dtype=np.float32)
    action[3 + axis_idx] = magnitude
    label = ["X", "Y", "Z"][axis_idx]

    # Snapshot before
    q_before = get_gripper_quat_world(env)
    _, base_ori_before = env.get_base_pose()

    # Step
    issue_action_and_step(env, action, n_steps=n_steps)

    # Snapshot after
    q_after = get_gripper_quat_world(env)

    # World-frame delta
    delta_str = describe_rotation_delta(q_before, q_after)

    # Convert world axis to base frame
    r_delta = R.from_quat(q_after) * R.from_quat(q_before).inv()
    rotvec_world = r_delta.as_rotvec()
    angle = float(np.linalg.norm(rotvec_world))
    if angle > 1e-6:
        axis_world = rotvec_world / angle
        axis_base = base_ori_before.T @ axis_world
    else:
        axis_base = np.zeros(3)

    logger.info(
        f"action[3+{axis_idx}]={magnitude:+.2f} ({label}-axis attempt, {n_steps} steps): "
        f"{delta_str}"
    )
    logger.info(
        f"  → axis_base=[{axis_base[0]:+.3f}, {axis_base[1]:+.3f}, {axis_base[2]:+.3f}], "
        f"effective_rad_per_step={angle / n_steps:.5f}"
    )
    return angle, axis_base


def main():
    env = EnvWrapper(EnvConfig())
    env.seed(3)
    env.reset()

    logger.info("=" * 60)
    logger.info("OSC_POSE ORIENTATION ACTION CONVENTION PROBE")
    logger.info("=" * 60)

    action_dim = env._env.action_dim
    base_idx = env._get_base_action_idx()
    logger.info(f"action_dim={action_dim}, base_idx={base_idx}")
    logger.info(
        "Action layout: assumed right_arm[0:6]=(Δpos[0:3], Δori[3:6]), "
        f"base[{base_idx}:{base_idx+3 if base_idx else '?'}], gripper[...]."
    )

    q0 = get_gripper_quat_world(env)
    logger.info(f"Initial gripper quat (xyzw) world: {q0}")
    r0 = R.from_quat(q0)
    logger.info(f"Initial gripper z-axis in world: {r0.apply([0, 0, 1])}")
    logger.info(f"Initial gripper x-axis in world: {r0.apply([1, 0, 0])}")
    logger.info("")

    # Probe each rotation axis independently
    for axis in range(3):
        logger.info(f"--- Probe rotation axis {axis} (X/Y/Z) ---")
        probe_axis(env, axis, magnitude=0.1, n_steps=30)
        logger.info("")
        # Recover (reset orientation by issuing opposite, partially)
        action_dim = env._env.action_dim
        recover = np.zeros(action_dim, dtype=np.float32)
        recover[3 + axis] = -0.1
        issue_action_and_step(env, recover, n_steps=20)

    logger.info("=" * 60)
    logger.info("INTERPRETATION GUIDE")
    logger.info("=" * 60)
    logger.info("If for action[3+k]=+m, observed rotation axis (base) is +k-th unit vector:")
    logger.info("  → CONVENTION = axis-angle in base frame (expected for OSC_POSE base ref)")
    logger.info("If axis is rotated 90°/-90°: probably world frame, not base.")
    logger.info("If small / no rotation: gain very small or OSC orientation locked.")
    logger.info("If chaotic rotation: probably Euler angles (less common).")

    try:
        env.close()
    except Exception:
        pass


def test_live_side_rotation():
    """After main() probe, verify that move_arm_to(approach_dir=[1,0,0]) actually
    rotates the gripper to face +x.

    Only runs after Task 5 (move_arm_to extension) is merged. Comment out the
    @skip marker once you reach Task 6.
    """
    env = EnvWrapper(EnvConfig())
    env.seed(3)
    env.reset()

    q_before = get_gripper_quat_world(env)
    z_world_before = R.from_quat(q_before).apply([0, 0, 1])
    logger.info(f"[live] Before: gripper z in world: {z_world_before}")

    current_pos = env.get_eef_pos()
    target = current_pos + np.array([0.05, 0.0, 0.0])
    env.move_arm_to(
        target, approach_dir=np.array([1.0, 0.0, 0.0]),
        max_steps=600, threshold_m=0.02,
    )

    q_after = get_gripper_quat_world(env)
    z_world_after = R.from_quat(q_after).apply([0, 0, 1])
    logger.info(f"[live] After:  gripper z in world: {z_world_after}")

    dot_x = float(np.dot(z_world_after, [1, 0, 0]))
    dot_minus_z = float(np.dot(z_world_after, [0, 0, -1]))
    logger.info(f"[live] dot(z_world, +x)={dot_x:.3f}, dot(z_world, -z)={dot_minus_z:.3f}")

    assert dot_x > 0.5, (
        f"Side rotation FAILED: gripper z should mostly point +x, got dot={dot_x:.3f}"
    )
    logger.info("[live] PASS: gripper rotated to face +x")

    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
    test_live_side_rotation()
