# Orientation-Aware Grasping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the 5 grasp strategies (top_down / gentle_side / handle_grasp / scoop_under / refuse) actually execute differently by adding end-effector orientation control to `move_arm_to` and converting `descend` into a direction-aware `approach`, unlocking workspace for thin/long objects.

**Architecture:** robosuite Panda OSC_POSE controller takes `action[0:6] = (Δpos, Δori_axis_angle)` in base frame. Current `move_arm_to` only sets `action[0:3]` and leaves `action[3:6] = 0` — orientation never changes. We add an optional `approach_dir` parameter to `move_arm_to` that computes the target gripper orientation (gripper z-axis aligned with `-approach_dir`) and feeds axis-angle deltas into `action[3:6]`. `descend` is replaced/wrapped by `approach(point_3d, approach_dir)`, which calls `move_arm_to` with the target position offset by `-approach_dir * margin` and the corresponding orientation. The 4 callers (`move_to_pre_grasp`, `descend`, `lift`, `action_executor.act`) are updated to pass `approach_dir` consistently.

**Tech Stack:** Python 3.10+, robosuite OSC_POSE, MuJoCo, scipy.spatial.transform.Rotation (already in deps), numpy, pytest.

**Spec:** `docs/superpowers/specs/2026-05-16-thin-object-grasping-analysis.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `scripts/debug_orientation_control.py` | Create | Standalone repl-style script to verify OSC_POSE orientation control axis convention |
| `src/env_wrapper.py:312-414` | Modify | `move_arm_to` accepts `approach_dir`, sets `action[3:6]` |
| `src/env_wrapper.py:1334-1361` | Modify | `move_to_pre_grasp` reads `candidate.approach_dir`, pre-grasp pos offset by `-approach_dir` |
| `src/env_wrapper.py:1363-1377` | Modify | Add `approach(point_3d, approach_dir, ...)`, keep `descend` as alias |
| `src/env_wrapper.py:1414-1448` | Modify | `lift` accepts `approach_dir` for retreat-along-approach option |
| `src/action_executor.py:91-188` | Modify | `act` passes `candidate.approach_dir` to `move_to_pre_grasp` / `approach` / `lift` |
| `src/grasp_planner.py:100-107` | Modify | Validate strategy `approach_dir` is non-degenerate; add `pre_grasp_offset_m` param |
| `tests/test_env_wrapper_orientation.py` | Create | Unit tests for orientation math (mocked sim) |
| `tests/test_action_executor_v1.py` | Modify | `FakeEnv` accepts `approach_dir`; new test verifies side approach calls correct trajectory |
| `tests/test_orientation_integration.py` | Create | Smoke test: side approach reaches lower z than top_down on a tall object |

---

## Pre-flight: Read These First

Before writing any code, read these files in full:

- [ ] `src/env_wrapper.py:312-414` (`move_arm_to` current implementation)
- [ ] `src/env_wrapper.py:1334-1377` (`move_to_pre_grasp`, `descend`)
- [ ] `src/env_wrapper.py:1013-1100` (`_descend_until_contact`)
- [ ] `src/action_executor.py:43-188` (`act` method, where strategies are applied)
- [ ] `src/grasp_planner.py:97-130` (`_STRATEGY_PARAMS`)
- [ ] `src/world_belief.py` — find `GraspCandidate` dataclass, confirm it has `approach_dir`
- [ ] `docs/superpowers/specs/2026-05-16-thin-object-grasping-analysis.md` (full spec)
- [ ] robosuite docs / source: `robosuite/controllers/parts/arm/osc.py` — confirm OSC_POSE `action[3:6]` is axis-angle in base frame or something else

---

## Task 1: Investigate OSC_POSE Orientation Convention (Mandatory, Cannot Skip)

**Why:** The whole plan depends on `action[3:6]` actually rotating the gripper. We need to confirm:
- Is it axis-angle, Euler angles, or quaternion delta?
- Is the rotation in world frame, base frame, or end-effector frame?
- What is the gain / units?

**Files:**
- Create: `scripts/debug_orientation_control.py`

- [ ] **Step 1: Find OSC_POSE controller source**

```bash
# Run in repo root:
fd osc /share/home/dlxt120210024/embodied/robosuite/robosuite/controllers
```

Expected output (sample): `osc.py` path. Open it and find where `action[3:6]` (or the corresponding indices for orientation) is consumed. Note exact convention.

- [ ] **Step 2: Create debug script that issues pure rotation actions**

```python
# scripts/debug_orientation_control.py
"""Standalone debug script: command pure rotation deltas to gripper and log
the resulting orientation change. Use to determine OSC_POSE axis-angle
convention before implementing approach_dir."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.spatial.transform import Rotation as R

from src.config_loader import load_config
from src.env_wrapper import EnvWrapper


def main():
    cfg = load_config("configs/default.yaml")
    env = EnvWrapper(cfg["simulator"])
    env.reset(seed=3)

    # Get initial gripper orientation (quaternion from sim)
    sim = env._env.sim
    site_id = sim.model.site_name2id("gripper0_right_grip_site")
    initial_xmat = sim.data.site_xmat[site_id].reshape(3, 3).copy()
    initial_quat = R.from_matrix(initial_xmat).as_quat()
    print(f"Initial gripper rotation matrix:\n{initial_xmat}")
    print(f"Initial quat (xyzw): {initial_quat}")

    # Issue pure rotation action: rotate +Y axis by small angle
    action_dim = env._env.action_dim
    base_idx = env._get_base_action_idx()
    print(f"action_dim={action_dim}, base_idx={base_idx}")

    # Convention candidates to test:
    # (a) action[3:6] is axis-angle (radians) in world frame
    # (b) action[3:6] is axis-angle in base frame
    # (c) action[3:6] is Euler XYZ (radians)

    for trial in range(3):
        action = np.zeros(action_dim, dtype=np.float32)
        action[3:6] = np.array([0.0, 0.1, 0.0])  # +Y rotation
        for _ in range(30):
            obs, _, _, _ = env._env.step(action)
        new_xmat = sim.data.site_xmat[site_id].reshape(3, 3).copy()
        new_quat = R.from_matrix(new_xmat).as_quat()
        delta = R.from_matrix(initial_xmat.T @ new_xmat).as_rotvec()
        print(f"Trial {trial}: delta rotvec = {delta}")
        initial_xmat = new_xmat

    env.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the script and document findings**

Run: `python scripts/debug_orientation_control.py 2>&1 | tee /tmp/osc_debug.log`

Expected: log shows rotation deltas. Document in a comment at the top of the file:
- Convention found (axis-angle / Euler / quaternion delta)
- Frame (world / base / EEF)
- Gain (e.g., `action[3:6] = 1.0` produces N radians per step)

**If the rotation does NOT change** the gripper orientation, escalate: check `composite_controller_factory.py`, the arm controller config json, and search for `input_ref_frame`. Do not proceed to Task 2 until confirmed working.

- [ ] **Step 4: Commit findings**

```bash
git add scripts/debug_orientation_control.py
git commit -m "debug: OSC_POSE orientation control axis-angle convention investigation"
```

---

## Task 2: Helper — Compute Target Gripper Quaternion from approach_dir

**Files:**
- Modify: `src/env_wrapper.py` (add helper method)
- Test: `tests/test_env_wrapper_orientation.py` (create)

**Math:** The gripper's local z-axis (in default configuration) points down (towards the object during top_down grasp). For arbitrary `approach_dir`, we want the gripper's local z-axis to align with `+approach_dir` (so the gripper "looks at" the object from the approach direction).

- [ ] **Step 1: Write failing test for `_approach_dir_to_quat`**

```python
# tests/test_env_wrapper_orientation.py
"""Tests for orientation math helpers in EnvWrapper."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R


class TestApproachDirToQuat:
    def test_top_down_returns_identity_like(self):
        """approach_dir = -z (gripper points down) should give base quaternion."""
        from src.env_wrapper import EnvWrapper
        q = EnvWrapper._approach_dir_to_quat(np.array([0.0, 0.0, -1.0]))
        # Gripper z-axis after rotation should point in +approach_dir = -z
        rot = R.from_quat(q)
        gripper_z_world = rot.apply(np.array([0.0, 0.0, 1.0]))
        # We pointed it at -z (down), so gripper z in world should be -z
        np.testing.assert_allclose(gripper_z_world, [0.0, 0.0, -1.0], atol=1e-6)

    def test_side_x_approach(self):
        """approach_dir = +x should rotate gripper so its z points in +x."""
        from src.env_wrapper import EnvWrapper
        q = EnvWrapper._approach_dir_to_quat(np.array([1.0, 0.0, 0.0]))
        rot = R.from_quat(q)
        gripper_z_world = rot.apply(np.array([0.0, 0.0, 1.0]))
        np.testing.assert_allclose(gripper_z_world, [1.0, 0.0, 0.0], atol=1e-6)

    def test_normalizes_input(self):
        """Non-unit approach_dir should still produce a valid quat."""
        from src.env_wrapper import EnvWrapper
        q = EnvWrapper._approach_dir_to_quat(np.array([2.0, 0.0, 0.0]))
        assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-6)
```

- [ ] **Step 2: Run test, confirm failure**

```bash
python -m pytest tests/test_env_wrapper_orientation.py::TestApproachDirToQuat -v
```

Expected: ImportError or AttributeError (method not defined).

- [ ] **Step 3: Implement `_approach_dir_to_quat` in EnvWrapper**

Add this static method to `src/env_wrapper.py` just before `move_arm_to`:

```python
    @staticmethod
    def _approach_dir_to_quat(approach_dir: np.ndarray) -> np.ndarray:
        """Compute gripper quaternion so its local +z axis points in approach_dir.

        Args:
            approach_dir: 3D unit vector (will be normalized) in world frame.

        Returns:
            Quaternion (xyzw) representing the target gripper rotation in world.
        """
        from scipy.spatial.transform import Rotation as R

        v = np.asarray(approach_dir, dtype=np.float64)
        norm = np.linalg.norm(v)
        if norm < 1e-9:
            # Degenerate input: return identity
            return np.array([0.0, 0.0, 0.0, 1.0])
        v = v / norm
        z_axis = np.array([0.0, 0.0, 1.0])
        cross = np.cross(z_axis, v)
        dot = float(np.dot(z_axis, v))
        if dot > 1.0 - 1e-9:
            # Already aligned
            return np.array([0.0, 0.0, 0.0, 1.0])
        if dot < -1.0 + 1e-9:
            # Anti-parallel: rotate 180° around any axis orthogonal to z
            return R.from_rotvec(np.pi * np.array([1.0, 0.0, 0.0])).as_quat()
        s = float(np.sqrt(2.0 * (1.0 + dot)))
        q_xyz = cross / s
        q_w = s / 2.0
        return np.array([q_xyz[0], q_xyz[1], q_xyz[2], q_w])
```

- [ ] **Step 4: Run tests, confirm pass**

```bash
python -m pytest tests/test_env_wrapper_orientation.py::TestApproachDirToQuat -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/env_wrapper.py tests/test_env_wrapper_orientation.py
git commit -m "feat(env_wrapper): _approach_dir_to_quat helper for orientation control"
```

---

## Task 3: Helper — Convert World Quaternion Delta to Action Axis-Angle

**Files:**
- Modify: `src/env_wrapper.py` (add helper)
- Modify: `tests/test_env_wrapper_orientation.py` (extend)

**Math:** OSC_POSE expects an axis-angle delta from current to target orientation. We need a helper that takes (current quat, target quat) and outputs the axis-angle vector to feed into `action[3:6]`. **The frame depends on Task 1 findings** — likely base frame.

- [ ] **Step 1: Write test for `_quat_delta_to_axis_angle`**

Add to `tests/test_env_wrapper_orientation.py`:

```python
class TestQuatDeltaToAxisAngle:
    def test_identity_returns_zero(self):
        from src.env_wrapper import EnvWrapper
        q_cur = np.array([0.0, 0.0, 0.0, 1.0])
        q_tgt = np.array([0.0, 0.0, 0.0, 1.0])
        out = EnvWrapper._quat_delta_to_axis_angle(q_cur, q_tgt)
        np.testing.assert_allclose(out, [0.0, 0.0, 0.0], atol=1e-9)

    def test_90deg_around_y(self):
        from src.env_wrapper import EnvWrapper
        from scipy.spatial.transform import Rotation as R
        q_cur = np.array([0.0, 0.0, 0.0, 1.0])
        q_tgt = R.from_rotvec([0.0, np.pi / 2, 0.0]).as_quat()
        out = EnvWrapper._quat_delta_to_axis_angle(q_cur, q_tgt)
        np.testing.assert_allclose(out, [0.0, np.pi / 2, 0.0], atol=1e-6)

    def test_shortest_path(self):
        """For target near current but with negated quat, take the short path."""
        from src.env_wrapper import EnvWrapper
        from scipy.spatial.transform import Rotation as R
        q_cur = np.array([0.0, 0.0, 0.0, 1.0])
        q_tgt = -R.from_rotvec([0.0, 0.1, 0.0]).as_quat()  # Negated
        out = EnvWrapper._quat_delta_to_axis_angle(q_cur, q_tgt)
        assert np.linalg.norm(out) < np.pi  # Took short path
```

- [ ] **Step 2: Run, confirm fail**

```bash
python -m pytest tests/test_env_wrapper_orientation.py::TestQuatDeltaToAxisAngle -v
```

- [ ] **Step 3: Implement `_quat_delta_to_axis_angle`**

Add to `src/env_wrapper.py`:

```python
    @staticmethod
    def _quat_delta_to_axis_angle(
        q_current: np.ndarray, q_target: np.ndarray,
    ) -> np.ndarray:
        """Compute axis-angle vector (radians) to rotate from current to target.

        Both quaternions in xyzw convention. Returns 3-vector suitable for
        action[3:6] in OSC_POSE (frame depends on controller config).
        """
        from scipy.spatial.transform import Rotation as R

        q_c = np.asarray(q_current, dtype=np.float64)
        q_t = np.asarray(q_target, dtype=np.float64)
        if np.dot(q_c, q_t) < 0:
            q_t = -q_t  # shortest path
        # q_delta = q_target * inv(q_current)
        r_c = R.from_quat(q_c)
        r_t = R.from_quat(q_t)
        r_delta = r_t * r_c.inv()
        return r_delta.as_rotvec().astype(np.float64)
```

- [ ] **Step 4: Run, confirm pass**

```bash
python -m pytest tests/test_env_wrapper_orientation.py::TestQuatDeltaToAxisAngle -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/env_wrapper.py tests/test_env_wrapper_orientation.py
git commit -m "feat(env_wrapper): _quat_delta_to_axis_angle helper for orientation deltas"
```

---

## Task 4: Get Current Gripper Quaternion from Sim

**Files:**
- Modify: `src/env_wrapper.py` (add `_get_eef_quat` method)
- Modify: `tests/test_env_wrapper_orientation.py`

- [ ] **Step 1: Write test (uses a mock sim)**

```python
class TestGetEEFQuat:
    def test_returns_quat_from_site_xmat(self):
        from src.env_wrapper import EnvWrapper
        from scipy.spatial.transform import Rotation as R

        wrapper = EnvWrapper.__new__(EnvWrapper)
        # Build a fake sim with a known rotation
        class FakeData:
            def __init__(self):
                rot = R.from_rotvec([0.0, np.pi / 4, 0.0])
                # site_xmat is row-major 9-vec
                self.site_xmat = {42: rot.as_matrix().flatten()}
        class FakeModel:
            def site_name2id(self, name):
                return 42
        class FakeSim:
            def __init__(self):
                self.model = FakeModel()
                self.data = FakeData()
        class FakeEnv2:
            sim = FakeSim()
        wrapper._env = FakeEnv2()

        q = wrapper._get_eef_quat()
        expected = R.from_rotvec([0.0, np.pi / 4, 0.0]).as_quat()
        np.testing.assert_allclose(q, expected, atol=1e-6)
```

- [ ] **Step 2: Run, confirm fail**

```bash
python -m pytest tests/test_env_wrapper_orientation.py::TestGetEEFQuat -v
```

- [ ] **Step 3: Implement `_get_eef_quat`**

Find the EEF site name first by searching:

```bash
grep -rn "grip_site" /share/home/dlxt120210024/embodied/robosuite/robosuite/models/grippers/ | head -5
```

Likely `gripper0_right_grip_site` (Panda right gripper). Confirm by checking existing `get_eef_pos` in `env_wrapper.py` to see which site it queries.

Add method to `env_wrapper.py` near `get_eef_pos`:

```python
    def _get_eef_quat(self) -> np.ndarray:
        """Return current gripper rotation as quaternion (xyzw, world frame)."""
        from scipy.spatial.transform import Rotation as R

        sim = self._env.sim
        # Use the same site as get_eef_pos
        site_name = self._eef_site_name()  # see Step 4
        site_id = sim.model.site_name2id(site_name)
        xmat = sim.data.site_xmat[site_id].reshape(3, 3)
        return R.from_matrix(xmat).as_quat().astype(np.float64)
```

- [ ] **Step 4: Refactor existing `get_eef_pos` to share `_eef_site_name`**

Look at `get_eef_pos` in `env_wrapper.py` and extract the site name lookup into:

```python
    def _eef_site_name(self) -> str:
        """The MuJoCo site name used as the end-effector reference point."""
        # Use the same string as get_eef_pos
        return "gripper0_right_grip_site"  # adjust to match existing code
```

Update `get_eef_pos` to call `self._eef_site_name()`. **Do NOT change its return value.** Run existing tests to confirm no regression:

```bash
python -m pytest tests/ -x -q
```

- [ ] **Step 5: Run new test, confirm pass**

```bash
python -m pytest tests/test_env_wrapper_orientation.py::TestGetEEFQuat -v
```

- [ ] **Step 6: Commit**

```bash
git add src/env_wrapper.py tests/test_env_wrapper_orientation.py
git commit -m "feat(env_wrapper): _get_eef_quat + _eef_site_name extraction"
```

---

## Task 5: Extend `move_arm_to` with Optional `approach_dir`

**Files:**
- Modify: `src/env_wrapper.py:312-414`
- Test: `tests/test_env_wrapper_orientation.py`

- [ ] **Step 1: Write failing integration-lite test (mock the inner step)**

```python
class TestMoveArmToWithApproachDir:
    def test_approach_dir_sets_action_3_to_6(self, monkeypatch):
        """When approach_dir is given, action[3:6] should be non-zero."""
        from src.env_wrapper import EnvWrapper
        # Build minimal wrapper
        wrapper = EnvWrapper.__new__(EnvWrapper)
        wrapper._latest_obs = {"dummy": True}

        captured_actions = []
        class FakeEnv2:
            action_dim = 12
            def step(self, action):
                captured_actions.append(action.copy())
                return {}, 0.0, False, {}
            class sim:
                class model:
                    @staticmethod
                    def site_name2id(name):
                        return 0
                class data:
                    site_xmat = [np.eye(3).flatten()]
                    site_xpos = [np.array([0.5, 0.0, 1.0])]
        wrapper._env = FakeEnv2()
        monkeypatch.setattr(wrapper, "_get_base_action_idx", lambda: None)
        monkeypatch.setattr(wrapper, "get_eef_pos", lambda: np.array([0.5, 0.0, 1.0]))
        monkeypatch.setattr(wrapper, "_get_eef_quat", lambda: np.array([0.0, 0.0, 0.0, 1.0]))
        monkeypatch.setattr(wrapper, "get_base_pose",
                            lambda: (np.zeros(3), np.eye(3)))
        monkeypatch.setattr(wrapper, "render", lambda: None)

        wrapper.move_arm_to(
            np.array([0.55, 0.0, 1.0]),
            approach_dir=np.array([1.0, 0.0, 0.0]),
            max_steps=2,
        )
        # First action should have non-zero rotation component
        assert np.linalg.norm(captured_actions[0][3:6]) > 1e-3
```

- [ ] **Step 2: Run, confirm fail**

```bash
python -m pytest tests/test_env_wrapper_orientation.py::TestMoveArmToWithApproachDir -v
```

- [ ] **Step 3: Modify `move_arm_to` signature and body**

Open `src/env_wrapper.py:312` and change:

```python
    def move_arm_to(
        self,
        target_pos_m,
        max_steps: int = 800,
        threshold_m: float = 0.02,
        approach_dir: np.ndarray | None = None,  # NEW
        ori_gain: float = 0.3,                    # NEW
    ) -> bool:
```

Inside the loop, **before** `action = np.zeros(action_dim, ...)`, compute target orientation once (outside the loop is fine, since approach_dir is fixed):

```python
        # ── Orientation target (Task 5) ──
        target_quat = None
        if approach_dir is not None:
            ad = np.asarray(approach_dir, dtype=np.float64)
            if np.linalg.norm(ad) > 1e-6:
                target_quat = self._approach_dir_to_quat(ad)
```

Inside the loop, **after** `action[0:3] = dir_base * step_size`, add:

```python
            # ── Orientation control (Task 5) ──
            if target_quat is not None:
                q_cur = self._get_eef_quat()
                ori_delta = self._quat_delta_to_axis_angle(q_cur, target_quat)
                # Clamp magnitude per step to avoid OSC instability
                max_ori_step = 0.2  # rad/step
                ori_norm = np.linalg.norm(ori_delta)
                if ori_norm > max_ori_step:
                    ori_delta = ori_delta * (max_ori_step / ori_norm)
                action[3:6] = (ori_delta * ori_gain).astype(np.float32)
```

**Important:** Frame conversion. If Task 1 found that `action[3:6]` is in **base frame** (not world), convert:

```python
                # World-frame delta → base-frame
                _, base_ori = self.get_base_pose()
                ori_delta = base_ori.T @ ori_delta
```

Apply this **before** the magnitude clamp. Use the result from Task 1.

- [ ] **Step 4: Run test, confirm pass**

```bash
python -m pytest tests/test_env_wrapper_orientation.py::TestMoveArmToWithApproachDir -v
```

- [ ] **Step 5: Run full test suite to verify no regression**

```bash
python -m pytest tests/ -x -q 2>&1 | tail -5
```

Expected: same pass count as before (263 + new orientation tests).

- [ ] **Step 6: Commit**

```bash
git add src/env_wrapper.py tests/test_env_wrapper_orientation.py
git commit -m "feat(env_wrapper): move_arm_to accepts approach_dir for end-effector orientation control"
```

---

## Task 6: Live Sim Verification — Rotate Gripper Sideways

**Why:** Unit tests with mocked sim can't catch controller-frame bugs. We must verify in real sim that `approach_dir=[1,0,0]` actually rotates the gripper to point sideways.

**Files:**
- Modify: `scripts/debug_orientation_control.py`

- [ ] **Step 1: Add a live test routine to the debug script**

Append to `scripts/debug_orientation_control.py`:

```python
def test_live_side_rotation():
    """Verify move_arm_to with approach_dir actually rotates the gripper."""
    cfg = load_config("configs/default.yaml")
    env = EnvWrapper(cfg["simulator"])
    env.reset(seed=3)

    initial_q = env._get_eef_quat()
    initial_z_world = R.from_quat(initial_q).apply([0, 0, 1])
    print(f"Initial gripper z in world: {initial_z_world}")

    # Move slightly with side approach_dir
    current_pos = env.get_eef_pos()
    target = current_pos + np.array([0.05, 0.0, 0.0])  # small +x move
    env.move_arm_to(target, approach_dir=np.array([1.0, 0.0, 0.0]),
                    max_steps=400, threshold_m=0.02)

    final_q = env._get_eef_quat()
    final_z_world = R.from_quat(final_q).apply([0, 0, 1])
    print(f"Final gripper z in world: {final_z_world}")

    # Expected: gripper z should now point closer to +x than to -z
    dot_with_x = np.dot(final_z_world, [1, 0, 0])
    dot_with_minus_z = np.dot(final_z_world, [0, 0, -1])
    print(f"Dot with +x: {dot_with_x:.3f}, dot with -z: {dot_with_minus_z:.3f}")
    assert dot_with_x > 0.5, f"Gripper should mostly face +x, got dot={dot_with_x}"


if __name__ == "__main__":
    main()
    test_live_side_rotation()
```

- [ ] **Step 2: Run live test on sim**

Run on the GPU server (where sim works):

```bash
python scripts/debug_orientation_control.py 2>&1 | tee /tmp/osc_live.log
```

Expected output: `Dot with +x: > 0.5` and assertion passes.

**If it fails:** Either (a) the convention from Task 1 was wrong, (b) the ori_gain is too low, or (c) the OSC controller has joint limits that prevent rotation. Debug by:
1. Try `ori_gain=1.0` (was 0.3)
2. Verify `action[3:6]` is being applied (log it in `move_arm_to`)
3. Try `approach_dir=[-1,0,0]` to confirm direction matters

Do NOT proceed to Task 7 until live rotation works.

- [ ] **Step 3: Commit the working live test**

```bash
git add scripts/debug_orientation_control.py
git commit -m "test: live verification of approach_dir end-effector rotation"
```

---

## Task 7: Add `approach` Method to EnvWrapper

**Files:**
- Modify: `src/env_wrapper.py:1363-1377` (`descend`)
- Test: `tests/test_env_wrapper_orientation.py`

**Design:** `approach(point_3d, approach_dir, ...)` is the new direction-aware grasp-approach primitive. `descend` becomes a thin wrapper for backward compatibility (approach_dir defaults to [0,0,-1]).

- [ ] **Step 1: Write test for `approach`**

```python
class TestApproach:
    def test_top_down_equivalent_to_old_descend(self, monkeypatch):
        """approach(p, [0,0,-1]) should behave like the old descend."""
        from src.env_wrapper import EnvWrapper
        wrapper = EnvWrapper.__new__(EnvWrapper)
        wrapper._latest_obs = {}

        calls = []
        monkeypatch.setattr(wrapper, "_descend_until_contact",
                            lambda tp, tb, **kw: (calls.append(("descend", tp.copy())), (True, tp[2]))[1])
        monkeypatch.setattr(wrapper, "_get_obj_type_map",
                            lambda: {"obj_main": "apple"})
        monkeypatch.setattr(wrapper, "move_arm_to", lambda *a, **kw: True)
        monkeypatch.setattr(wrapper, "get_eef_pos",
                            lambda: np.array([0.5, 0.0, 1.0]))

        ok, z = wrapper.approach(
            np.array([0.5, 0.0, 0.9]),
            approach_dir=np.array([0.0, 0.0, -1.0]),
            target_label="apple", margin_m=0.015,
        )
        assert ok is True
        # The descend was called with target z = 0.9 - 0.015 = 0.885
        assert len(calls) == 1
        np.testing.assert_allclose(calls[0][1][2], 0.885, atol=1e-6)

    def test_side_approach_uses_move_arm_to(self, monkeypatch):
        """For side approach, should use move_arm_to with approach_dir, not descend."""
        from src.env_wrapper import EnvWrapper
        wrapper = EnvWrapper.__new__(EnvWrapper)

        move_calls = []
        def fake_move(target, **kw):
            move_calls.append((target.copy(), kw.get("approach_dir")))
            return True
        monkeypatch.setattr(wrapper, "move_arm_to", fake_move)
        monkeypatch.setattr(wrapper, "_get_obj_type_map", lambda: {})
        monkeypatch.setattr(wrapper, "get_eef_pos",
                            lambda: np.array([0.6, 0.0, 0.93]))

        ok, z = wrapper.approach(
            np.array([0.5, 0.0, 0.93]),
            approach_dir=np.array([1.0, 0.0, 0.0]),
            target_label=None, margin_m=0.0,
        )
        assert ok is True
        # Should have called move_arm_to with approach_dir = [1,0,0]
        assert len(move_calls) >= 1
        np.testing.assert_allclose(move_calls[-1][1], [1.0, 0.0, 0.0])
```

- [ ] **Step 2: Run, confirm fail**

```bash
python -m pytest tests/test_env_wrapper_orientation.py::TestApproach -v
```

- [ ] **Step 3: Implement `approach`**

In `src/env_wrapper.py`, replace existing `descend` (lines 1363-1377) with:

```python
    def approach(
        self,
        point_3d,
        approach_dir: np.ndarray,
        target_label: Optional[str] = None,
        step_z: float = 0.01,
        max_steps: int = 35,
        margin_m: float = 0.015,
    ) -> tuple[bool, float]:
        """Direction-aware approach to point_3d along -approach_dir.

        For top_down (approach_dir = [0,0,-1]): equivalent to old descend.
        For side (approach_dir = [1,0,0] etc): moves horizontally toward point_3d
        with gripper rotated to face approach_dir.

        Args:
            point_3d: 3D grasp point in world frame.
            approach_dir: unit vector pointing FROM the approach start TO the object.
            target_label: optional label for contact-based stopping.
            margin_m: extra distance past point_3d along approach_dir.

        Returns:
            (success, final_eef_z)
        """
        ad = np.asarray(approach_dir, dtype=np.float32)
        ad_norm = float(np.linalg.norm(ad))
        if ad_norm < 1e-6:
            ad = np.array([0.0, 0.0, -1.0], dtype=np.float32)
            ad_norm = 1.0
        ad_unit = ad / ad_norm

        target = np.asarray(point_3d, dtype=np.float32).copy()
        if margin_m > 0:
            target = target + ad_unit * margin_m
            logger.info(
                f"[approach] margin={margin_m:.3f}m along {ad_unit}, "
                f"adjusted target={target}"
            )

        # Top-down path: keep specialized contact-based descent
        is_top_down = ad_unit[2] < -0.9 and abs(ad_unit[0]) < 0.1 and abs(ad_unit[1]) < 0.1
        if is_top_down:
            target_body: Optional[str] = None
            if target_label:
                try:
                    type_map = self._get_obj_type_map()
                    for body, cat in type_map.items():
                        if cat == target_label:
                            target_body = body
                            break
                except Exception as e:
                    logger.debug(f"[approach] type_map lookup failed: {e}")
            if target_body:
                return self._descend_until_contact(
                    target, target_body, step_z=step_z, max_steps=max_steps,
                )
            ok = self.move_arm_to(target, threshold_m=0.01, max_steps=200,
                                  approach_dir=ad_unit)
            return bool(ok), float(self.get_eef_pos()[2])

        # Side/arbitrary approach: simple move_arm_to with orientation control
        ok = self.move_arm_to(
            target, threshold_m=0.01, max_steps=400, approach_dir=ad_unit,
        )
        return bool(ok), float(self.get_eef_pos()[2])

    def descend(
        self, point_3d, target_label: Optional[str] = None,
        step_z: float = 0.01, max_steps: int = 25,
        margin_m: float = 0.015,
    ) -> tuple[bool, float]:
        """Backward-compatible wrapper: top-down approach via approach()."""
        return self.approach(
            point_3d,
            approach_dir=np.array([0.0, 0.0, -1.0]),
            target_label=target_label,
            step_z=step_z, max_steps=max_steps, margin_m=margin_m,
        )
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_env_wrapper_orientation.py::TestApproach -v
python -m pytest tests/ -x -q 2>&1 | tail -5
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/env_wrapper.py tests/test_env_wrapper_orientation.py
git commit -m "feat(env_wrapper): approach() with approach_dir, descend is now wrapper"
```

---

## Task 8: Update `move_to_pre_grasp` to Use approach_dir

**Files:**
- Modify: `src/env_wrapper.py:1334-1361`
- Test: extend `tests/test_env_wrapper_orientation.py`

**Logic:**
- Old `move_to_pre_grasp(candidate, height_m=0.05)`: positions EEF at `(obj_x, obj_y, obj_z + height_m)` — always above.
- New: positions EEF at `point_3d - approach_dir * pre_grasp_offset` with orientation locked to approach_dir.

- [ ] **Step 1: Add test**

```python
class TestMoveToPreGrasp:
    def test_top_down_pre_grasp_above(self, monkeypatch):
        from src.env_wrapper import EnvWrapper
        from src.world_belief import GraspCandidate
        wrapper = EnvWrapper.__new__(EnvWrapper)
        moves = []
        monkeypatch.setattr(wrapper, "move_arm_to",
                            lambda t, **kw: moves.append((t.copy(), kw)) or True)
        monkeypatch.setattr(wrapper, "_gripper_action", lambda *a, **kw: None)
        monkeypatch.setattr(wrapper, "get_eef_pos",
                            lambda: np.array([0.5, 0.0, 1.0]))

        c = GraspCandidate(
            point_3d=np.array([0.5, 0.0, 0.9]),
            approach_dir=np.array([0.0, 0.0, -1.0]),
            finger_width_m=0.04, score=0.9, source="test",
        )
        wrapper.move_to_pre_grasp(c, height_m=0.05)
        # Final move should be at (0.5, 0.0, 0.95) - 5cm above object
        last_pos = moves[-1][0]
        np.testing.assert_allclose(last_pos, [0.5, 0.0, 0.95], atol=1e-6)

    def test_side_pre_grasp_offset_horizontally(self, monkeypatch):
        from src.env_wrapper import EnvWrapper
        from src.world_belief import GraspCandidate
        wrapper = EnvWrapper.__new__(EnvWrapper)
        moves = []
        monkeypatch.setattr(wrapper, "move_arm_to",
                            lambda t, **kw: moves.append((t.copy(), kw)) or True)
        monkeypatch.setattr(wrapper, "_gripper_action", lambda *a, **kw: None)
        monkeypatch.setattr(wrapper, "get_eef_pos",
                            lambda: np.array([0.5, 0.0, 1.0]))

        c = GraspCandidate(
            point_3d=np.array([0.5, 0.0, 0.93]),
            approach_dir=np.array([1.0, 0.0, 0.0]),
            finger_width_m=0.03, score=0.9, source="test",
        )
        wrapper.move_to_pre_grasp(c, height_m=0.10)  # height_m → pre-grasp offset
        # Final move: 10cm BEHIND in -x direction at same z
        last_pos = moves[-1][0]
        np.testing.assert_allclose(last_pos, [0.4, 0.0, 0.93], atol=1e-6)
```

- [ ] **Step 2: Run, confirm fail**

```bash
python -m pytest tests/test_env_wrapper_orientation.py::TestMoveToPreGrasp -v
```

- [ ] **Step 3: Modify `move_to_pre_grasp`**

Replace the body of `move_to_pre_grasp` (lines 1334-1361) with:

```python
    def move_to_pre_grasp(self, candidate, height_m: float = 0.05) -> bool:
        """Move EEF to pre-grasp position offset by -approach_dir * height_m.

        For top_down (approach_dir=[0,0,-1]): pre-grasp at (obj_xy, obj_z + height_m).
        For side approach: pre-grasp at obj - approach_dir * height_m.
        Also commands gripper to be open.
        """
        approach_dir = np.asarray(
            getattr(candidate, "approach_dir", [0.0, 0.0, -1.0]),
            dtype=np.float32,
        )
        n = float(np.linalg.norm(approach_dir))
        if n < 1e-6:
            approach_dir = np.array([0.0, 0.0, -1.0], dtype=np.float32)
            n = 1.0
        ad_unit = approach_dir / n

        target_pos = np.asarray(candidate.point_3d, dtype=np.float32).copy()
        # Pre-grasp position: offset opposite to approach direction
        pre_pos = target_pos - ad_unit * height_m

        # Base approach for far targets: drive base toward (target_xy - 0.4m * xy_approach)
        try:
            eef = self.get_eef_pos()
            xy_approach = np.array([ad_unit[0], ad_unit[1], 0.0], dtype=np.float32)
            xy_norm = float(np.linalg.norm(xy_approach))
            base_target = np.array([
                float(target_pos[0]) - (xy_approach[0] / max(xy_norm, 1e-6)) * 0.4 if xy_norm > 0.1 else float(target_pos[0]) - 0.4,
                float(target_pos[1]) - (xy_approach[1] / max(xy_norm, 1e-6)) * 0.4 if xy_norm > 0.1 else float(target_pos[1]),
                float(eef[2]),
            ], dtype=np.float32)
            self.move_arm_to(base_target, threshold_m=0.15, max_steps=600)
        except Exception as e:
            logger.debug(f"[pre_grasp] base approach failed: {e}")

        # Open gripper
        try:
            self._gripper_action(-1.0, n_steps=8)
        except Exception:
            pass

        # Move to pre-grasp with orientation
        return self.move_arm_to(
            pre_pos, threshold_m=0.06, approach_dir=ad_unit,
        )
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_env_wrapper_orientation.py::TestMoveToPreGrasp -v
python -m pytest tests/ -x -q 2>&1 | tail -5
```

- [ ] **Step 5: Commit**

```bash
git add src/env_wrapper.py tests/test_env_wrapper_orientation.py
git commit -m "feat(env_wrapper): move_to_pre_grasp uses approach_dir for pre-grasp position + orientation"
```

---

## Task 9: Update ActionExecutor to Pass approach_dir

**Files:**
- Modify: `src/action_executor.py:91-188` (`act` method)
- Modify: `tests/test_action_executor_v1.py`

- [ ] **Step 1: Add `approach_dir` handling to FakeEnv in existing tests**

In `tests/test_action_executor_v1.py`, modify `FakeEnv.descend` and add `approach`:

```python
    def descend(self, point_3d, target_label=None, **kwargs):
        self.calls.append("descend")
        if self.descend_ok:
            return True, point_3d[2]
        return False, point_3d[2] + 0.03

    def approach(self, point_3d, approach_dir, target_label=None, **kwargs):
        self.calls.append(f"approach[{approach_dir[0]:.0f},{approach_dir[1]:.0f},{approach_dir[2]:.0f}]")
        if self.descend_ok:
            return True, float(point_3d[2])
        return False, float(point_3d[2]) + 0.03

    def move_to_pre_grasp(self, candidate) -> bool:
        # Track approach_dir from candidate
        ad = getattr(candidate, "approach_dir", None)
        if ad is not None:
            self.calls.append(f"pre_grasp[{ad[0]:.0f},{ad[1]:.0f},{ad[2]:.0f}]")
        else:
            self.calls.append("move_to_pre_grasp")
        return self.ik_ok
```

- [ ] **Step 2: Add new test verifying ActionExecutor uses approach_dir**

```python
    def test_side_approach_calls_approach_not_descend(self):
        """When candidate.approach_dir = +x, act() should call env.approach with +x."""
        from src.action_executor import ActionExecutor
        from src.world_belief import DecomposedTask, GraspCandidate, Hypothesis
        import numpy as np

        env = FakeEnv()
        exe = ActionExecutor(scene_describer=None)
        c = GraspCandidate(
            point_3d=np.array([0.5, 0, 0.9]),
            approach_dir=np.array([1.0, 0.0, 0.0]),
            finger_width_m=0.03, score=0.9, source="side_test",
        )
        h = Hypothesis(
            object_id="o0", label="spoon",
            label_alternatives=[("spoon", 0.9)], label_entropy=0.1,
            position_3d=np.array([0.5, 0, 0.9]), position_std_m=0.02,
            grasp_candidates=[c],
        )
        exe.act(h, DecomposedTask(primary_target="spoon"), env)
        # Verify approach was called with side direction
        assert any("approach[1,0,0]" in call for call in env.calls), \
            f"Expected side approach in calls, got: {env.calls}"
```

- [ ] **Step 3: Run, confirm fail**

```bash
python -m pytest tests/test_action_executor_v1.py::TestAct::test_side_approach_calls_approach_not_descend -v
```

- [ ] **Step 4: Modify ActionExecutor.act**

In `src/action_executor.py`, find where `env.descend(...)` is called (around line 91). Replace with:

```python
        approach_dir = np.asarray(
            getattr(candidate, "approach_dir", [0.0, 0.0, -1.0]),
            dtype=np.float32,
        )
        descend_ok, z_actual = env.approach(
            candidate.point_3d,
            approach_dir=approach_dir,
            target_label=getattr(target, "label", None),
            margin_m=margin_m,
        )
```

Also in the z-stall recovery section (where we currently call `env.descend` again after base reposition), use `env.approach` similarly:

```python
                descend_ok2, z_actual = env.approach(
                    candidate.point_3d,
                    approach_dir=approach_dir,
                    target_label=getattr(target, "label", None),
                    margin_m=margin_m,
                )
```

- [ ] **Step 5: Run tests**

```bash
python -m pytest tests/test_action_executor_v1.py -v
python -m pytest tests/ -x -q 2>&1 | tail -5
```

- [ ] **Step 6: Commit**

```bash
git add src/action_executor.py tests/test_action_executor_v1.py
git commit -m "feat(action_executor): pass candidate.approach_dir to env.approach for direction-aware grasping"
```

---

## Task 10: Update `lift` to Use approach_dir Retreat (Optional Enhancement)

**Files:**
- Modify: `src/env_wrapper.py:1414-1448` (`lift`)
- Test: extend `tests/test_env_wrapper_orientation.py`

**Why:** For side grasps, after closing the gripper we want to retreat **back along -approach_dir** (i.e., away from the object), not straight up. Straight up may drag the object across the table.

- [ ] **Step 1: Add test**

```python
class TestLiftWithApproach:
    def test_default_lift_is_vertical(self, monkeypatch):
        from src.env_wrapper import EnvWrapper
        wrapper = EnvWrapper.__new__(EnvWrapper)
        moves = []
        monkeypatch.setattr(wrapper, "move_arm_to",
                            lambda t, **kw: moves.append(t.copy()) or True)
        monkeypatch.setattr(wrapper, "get_eef_pos",
                            lambda: np.array([0.5, 0.0, 0.93]) if not moves else np.array([0.5, 0.0, 1.03]))
        ok, z = wrapper.lift(height_m=0.10)
        # Last move target should be roughly (0.5, 0, 1.03)
        assert moves[-1][2] > 0.95

    def test_lift_with_side_approach_retreats_horizontally(self, monkeypatch):
        from src.env_wrapper import EnvWrapper
        wrapper = EnvWrapper.__new__(EnvWrapper)
        moves = []
        monkeypatch.setattr(wrapper, "move_arm_to",
                            lambda t, **kw: moves.append(t.copy()) or True)
        monkeypatch.setattr(wrapper, "get_eef_pos",
                            lambda: np.array([0.5, 0.0, 0.93]))
        ok, z = wrapper.lift(height_m=0.10, approach_dir=np.array([1.0, 0.0, 0.0]))
        # First moves should be -x retreat (4 micro), then z up
        first = moves[0]
        # Pre-retreat should reduce x
        assert first[0] < 0.5
```

- [ ] **Step 2: Run, confirm fail**

```bash
python -m pytest tests/test_env_wrapper_orientation.py::TestLiftWithApproach -v
```

- [ ] **Step 3: Modify `lift`**

Replace `lift` body in `src/env_wrapper.py` with:

```python
    def lift(
        self, height_m: float = 0.10,
        approach_dir: np.ndarray | None = None,
    ) -> tuple[bool, float]:
        """Lift the grasped object.

        Top-down (default or approach_dir~[0,0,-1]): two-phase vertical lift.
        Side approach: retreat horizontally along -approach_dir first, then lift z.
        """
        try:
            curr = self.get_eef_pos().copy()
            start_z = float(curr[2])

            # If side approach, retreat horizontally first by 5cm
            if approach_dir is not None:
                ad = np.asarray(approach_dir, dtype=np.float32)
                if np.linalg.norm(ad) > 1e-6:
                    ad_unit = ad / np.linalg.norm(ad)
                    # Only retreat if approach has a horizontal component
                    horizontal_norm = float(np.linalg.norm(ad_unit[:2]))
                    if horizontal_norm > 0.3:
                        retreat = curr - ad_unit * 0.05
                        self.move_arm_to(
                            retreat, threshold_m=0.01, max_steps=200,
                            approach_dir=ad_unit,
                        )
                        curr = self.get_eef_pos().copy()

            # Gentle micro-step lift (4 × 5mm)
            gentle_total = min(0.02, height_m)
            micro_step = 0.005
            n_micro = max(1, int(gentle_total / micro_step))
            for k in range(n_micro):
                target_z = curr[2] + (k + 1) * micro_step
                target = np.array([curr[0], curr[1], target_z], dtype=np.float32)
                self.move_arm_to(target, threshold_m=0.003, max_steps=80)

            # Remainder
            remaining = height_m - gentle_total
            if remaining > 0.005:
                final_target = np.array(
                    [curr[0], curr[1], curr[2] + height_m], dtype=np.float32,
                )
                self.move_arm_to(final_target, threshold_m=0.02, max_steps=200)

            final_z = float(self.get_eef_pos()[2])
            ok = final_z > start_z + height_m * 0.5
            return ok, final_z
        except Exception as e:
            logger.warning(f"[lift] failed: {e}")
            return False, 0.0
```

- [ ] **Step 4: Pass approach_dir from ActionExecutor**

In `src/action_executor.py`, change `lift_ok, final_z = env.lift()` calls to:

```python
        lift_ok, final_z = env.lift(approach_dir=approach_dir)
```

(Both normal-path and z-stall recovery paths.)

- [ ] **Step 5: Run tests**

```bash
python -m pytest tests/test_env_wrapper_orientation.py::TestLiftWithApproach -v
python -m pytest tests/ -x -q 2>&1 | tail -5
```

- [ ] **Step 6: Commit**

```bash
git add src/env_wrapper.py src/action_executor.py tests/test_env_wrapper_orientation.py
git commit -m "feat(env_wrapper): lift accepts approach_dir to retreat horizontally for side grasps"
```

---

## Task 11: Live Sim Validation on wooden_spoon seed=3

**Files:** None modified; this is verification only.

- [ ] **Step 1: Pull latest, prepare recording**

On GPU server:

```bash
git pull
```

- [ ] **Step 2: Run the seed=3 scenario**

```bash
python scripts/record_video.py \
  --scenario random_seed_3 \
  --scenarios-config logs/long_generalization/overnight-gen-50-v2/scenarios.yaml \
  --multi -o results/videos/demo_orientation_aware.mp4
```

- [ ] **Step 3: Inspect logs**

Look for these markers indicating side approach actually executed:

```text
[approach] margin=0.010m along [1, 0, 0], adjusted target=...
[move_arm_to] ... approach_dir=[1, 0, 0]
[close_gripper] grasp confirmed + squeezed at step ...
[act] post-lift verified: obj Δz=0.0xx
```

If post-lift Δz > 0.02m for the spoon → **success**, plan goal achieved.

- [ ] **Step 4: Failure-mode analysis if still failing**

If still failing, check:
- Did the gripper actually rotate? Inspect `_get_eef_quat` outputs in log.
- Is the orientation slow to converge? Increase `ori_gain` from 0.3 to 0.5.
- Is the side approach reaching the object? Log `descended` distance in `move_arm_to`.

Document findings in `docs/superpowers/specs/2026-05-16-thin-object-grasping-analysis.md` Section 7 (next steps).

- [ ] **Step 5: Commit any tuning + summary**

```bash
git add -p  # interactive: only stage tuning constants if any
git commit -m "validation(orientation): wooden_spoon seed=3 results + tuning if needed"
git push origin main
```

---

## Task 12: Regression Test — Confirm Other Seeds Still Pass

**Files:** None modified.

- [ ] **Step 1: Pick 3 seeds known to succeed previously**

From the last green run (e.g., `batch_20260512_005101.json`), pick 3 seeds where simple objects succeed (orange, juice, etc.).

- [ ] **Step 2: Run them individually**

```bash
for SEED in 101 105 110; do
  python eval/run_fixed.py \
    --scenario random_seed_$SEED \
    --scenarios-config logs/long_generalization/overnight-gen-50-v2/scenarios.yaml \
    --log-level WARNING
done
```

- [ ] **Step 3: Verify all 3 succeed**

If any regress, this is a blocker — debug before merging. Common regression: orientation control in top_down (`approach_dir=[0,0,-1]`) introducing rotation drift. Fix by skipping orientation control when `approach_dir` is essentially `-z`:

In `move_arm_to`, before setting `target_quat`:

```python
        # Skip orientation if approach is essentially top-down (matches current gripper default)
        if approach_dir is not None:
            ad = np.asarray(approach_dir, dtype=np.float64)
            if ad[2] < -0.9 and abs(ad[0]) < 0.1 and abs(ad[1]) < 0.1:
                approach_dir = None  # treat as no orientation control
```

- [ ] **Step 4: Commit if any fixes**

```bash
git add src/env_wrapper.py
git commit -m "fix(env_wrapper): skip orientation control for top_down (matches default), prevents regression"
```

---

## Self-Review (Run Before Handing Off)

- [ ] **Spec coverage:** All 4 sections of the spec's "路径 A" mapped to tasks?
  - Orientation control in OSC → Tasks 1, 5, 6 ✓
  - `descend` → `approach` → Task 7 ✓
  - `move_to_pre_grasp` direction-aware → Task 8 ✓
  - Retreat sequence → Task 10 ✓
- [ ] **Placeholder scan:** No TBD / TODO / "implement later" anywhere? ✓
- [ ] **Type consistency:** `approach_dir` is `np.ndarray` everywhere; `point_3d` is `np.ndarray`; quat is `(x,y,z,w)` everywhere ✓
- [ ] **Order:** Tasks 1-4 are pure helpers, 5-6 add orientation to `move_arm_to`, 7-9 wire through to `approach` and `ActionExecutor`, 10 is the lift refinement, 11-12 are live validation ✓

---

## Rollback Strategy

If at any point the live test in Task 6 fails repeatedly:

1. **Revert to pre-Task-5 commit**: `git revert <task5-commit>`
2. **Keep Tasks 1-4 helpers** (they're pure math, safe to keep)
3. **Document the OSC convention finding** in the analysis spec
4. **Pivot to Path B** (depth_margin tuning, from analysis spec section 4.2)

---

## Estimated Effort

| Tasks | Hours (single engineer) |
|-------|-------------------------|
| 1 (OSC investigation) | 2-4 (most uncertain) |
| 2-4 (helpers) | 2 |
| 5 (move_arm_to extension) | 2 |
| 6 (live verification) | 1-3 (depends on debugging) |
| 7-9 (approach + integration) | 3 |
| 10 (lift retreat) | 1 |
| 11-12 (validation) | 1-2 |
| **Total** | **12-17 hours** (1.5-2 days) |
