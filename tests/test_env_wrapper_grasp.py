from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.env_wrapper import EnvWrapper  # noqa: E402


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
