"""Probe PandaOmron action channels by measuring EEF response.

Usage on GPU:
    MUJOCO_GL=egl python scripts/probe_arm_action_response.py

This intentionally avoids LLM/VLM and full agent logic. It builds the same
RoboCasa env, resets with seed=42, then applies short pulses to each action
index and prints how robot0_eef_pos changes.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils import load_dotenv  # noqa: E402

load_dotenv(str(REPO_ROOT / ".env"))


def _load_config() -> dict:
    import yaml

    with open(REPO_ROOT / "configs" / "default.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_env(config: dict):
    from src.env_wrapper import EnvConfig, EnvWrapper

    sim_cfg = config.get("simulator", {})
    cams = sim_cfg.get("camera_names")
    env_cfg = EnvConfig(
        env_name=sim_cfg.get("env_name", "PickPlaceCounterToCabinet"),
        robots=sim_cfg.get("robots", "PandaMobile"),
        image_width=sim_cfg.get("image_width", 256),
        image_height=sim_cfg.get("image_height", 256),
        camera_names=(
            tuple(cams) if cams
            else EnvConfig.__dataclass_fields__["camera_names"].default
        ),
    )
    return EnvWrapper(env_cfg)


def _print_layout(env) -> None:
    print("\n=== Action layout ===")
    robot = env._env.robots[0]
    idx = 0
    for part_name, ctrl in robot.composite_controller.part_controllers.items():
        dim = ctrl.control_dim
        print(f"[{idx:02d}:{idx + dim:02d}] {part_name:16s} dim={dim} type={type(ctrl).__name__}")
        idx += dim
    print(f"env.action_dim={env._env.action_dim}")


def _eef(env) -> np.ndarray:
    return np.asarray(env._latest_obs["robot0_eef_pos"], dtype=np.float64)


def _pulse(env, action_idx: int, value: float, steps: int = 25) -> np.ndarray:
    start = _eef(env)
    action = np.zeros(env._env.action_dim, dtype=np.float32)
    action[action_idx] = float(value)
    for _ in range(steps):
        obs, _, _, _ = env._env.step(action)
        env._latest_obs = obs
    return _eef(env) - start


def _reset(env) -> None:
    env.seed(42)
    env.reset()


def main() -> int:
    if not os.environ.get("DEEPSEEK_API_KEY"):
        os.environ["DEEPSEEK_API_KEY"] = "sk-dummy-probe-only"

    config = _load_config()
    env = _build_env(config)
    _reset(env)
    _print_layout(env)

    print("\n=== Initial state ===")
    print(f"eef={_eef(env).round(4).tolist()}")

    print("\n=== Single-channel positive pulse response ===")
    print("idx  value  delta_eef_xyz")
    for idx in range(env._env.action_dim):
        _reset(env)
        delta = _pulse(env, idx, +0.5)
        print(f"{idx:02d}   +0.5   {delta.round(4).tolist()}")

    print("\n=== Single-channel negative pulse response ===")
    print("idx  value  delta_eef_xyz")
    for idx in range(env._env.action_dim):
        _reset(env)
        delta = _pulse(env, idx, -0.5)
        print(f"{idx:02d}   -0.5   {delta.round(4).tolist()}")

    print("\n=== move_arm_to smoke: +10cm z from reset ===")
    _reset(env)
    start = _eef(env)
    ok = env.move_arm_to(start + np.array([0.0, 0.0, 0.10]), threshold_m=0.02, max_steps=120)
    end = _eef(env)
    print(f"ok={ok} start={start.round(4).tolist()} end={end.round(4).tolist()} delta={(end-start).round(4).tolist()}")

    print("\n=== move_arm_to smoke: -10cm z from reset ===")
    _reset(env)
    start = _eef(env)
    ok = env.move_arm_to(start + np.array([0.0, 0.0, -0.10]), threshold_m=0.02, max_steps=120)
    end = _eef(env)
    print(f"ok={ok} start={start.round(4).tolist()} end={end.round(4).tolist()} delta={(end-start).round(4).tolist()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
