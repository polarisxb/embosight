"""Probe arm OSC response after navigate_base_to in the lemon seed.

Usage on GPU:
    MUJOCO_GL=egl python scripts/probe_arm_after_navigation.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

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


def _arr(x: Any) -> str:
    try:
        a = np.asarray(x, dtype=np.float64)
        return str(a.round(5).tolist())
    except Exception:
        return repr(x)


def _eef(env) -> np.ndarray:
    return np.asarray(env._latest_obs["robot0_eef_pos"], dtype=np.float64)


def _base_pose_str(env) -> str:
    pose = env._read_real_base_pose()
    if pose is None:
        return "None"
    pos, ori = pose
    yaw = float(np.degrees(np.arctan2(float(ori[1, 0]), float(ori[0, 0]))))
    return f"pos={_arr(pos)} yaw={yaw:.2f}deg"


def _right_controller(env):
    robot = env._env.robots[0]
    ctrls = robot.composite_controller.part_controllers
    if "right" in ctrls:
        return ctrls["right"]
    for name, ctrl in ctrls.items():
        lname = name.lower()
        if "right" in lname and "gripper" not in lname:
            return ctrl
    return None


def _print_layout(env) -> None:
    print("\n=== Action layout ===")
    robot = env._env.robots[0]
    idx = 0
    for part_name, ctrl in robot.composite_controller.part_controllers.items():
        dim = int(ctrl.control_dim)
        print(f"[{idx:02d}:{idx + dim:02d}] {part_name:16s} dim={dim} type={type(ctrl).__name__}")
        idx += dim
    print(f"env.action_dim={env._env.action_dim}")


def _dump_controller(env, label: str) -> None:
    ctrl = _right_controller(env)
    print(f"\n=== Right controller: {label} ===")
    if ctrl is None:
        print("right controller: None")
        return
    print(f"type={type(ctrl).__name__}")
    for attr in (
        "input_ref_frame",
        "input_type",
        "control_delta",
        "uncoupling",
        "control_dim",
        "input_min",
        "input_max",
        "output_min",
        "output_max",
        "goal_pos",
        "goal_ori",
        "goal_ori_quat",
        "goal",
        "ee_pos",
        "ee_ori_mat",
        "ref_pos",
        "ref_ori_mat",
    ):
        if hasattr(ctrl, attr):
            print(f"{attr}={_arr(getattr(ctrl, attr))}")
    goal_attrs = [a for a in dir(ctrl) if "goal" in a.lower() and not a.startswith("__")]
    print(f"goal_attrs={goal_attrs}")
    reset_methods = [m for m in ("reset_goal", "reset", "update_initial_joints") if hasattr(ctrl, m)]
    print(f"reset_methods={reset_methods}")


def _find_lemon(env) -> tuple[str, np.ndarray]:
    tmap = env._get_obj_type_map()
    for body, cat in tmap.items():
        if str(cat).lower() == "lemon":
            pos = env._get_body_pos(body)
            if pos is None:
                raise RuntimeError(f"lemon body {body!r} has no position")
            return body, np.asarray(pos, dtype=np.float64)
    raise RuntimeError(f"lemon not found in object map: {tmap}")


def _reset_seed42(env) -> None:
    env.seed(42)
    env.reset()


def _new_seed42_env(config: dict):
    env = _build_env(config)
    _reset_seed42(env)
    return env


def _close_env(env) -> None:
    backend = getattr(env, "_env", None)
    close = getattr(backend, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


def _sync_zero(env, steps: int = 1) -> None:
    action = np.zeros(env._env.action_dim, dtype=np.float32)
    for _ in range(steps):
        obs, _, _, _ = env._env.step(action)
        env._latest_obs = obs


def _prepare_navigated(env) -> tuple[str, np.ndarray]:
    body, lemon_pos = _find_lemon(env)
    ok = env.navigate_base_to(lemon_pos[:2], offset_m=0.30)
    _sync_zero(env, steps=2)
    print(f"\nprepared navigate ok={ok} lemon_body={body} lemon_pos={_arr(lemon_pos)}")
    print(f"base={_base_pose_str(env)} eef={_arr(_eef(env))}")
    return body, lemon_pos


def _pulse(env, idx: int, value: float, steps: int = 30) -> np.ndarray:
    start = _eef(env)
    action = np.zeros(env._env.action_dim, dtype=np.float32)
    action[idx] = float(value)
    for _ in range(steps):
        obs, _, _, _ = env._env.step(action)
        env._latest_obs = obs
    return _eef(env) - start


def _call_controller_reset(env) -> None:
    ctrl = _right_controller(env)
    if ctrl is None:
        print("controller reset: no right controller")
        return
    called: list[str] = []
    for method_name in ("reset_goal", "reset"):
        method = getattr(ctrl, method_name, None)
        if method is None:
            continue
        try:
            method()
            called.append(method_name)
        except TypeError as e:
            print(f"controller {method_name} skipped TypeError: {e}")
        except Exception as e:
            print(f"controller {method_name} failed: {type(e).__name__}: {e}")
    print(f"controller reset called={called}")


def _probe_reset_state(config: dict) -> None:
    print("\n=== Baseline pulses from reset state ===")
    for idx in (0, 1, 2):
        for value in (+0.5, -0.5):
            env = _new_seed42_env(config)
            try:
                body, lemon_pos = _find_lemon(env)
                start = _eef(env)
                delta = _pulse(env, idx, value)
                print(f"reset idx={idx} value={value:+.1f} body={body} lemon={_arr(lemon_pos)} start={_arr(start)} delta={_arr(delta)} end={_arr(_eef(env))}")
            finally:
                _close_env(env)


def _probe_navigated_state(config: dict, reset_controller: bool = False) -> None:
    title = "navigated after controller reset" if reset_controller else "navigated raw"
    print(f"\n=== Pulses from {title} ===")
    for idx in (0, 1, 2):
        for value in (+0.5, -0.5):
            env = _new_seed42_env(config)
            try:
                _prepare_navigated(env)
                _dump_controller(env, "before optional reset")
                if reset_controller:
                    _call_controller_reset(env)
                    _sync_zero(env, steps=1)
                    _dump_controller(env, "after controller reset + zero step")
                start = _eef(env)
                delta = _pulse(env, idx, value)
                print(f"{title} idx={idx} value={value:+.1f} start={_arr(start)} delta={_arr(delta)} end={_arr(_eef(env))}")
            finally:
                _close_env(env)


def _probe_move_arm_after_navigation(config: dict) -> None:
    print("\n=== move_arm_to smoke after navigation ===")
    for vec, name in (
        (np.array([0.0, 0.0, 0.10]), "+10cm z"),
        (np.array([0.0, 0.0, -0.10]), "-10cm z"),
        (np.array([-0.10, 0.0, 0.0]), "-10cm x"),
    ):
        env = _new_seed42_env(config)
        try:
            _prepare_navigated(env)
            start = _eef(env)
            target = start + vec
            ok = env.move_arm_to(target, threshold_m=0.02, max_steps=160)
            end = _eef(env)
            print(f"move_arm_to {name}: ok={ok} start={_arr(start)} target={_arr(target)} delta={_arr(end - start)} end={_arr(end)}")
        finally:
            _close_env(env)


def main() -> int:
    if not os.environ.get("DEEPSEEK_API_KEY"):
        os.environ["DEEPSEEK_API_KEY"] = "sk-dummy-probe-only"

    config = _load_config()
    env = _new_seed42_env(config)
    try:
        _print_layout(env)
        body, lemon_pos = _find_lemon(env)
        print(f"\nseed=42 lemon_body={body} lemon_pos={_arr(lemon_pos)}")
        print(f"initial base={_base_pose_str(env)} eef={_arr(_eef(env))}")
        _dump_controller(env, "after reset")
    finally:
        _close_env(env)

    _probe_reset_state(config)
    _probe_navigated_state(config, reset_controller=False)
    _probe_navigated_state(config, reset_controller=True)
    _probe_move_arm_after_navigation(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
