"""Probe EEF z reach limit at the lemon scene (seed=42) for varied base offsets.

Goal
----
After torso assist + base nudge (commits aa4f4042 + de4e53d), the lemon top-down
grasp still stops at EEF z ~0.940-0.945, while lemon center is at z~0.932 and
lemon top at z~0.957. We need to know:

  1. Is z=0.945 truly the kinematic floor, or just the floor with the current
     0.55m navigate offset?
  2. Does reducing offset open up lower z but break OSC (the original reason
     0.55m was picked, per action_executor.py:95-98)?
  3. Is the torso fully saturated at every offset?

Method
------
For each base offset in a sweep:
  1. fresh reset (seed=42), find lemon body
  2. navigate_base_to(lemon_xy, offset_m=X)
  3. record start state (eef, torso qpos, arm qpos)
  4. command move_arm_to(target = lemon_xy + (lemon_z - 0.025)), drive_base=False
     - that's the same descend target used in the failing run
  5. record final EEF z, Δz to target, torso qpos (saturation indicator)
  6. issue a final OSC -z pulse (action[2]=-0.5, 30 steps) to test if z control
     is still alive after the descend
  7. print one CSV row per offset

Usage on GPU
------------
    MUJOCO_GL=egl python scripts/probe_eef_z_limit.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils import load_dotenv  # noqa: E402

load_dotenv(str(REPO_ROOT / ".env"))


# ---------------------------------------------------------------------------
# Setup helpers (mirrored from probe_arm_after_navigation.py for self-contained
# runs).
# ---------------------------------------------------------------------------


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


def _new_seed42_env(config: dict):
    env = _build_env(config)
    env.seed(42)
    env.reset()
    return env


def _close_env(env) -> None:
    backend = getattr(env, "_env", None)
    close = getattr(backend, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


def _arr(x: Any, ndigits: int = 4) -> str:
    try:
        a = np.asarray(x, dtype=np.float64).round(ndigits)
        return str(a.tolist())
    except Exception:
        return repr(x)


def _eef(env) -> np.ndarray:
    return np.asarray(env._latest_obs["robot0_eef_pos"], dtype=np.float64)


def _find_lemon(env) -> tuple[str, np.ndarray]:
    tmap = env._get_obj_type_map()
    for body, cat in tmap.items():
        if str(cat).lower() == "lemon":
            pos = env._get_body_pos(body)
            if pos is None:
                raise RuntimeError(f"lemon body {body!r} has no position")
            return body, np.asarray(pos, dtype=np.float64)
    raise RuntimeError(f"lemon not found in object map: {tmap}")


def _torso_qpos(env) -> Optional[float]:
    """Read torso joint qpos (the joint commanded by torso assist)."""
    try:
        sim = env._env.sim
        for jname in ("robot0_torso_joint0", "robot0_torso_joint", "torso_joint0"):
            try:
                jid = sim.model.joint_name2id(jname)
                qpos_idx = sim.model.jnt_qposadr[jid]
                return float(sim.data.qpos[qpos_idx])
            except Exception:
                continue
    except Exception:
        pass
    return None


def _arm_qpos(env) -> Optional[np.ndarray]:
    """Read 7-DoF right arm joint qpos (Panda)."""
    try:
        sim = env._env.sim
        qpos = []
        for i in range(7):
            for name_pat in (
                f"robot0_right_j{i}",
                f"robot0_joint{i + 1}",
                f"robot0_panda_joint{i + 1}",
            ):
                try:
                    jid = sim.model.joint_name2id(name_pat)
                    qpos.append(float(sim.data.qpos[sim.model.jnt_qposadr[jid]]))
                    break
                except Exception:
                    continue
            else:
                return None
        return np.asarray(qpos, dtype=np.float64)
    except Exception:
        return None


def _sync_zero(env, steps: int = 1) -> None:
    action = np.zeros(env._env.action_dim, dtype=np.float32)
    for _ in range(steps):
        obs, _, _, _ = env._env.step(action)
        env._latest_obs = obs


def _z_pulse(env, value: float = -0.5, steps: int = 30) -> float:
    """Hold action[2]=value for `steps` steps. Returns Δz of EEF."""
    start_z = float(_eef(env)[2])
    action = np.zeros(env._env.action_dim, dtype=np.float32)
    action[2] = float(value)
    for _ in range(steps):
        obs, _, _, _ = env._env.step(action)
        env._latest_obs = obs
    return float(_eef(env)[2]) - start_z


# ---------------------------------------------------------------------------
# Core probe
# ---------------------------------------------------------------------------


def _probe_one_offset(config: dict, offset_m: float) -> dict:
    env = _new_seed42_env(config)
    try:
        body, lemon = _find_lemon(env)
        nav_ok = env.navigate_base_to(lemon[:2], offset_m=offset_m)
        _sync_zero(env, steps=2)

        start_eef = _eef(env).copy()
        start_torso = _torso_qpos(env)
        start_arm = _arm_qpos(env)

        # Target = same as action_executor's adjusted descend target
        # (top_down: lemon_xy, lemon_z - 0.025m margin).
        target = np.array(
            [float(lemon[0]), float(lemon[1]), float(lemon[2]) - 0.025],
            dtype=np.float32,
        )

        # Run move_arm_to with the same params action_executor uses for
        # descend: drive_base=False, threshold 2 mm, max_steps 300.
        move_ok = False
        try:
            move_ok = bool(
                env.move_arm_to(
                    target,
                    threshold_m=0.002,
                    max_steps=300,
                    drive_base=False,
                )
            )
        except Exception as e:
            print(f"  move_arm_to raised: {type(e).__name__}: {e}")

        end_eef = _eef(env).copy()
        end_torso = _torso_qpos(env)
        end_arm = _arm_qpos(env)

        # Final z pulse: is OSC z control still alive after settling?
        z_pulse_delta = _z_pulse(env, value=-0.5, steps=30)

        result = {
            "offset_m": offset_m,
            "nav_ok": bool(nav_ok),
            "lemon_xyz": lemon.tolist(),
            "target_z": float(target[2]),
            "start_eef": start_eef.tolist(),
            "end_eef": end_eef.tolist(),
            "delta_z_to_target": float(end_eef[2] - target[2]),
            "delta_z_to_lemon_center": float(end_eef[2] - float(lemon[2])),
            "start_torso_qpos": start_torso,
            "end_torso_qpos": end_torso,
            "torso_excursion": (
                None
                if (start_torso is None or end_torso is None)
                else float(end_torso - start_torso)
            ),
            "start_arm_qpos": (None if start_arm is None else start_arm.tolist()),
            "end_arm_qpos": (None if end_arm is None else end_arm.tolist()),
            "move_arm_ok": move_ok,
            "z_pulse_delta": z_pulse_delta,
        }
        return result
    finally:
        _close_env(env)


def _format_row(r: dict) -> str:
    eef = r["end_eef"]
    return (
        f"offset={r['offset_m']:.2f}m  "
        f"nav_ok={int(r['nav_ok'])}  "
        f"end_z={eef[2]:.4f}  "
        f"Δz_target={r['delta_z_to_target']:+.4f}m  "
        f"Δz_lemon_ctr={r['delta_z_to_lemon_center']:+.4f}m  "
        f"torso[start→end]={r['start_torso_qpos']}→{r['end_torso_qpos']}  "
        f"move_ok={int(r['move_arm_ok'])}  "
        f"z_pulse_Δ={r['z_pulse_delta']:+.4f}m"
    )


def main() -> int:
    if not os.environ.get("DEEPSEEK_API_KEY"):
        os.environ["DEEPSEEK_API_KEY"] = "sk-dummy-probe-only"

    config = _load_config()

    # Reference scan: print lemon position once with a fresh env.
    env = _new_seed42_env(config)
    try:
        body, lemon = _find_lemon(env)
        print("\n=== Scene ===")
        print(f"seed=42 lemon_body={body} lemon_xyz={_arr(lemon)}")
        print(f"  lemon top z ≈ {float(lemon[2]) + 0.025:.4f}")
        print(f"  lemon ctr z  = {float(lemon[2]):.4f}")
        print(f"  lemon bot z ≈ {float(lemon[2]) - 0.025:.4f}")
        print(f"  descend target (ctr-0.025) = {float(lemon[2]) - 0.025:.4f}")
        print(f"  initial eef  = {_arr(_eef(env))}")
        print(f"  initial torso qpos = {_torso_qpos(env)}")
    finally:
        _close_env(env)

    print("\n=== Z-reach sweep over base offset ===")
    print("(target_z = lemon_ctr - 0.025; positive Δz_target = EEF stopped ABOVE target)")
    print()

    offsets = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
    results = []
    for off in offsets:
        print(f"--- offset {off:.2f}m ---")
        r = _probe_one_offset(config, off)
        results.append(r)
        print("  " + _format_row(r))

    print("\n=== Summary table ===")
    print(
        f"{'offset':>7s}  {'end_z':>7s}  {'Δz_tgt':>8s}  "
        f"{'Δz_ctr':>8s}  {'torso_excursion':>15s}  "
        f"{'move_ok':>7s}  {'z_pulse_Δ':>10s}"
    )
    for r in results:
        torso_str = (
            "n/a" if r["torso_excursion"] is None
            else f"{r['torso_excursion']:+.4f}"
        )
        print(
            f"{r['offset_m']:>7.2f}  "
            f"{r['end_eef'][2]:>7.4f}  "
            f"{r['delta_z_to_target']:>+8.4f}  "
            f"{r['delta_z_to_lemon_center']:>+8.4f}  "
            f"{torso_str:>15s}  "
            f"{int(r['move_arm_ok']):>7d}  "
            f"{r['z_pulse_delta']:>+10.4f}"
        )

    print()
    print("=== Decision criteria ===")
    print("  Δz_ctr ≤ +0.005m AND |z_pulse_Δ| ≥ 0.005m  → safe new offset")
    print("  Δz_ctr > +0.010m at all offsets             → kinematic floor;")
    print("                                                  pick alt grasp geometry.")
    print("  torso_excursion ≈ 0 at every offset         → torso saturated from init;")
    print("                                                  raise initial torso qpos.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
