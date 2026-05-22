"""Phase 1 probe: 探查 RoboCasa kitchen 场景中 mobilebase 的 joint 结构.

Used to design navigate_base_to (Phase 2) teleport implementation. Outputs:
- mobilebase joint 命名清单
- qpos / qvel address
- joint type & axis (slide vs hinge vs free)
- 真实 base body xpos vs robot.base_pos (anchor)
- arm joint addresses (for future arm-home-pose reset if needed)

Usage:
    DEEPSEEK_API_KEY=sk-xxx MUJOCO_GL=egl \
        python scripts/probe_mobilebase_joints.py

The probe does not require an LLM call but DEEPSEEK_API_KEY env must be
set because EnvWrapper / config validation reads it. MUJOCO_GL=egl is
needed for headless GPU rendering.
"""

import os
import sys
from pathlib import Path

# Make repo root importable when run as script
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Load .env (DEEPSEEK_API_KEY etc.) using project utility
from src.utils import load_dotenv  # noqa: E402

load_dotenv(str(REPO_ROOT / ".env"))


def _load_config():
    """Load default config used by run_fixed pipeline."""
    import yaml

    config_path = Path("configs/default.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_env(config: dict):
    """Build EnvWrapper with same setup as scripts/run_agent._build_env."""
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


def _probe_joints(env) -> None:
    """Print mobilebase joint structure."""
    sim = env._env.sim
    model = sim.model

    print("=" * 70)
    print("PROBE 1: All joints with 'mobile' or 'base' in name")
    print("=" * 70)
    print(f"{'name':40} {'type':6} {'qpos_addr':10} {'qvel_addr':10} {'axis':20}")
    print("-" * 70)

    JTYPE = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}
    found_any = False
    for jid in range(model.njnt):
        name = model.joint_id2name(jid)
        if not name:
            continue
        if "mobile" not in name.lower() and "base" not in name.lower():
            continue
        found_any = True
        jtype = int(model.jnt_type[jid])
        qpos_addr = int(model.jnt_qposadr[jid])
        qvel_addr = int(model.jnt_dofadr[jid])
        axis = tuple(round(float(a), 3) for a in model.jnt_axis[jid])
        print(f"{name:40} {JTYPE.get(jtype, '?'):6} {qpos_addr:<10} {qvel_addr:<10} {axis}")

    if not found_any:
        print("  (no joint with 'mobile' or 'base' in name)")


def _probe_robot_joints(env) -> None:
    """Print all robot joints to find arm joint range."""
    sim = env._env.sim
    model = sim.model

    print()
    print("=" * 70)
    print("PROBE 2: All robot0_* joints (arm + gripper)")
    print("=" * 70)
    print(f"{'name':40} {'type':6} {'qpos_addr':10} {'qvel_addr':10}")
    print("-" * 70)

    JTYPE = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}
    count = 0
    for jid in range(model.njnt):
        name = model.joint_id2name(jid)
        if not name or "robot0_" not in name.lower():
            continue
        # Skip anything we already printed in PROBE 1
        if "mobile" in name.lower() or name.endswith("_base"):
            continue
        jtype = int(model.jnt_type[jid])
        qpos_addr = int(model.jnt_qposadr[jid])
        qvel_addr = int(model.jnt_dofadr[jid])
        print(f"{name:40} {JTYPE.get(jtype, '?'):6} {qpos_addr:<10} {qvel_addr:<10}")
        count += 1
    if count == 0:
        print("  (no robot0_* arm/gripper joints found - check naming)")


def _probe_base_bodies(env) -> None:
    """Print real base body world position vs robot.base_pos anchor."""
    import numpy as np

    sim = env._env.sim
    model = sim.model
    data = sim.data
    robot = env._env.robots[0]

    print()
    print("=" * 70)
    print("PROBE 3: Base body world positions (anchor vs real)")
    print("=" * 70)

    # robot.base_pos (the anchor)
    base_pos = np.asarray(robot.base_pos, dtype=np.float32)
    print(f"  robot.base_pos (anchor):       {tuple(round(float(x), 3) for x in base_pos)}")
    try:
        base_ori = np.asarray(robot.base_ori, dtype=np.float32)
        print(f"  robot.base_ori (anchor xmat):")
        for row in base_ori:
            print(f"    {tuple(round(float(x), 3) for x in row)}")
    except Exception as e:
        print(f"  robot.base_ori unavailable: {e}")

    idn = getattr(robot, "idn", 0)
    candidates = [
        f"mobilebase{idn}_base",
        f"mobilebase{idn}_root",
        f"robot{idn}_base",
        f"robot{idn}_root",
        f"robot{idn}_mobilebase",
        f"robot{idn}_floating_base",
    ]
    print()
    for body_name in candidates:
        try:
            bid = model.body_name2id(body_name)
            xpos = data.body_xpos[bid]
            xmat = data.body_xmat[bid].reshape(3, 3)
            print(f"  body '{body_name}' (id={bid}):")
            print(f"    xpos = {tuple(round(float(x), 3) for x in xpos)}")
            print(f"    xmat row0 = {tuple(round(float(x), 3) for x in xmat[0])}")
        except (KeyError, ValueError):
            print(f"  body '{body_name}': NOT FOUND")


def _probe_action_layout(env) -> None:
    """Print action vector layout from composite_controller."""
    print()
    print("=" * 70)
    print("PROBE 4: Action vector layout (composite_controller)")
    print("=" * 70)
    try:
        robot = env._env.robots[0]
        idx = 0
        for part_name, ctrl in robot.composite_controller.part_controllers.items():
            dim = ctrl.control_dim
            print(f"  [{idx:3}:{idx + dim:3}] part='{part_name}' dim={dim} type={type(ctrl).__name__}")
            idx += dim
        print(f"  total action_dim = {idx}")
        print(f"  env.action_dim   = {env._env.action_dim}")
    except Exception as e:
        print(f"  ERROR: {e}")


def _probe_arm_init_qpos(env) -> None:
    """Print arm init_qpos for future arm-home-pose reset."""
    print()
    print("=" * 70)
    print("PROBE 5: Arm init_qpos (for potential home-pose reset)")
    print("=" * 70)
    try:
        robot = env._env.robots[0]
        if hasattr(robot, "init_qpos"):
            init = robot.init_qpos
            print(f"  robot.init_qpos = {init}")
        else:
            print(f"  robot.init_qpos: NOT available")
    except Exception as e:
        print(f"  ERROR: {e}")


def main() -> int:
    # load_dotenv at module top already set DEEPSEEK_API_KEY if .env exists.
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("WARNING: DEEPSEEK_API_KEY not set (no .env or env). "
              "Using dummy value for probe (no LLM call needed).")
        os.environ["DEEPSEEK_API_KEY"] = "sk-dummy-probe-only"

    print("Loading config and building env (this takes ~30s for first RoboCasa init)...")
    config = _load_config()
    env = _build_env(config)
    env.seed(42)
    env.reset()
    print("Env ready.\n")

    _probe_joints(env)
    _probe_robot_joints(env)
    _probe_base_bodies(env)
    _probe_action_layout(env)
    _probe_arm_init_qpos(env)

    print()
    print("=" * 70)
    print("Probe complete. Copy outputs into docs/07_navigation_refactor_design.md")
    print("appendix D.2 'Phase 1 Probe Results'.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
