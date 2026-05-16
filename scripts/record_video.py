#!/usr/bin/env python3
"""录制一次 EmboSight 场景执行视频

用法:
  python scripts/record_video.py --scenario random_seed_3 --output results/videos/demo.mp4
  python scripts/record_video.py --scenario fixed_seed_discover_001 --fps 10

会在 agent 执行每一步时抓取 agentview_center 帧，最终拼成 mp4。
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger(__name__)


def record_episode(
    scenario_id: str,
    scenarios_config: str,
    config_path: str,
    agent_config_path: str,
    output_path: str,
    fps: int = 10,
    camera: str = "robot0_agentview_center",
    resolution: tuple[int, int] = (256, 256),
) -> None:
    """运行一个 episode 并录制视频"""
    import imageio

    from eval.run_fixed import (
        build_agent,
        load_scenario,
        reset_until_expected,
        rewrite_query_for_actual_object,
    )
    from scripts.run_agent import _build_env

    scenario = load_scenario(scenarios_config, scenario_id)
    top_cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    agent_cfg = yaml.safe_load(Path(agent_config_path).read_text(encoding="utf-8")) or {}
    if scenario.get("env_name"):
        top_cfg.setdefault("simulator", {})["env_name"] = scenario["env_name"]

    # 覆盖分辨率
    top_cfg.setdefault("simulator", {})["image_width"] = resolution[0]
    top_cfg["simulator"]["image_height"] = resolution[1]
    # 确保录制相机在列表中
    cam_list = top_cfg["simulator"].get("camera_names", [])
    if camera not in cam_list:
        cam_list.append(camera)
        top_cfg["simulator"]["camera_names"] = cam_list

    env = _build_env(top_cfg)
    actual_object, _ = reset_until_expected(
        env, expected_object=scenario.get("expected_object"), seed=scenario.get("seed"),
    )
    query = rewrite_query_for_actual_object(
        str(scenario.get("query", "pick up anything")),
        str(scenario.get("user_mode", "fake_from_robocasa")),
        actual_object,
    )

    # 收集帧的钩子：monkey-patch env.step 来抓帧
    frames: list[np.ndarray] = []
    img_key = f"{camera}_image"

    # 抓初始帧
    init_img = env._latest_obs.get(img_key)
    if init_img is not None:
        frames.append(init_img.copy())

    original_step = env._env.step

    def _hooked_step(action):
        obs, reward, done, info = original_step(action)
        env._latest_obs = obs
        img = obs.get(img_key)
        if img is not None:
            frames.append(img.copy())
        return obs, reward, done, info

    env._env.step = _hooked_step

    # 运行 agent
    agent = build_agent(top_cfg, agent_cfg, env, "fake_from_robocasa", query)
    result = agent.run(query, env)

    print(f"\n{'='*50}")
    print(f"Episode finished: {'SUCCESS' if result.success else 'FAIL'}")
    print(f"Steps: {result.n_steps}, Time: {result.elapsed_seconds:.1f}s")
    print(f"Frames captured: {len(frames)}")
    print(f"{'='*50}")

    # 写视频
    if not frames:
        print("ERROR: No frames captured!")
        return

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    writer = imageio.get_writer(str(out), fps=fps, codec="libx264", quality=8)
    for frame in frames:
        # imageio 需要 uint8 RGB
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        if frame.ndim == 3 and frame.shape[2] == 4:  # RGBA -> RGB
            frame = frame[:, :, :3]
        writer.append_data(frame)
    writer.close()

    print(f"\nVideo saved: {out} ({len(frames)} frames @ {fps}fps = {len(frames)/fps:.1f}s)")
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Record EmboSight episode video")
    parser.add_argument("--scenario", required=True, help="Scenario ID (e.g. random_seed_3)")
    parser.add_argument("--scenarios-config", default="configs/eval_scenarios.yaml")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--agent-config", default="configs/agent.yaml")
    parser.add_argument("--output", "-o", default=None,
                        help="Output mp4 path (default: results/videos/<scenario>.mp4)")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--camera", default="robot0_agentview_center",
                        help="Camera to record from (use robot0_frontview for 3D overview)")
    parser.add_argument("--resolution", type=int, nargs=2, default=[256, 256],
                        metavar=("W", "H"),
                        help="Render resolution (default: 256 256, use 640 480 or 1280 720 for HD)")
    parser.add_argument("--hd", action="store_true",
                        help="Shortcut for --camera robot0_frontview --resolution 1280 720 --fps 15")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    # --hd 快捷方式：3D 全景高清
    if args.hd:
        args.camera = "robot0_frontview"
        args.resolution = [1280, 720]
        args.fps = 15

    output = args.output or f"results/videos/{args.scenario}.mp4"

    record_episode(
        scenario_id=args.scenario,
        scenarios_config=args.scenarios_config,
        config_path=args.config,
        agent_config_path=args.agent_config,
        output_path=output,
        fps=args.fps,
        camera=args.camera,
        resolution=tuple(args.resolution),
    )


if __name__ == "__main__":
    main()
