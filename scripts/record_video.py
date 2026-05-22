#!/usr/bin/env python3
"""Record one EmboSight RoboCasa episode as an MP4 video.

Examples:
  python scripts/record_video.py --scenario fixed_seed_discover_001 --fps 10
  python scripts/record_video.py --scenario fixed_lemon_001 --multi --require-success
  python scripts/record_video.py --scenario fixed_seed_discover_001 --hd -o results/videos/demo.mp4
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RecordingResult:
    """Summary returned by a video recording run."""

    scenario_id: str
    success: bool
    output_path: str
    video_saved: bool
    frames_captured: int
    steps: int | None
    elapsed_seconds: float | None
    failure_reason: str | None = None

    def cli_ok(self, require_success: bool) -> bool:
        if not self.video_saved:
            return False
        return self.success if require_success else True


def _camera_names(camera: str) -> list[str]:
    return [c.strip() for c in camera.split(",") if c.strip()]


def _grab_frame(obs: dict, img_keys: list[str]) -> np.ndarray | None:
    imgs = []
    for key in img_keys:
        img = obs.get(key)
        if img is not None:
            imgs.append(img.copy())
    if not imgs:
        return None
    if len(imgs) == 1:
        return imgs[0]
    return np.concatenate(imgs, axis=1)


def _write_video(frames: list[np.ndarray], output_path: str, fps: int) -> None:
    import imageio

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    writer = imageio.get_writer(str(out), fps=fps, codec="libx264", quality=8)
    try:
        for frame in frames:
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            if frame.ndim == 3 and frame.shape[2] == 4:
                frame = frame[:, :, :3]
            # MuJoCo image origin is bottom-left; flip for normal video playback.
            writer.append_data(np.flipud(frame).copy())
    finally:
        writer.close()


def _delete_stale_output(output_path: str) -> None:
    path = Path(output_path)
    if path.exists():
        path.unlink()


def record_episode(
    scenario_id: str,
    scenarios_config: str,
    config_path: str,
    agent_config_path: str,
    output_path: str,
    fps: int = 10,
    camera: str = "robot0_agentview_center",
    resolution: tuple[int, int] = (256, 256),
    require_success: bool = False,
    min_frames: int = 1,
) -> RecordingResult:
    """Run one scenario and optionally save its recorded frames.

    When require_success is true, failed episodes are not written to output_path.
    This makes the script safe for contest-demo recording pipelines where stale
    failed videos must not be mistaken for a successful take.
    """
    from eval.run_fixed import (
        build_agent,
        load_scenario,
        reset_until_expected,
        rewrite_query_for_actual_object,
    )
    from scripts.run_agent import _build_env

    scenario = load_scenario(scenarios_config, scenario_id)
    user_mode = str(scenario.get("user_mode", "fake_from_robocasa"))

    top_cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    agent_cfg = yaml.safe_load(Path(agent_config_path).read_text(encoding="utf-8")) or {}
    if scenario.get("env_name"):
        top_cfg.setdefault("simulator", {})["env_name"] = scenario["env_name"]

    top_cfg.setdefault("simulator", {})["image_width"] = resolution[0]
    top_cfg["simulator"]["image_height"] = resolution[1]

    cameras = _camera_names(camera)
    cam_list = list(top_cfg["simulator"].get("camera_names", []))
    for cam in cameras:
        if cam not in cam_list:
            cam_list.append(cam)
    top_cfg["simulator"]["camera_names"] = cam_list
    img_keys = [f"{cam}_image" for cam in cameras]

    env = None
    frames: list[np.ndarray] = []
    result = None
    try:
        env = _build_env(top_cfg)
        actual_object, _ = reset_until_expected(
            env,
            expected_object=scenario.get("expected_object"),
            seed=scenario.get("seed"),
            max_resets=int(scenario.get("max_resets", 1)),
        )
        query = rewrite_query_for_actual_object(
            str(scenario.get("query", "pick up anything")),
            user_mode,
            actual_object,
        )

        latest_obs = getattr(env, "_latest_obs", None)
        if latest_obs is not None:
            init_frame = _grab_frame(latest_obs, img_keys)
            if init_frame is not None:
                frames.append(init_frame)

        original_step = env._env.step

        def _hooked_step(action):
            obs, reward, done, info = original_step(action)
            env._latest_obs = obs
            frame = _grab_frame(obs, img_keys)
            if frame is not None:
                frames.append(frame)
            return obs, reward, done, info

        env._env.step = _hooked_step

        agent = build_agent(top_cfg, agent_cfg, env, user_mode, query)
        result = agent.run(query, env)

        print(f"\n{'=' * 50}")
        print(f"Episode finished: {'SUCCESS' if result.success else 'FAIL'}")
        print(f"Steps: {result.n_steps}, Time: {result.elapsed_seconds:.1f}s")
        print(f"Frames captured: {len(frames)}")
        print(f"{'=' * 50}")

        enough_frames = len(frames) >= min_frames
        should_save = enough_frames and (result.success or not require_success)
        if not should_save:
            _delete_stale_output(output_path)
            if not enough_frames:
                print(
                    f"ERROR: captured {len(frames)} frame(s), "
                    f"but --min-frames requires {min_frames}."
                )
            elif require_success:
                print("Episode failed; --require-success prevents saving this take.")
            return RecordingResult(
                scenario_id=scenario_id,
                success=bool(result.success),
                output_path=output_path,
                video_saved=False,
                frames_captured=len(frames),
                steps=int(result.n_steps),
                elapsed_seconds=float(result.elapsed_seconds),
                failure_reason=getattr(result, "failure_reason", None),
            )

        _write_video(frames, output_path, fps)
        print(
            f"\nVideo saved: {output_path} "
            f"({len(frames)} frames @ {fps}fps = {len(frames) / fps:.1f}s)"
        )
        return RecordingResult(
            scenario_id=scenario_id,
            success=bool(result.success),
            output_path=output_path,
            video_saved=True,
            frames_captured=len(frames),
            steps=int(result.n_steps),
            elapsed_seconds=float(result.elapsed_seconds),
            failure_reason=getattr(result, "failure_reason", None),
        )
    finally:
        if env is not None:
            close = getattr(env, "close", None)
            if callable(close):
                close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record one EmboSight episode video")
    parser.add_argument("--scenario", required=True, help="Scenario ID from configs/eval_scenarios.yaml")
    parser.add_argument("--scenarios-config", default="configs/eval_scenarios.yaml")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--agent-config", default="configs/agent.yaml")
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output mp4 path (default: results/videos/<scenario>.mp4)",
    )
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--min-frames", type=int, default=1)
    parser.add_argument(
        "--require-success",
        action="store_true",
        help="Only save the video and exit 0 if the episode result is success.",
    )
    parser.add_argument(
        "--camera",
        default="robot0_agentview_center",
        help=(
            "Camera(s) to record, comma-separated for tiled view. "
            "Example: robot0_frontview,robot0_eye_in_hand,robot0_agentview_left"
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        default=[256, 256],
        metavar=("W", "H"),
        help="Per-camera render resolution (default: 256 256).",
    )
    parser.add_argument(
        "--hd",
        action="store_true",
        help="Shortcut for --camera robot0_frontview --resolution 1280 720 --fps 15.",
    )
    parser.add_argument(
        "--multi",
        action="store_true",
        help="Shortcut for a 3-camera tiled kitchen view.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    if args.multi:
        args.camera = "robot0_frontview,robot0_eye_in_hand,robot0_agentview_left"
        args.resolution = [480, 480]
        args.fps = 10

    if args.hd:
        args.camera = "robot0_frontview"
        args.resolution = [1280, 720]
        args.fps = 15

    output = args.output or f"results/videos/{args.scenario}.mp4"

    recording = record_episode(
        scenario_id=args.scenario,
        scenarios_config=args.scenarios_config,
        config_path=args.config,
        agent_config_path=args.agent_config,
        output_path=output,
        fps=args.fps,
        camera=args.camera,
        resolution=tuple(args.resolution),
        require_success=args.require_success,
        min_frames=max(1, args.min_frames),
    )
    return 0 if recording.cli_ok(args.require_success) else 1


if __name__ == "__main__":
    sys.exit(main())
