#!/usr/bin/env python3
"""Record a contest-ready successful RoboCasa kitchen demo video.

The script tries candidate scenarios one by one in fresh Python subprocesses.
Only a successful episode with enough captured frames is copied to the final
output path.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent


def default_candidate_scenarios() -> list[str]:
    return [
        "fixed_lemon_001",
        "fixed_seed_discover_001",
        "fixed_seed_discover_002",
        "fixed_seed_discover_003",
        "fixed_seed_discover_004",
        "fixed_seed_discover_005",
        "fixed_seed_discover_006",
        "fixed_seed_discover_007",
        "fixed_seed_discover_008",
        "fixed_seed_discover_009",
        "fixed_seed_discover_010",
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Try fixed RoboCasa kitchen scenarios until one successful MP4 is recorded.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="results/videos/embosight_kitchen_success.mp4",
        help="Final successful MP4 path.",
    )
    parser.add_argument(
        "--candidates",
        nargs="*",
        default=None,
        help="Scenario IDs to try in order. Defaults to a graspable kitchen-first list.",
    )
    parser.add_argument("--scenarios-config", default="configs/eval_scenarios.yaml")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--agent-config", default="configs/agent.yaml")
    parser.add_argument("--attempt-dir", default="results/videos/attempts")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--min-frames", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--camera",
        default="robot0_frontview",
        help="Camera(s) to record when --hd/--multi is not used.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        default=[1280, 720],
        metavar=("W", "H"),
        help="Per-camera render resolution when --hd/--multi is not used.",
    )
    parser.add_argument("--hd", action="store_true", help="Record front-view HD video.")
    parser.add_argument(
        "--multi",
        action="store_true",
        help="Record a 3-camera tiled view: front, wrist, and side.",
    )
    return parser.parse_args(argv)


def build_record_command(
    args: argparse.Namespace,
    scenario_id: str,
    output_path: Path,
) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/record_video.py",
        "--scenario",
        scenario_id,
        "--scenarios-config",
        args.scenarios_config,
        "--config",
        args.config,
        "--agent-config",
        args.agent_config,
        "--output",
        str(output_path),
        "--fps",
        str(args.fps),
        "--min-frames",
        str(args.min_frames),
        "--require-success",
        "--log-level",
        args.log_level,
    ]
    if args.hd:
        cmd.append("--hd")
    elif args.multi:
        cmd.append("--multi")
    else:
        cmd.extend([
            "--camera",
            args.camera,
            "--resolution",
            str(args.resolution[0]),
            str(args.resolution[1]),
        ])
    return cmd


def _copy_successful_attempt(attempt_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if attempt_path.resolve() == output_path.resolve():
        return
    shutil.copy2(attempt_path, output_path)


def _write_summary(output_path: Path, summary: dict[str, Any]) -> Path:
    summary_path = output_path.with_suffix(".json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_path = Path(args.output)
    attempt_dir = Path(args.attempt_dir)
    attempt_dir.mkdir(parents=True, exist_ok=True)

    candidates = args.candidates or default_candidate_scenarios()
    attempts: list[dict[str, Any]] = []
    started_at = datetime.now().isoformat(timespec="seconds")

    for index, scenario_id in enumerate(candidates, start=1):
        attempt_path = attempt_dir / f"{index:02d}_{scenario_id}.mp4"
        if attempt_path.exists():
            attempt_path.unlink()

        cmd = build_record_command(args, scenario_id, attempt_path)
        print(f"\n[{index}/{len(candidates)}] Recording candidate: {scenario_id}")
        print("Command:", " ".join(cmd))

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(REPO_ROOT),
                timeout=args.timeout,
            )
            returncode = proc.returncode
            error = None
        except subprocess.TimeoutExpired:
            returncode = 124
            error = f"timeout after {args.timeout}s"

        saved = attempt_path.exists() and attempt_path.stat().st_size > 0
        attempts.append({
            "scenario_id": scenario_id,
            "returncode": returncode,
            "attempt_video": str(attempt_path),
            "video_saved": saved,
            "error": error,
        })

        if returncode == 0 and saved:
            _copy_successful_attempt(attempt_path, output_path)
            summary = {
                "success": True,
                "selected_scenario": scenario_id,
                "output": str(output_path),
                "started_at": started_at,
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "attempts": attempts,
            }
            summary_path = _write_summary(output_path, summary)
            print(f"\nSuccessful kitchen demo saved: {output_path}")
            print(f"Summary saved: {summary_path}")
            return 0

        print(f"Candidate failed or did not produce a valid video: {scenario_id}")

    summary = {
        "success": False,
        "selected_scenario": None,
        "output": str(output_path),
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "attempts": attempts,
    }
    summary_path = _write_summary(output_path, summary)
    print("\nNo successful kitchen demo video was produced.")
    print(f"Summary saved: {summary_path}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
