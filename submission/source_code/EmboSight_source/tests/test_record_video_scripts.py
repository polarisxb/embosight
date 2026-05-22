from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
RECORD_VIDEO = REPO_ROOT / "scripts" / "record_video.py"
RECORD_SUCCESS_VIDEO = REPO_ROOT / "scripts" / "record_success_video.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_record_video_cli_supports_success_only_mode() -> None:
    text = RECORD_VIDEO.read_text(encoding="utf-8")

    assert "--require-success" in text
    assert "--min-frames" in text
    assert "RecordingResult" in text


def test_record_video_respects_scenario_user_mode_when_building_agent() -> None:
    text = RECORD_VIDEO.read_text(encoding="utf-8")

    assert 'str(scenario.get("user_mode", "fake_from_robocasa"))' in text
    assert 'build_agent(top_cfg, agent_cfg, env, user_mode, query)' in text


def test_record_video_multi_preset_preserves_explicit_fps() -> None:
    module = _load_module(RECORD_VIDEO, "record_video_presets")
    args = module.parse_args([
        "--scenario", "fixed_lemon_001",
        "--multi",
        "--fps", "24",
    ])

    module.apply_video_preset_args(args)

    assert args.camera == "robot0_frontview,robot0_eye_in_hand,robot0_agentview_left"
    assert args.resolution == [480, 480]
    assert args.fps == 24


def test_record_video_hd_preset_uses_default_hd_fps_when_unspecified() -> None:
    module = _load_module(RECORD_VIDEO, "record_video_hd_presets")
    args = module.parse_args([
        "--scenario", "fixed_lemon_001",
        "--hd",
    ])

    module.apply_video_preset_args(args)

    assert args.camera == "robot0_frontview"
    assert args.resolution == [1280, 720]
    assert args.fps == 15


def test_record_success_video_defaults_to_graspable_kitchen_scene() -> None:
    module = _load_module(RECORD_SUCCESS_VIDEO, "record_success_video")

    candidates = module.default_candidate_scenarios()

    assert candidates[0] == "fixed_lemon_001"
    assert "fixed_seed_discover_001" in candidates


def test_record_success_video_builds_success_checked_record_command(tmp_path) -> None:
    module = _load_module(RECORD_SUCCESS_VIDEO, "record_success_video_cmd")
    args = module.parse_args([
        "--output", str(tmp_path / "demo.mp4"),
        "--config", "configs/default.yaml",
        "--agent-config", "configs/agent.yaml",
        "--scenarios-config", "configs/eval_scenarios.yaml",
        "--multi",
        "--fps", "12",
        "--frame-repeat", "2",
        "--min-frames", "8",
    ])

    cmd = module.build_record_command(
        args=args,
        scenario_id="fixed_lemon_001",
        output_path=tmp_path / "attempt.mp4",
    )

    assert cmd[0] == sys.executable
    assert "scripts/record_video.py" in cmd
    assert "--require-success" in cmd
    assert "--scenario" in cmd
    assert "fixed_lemon_001" in cmd
    assert "--output" in cmd
    assert str(tmp_path / "attempt.mp4") in cmd
    assert "--multi" in cmd
    assert "--frame-repeat" in cmd
    assert "2" in cmd
    assert "--min-frames" in cmd
    assert "8" in cmd
