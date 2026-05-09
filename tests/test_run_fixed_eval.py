from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


RUN_FIXED_PATH = Path(__file__).parent.parent / "eval" / "run_fixed.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_fixed_eval", str(RUN_FIXED_PATH))
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_fixed_module_loads():
    module = _load_module()
    assert hasattr(module, "load_scenario")


def test_load_scenario_by_id(tmp_path):
    module = _load_module()
    path = tmp_path / "scenarios.yaml"
    path.write_text(
        "scenarios:\n"
        "  - id: fixed_apple_001\n"
        "    seed: 42\n"
        "    env_name: PickPlaceCounterToCabinet\n"
        "    query: pick up anything\n"
        "    expected_object: apple\n"
        "    user_mode: fake_from_robocasa\n"
        "    max_resets: 3\n",
        encoding="utf-8",
    )

    scenario = module.load_scenario(path, "fixed_apple_001")

    assert scenario["id"] == "fixed_apple_001"
    assert scenario["seed"] == 42
    assert scenario["expected_object"] == "apple"


def test_load_scenario_missing_id_raises(tmp_path):
    module = _load_module()
    path = tmp_path / "scenarios.yaml"
    path.write_text("scenarios: []\n", encoding="utf-8")

    with pytest.raises(KeyError):
        module.load_scenario(path, "missing")
