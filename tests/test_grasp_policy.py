from pathlib import Path

import yaml

from src.grasp_policy import resolve_grasp_policy


def test_agent_config_defaults_to_legacy_policy():
    data = yaml.safe_load(Path("configs/agent.yaml").read_text(encoding="utf-8"))

    assert data["grasp_policy"] == {
        "mode": "legacy",
        "enabled_profiles": [],
    }


def test_legacy_policy_returns_execution_params_unchanged():
    decision = resolve_grasp_policy(
        config={
            "mode": "legacy",
            "enabled_profiles": ["small_round_slippery"],
        },
        grasp_profile="small_round_slippery",
        depth_margin_m=0.025,
        squeeze_extra_steps=4,
    )

    assert decision.depth_margin_m == 0.025
    assert decision.squeeze_extra_steps == 4
    assert decision.applied is False
    assert decision.diagnostic() == {
        "grasp_policy_mode": "legacy",
        "grasp_policy_applied": False,
        "grasp_policy_profile": "small_round_slippery",
        "legacy_depth_margin_m": 0.025,
        "legacy_squeeze_extra_steps": 4,
    }


def test_profiled_policy_leaves_disabled_profile_unchanged():
    decision = resolve_grasp_policy(
        config={
            "mode": "profiled",
            "enabled_profiles": ["small_round_slippery"],
        },
        grasp_profile="wide_ungraspable",
        depth_margin_m=0.025,
        squeeze_extra_steps=4,
    )

    assert decision.depth_margin_m == 0.025
    assert decision.squeeze_extra_steps == 4
    assert decision.applied is False


def test_profiled_policy_applies_small_round_slippery_when_enabled():
    decision = resolve_grasp_policy(
        config={
            "mode": "profiled",
            "enabled_profiles": ["small_round_slippery"],
        },
        grasp_profile="small_round_slippery",
        depth_margin_m=0.025,
        squeeze_extra_steps=4,
    )

    assert decision.depth_margin_m == 0.010
    assert decision.squeeze_extra_steps == 16
    assert decision.applied is True
    assert decision.diagnostic() == {
        "grasp_policy_mode": "profiled",
        "grasp_policy_applied": True,
        "grasp_policy_profile": "small_round_slippery",
        "legacy_depth_margin_m": 0.025,
        "legacy_squeeze_extra_steps": 4,
    }


def test_small_round_slippery_policy_preserves_stronger_legacy_squeeze():
    decision = resolve_grasp_policy(
        config={
            "mode": "profiled",
            "enabled_profiles": ["small_round_slippery"],
        },
        grasp_profile="small_round_slippery",
        depth_margin_m=0.015,
        squeeze_extra_steps=18,
    )

    assert decision.depth_margin_m == 0.010
    assert decision.squeeze_extra_steps == 18
    assert decision.applied is True
