from pathlib import Path

import yaml

from src.grasp_policy import resolve_grasp_policy


def test_agent_config_defaults_to_legacy_policy():
    data = yaml.safe_load(Path("configs/agent.yaml").read_text(encoding="utf-8"))

    assert data["grasp_policy"] == {
        "mode": "legacy",
        "enabled_profiles": [],
        "actionability_diagnostics": False,
        "actionability_gate": False,
        "execution_recovery_diagnostics": False,
        "execution_recovery_gate": False,
        "execution_recovery_max_attempts": 1,
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


def test_actionability_flags_default_to_disabled():
    from src.grasp_policy import (
        actionability_diagnostics_enabled,
        actionability_gate_enabled,
    )

    assert actionability_diagnostics_enabled(None) is False
    assert actionability_gate_enabled(None, "small_round_slippery") is False


def test_actionability_diagnostics_can_be_enabled_in_profiled_mode():
    from src.grasp_policy import actionability_diagnostics_enabled

    assert actionability_diagnostics_enabled({
        "mode": "profiled",
        "enabled_profiles": ["small_round_slippery"],
        "actionability_diagnostics": True,
    }) is True


def test_actionability_gate_requires_profiled_enabled_profile():
    from src.grasp_policy import actionability_gate_enabled

    config = {
        "mode": "profiled",
        "enabled_profiles": ["small_round_slippery"],
        "actionability_gate": True,
    }

    assert actionability_gate_enabled(config, "small_round_slippery") is True
    assert actionability_gate_enabled(config, "thin_flat") is False
    assert actionability_gate_enabled(
        {"mode": "legacy", "actionability_gate": True},
        "small_round_slippery",
    ) is False


def test_actionability_gate_is_limited_to_small_round_slippery():
    from src.grasp_policy import actionability_gate_enabled

    config = {
        "mode": "profiled",
        "enabled_profiles": ["small_round_slippery", "thin_flat"],
        "actionability_gate": True,
    }

    assert actionability_gate_enabled(config, "thin_flat") is False


def test_execution_recovery_policy_requires_profiled_mode():
    from src.grasp_policy import (
        execution_recovery_diagnostics_enabled,
        execution_recovery_gate_enabled,
        execution_recovery_max_attempts,
    )

    legacy = {
        "mode": "legacy",
        "execution_recovery_diagnostics": True,
        "execution_recovery_gate": True,
        "execution_recovery_max_attempts": 2,
    }
    profiled = {
        "mode": "profiled",
        "execution_recovery_diagnostics": True,
        "execution_recovery_gate": True,
        "execution_recovery_max_attempts": 2,
    }

    assert execution_recovery_diagnostics_enabled(legacy) is False
    assert execution_recovery_gate_enabled(legacy) is False
    assert execution_recovery_diagnostics_enabled(profiled) is True
    assert execution_recovery_gate_enabled(profiled) is True
    assert execution_recovery_max_attempts(profiled) == 2


def test_execution_recovery_gate_is_not_limited_by_enabled_profiles():
    from src.grasp_policy import execution_recovery_gate_enabled

    config = {
        "mode": "profiled",
        "enabled_profiles": [],
        "execution_recovery_gate": True,
    }

    assert execution_recovery_gate_enabled(config) is True


def test_execution_recovery_max_attempts_is_clamped():
    from src.grasp_policy import execution_recovery_max_attempts

    assert execution_recovery_max_attempts(None) == 1
    assert execution_recovery_max_attempts({"execution_recovery_max_attempts": -1}) == 0
    assert execution_recovery_max_attempts({"execution_recovery_max_attempts": 9}) == 3
