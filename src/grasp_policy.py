from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GraspPolicyDecision:
    depth_margin_m: float
    squeeze_extra_steps: int
    mode: str
    applied: bool
    profile: str
    legacy_depth_margin_m: float
    legacy_squeeze_extra_steps: int

    def diagnostic(self) -> dict[str, Any]:
        return {
            "grasp_policy_mode": self.mode,
            "grasp_policy_applied": self.applied,
            "grasp_policy_profile": self.profile,
            "legacy_depth_margin_m": self.legacy_depth_margin_m,
            "legacy_squeeze_extra_steps": self.legacy_squeeze_extra_steps,
        }


def resolve_grasp_policy(
    config: dict[str, Any] | None,
    grasp_profile: str | None,
    depth_margin_m: float,
    squeeze_extra_steps: int,
) -> GraspPolicyDecision:
    mode = _mode(config)
    profile = _profile_name(grasp_profile)
    legacy_depth = float(depth_margin_m)
    legacy_squeeze = int(squeeze_extra_steps)

    if mode != "profiled" or profile not in _enabled_profiles(config):
        return GraspPolicyDecision(
            depth_margin_m=legacy_depth,
            squeeze_extra_steps=legacy_squeeze,
            mode=mode,
            applied=False,
            profile=profile,
            legacy_depth_margin_m=legacy_depth,
            legacy_squeeze_extra_steps=legacy_squeeze,
        )

    if profile == "small_round_slippery":
        return GraspPolicyDecision(
            depth_margin_m=0.010,
            squeeze_extra_steps=max(legacy_squeeze, 16),
            mode=mode,
            applied=True,
            profile=profile,
            legacy_depth_margin_m=legacy_depth,
            legacy_squeeze_extra_steps=legacy_squeeze,
        )

    return GraspPolicyDecision(
        depth_margin_m=legacy_depth,
        squeeze_extra_steps=legacy_squeeze,
        mode=mode,
        applied=False,
        profile=profile,
        legacy_depth_margin_m=legacy_depth,
        legacy_squeeze_extra_steps=legacy_squeeze,
    )


def _mode(config: dict[str, Any] | None) -> str:
    if not isinstance(config, dict):
        return "legacy"
    mode = str(config.get("mode") or "legacy").strip().lower()
    return mode if mode in {"legacy", "profiled"} else "legacy"


def _enabled_profiles(config: dict[str, Any] | None) -> set[str]:
    if not isinstance(config, dict):
        return set()
    profiles = config.get("enabled_profiles") or []
    if not isinstance(profiles, list):
        return set()
    return {_profile_name(profile) for profile in profiles}


def _profile_name(profile: Any) -> str:
    text = str(profile).strip() if profile is not None else ""
    return text if text else "unknown"
