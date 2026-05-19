from __future__ import annotations

from dataclasses import dataclass
from typing import Any


_CANDIDATE_ATTEMPT_DIAGNOSTIC_ATTR = "_embosight_attempt_diagnostic"


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


@dataclass(frozen=True)
class CandidateSourcePolicyDecision:
    candidates: list[Any]
    policy: str
    applied: bool
    legacy_first_candidate_source: str | None
    final_first_candidate_source: str | None

    def diagnostic(self) -> dict[str, Any]:
        return {
            "candidate_source_policy": self.policy,
            "candidate_source_policy_applied": self.applied,
            "legacy_first_candidate_source": self.legacy_first_candidate_source,
            "final_first_candidate_source": self.final_first_candidate_source,
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


def apply_candidate_source_policy(
    config: dict[str, Any] | None,
    grasp_profile: str | None,
    selected_strategy: str | None,
    candidates: list[Any],
) -> CandidateSourcePolicyDecision:
    legacy_order = list(candidates)
    legacy_first = _candidate_source(legacy_order[0]) if legacy_order else None
    final_order = list(legacy_order)
    policy = "legacy"
    applied = False

    profile = _profile_name(grasp_profile)
    if (
        _mode(config) == "profiled"
        and profile == "small_round_slippery"
        and profile in _enabled_profiles(config)
    ):
        policy = "prefer_selected_strategy_candidate"
        strategy_source = _selected_strategy_source(selected_strategy)
        if strategy_source:
            strategy_idx = _first_source_index(final_order, strategy_source)
            vlm_idx = _first_source_index(final_order, "vlm_top_grasp")
            if strategy_idx is not None and vlm_idx is not None and vlm_idx < strategy_idx:
                selected = final_order.pop(strategy_idx)
                final_order.insert(vlm_idx, selected)
                applied = True

    final_first = _candidate_source(final_order[0]) if final_order else None
    return CandidateSourcePolicyDecision(
        candidates=final_order,
        policy=policy,
        applied=applied,
        legacy_first_candidate_source=legacy_first,
        final_first_candidate_source=final_first,
    )


def merge_candidate_attempt_diagnostic(candidate: Any, diagnostic: dict[str, Any]) -> None:
    if candidate is None or not diagnostic:
        return
    existing = getattr(candidate, _CANDIDATE_ATTEMPT_DIAGNOSTIC_ATTR, None)
    if not isinstance(existing, dict):
        existing = {}
    merged = dict(existing)
    merged.update(diagnostic)
    setattr(candidate, _CANDIDATE_ATTEMPT_DIAGNOSTIC_ATTR, merged)


def _candidate_source(candidate: Any) -> str:
    return str(getattr(candidate, "source", "unknown"))


def _first_source_index(candidates: list[Any], source: str) -> int | None:
    for idx, candidate in enumerate(candidates):
        if _candidate_source(candidate) == source:
            return idx
    return None


def _selected_strategy_source(selected_strategy: str | None) -> str | None:
    strategy = str(selected_strategy or "").strip()
    if not strategy or strategy == "unknown":
        return None
    return f"strategy_{strategy}"
