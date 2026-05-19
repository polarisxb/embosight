"""Diagnostic-only object profile classification for grasp analysis."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

import numpy as np


class GraspProfile(str, Enum):
    SMALL_ROUND_SLIPPERY = "small_round_slippery"
    THIN_FLAT = "thin_flat"
    WIDE_UNGRASPABLE = "wide_ungraspable"
    HANDLED = "handled"
    FRAGILE_SOFT = "fragile_soft"
    DEFAULT_RIGID = "default_rigid"
    UNKNOWN = "unknown"


@dataclass
class GraspProfileResult:
    profile: GraspProfile
    confidence: float
    reasons: list[str] = field(default_factory=list)
    execution_overrides: dict[str, Any] = field(default_factory=dict)

    def to_diagnostic(self) -> dict[str, Any]:
        return {
            "grasp_profile": self.profile.value,
            "grasp_profile_confidence": float(self.confidence),
            "grasp_profile_reasons": list(self.reasons),
        }


def classify_grasp_profile(
    hyp: Any,
    candidate: Any = None,
    object_size_m: Iterable[float] | None = None,
    gripper_max_width_m: float = 0.08,
) -> GraspProfileResult:
    """Classify a grasp target for diagnostics without changing execution."""
    _ = candidate
    label = getattr(hyp, "label", "")
    visible = getattr(hyp, "visible_features", "")
    strategy = getattr(hyp, "grasp_strategy", None)
    slip = str(getattr(strategy, "slip_risk", "medium") or "medium").lower()
    tokens = _text_tokens(label, visible, slip)
    size = _size_tuple(object_size_m)

    round_terms = {"round", "sphere", "spherical", "lemon", "orange", "lime", "apple"}
    slippery_terms = {"smooth", "slippery", "waxy", "glossy", "high"}
    is_round_slippery = (round_terms & tokens) and (
        slip == "high" or slippery_terms & tokens
    )
    size_allows_small_round = size is None or max(size) <= 0.12
    if is_round_slippery and size_allows_small_round:
        return GraspProfileResult(
            profile=GraspProfile.SMALL_ROUND_SLIPPERY,
            confidence=0.85,
            reasons=["round", "slippery", "small"],
        )

    if {"handle", "handled", "mug", "cup", "pan", "jug"} & tokens:
        return GraspProfileResult(
            profile=GraspProfile.HANDLED,
            confidence=0.8,
            reasons=["handle_semantics"],
        )

    if {"bread", "cake", "soft", "fragile", "delicate", "squishy"} & tokens:
        return GraspProfileResult(
            profile=GraspProfile.FRAGILE_SOFT,
            confidence=0.75,
            reasons=["fragile_or_soft_semantics"],
        )

    if size is not None:
        max_xy = max(size[0], size[1])
        min_dim = min(size)
        if min_dim <= 0.015 and max_xy >= 0.05:
            return GraspProfileResult(
                profile=GraspProfile.THIN_FLAT,
                confidence=0.75,
                reasons=["thin_aabb"],
            )
        if max_xy > float(gripper_max_width_m):
            return GraspProfileResult(
                profile=GraspProfile.WIDE_UNGRASPABLE,
                confidence=0.9,
                reasons=["width_exceeds_gripper"],
            )

    return GraspProfileResult(
        profile=GraspProfile.DEFAULT_RIGID,
        confidence=0.5,
        reasons=["default"],
    )


def _text_tokens(*parts: object) -> set[str]:
    text = " ".join(str(part or "").lower() for part in parts)
    for ch in ",.;:/()[]{}_-":
        text = text.replace(ch, " ")
    return {token for token in text.split() if token}


def _size_tuple(object_size_m: Iterable[float] | None) -> tuple[float, float, float] | None:
    if object_size_m is None:
        return None
    arr = np.asarray(list(object_size_m), dtype=np.float32).reshape(-1)
    if arr.shape[0] < 3 or not np.all(np.isfinite(arr[:3])):
        return None
    return float(arr[0]), float(arr[1]), float(arr[2])
