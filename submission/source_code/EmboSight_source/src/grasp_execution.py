"""Pre-grasp diagnostic types and approach-frame error decomposition.

This module is intentionally free of RoboCasa / robosuite imports so that
the geometry helpers and dataclasses can be unit-tested without a simulator.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ─── Reason constants ────────────────────────────────────────────────────

PRE_GRASP_STRICT_OK = "strict_ok"
PRE_GRASP_SAFE_HANDOFF = "safe_handoff"
PRE_GRASP_LATERAL_MISALIGNED = "lateral_misaligned"
PRE_GRASP_AXIS_GAP_TOO_SMALL = "axis_gap_too_small"
PRE_GRASP_AXIS_GAP_TOO_LARGE = "axis_gap_too_large"
PRE_GRASP_BELOW_GRASP_POINT = "below_grasp_point"
PRE_GRASP_UNREACHABLE = "pre_grasp_unreachable"

# ─── Dataclasses ─────────────────────────────────────────────────────────


@dataclass
class PreGraspDecomposition:
    """Raw approach-frame error breakdown."""

    total_error_m: float
    lateral_error_m: float
    axis_error_m: float
    approach_gap_m: float


@dataclass
class PreGraspResult:
    """Structured outcome of a diagnostic pre-grasp attempt."""

    ok: bool
    handoff_ok: bool
    needs_recovery: bool
    reason: str
    final_eef: np.ndarray
    pre_pos: np.ndarray
    grasp_point: np.ndarray
    approach_dir: np.ndarray
    total_error_m: float
    lateral_error_m: float
    axis_error_m: float
    approach_gap_m: float
    lateral_limit_m: float
    min_approach_gap_m: float
    max_approach_gap_m: float
    move_ok: bool


# ─── Pure helpers ────────────────────────────────────────────────────────


def normalize_approach_dir(approach_dir) -> np.ndarray:
    """Normalize approach direction; fall back to top-down [0,0,-1]."""
    ad = np.asarray(approach_dir, dtype=np.float32).ravel()[:3]
    norm = float(np.linalg.norm(ad))
    if norm < 1e-6:
        return np.array([0.0, 0.0, -1.0], dtype=np.float32)
    return (ad / norm).astype(np.float32)


def decompose_pre_grasp_error(
    final_eef, pre_pos, grasp_point, approach_dir,
) -> PreGraspDecomposition:
    """Decompose EEF residual into approach-frame components.

    Args:
        final_eef:    actual EEF position after motion (world frame).
        pre_pos:      intended pre-grasp position (world frame).
        grasp_point:  target grasp point on the object (world frame).
        approach_dir: unit approach direction (pre-grasp → object).

    Returns:
        PreGraspDecomposition with lateral, axis, and gap metrics.
    """
    p = np.asarray(final_eef, dtype=np.float32)[:3]
    pre = np.asarray(pre_pos, dtype=np.float32)[:3]
    g = np.asarray(grasp_point, dtype=np.float32)[:3]
    ad = normalize_approach_dir(approach_dir)

    pre_error = pre - p
    axis_signed = float(np.dot(pre_error, ad))
    lateral_vec = pre_error - axis_signed * ad
    approach_gap = float(np.dot(g - p, ad))

    return PreGraspDecomposition(
        total_error_m=float(np.linalg.norm(pre_error)),
        lateral_error_m=float(np.linalg.norm(lateral_vec)),
        axis_error_m=abs(axis_signed),
        approach_gap_m=approach_gap,
    )


def lateral_limit_for_finger_width(finger_width_m: float | None) -> float:
    """Candidate-scale lateral handoff limit, clamped to [0.015, 0.045]."""
    width = 0.04 if finger_width_m is None else float(finger_width_m)
    return float(np.clip(0.5 * width, 0.015, 0.045))


def evaluate_pre_grasp_handoff(
    *,
    move_ok: bool,
    final_eef,
    pre_pos,
    grasp_point,
    approach_dir,
    finger_width_m: float | None,
    height_m: float,
) -> PreGraspResult:
    """Evaluate whether a pre-grasp attempt can safely hand off to approach.

    Uses approach-frame decomposition and candidate-scale guards instead of
    a single global Euclidean threshold.
    """
    ad = normalize_approach_dir(approach_dir)
    d = decompose_pre_grasp_error(final_eef, pre_pos, grasp_point, ad)
    lateral_limit = lateral_limit_for_finger_width(finger_width_m)
    min_gap = 0.010
    max_gap = max(float(height_m) + 0.030, 0.080)

    p = np.asarray(final_eef, dtype=np.float32)[:3]
    g = np.asarray(grasp_point, dtype=np.float32)[:3]

    if move_ok:
        reason = PRE_GRASP_STRICT_OK
        handoff_ok = True
        needs_recovery = False
    elif ad[2] < -0.9 and p[2] < g[2]:
        reason = PRE_GRASP_BELOW_GRASP_POINT
        handoff_ok = False
        needs_recovery = False
    elif d.approach_gap_m < min_gap:
        reason = PRE_GRASP_AXIS_GAP_TOO_SMALL
        handoff_ok = False
        needs_recovery = False
    elif d.approach_gap_m > max_gap:
        reason = PRE_GRASP_AXIS_GAP_TOO_LARGE
        handoff_ok = False
        needs_recovery = False
    elif d.lateral_error_m <= lateral_limit:
        reason = PRE_GRASP_SAFE_HANDOFF
        handoff_ok = True
        needs_recovery = False
    else:
        reason = PRE_GRASP_LATERAL_MISALIGNED
        handoff_ok = False
        needs_recovery = True

    return PreGraspResult(
        ok=bool(move_ok),
        handoff_ok=handoff_ok,
        needs_recovery=needs_recovery,
        reason=reason,
        final_eef=p,
        pre_pos=np.asarray(pre_pos, dtype=np.float32)[:3],
        grasp_point=g,
        approach_dir=ad,
        total_error_m=d.total_error_m,
        lateral_error_m=d.lateral_error_m,
        axis_error_m=d.axis_error_m,
        approach_gap_m=d.approach_gap_m,
        lateral_limit_m=lateral_limit,
        min_approach_gap_m=min_gap,
        max_approach_gap_m=max_gap,
        move_ok=bool(move_ok),
    )
