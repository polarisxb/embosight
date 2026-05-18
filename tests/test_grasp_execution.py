"""Tests for src/grasp_execution: approach-frame decomposition and handoff gates."""
import numpy as np
import pytest

from src.grasp_execution import (
    PRE_GRASP_AXIS_GAP_TOO_LARGE,
    PRE_GRASP_AXIS_GAP_TOO_SMALL,
    PRE_GRASP_BELOW_GRASP_POINT,
    PRE_GRASP_LATERAL_MISALIGNED,
    PRE_GRASP_SAFE_HANDOFF,
    PRE_GRASP_STRICT_OK,
    PRE_GRASP_UNREACHABLE,
    PreGraspDecomposition,
    PreGraspResult,
    decompose_pre_grasp_error,
    evaluate_pre_grasp_handoff,
    lateral_limit_for_finger_width,
    normalize_approach_dir,
)


# ─── decompose_pre_grasp_error ───────────────────────────────────────────


class TestDecomposition:
    def test_top_down_xy_error_maps_to_lateral(self):
        """XY offset from pre-grasp target becomes lateral error for top-down."""
        final_eef = np.array([0.13, -2.80, 0.98], dtype=np.float32)
        pre_pos = np.array([0.125, -2.86, 0.982], dtype=np.float32)
        grasp_point = np.array([0.125, -2.86, 0.932], dtype=np.float32)
        approach_dir = np.array([0.0, 0.0, -1.0], dtype=np.float32)

        d = decompose_pre_grasp_error(final_eef, pre_pos, grasp_point, approach_dir)

        # lateral ≈ sqrt(0.005^2 + 0.06^2) ≈ 0.060
        assert 0.055 < d.lateral_error_m < 0.065
        # axis error ≈ |0.982 - 0.98| = 0.002
        assert d.axis_error_m < 0.01
        # approach gap: EEF above grasp point by ~0.048
        assert 0.04 < d.approach_gap_m < 0.06

    def test_top_down_z_error_maps_to_axis(self):
        """Z offset from pre-grasp target becomes axis error for top-down."""
        final_eef = np.array([0.5, 0.0, 1.05], dtype=np.float32)
        pre_pos = np.array([0.5, 0.0, 1.00], dtype=np.float32)
        grasp_point = np.array([0.5, 0.0, 0.95], dtype=np.float32)
        approach_dir = np.array([0.0, 0.0, -1.0], dtype=np.float32)

        d = decompose_pre_grasp_error(final_eef, pre_pos, grasp_point, approach_dir)

        assert d.lateral_error_m < 0.001
        assert 0.04 < d.axis_error_m < 0.06

    def test_side_approach_horizontal_error_maps_to_axis(self):
        """Side approach: forward-axis residual becomes axis error."""
        final_eef = np.array([0.60, 0.0, 0.9], dtype=np.float32)
        pre_pos = np.array([0.50, 0.0, 0.9], dtype=np.float32)
        grasp_point = np.array([0.40, 0.0, 0.9], dtype=np.float32)
        approach_dir = np.array([-1.0, 0.0, 0.0], dtype=np.float32)

        d = decompose_pre_grasp_error(final_eef, pre_pos, grasp_point, approach_dir)

        assert d.lateral_error_m < 0.001
        # axis error: pre-pos - final in approach direction = 0.10
        assert 0.09 < d.axis_error_m < 0.11
        # approach gap: EEF is 0.2m behind grasp point along -x
        assert 0.19 < d.approach_gap_m < 0.21

    def test_zero_approach_dir_normalizes_to_top_down(self):
        """Zero approach direction falls back to [0,0,-1]."""
        ad = normalize_approach_dir(np.zeros(3))
        assert ad[2] == pytest.approx(-1.0)
        assert float(np.linalg.norm(ad)) == pytest.approx(1.0)


# ─── lateral_limit_for_finger_width ─────────────────────────────────────


class TestLateralLimit:
    def test_default_finger_width(self):
        assert lateral_limit_for_finger_width(None) == pytest.approx(0.02)

    def test_wide_finger_is_clamped(self):
        # 0.5 * 0.12 = 0.06 → clamped to 0.045
        assert lateral_limit_for_finger_width(0.12) == pytest.approx(0.045)

    def test_narrow_finger_is_clamped(self):
        # 0.5 * 0.02 = 0.01 → clamped to 0.015
        assert lateral_limit_for_finger_width(0.02) == pytest.approx(0.015)


# ─── evaluate_pre_grasp_handoff ──────────────────────────────────────────


class TestEvaluateHandoff:
    def _common_kwargs(self, **overrides):
        defaults = dict(
            move_ok=False,
            final_eef=np.array([0.505, 0.01, 1.00], dtype=np.float32),
            pre_pos=np.array([0.50, 0.00, 1.00], dtype=np.float32),
            grasp_point=np.array([0.50, 0.00, 0.95], dtype=np.float32),
            approach_dir=np.array([0.0, 0.0, -1.0], dtype=np.float32),
            finger_width_m=0.04,
            height_m=0.05,
        )
        defaults.update(overrides)
        return defaults

    def test_move_ok_returns_strict_ok(self):
        r = evaluate_pre_grasp_handoff(**self._common_kwargs(move_ok=True))
        assert r.ok is True
        assert r.handoff_ok is True
        assert r.reason == PRE_GRASP_STRICT_OK

    def test_small_lateral_allows_handoff(self):
        # lateral ≈ sqrt(0.005^2 + 0.01^2) ≈ 0.011 < 0.02 limit
        r = evaluate_pre_grasp_handoff(**self._common_kwargs())
        assert r.handoff_ok is True
        assert r.reason == PRE_GRASP_SAFE_HANDOFF

    def test_large_lateral_requires_recovery(self):
        # XY offset 6cm → lateral ≈ 0.06 > 0.02
        r = evaluate_pre_grasp_handoff(**self._common_kwargs(
            final_eef=np.array([0.56, 0.0, 1.00], dtype=np.float32),
        ))
        assert r.handoff_ok is False
        assert r.needs_recovery is True
        assert r.reason == PRE_GRASP_LATERAL_MISALIGNED

    def test_approach_gap_too_small_no_handoff(self):
        # EEF almost at grasp point z
        r = evaluate_pre_grasp_handoff(**self._common_kwargs(
            final_eef=np.array([0.50, 0.00, 0.955], dtype=np.float32),
        ))
        assert r.handoff_ok is False
        assert r.needs_recovery is False
        assert r.reason == PRE_GRASP_AXIS_GAP_TOO_SMALL

    def test_below_grasp_point_no_handoff(self):
        # EEF below grasp z for top-down
        r = evaluate_pre_grasp_handoff(**self._common_kwargs(
            final_eef=np.array([0.50, 0.00, 0.94], dtype=np.float32),
        ))
        assert r.handoff_ok is False
        assert r.needs_recovery is False
        assert r.reason == PRE_GRASP_BELOW_GRASP_POINT

    def test_approach_gap_too_large_no_handoff(self):
        # EEF very far above grasp
        r = evaluate_pre_grasp_handoff(**self._common_kwargs(
            final_eef=np.array([0.50, 0.00, 1.20], dtype=np.float32),
        ))
        assert r.handoff_ok is False
        assert r.needs_recovery is False
        assert r.reason == PRE_GRASP_AXIS_GAP_TOO_LARGE
