import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.grasp_profile import GraspProfile, classify_grasp_profile
from src.world_belief import GraspCandidate, GraspStrategy, Hypothesis


def _hyp(label: str, visible: str = "", slip: str = "medium") -> Hypothesis:
    return Hypothesis(
        object_id=f"{label}_main",
        label=label,
        label_alternatives=[(label, 0.95)],
        label_entropy=0.1,
        position_3d=np.array([0.5, 0.0, 0.9], dtype=np.float32),
        position_std_m=0.02,
        visible_features=visible,
        grasp_strategy=GraspStrategy(strategy="top_down", slip_risk=slip),
    )


def _candidate(width: float = 0.04) -> GraspCandidate:
    return GraspCandidate(
        point_3d=np.array([0.5, 0.0, 0.9], dtype=np.float32),
        approach_dir=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        finger_width_m=width,
        score=1.0,
        source="strategy_top_down",
    )


def test_lemon_is_small_round_slippery():
    result = classify_grasp_profile(
        _hyp("lemon", "round yellow smooth waxy fruit", "high"),
        _candidate(),
        object_size_m=(0.058, 0.055, 0.058),
    )

    assert result.profile == GraspProfile.SMALL_ROUND_SLIPPERY
    assert result.confidence >= 0.7
    assert "round" in result.reasons
    assert result.execution_overrides == {}


def test_wide_object_is_ungraspable_when_width_exceeds_gripper():
    result = classify_grasp_profile(
        _hyp("tupperware", "wide rectangular container", "medium"),
        _candidate(width=0.04),
        object_size_m=(0.14, 0.10, 0.05),
        gripper_max_width_m=0.08,
    )

    assert result.profile == GraspProfile.WIDE_UNGRASPABLE
    assert "width_exceeds_gripper" in result.reasons
    assert result.execution_overrides == {}


def test_mug_with_handle_is_handled():
    result = classify_grasp_profile(
        _hyp("mug", "ceramic cup with handle", "medium"),
        _candidate(width=0.04),
        object_size_m=(0.07, 0.09, 0.10),
    )

    assert result.profile == GraspProfile.HANDLED
    assert result.execution_overrides == {}


def test_bread_like_object_is_fragile_soft():
    result = classify_grasp_profile(
        _hyp("bread", "soft squishy loaf", "low"),
        _candidate(width=0.06),
        object_size_m=(0.07, 0.05, 0.04),
    )

    assert result.profile == GraspProfile.FRAGILE_SOFT
    assert result.execution_overrides == {}


def test_unknown_object_defaults_without_side_effects():
    result = classify_grasp_profile(
        _hyp("object", "", "medium"),
        _candidate(width=0.04),
        object_size_m=None,
    )

    assert result.profile == GraspProfile.DEFAULT_RIGID
    assert result.execution_overrides == {}
