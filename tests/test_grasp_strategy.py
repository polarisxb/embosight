"""Tests for LLM-driven grasp strategy selection."""
from __future__ import annotations

import numpy as np
import pytest

from src.world_belief import GraspStrategy, Hypothesis
from src.grasp_planner import GraspPlanner
from tests._mocks import MockLLM, MockVLM


def _hyp(label: str, safety_dist: dict = None, visible_features: str = "",
         obj_id: str = "obj_0") -> Hypothesis:
    return Hypothesis(
        object_id=obj_id,
        label=label,
        label_alternatives=[(label, 0.8)],
        label_entropy=0.3,
        position_3d=np.array([0.3, -3.0, 0.95]),
        position_std_m=0.02,
        safety_dist=safety_dist or {"safe": 0.9, "fragile": 0.1},
        safety_entropy=0.3,
        visible_features=visible_features,
    )


class _FakeEnv:
    def is_reachable(self, point, approach):
        return True
    def get_base_pose(self):
        return np.array([0.0, 0.0, 0.0]), np.eye(3)
    def get_eef_pos(self):
        return np.array([0.3, 0.0, 1.0])
    def observe(self, vp):
        return None
    def eye_in_hand_viewpoint(self):
        return None


class TestSelectStrategy:
    def test_returns_strategy_from_llm(self):
        llm = MockLLM(responses=[
            '{"strategy": "gentle_side", "approach_axis": "x", '
            '"reasoning": "cake is fragile", "speech": "这是蛋糕，我从侧面轻拿"}'
        ])
        planner = GraspPlanner(vlm=MockVLM([]), env=_FakeEnv(), llm=llm)
        h = _hyp("cake", {"fragile": 0.8, "safe": 0.2}, "brown soft object")
        strategy = planner.select_strategy(h)

        assert strategy.strategy == "gentle_side"
        assert "fragile" in strategy.reasoning or "cake" in strategy.reasoning
        assert strategy.speech != ""

    def test_fallback_to_top_down_on_invalid_strategy(self):
        llm = MockLLM(responses=[
            '{"strategy": "invalid_strategy", "reasoning": "test"}'
        ])
        planner = GraspPlanner(vlm=MockVLM([]), env=_FakeEnv(), llm=llm)
        h = _hyp("apple")
        strategy = planner.select_strategy(h)

        assert strategy.strategy == "top_down"

    def test_fallback_to_top_down_without_llm(self):
        planner = GraspPlanner(vlm=MockVLM([]), env=_FakeEnv(), llm=None)
        h = _hyp("apple")
        strategy = planner.select_strategy(h)

        assert strategy.strategy == "top_down"
        assert "no LLM" in strategy.reasoning

    def test_refuse_strategy(self):
        llm = MockLLM(responses=[
            '{"strategy": "refuse", "reasoning": "too hot", "speech": "这个锅太烫了，不能拿"}'
        ])
        planner = GraspPlanner(vlm=MockVLM([]), env=_FakeEnv(), llm=llm)
        h = _hyp("pot", {"hot": 0.9, "safe": 0.1}, "steaming metal pot")
        strategy = planner.select_strategy(h)

        assert strategy.strategy == "refuse"


class TestStrategyDrivenPlan:
    def test_strategy_candidate_has_highest_score(self):
        planner = GraspPlanner(vlm=MockVLM([]), env=_FakeEnv(), llm=None)
        h = _hyp("cake")
        h.grasp_strategy = GraspStrategy(
            strategy="gentle_side", reasoning="fragile", speech="侧面拿",
        )
        cands = planner.plan(h)

        assert len(cands) >= 2
        # strategy candidate should be first (highest score)
        assert cands[0].source == "strategy_gentle_side"
        assert cands[0].approach_dir[2] == 0.0  # horizontal (side approach)
        assert float(np.linalg.norm(cands[0].approach_dir)) > 0.99  # unit vector
        assert cands[0].finger_width_m == 0.06  # wider for gentle

    def test_no_strategy_uses_geometric_centroid(self):
        planner = GraspPlanner(vlm=MockVLM([]), env=_FakeEnv(), llm=None)
        h = _hyp("apple")
        # No strategy set
        cands = planner.plan(h)

        assert len(cands) >= 1
        assert cands[0].source == "geometric_centroid"

    def test_refuse_strategy_skipped_in_plan(self):
        planner = GraspPlanner(vlm=MockVLM([]), env=_FakeEnv(), llm=None)
        h = _hyp("pot")
        h.grasp_strategy = GraspStrategy(strategy="refuse", speech="太烫了")
        cands = planner.plan(h)

        # refuse 不生成策略候选, 只有 geometric_centroid 兜底
        sources = [c.source for c in cands]
        assert "strategy_refuse" not in sources
        assert "geometric_centroid" in sources


class TestGraspStrategyDataclass:
    def test_default_values(self):
        gs = GraspStrategy(strategy="top_down")
        assert gs.approach_axis == "z"
        assert gs.reasoning == ""
        assert gs.speech == ""

    def test_hypothesis_has_visible_features(self):
        h = _hyp("cake", visible_features="brown rectangular sealed package")
        assert h.visible_features == "brown rectangular sealed package"

    def test_hypothesis_grasp_strategy_field(self):
        h = _hyp("cake")
        assert h.grasp_strategy is None
        h.grasp_strategy = GraspStrategy(strategy="gentle_side", speech="轻拿")
        assert h.grasp_strategy.strategy == "gentle_side"
