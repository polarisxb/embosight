"""Tests for LLM-driven grasp strategy selection."""
from __future__ import annotations

from typing import get_args, get_type_hints

import numpy as np

from src.world_belief import GraspCandidate, GraspStrategy, Hypothesis
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


class _GeometryAwareEnv(_FakeEnv):
    def __init__(self):
        self.grasp_pose_calls: list[tuple[str, tuple[float, float, float]]] = []

    def _get_obj_type_map(self):
        return {
            "obj_main": "apple",
            "distr_counter_main": "lemon",
        }

    def _compute_grasp_pose(self, body_name: str, fallback_pos: np.ndarray) -> np.ndarray:
        self.grasp_pose_calls.append((
            body_name,
            tuple(np.asarray(fallback_pos, dtype=np.float32).tolist()),
        ))
        return np.array([0.31, -2.99, 0.982], dtype=np.float32)


def test_grasp_candidate_source_literal_allows_tilted_strategy():
    source_type = get_type_hints(GraspCandidate)["source"]
    assert "strategy_tilted_grasp" in get_args(source_type)


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


class TestAdaptiveForceParams:
    """DeliGrasp-inspired adaptive force params (Step 0+1 of slip prevention)."""

    def test_derive_force_params_lemon_high_slip(self):
        """High slip_risk + ~lemon mass → ≥18 squeeze, 0.025m descent margin."""
        squeeze, margin = GraspPlanner._derive_force_params(
            "top_down", mass_g=100.0, slip_risk="high",
        )
        # mass 100g → 2 steps; slip high → 16 steps. Total 18.
        assert squeeze == 18, f"expected 18 squeeze for lemon, got {squeeze}"
        # Round/slippery objects should be closed shallowly instead of
        # pushed below center on the first attempt.
        assert abs(margin - 0.010) < 1e-6, (
            f"expected 0.010m margin for high slip, got {margin}"
        )

    def test_derive_force_params_bread_low_slip(self):
        """Low slip_risk + light mass → minimal squeeze, default margin."""
        squeeze, margin = GraspPlanner._derive_force_params(
            "top_down", mass_g=50.0, slip_risk="low",
        )
        # mass 50g → 1 step; slip low → 0 steps. Total 1.
        assert squeeze == 1, f"expected ~1 squeeze for bread, got {squeeze}"
        # base 0.015 + 0 = 0.015
        assert abs(margin - 0.015) < 1e-6

    def test_derive_force_params_squeeze_clamped_to_30(self):
        """Heavy + high slip should clamp squeeze at 30 (not unbounded)."""
        squeeze, _ = GraspPlanner._derive_force_params(
            "top_down", mass_g=2000.0, slip_risk="high",
        )
        assert squeeze == 30, f"expected squeeze clamped to 30, got {squeeze}"

    def test_derive_force_params_gentle_side_uses_strategy_default(self):
        """gentle_side keeps shallow 0.010m margin even with high slip."""
        squeeze, margin = GraspPlanner._derive_force_params(
            "gentle_side", mass_g=100.0, slip_risk="high",
        )
        # gentle_side strategy default is 0.010m, not adjusted by slip_risk
        assert abs(margin - 0.010) < 1e-6, (
            f"gentle_side should keep 0.010m margin, got {margin}"
        )
        # squeeze still scales with mass + slip
        assert squeeze == 18

    def test_derive_force_params_invalid_slip_risk_defaults_medium(self):
        squeeze, margin = GraspPlanner._derive_force_params(
            "top_down", mass_g=100.0, slip_risk="UNKNOWN",
        )
        # falls back to medium: mass 2 + medium 8 = 10 squeeze; margin 0.020
        assert squeeze == 10
        assert abs(margin - 0.020) < 1e-6

    def test_select_strategy_parses_mass_and_slip_risk(self):
        llm = MockLLM(responses=[
            '{"strategy": "top_down", "approach_axis": "z", '
            '"mass_g": 100, "slip_risk": "high", '
            '"reasoning": "lemon is round and slippery", '
            '"speech": "这是柠檬，我从上方拿"}'
        ])
        planner = GraspPlanner(vlm=MockVLM([]), env=_FakeEnv(), llm=llm)
        h = _hyp("lemon", visible_features="round yellow fruit")
        strategy = planner.select_strategy(h)

        assert strategy.strategy == "top_down"
        assert strategy.mass_g == 100.0
        assert strategy.slip_risk == "high"
        assert strategy.squeeze_extra_steps == 18  # mass 2 + risk 16
        assert abs(strategy.depth_margin_m - 0.010) < 1e-6

    def test_select_strategy_missing_mass_falls_back_to_default(self):
        llm = MockLLM(responses=[
            '{"strategy": "top_down", "approach_axis": "z", '
            '"reasoning": "object", "speech": "拿取"}'
        ])
        planner = GraspPlanner(vlm=MockVLM([]), env=_FakeEnv(), llm=llm)
        h = _hyp("apple")
        strategy = planner.select_strategy(h)

        # Default mass_g=100, slip_risk="medium" → squeeze=10, margin=0.020
        assert strategy.mass_g == 100.0
        assert strategy.slip_risk == "medium"
        assert strategy.squeeze_extra_steps == 10
        assert abs(strategy.depth_margin_m - 0.020) < 1e-6

    def test_select_strategy_clamps_extreme_mass(self):
        llm = MockLLM(responses=[
            '{"strategy": "top_down", "mass_g": 10000, "slip_risk": "low", '
            '"reasoning": "x", "speech": "y"}'
        ])
        planner = GraspPlanner(vlm=MockVLM([]), env=_FakeEnv(), llm=llm)
        h = _hyp("anvil")
        strategy = planner.select_strategy(h)

        # Clamped to 2000g
        assert strategy.mass_g == 2000.0
        # 2000/50 = 40 → clamped to 30
        assert strategy.squeeze_extra_steps == 30


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
        assert cands[0].approach_dir[2] < 0  # tilted downward (~15°)
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


class TestBannedStrategies:
    def test_parse_banned_from_memory(self):
        advice = "wooden_spoon: avoid top_down (slipped x4), avoid handle_grasp (slipped x2)"
        banned = GraspPlanner._parse_banned_strategies(advice)
        assert "top_down" in banned      # x4 >= 3
        assert "handle_grasp" not in banned  # x2 < 3

    def test_banned_strategy_not_selected(self):
        llm = MockLLM(responses=[
            '{"strategy": "top_down", "reasoning": "best", "speech": "上方拿"}'
        ])
        planner = GraspPlanner(vlm=MockVLM([]), env=_FakeEnv(), llm=llm)
        h = _hyp("spoon")
        advice = "spoon: avoid top_down (slipped x5)"
        strategy = planner.select_strategy(h, memory_advice=advice)
        # LLM chose top_down but it's banned → overridden
        assert strategy.strategy != "top_down"

    def test_no_llm_respects_ban(self):
        planner = GraspPlanner(vlm=MockVLM([]), env=_FakeEnv(), llm=None)
        h = _hyp("spoon")
        advice = "spoon: avoid top_down (slipped x3)"
        strategy = planner.select_strategy(h, memory_advice=advice)
        assert strategy.strategy != "top_down"

    def test_memory_driven_tilted_discovery(self):
        """记忆系统驱动的策略发现: top_down 被禁 → tilted_grasp 是下一个 fallback"""
        planner = GraspPlanner(vlm=MockVLM([]), env=_FakeEnv(), llm=None)
        h = _hyp("spoon")
        advice = "wooden_spoon: avoid top_down (slipped x5)"
        strategy = planner.select_strategy(h, memory_advice=advice)
        assert strategy.strategy == "tilted_grasp"

    def test_grasp_point_offset_handle(self):
        planner = GraspPlanner(vlm=MockVLM([]), env=_FakeEnv(), llm=None)
        h = _hyp("spoon")
        h.grasp_strategy = GraspStrategy(strategy="handle_grasp", speech="侧抓")
        cands = planner.plan(h)
        strat_cand = [c for c in cands if c.source == "strategy_handle_grasp"]
        assert len(strat_cand) == 1
        # upright → z offset +3cm from centroid
        assert strat_cand[0].point_3d[2] > h.position_3d[2]

    def test_grasp_point_offset_top_down_low_slip_keeps_bowl_offset(self):
        """Low slip_risk objects (bowls / boxes / wide ends) keep the -1.5cm offset."""
        planner = GraspPlanner(vlm=MockVLM([]), env=_FakeEnv(), llm=None)
        h = _hyp("bowl")
        h.grasp_strategy = GraspStrategy(
            strategy="top_down", slip_risk="low", speech="上方拿",
        )
        cands = planner.plan(h)
        strat_cand = [c for c in cands if c.source == "strategy_top_down"]
        assert len(strat_cand) == 1
        # low slip → keep historical "bowl end" -1.5cm offset
        assert abs(strat_cand[0].point_3d[2] - (h.position_3d[2] - 0.015)) < 1e-6

    def test_grasp_point_top_down_high_slip_skips_bowl_offset(self):
        """High slip_risk (round/smooth, e.g. lemon) must NOT lower the grasp point.

        Regression test: hardcoded -1.5cm + depth_margin together drove the descend
        target into the counter for lemon, causing slipped_lift (commit 4bfda16).
        """
        planner = GraspPlanner(vlm=MockVLM([]), env=_FakeEnv(), llm=None)
        h = _hyp("lemon")
        h.grasp_strategy = GraspStrategy(
            strategy="top_down", slip_risk="high", speech="上方拿",
        )
        cands = planner.plan(h)
        strat_cand = [c for c in cands if c.source == "strategy_top_down"]
        assert len(strat_cand) == 1
        # high slip → grasp at geometric center, no z lowering
        assert abs(strat_cand[0].point_3d[2] - h.position_3d[2]) < 1e-6

    def test_top_down_lemon_uses_geometry_pose_for_distractor_body(self):
        """Fixed lemon is a distractor body; use AABB grasp pose, not body origin."""
        env = _GeometryAwareEnv()
        planner = GraspPlanner(vlm=MockVLM([]), env=env, llm=None)
        h = _hyp("lemon")
        h.grasp_strategy = GraspStrategy(
            strategy="top_down", slip_risk="high", speech="top grasp",
        )

        cands = planner.plan(h, env=env)

        strat_cand = [c for c in cands if c.source == "strategy_top_down"]
        assert len(strat_cand) == 1
        assert env.grasp_pose_calls == [(
            "distr_counter_main",
            tuple(h.position_3d.astype(np.float32).tolist()),
        )]
        np.testing.assert_allclose(
            strat_cand[0].point_3d,
            np.array([0.31, -2.99, 0.982], dtype=np.float32),
        )

    def test_grasp_point_top_down_medium_slip_skips_bowl_offset(self):
        """Medium slip_risk also skips the bowl offset (safer default for fruits)."""
        planner = GraspPlanner(vlm=MockVLM([]), env=_FakeEnv(), llm=None)
        h = _hyp("apple")
        h.grasp_strategy = GraspStrategy(
            strategy="top_down", slip_risk="medium", speech="上方拿",
        )
        cands = planner.plan(h)
        strat_cand = [c for c in cands if c.source == "strategy_top_down"]
        assert len(strat_cand) == 1
        assert abs(strat_cand[0].point_3d[2] - h.position_3d[2]) < 1e-6

    def test_grasp_point_top_down_default_slip_skips_bowl_offset(self):
        """Default slip_risk='medium' (no LLM input) → safer no-offset behavior."""
        planner = GraspPlanner(vlm=MockVLM([]), env=_FakeEnv(), llm=None)
        h = _hyp("spoon")
        # Default GraspStrategy(slip_risk="medium") → no bowl offset
        h.grasp_strategy = GraspStrategy(strategy="top_down", speech="上方拿")
        cands = planner.plan(h)
        strat_cand = [c for c in cands if c.source == "strategy_top_down"]
        assert len(strat_cand) == 1
        assert abs(strat_cand[0].point_3d[2] - h.position_3d[2]) < 1e-6

    def test_tilted_grasp_approach_dir(self):
        planner = GraspPlanner(vlm=MockVLM([]), env=_FakeEnv(), llm=None)
        h = _hyp("spoon")
        h.grasp_strategy = GraspStrategy(strategy="tilted_grasp", speech="斜拿")
        cands = planner.plan(h)
        strat_cand = [c for c in cands if c.source == "strategy_tilted_grasp"]
        assert len(strat_cand) == 1
        ad = strat_cand[0].approach_dir
        # mostly vertical (z dominant), some horizontal component
        assert ad[2] < -0.7    # strong downward component
        assert max(abs(ad[0]), abs(ad[1])) > 0.3  # non-trivial horizontal

    def test_tilted_grasp_z_offset(self):
        planner = GraspPlanner(vlm=MockVLM([]), env=_FakeEnv(), llm=None)
        h = _hyp("spoon")
        h.grasp_strategy = GraspStrategy(strategy="tilted_grasp", speech="斜拿")
        cands = planner.plan(h)
        strat_cand = [c for c in cands if c.source == "strategy_tilted_grasp"]
        assert len(strat_cand) == 1
        # upright → z offset -2cm from centroid
        assert strat_cand[0].point_3d[2] < h.position_3d[2] - 0.01


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
