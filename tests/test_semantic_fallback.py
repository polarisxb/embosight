"""Tests for CLIP semantic injection and LLM semantic fallback."""
from __future__ import annotations

import numpy as np
import pytest

from src.world_belief import (
    DecomposedTask,
    Hypothesis,
    WorldBelief,
)


def _hyp(label: str, alts: list[tuple[str, float]], obj_id: str = "obj_0") -> Hypothesis:
    return Hypothesis(
        object_id=obj_id,
        label=label,
        label_alternatives=alts,
        label_entropy=0.9,
        position_3d=np.array([0.3, -3.0, 0.95]),
        position_std_m=0.02,
        bbox_per_view={"v0": (50, 50, 150, 150)},
        observed_in_views=["v0"],
    )


# ──────────────────────────────────────
# CLIPScorer unit tests
# ──────────────────────────────────────

class TestCLIPScorer:
    def test_score_crops_empty_bboxes(self):
        from src.clip_scorer import CLIPScorer
        scorer = CLIPScorer()
        assert scorer.score_crops("nonexistent.png", [], "cake") == []

    def test_score_crops_returns_correct_length(self):
        """score_crops returns zeros when model can't load (no real model in CI)."""
        from src.clip_scorer import CLIPScorer
        scorer = CLIPScorer(model_name="nonexistent-model")
        scores = scorer.score_crops("dummy.png", [(0, 0, 10, 10), (0, 0, 20, 20)], "cake")
        assert len(scores) == 2


# ──────────────────────────────────────
# CLIP injection in perception
# ──────────────────────────────────────

class MockCLIPScorer:
    """Mock CLIPScorer that returns pre-set scores."""
    INJECT_THRESHOLD = 0.23

    def __init__(self, scores: list[float]):
        self._scores = scores

    def score_crops(self, image_path, bboxes, text_query):
        return self._scores[:len(bboxes)]


class TestCLIPInjection:
    def test_inject_adds_target_to_alternatives(self):
        from src.perception import QueryAwareGrounder, _label_key, _shannon

        scorer = MockCLIPScorer(scores=[0.30])  # above threshold
        grounder = QueryAwareGrounder.__new__(QueryAwareGrounder)
        grounder._clip_scorer = scorer

        h = _hyp("chocolate", [("chocolate", 0.6), ("candy", 0.4)])
        grounder._inject_clip_scores([h], "dummy.png", "cake")

        # "cake" should be injected into alternatives
        labels = [lbl for lbl, _ in h.label_alternatives]
        assert "cake" in labels

    def test_inject_skips_if_target_already_present(self):
        from src.perception import QueryAwareGrounder

        scorer = MockCLIPScorer(scores=[0.30])
        grounder = QueryAwareGrounder.__new__(QueryAwareGrounder)
        grounder._clip_scorer = scorer

        h = _hyp("cake", [("cake", 0.6), ("pastry", 0.4)])
        old_alts = list(h.label_alternatives)
        grounder._inject_clip_scores([h], "dummy.png", "cake")

        # should not modify since "cake" already present
        assert h.label_alternatives == old_alts

    def test_inject_skips_below_threshold(self):
        from src.perception import QueryAwareGrounder

        scorer = MockCLIPScorer(scores=[0.10])  # below threshold
        grounder = QueryAwareGrounder.__new__(QueryAwareGrounder)
        grounder._clip_scorer = scorer

        h = _hyp("chocolate", [("chocolate", 0.6), ("candy", 0.4)])
        old_labels = {lbl for lbl, _ in h.label_alternatives}
        grounder._inject_clip_scores([h], "dummy.png", "cake")

        # should not inject
        new_labels = {lbl for lbl, _ in h.label_alternatives}
        assert new_labels == old_labels

    def test_injected_target_found_by_belief_target(self):
        """After CLIP injection, WorldBelief.target() should find the hypothesis."""
        from src.perception import QueryAwareGrounder

        scorer = MockCLIPScorer(scores=[0.30])
        grounder = QueryAwareGrounder.__new__(QueryAwareGrounder)
        grounder._clip_scorer = scorer

        h = _hyp("chocolate", [("chocolate", 0.6), ("candy", 0.4)])
        grounder._inject_clip_scores([h], "dummy.png", "cake")

        b = WorldBelief(user_query="pick up the cake")
        b.decomposed = DecomposedTask(primary_target="cake")
        b.hypotheses = [h]

        assert b.target() is not None
        assert b.target() is h


# ──────────────────────────────────────
# LLM semantic fallback
# ──────────────────────────────────────

class TestLLMSemanticFallback:
    def _make_agent(self, llm_response: str):
        from tests._mocks import MockLLM
        from src.active_planner import ViewpointLibrary
        from src.agent import EmboSightAgent
        vp_lib = ViewpointLibrary(config_path="configs/viewpoints.yaml")
        llm = MockLLM(responses=[llm_response])
        return EmboSightAgent.with_test_doubles(
            vp_lib=vp_lib, llm=llm,
        )

    def test_fallback_injects_target_on_match(self):
        agent = self._make_agent("chocolate")
        b = WorldBelief(user_query="pick up the cake")
        b.decomposed = DecomposedTask(primary_target="cake")
        b.hypotheses = [_hyp("chocolate", [("chocolate", 0.6), ("candy", 0.4)])]

        result = agent._llm_semantic_fallback(b)
        assert result is True
        labels = [lbl for lbl, _ in b.hypotheses[0].label_alternatives]
        assert "cake" in labels

    def test_fallback_returns_false_on_none(self):
        agent = self._make_agent("none")
        b = WorldBelief(user_query="pick up the cake")
        b.decomposed = DecomposedTask(primary_target="cake")
        b.hypotheses = [_hyp("book", [("book", 0.8), ("notebook", 0.2)])]

        result = agent._llm_semantic_fallback(b)
        assert result is False

    def test_fallback_returns_false_on_empty_hypotheses(self):
        agent = self._make_agent("whatever")
        b = WorldBelief(user_query="pick up the cake")
        b.decomposed = DecomposedTask(primary_target="cake")
        b.hypotheses = []

        result = agent._llm_semantic_fallback(b)
        assert result is False

    def test_fallback_then_target_finds_it(self):
        """After LLM fallback, target() should find the hypothesis."""
        agent = self._make_agent("chocolate")
        b = WorldBelief(user_query="pick up the cake")
        b.decomposed = DecomposedTask(primary_target="cake")
        b.hypotheses = [_hyp("chocolate", [("chocolate", 0.6), ("candy", 0.4)])]

        # Before fallback: target() should not find it
        assert b.target() is None

        # After fallback
        agent._llm_semantic_fallback(b)
        assert b.target() is not None
        assert b.target() is b.hypotheses[0]
