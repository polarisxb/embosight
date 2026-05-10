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
    """Mock CLIPScorer that returns pre-set scores.

    支持 per-query 不同的分数: scores 可以是 list[float] (单 query)
    或 dict[str, list[float]] (多 query, 按 text_query 查表)。
    """
    INJECT_THRESHOLD = 0.23

    def __init__(self, scores):
        self._scores = scores

    def score_crops(self, image_path, bboxes, text_query):
        if isinstance(self._scores, dict):
            scores = self._scores.get(text_query, [0.0] * len(bboxes))
        else:
            scores = self._scores
        return scores[:len(bboxes)]

    def score_crops_multi(self, image_path, bboxes, text_queries):
        return [self.score_crops(image_path, bboxes, q) for q in text_queries]


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


class TestCLIPMultiQueryInjection:
    """Synonym 查询路径: primary 低分但 synonym 高分时, 应注入 primary。"""

    def test_synonym_higher_than_primary(self):
        """primary='tangerine' 低于阈值, synonym='orange' 高于阈值 → 注入 tangerine。"""
        from src.perception import QueryAwareGrounder

        # tangerine 0.18 (低于 0.20 阈值), orange 0.28 (高于阈值)
        scorer = MockCLIPScorer(scores={
            "tangerine": [0.18],
            "orange": [0.28],
            "mandarin": [0.10],
        })
        grounder = QueryAwareGrounder.__new__(QueryAwareGrounder)
        grounder._clip_scorer = scorer

        h = _hyp("fruit", [("fruit", 0.7), ("citrus", 0.3)])
        grounder._inject_clip_scores(
            [h], "dummy.png", "tangerine",
            synonyms=["orange", "mandarin"],
        )
        labels = [lbl for lbl, _ in h.label_alternatives]
        # 注入的是 primary 名 (tangerine), 而不是 synonym (orange)
        assert "tangerine" in labels

    def test_synonym_already_in_label_skips_injection(self):
        """label 已含 synonym → 跳过注入 (避免重复)。"""
        from src.perception import QueryAwareGrounder

        scorer = MockCLIPScorer(scores={
            "tangerine": [0.30],
            "orange": [0.40],
        })
        grounder = QueryAwareGrounder.__new__(QueryAwareGrounder)
        grounder._clip_scorer = scorer

        # h.label='orange' 已经匹配 synonym 'orange'
        h = _hyp("orange", [("orange", 0.85), ("fruit", 0.15)])
        old_alts = list(h.label_alternatives)
        grounder._inject_clip_scores(
            [h], "dummy.png", "tangerine", synonyms=["orange"],
        )
        # 既然 synonym 已匹配, 不应再注入
        assert h.label_alternatives == old_alts

    def test_relaxed_threshold_for_synonym_query(self):
        """synonym 查询用放宽阈值 0.20; primary 查询仍用严格 0.23。"""
        from src.perception import QueryAwareGrounder

        # primary "tangerine" 0.21 < strict 0.23 → 不够
        # synonym "orange" 0.21 >= relaxed 0.20 → 够, 注入 "tangerine"
        scorer = MockCLIPScorer(scores={
            "tangerine": [0.10],
            "orange": [0.21],
        })
        grounder = QueryAwareGrounder.__new__(QueryAwareGrounder)
        grounder._clip_scorer = scorer

        h = _hyp("food", [("food", 0.8), ("snack", 0.2)])
        grounder._inject_clip_scores(
            [h], "dummy.png", "tangerine", synonyms=["orange"],
        )
        labels = [lbl for lbl, _ in h.label_alternatives]
        assert "tangerine" in labels

    def test_primary_uses_strict_threshold_even_with_synonyms(self):
        """primary 0.21 < strict 0.23, 即使有 synonyms 也不因放宽而注入。"""
        from src.perception import QueryAwareGrounder

        # primary "tangerine" 0.21 是最高分, 但 < strict 0.23
        # synonym "orange" 0.15 < relaxed 0.20
        scorer = MockCLIPScorer(scores={
            "tangerine": [0.21],
            "orange": [0.15],
        })
        grounder = QueryAwareGrounder.__new__(QueryAwareGrounder)
        grounder._clip_scorer = scorer

        h = _hyp("food", [("food", 0.8)])
        old_alts = list(h.label_alternatives)
        grounder._inject_clip_scores(
            [h], "dummy.png", "tangerine", synonyms=["orange"],
        )
        # primary 分数虽然最高但 < strict 阈值, 不应注入
        assert h.label_alternatives == old_alts

    def test_no_synonyms_uses_strict_threshold(self):
        """无 synonyms 时仍用 0.23 阈值 (向后兼容)。"""
        from src.perception import QueryAwareGrounder

        scorer = MockCLIPScorer(scores={"tangerine": [0.21]})
        grounder = QueryAwareGrounder.__new__(QueryAwareGrounder)
        grounder._clip_scorer = scorer

        h = _hyp("food", [("food", 0.8)])
        old_alts = list(h.label_alternatives)
        grounder._inject_clip_scores([h], "dummy.png", "tangerine")
        # 0.21 < 0.23, 严格阈值下不应注入
        assert h.label_alternatives == old_alts


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
