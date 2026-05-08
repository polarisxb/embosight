"""QueryAwareGrounder 单元测试 (observe / parse / 温度缩放)。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import math

import pytest

from tests._mocks import MockLLM, MockVLM


@pytest.fixture
def tmp_image(tmp_path):
    from PIL import Image
    p = tmp_path / "img.png"
    Image.new("RGB", (256, 256), (200, 100, 50)).save(p)
    return str(p)


def _make_vlm_json(objects):
    return json.dumps({"objects": objects})


def _make_grounder(label_temperature: float = 1.0, vlm=None):
    from src.perception import QueryAwareGrounder
    from src.vlm_cache import VLMCache
    return QueryAwareGrounder(
        vlm=vlm or MockVLM([]), llm=MockLLM([]),
        cache=VLMCache(), label_temperature=label_temperature,
    )


class TestParse:
    def test_basic_parse(self):
        g = _make_grounder()
        raw = _make_vlm_json([
            {"bbox_2d": [10, 10, 50, 50], "label": "apple",
             "alternatives": [["apple", 0.7], ["pear", 0.3]],
             "confidence": 0.9, "visible_features": "red round"},
        ])
        hyps = g._parse_to_hypotheses(raw, viewpoint=None, env=None)
        assert len(hyps) == 1
        assert hyps[0].label == "apple"
        labels = [lbl for lbl, _ in hyps[0].label_alternatives]
        assert "apple" in labels and "pear" in labels
        # 温度=1, sum 已经=1, 概率原样
        d = dict(hyps[0].label_alternatives)
        assert d["apple"] == pytest.approx(0.7, abs=1e-2)

    def test_temperature_scaling_flattens(self):
        """τ>1 让 0.95 概率被压平。"""
        g = _make_grounder(label_temperature=2.0)
        raw = _make_vlm_json([
            {"bbox_2d": [0, 0, 1, 1], "label": "x",
             "alternatives": [["x", 0.95], ["y", 0.04], ["z", 0.01]],
             "confidence": 0.9, "visible_features": "f"},
        ])
        hyps = g._parse_to_hypotheses(raw, viewpoint=None, env=None)
        # τ=2: p_i' ∝ p_i^0.5; top1 归一化后落在 (0.70, 0.90)
        top1_prob = hyps[0].label_alternatives[0][1]
        assert 0.70 < top1_prob < 0.90

    def test_entropy_computation(self):
        g = _make_grounder()
        raw = _make_vlm_json([
            {"bbox_2d": [0, 0, 1, 1], "label": "x",
             "alternatives": [["x", 0.5], ["y", 0.5]],
             "confidence": 0.9, "visible_features": "f"},
        ])
        hyps = g._parse_to_hypotheses(raw, viewpoint=None, env=None)
        # H(0.5, 0.5) = ln(2) ≈ 0.693
        assert hyps[0].label_entropy == pytest.approx(math.log(2), abs=0.01)

    def test_malformed_json_returns_empty(self):
        g = _make_grounder()
        hyps = g._parse_to_hypotheses("not json", viewpoint=None, env=None)
        assert hyps == []

    def test_alternatives_sum_normalized(self):
        """alternatives 和不等于 1 的也能解析 + 归一化。"""
        g = _make_grounder()
        raw = _make_vlm_json([
            {"bbox_2d": [0, 0, 1, 1], "label": "x",
             "alternatives": [["x", 0.4], ["y", 0.4]],   # sum=0.8
             "confidence": 0.9, "visible_features": "f"},
        ])
        hyps = g._parse_to_hypotheses(raw, viewpoint=None, env=None)
        s = sum(p for _, p in hyps[0].label_alternatives)
        assert s == pytest.approx(1.0, abs=1e-3)

    def test_multiple_objects_parsed(self):
        """多物体应都被解析为独立 Hypothesis。"""
        g = _make_grounder()
        raw = _make_vlm_json([
            {"bbox_2d": [0, 0, 10, 10], "label": "apple",
             "alternatives": [["apple", 1.0]],
             "confidence": 0.9, "visible_features": "red"},
            {"bbox_2d": [20, 20, 40, 40], "label": "knife",
             "alternatives": [["knife", 1.0]],
             "confidence": 0.9, "visible_features": "shiny"},
        ])
        hyps = g._parse_to_hypotheses(raw, viewpoint=None, env=None)
        assert len(hyps) == 2
        ids = {h.object_id for h in hyps}
        assert len(ids) == 2  # 唯一 id
        labels = {h.label for h in hyps}
        assert labels == {"apple", "knife"}


class TestPromptBuild:
    def test_inject_target_and_constraints(self):
        from src.world_belief import Constraint
        g = _make_grounder()
        prompt = g._build_query_aware_prompt(
            primary_target="削皮器",
            constraints=[Constraint(kind="avoid", target_label="knife",
                                    reason="用户说避开")],
            img_w=512, img_h=384,
        )
        assert "削皮器" in prompt
        assert "knife" in prompt
        assert "512" in prompt
        assert "384" in prompt


class TestObserve:
    def test_observe_calls_vlm_with_query(self, tmp_image):
        """observe 应注入 query 到 prompt (根因①)。"""
        from src.world_belief import DecomposedTask, WorldBelief

        class FakeVP:
            name = "robot0_agentview_center"

        class FakeObs:
            image_path = tmp_image

        class FakeEnv:
            def observe(self, vp): return FakeObs()
            def viewpoint_intrinsics(self, vp): return None

        vlm = MockVLM(responses=[_make_vlm_json([
            {"bbox_2d": [50, 50, 100, 100], "label": "apple",
             "alternatives": [["apple", 0.8], ["other", 0.2]],
             "confidence": 0.9, "visible_features": "red"},
        ])])
        from src.perception import QueryAwareGrounder
        from src.vlm_cache import VLMCache
        g = QueryAwareGrounder(vlm=vlm, llm=MockLLM([]),
                               cache=VLMCache(), label_temperature=1.0)
        belief = WorldBelief(user_query="拿苹果")
        belief.decomposed = DecomposedTask(primary_target="apple")
        ev = g.observe(FakeVP(), FakeEnv(), belief)
        assert ev.source == "vlm_ground"
        assert "apple" in vlm.calls[0][1]   # prompt 含 "apple"
        assert ev.raw_payload["viewpoint"] == "robot0_agentview_center"
        assert len(ev.raw_payload["hypotheses"]) == 1

    def test_observe_uses_cache(self, tmp_image):
        """同一 (image, prompt) 第二次不再调 VLM。"""
        from src.perception import QueryAwareGrounder
        from src.vlm_cache import VLMCache
        from src.world_belief import DecomposedTask, WorldBelief

        class FakeVP:
            name = "v1"

        class FakeObs:
            image_path = tmp_image

        class FakeEnv:
            def observe(self, vp): return FakeObs()
            def viewpoint_intrinsics(self, vp): return None

        vlm = MockVLM(responses=[_make_vlm_json([])] * 5)
        cache = VLMCache(max_size=10)
        g = QueryAwareGrounder(vlm=vlm, llm=MockLLM([]),
                               cache=cache, label_temperature=1.0)
        belief = WorldBelief(user_query="拿苹果")
        belief.decomposed = DecomposedTask(primary_target="apple")
        g.observe(FakeVP(), FakeEnv(), belief)
        first_call_count = len(vlm.calls)
        g.observe(FakeVP(), FakeEnv(), belief)
        assert len(vlm.calls) == first_call_count   # 没多调

    def test_observe_failed_returns_failed_evidence(self, tmp_image):
        """VLM 抛异常 → Evidence(source='vlm_failed') (Edge 9.8)。"""
        from src.perception import QueryAwareGrounder
        from src.vlm_cache import VLMCache
        from src.world_belief import DecomposedTask, WorldBelief

        class BadVLM:
            calls: list = []
            def describe(self, *a, **kw):
                raise RuntimeError("VLM down")

        class FakeVP:
            name = "v1"

        class FakeObs:
            image_path = tmp_image

        class FakeEnv:
            def observe(self, vp): return FakeObs()
            def viewpoint_intrinsics(self, vp): return None

        g = QueryAwareGrounder(vlm=BadVLM(), llm=MockLLM([]),
                               cache=VLMCache())
        belief = WorldBelief(user_query="x")
        belief.decomposed = DecomposedTask(primary_target="apple")
        ev = g.observe(FakeVP(), FakeEnv(), belief)
        assert ev.source == "vlm_failed"
        assert "error" in ev.raw_payload
