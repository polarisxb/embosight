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

    def test_sim_position_uses_matching_body_for_label(self):
        g = _make_grounder()
        raw = _make_vlm_json([
            {"bbox_2d": [600, 300, 700, 400], "label": "shaker",
             "alternatives": [["shaker", 0.85]],
             "confidence": 0.9, "visible_features": "metal"},
        ])

        class FakeEnv:
            body_requested = None

            def _get_obj_type_map(self):
                return {"obj_main": "shaker"}

            def _get_body_pos(self, body_name):
                self.body_requested = body_name
                if body_name == "obj_main":
                    return [0.4, 0.2, 0.85]
                return None

        env = FakeEnv()
        hyps = g._parse_to_hypotheses(raw, viewpoint=None, env=env)
        assert env.body_requested == "obj_main"
        assert hyps[0].position_3d.tolist() == pytest.approx([0.4, 0.2, 0.85])


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


class TestReObserve:
    def test_zoom_in_uses_zoom_prompt(self, tmp_image):
        import numpy as np
        from src.perception import QueryAwareGrounder
        from src.vlm_cache import VLMCache
        from src.world_belief import (
            DecomposedTask, Hypothesis, WorldBelief,
        )
        vlm = MockVLM(responses=[json.dumps({
            "label": "apple",
            "alternatives": [["apple", 0.9], ["pear", 0.1]],
            "visible_features": "shiny red",
        })])

        class FakeVP:
            name = "v0"
        vp_lib = [FakeVP()]
        g = QueryAwareGrounder(
            vlm=vlm, llm=MockLLM([]),
            cache=VLMCache(), label_temperature=1.0,
            viewpoint_lib=vp_lib,
        )
        h = Hypothesis(
            object_id="o0", label="apple",
            label_alternatives=[("apple", 0.5), ("kiwi", 0.5)],
            label_entropy=0.69,
            position_3d=np.array([0.5, 0, 0.9]), position_std_m=0.05,
            bbox_per_view={"v0": (50, 50, 100, 100)},
            observed_in_views=["v0"],
        )

        class FakeEnv:
            def observe(self, vp):
                return type("O", (), {"image_path": tmp_image})()
            def viewpoint_intrinsics(self, vp): return None
        belief = WorldBelief(user_query="x")
        belief.decomposed = DecomposedTask(primary_target="apple")
        ev = g.re_observe(h, "zoom_in", FakeEnv(), belief)
        assert ev.source == "vlm_zoom"
        assert ev.raw_payload.get("hypotheses"), "zoom 应该返回更新后的 hypothesis dict"
        new_alts = ev.raw_payload["hypotheses"][0]["label_alternatives"]
        # alts 来自 vlm 重新评估
        labels = [lbl for lbl, _ in new_alts]
        assert "apple" in labels

    def test_parallax_view_returns_evidence(self, tmp_image):
        import numpy as np
        from src.perception import QueryAwareGrounder
        from src.vlm_cache import VLMCache
        from src.world_belief import (
            DecomposedTask, Hypothesis, WorldBelief,
        )
        vlm = MockVLM(responses=[json.dumps({
            "bbox_2d": [60, 60, 110, 110], "confidence": 0.85,
        })])

        class FakeVP0:
            name = "v0"

        class FakeVP1:
            name = "v1"
        vp_lib = [FakeVP0(), FakeVP1()]
        g = QueryAwareGrounder(
            vlm=vlm, llm=MockLLM([]),
            cache=VLMCache(), viewpoint_lib=vp_lib,
        )
        h = Hypothesis(
            object_id="o0", label="apple",
            label_alternatives=[("apple", 0.95)],
            label_entropy=0.1,
            position_3d=np.array([0.5, 0, 0.9]), position_std_m=0.20,
            observed_in_views=["v0"],
        )

        class FakeEnv:
            def observe(self, vp):
                return type("O", (), {"image_path": tmp_image})()
            def viewpoint_intrinsics(self, vp): return None
        belief = WorldBelief(user_query="x")
        belief.decomposed = DecomposedTask(primary_target="apple")
        ev = g.re_observe(h, "parallax_view", FakeEnv(), belief)
        assert ev.source == "vlm_zoom"
        assert ev.raw_payload.get("viewpoint") == "v1"

    def test_unknown_strategy_raises(self):
        import numpy as np
        from src.perception import QueryAwareGrounder
        from src.vlm_cache import VLMCache
        from src.world_belief import (
            DecomposedTask, Hypothesis, WorldBelief,
        )
        g = QueryAwareGrounder(vlm=MockVLM([]), llm=MockLLM([]),
                               cache=VLMCache())
        h = Hypothesis(
            object_id="o0", label="x",
            label_alternatives=[("x", 1.0)], label_entropy=0.0,
            position_3d=np.zeros(3), position_std_m=0.05,
        )
        belief = WorldBelief(user_query="x")
        belief.decomposed = DecomposedTask(primary_target="x")
        with pytest.raises(ValueError):
            g.re_observe(h, "unknown_strategy", env=None, belief=belief)


class TestVerifyGrasp:
    def test_verify_match(self, tmp_image):
        import numpy as np
        from src.perception import QueryAwareGrounder
        from src.vlm_cache import VLMCache
        from src.world_belief import Hypothesis
        vlm = MockVLM(responses=[json.dumps({
            "is_match": True, "confidence": 0.9, "actual_guess": "",
        })])
        g = QueryAwareGrounder(vlm=vlm, llm=MockLLM([]),
                               cache=VLMCache())
        h = Hypothesis(
            object_id="o0", label="apple",
            label_alternatives=[("apple", 0.95)], label_entropy=0.1,
            position_3d=np.zeros(3), position_std_m=0.05,
        )

        class FakeEnv:
            def observe(self, vp):
                return type("O", (), {"image_path": tmp_image})()
            def eye_in_hand_viewpoint(self):
                return type("VP", (), {"name": "eye_in_hand"})()
        ok, conf = g.verify_grasp(h, FakeEnv())
        assert ok is True
        assert conf == pytest.approx(0.9)

    def test_verify_mismatch(self, tmp_image):
        import numpy as np
        from src.perception import QueryAwareGrounder
        from src.vlm_cache import VLMCache
        from src.world_belief import Hypothesis
        vlm = MockVLM(responses=[json.dumps({
            "is_match": False, "confidence": 0.7, "actual_guess": "pear",
        })])
        g = QueryAwareGrounder(vlm=vlm, llm=MockLLM([]),
                               cache=VLMCache())
        h = Hypothesis(
            object_id="o0", label="apple",
            label_alternatives=[("apple", 0.95)], label_entropy=0.1,
            position_3d=np.zeros(3), position_std_m=0.05,
        )

        class FakeEnv:
            def observe(self, vp):
                return type("O", (), {"image_path": tmp_image})()
            def eye_in_hand_viewpoint(self):
                return type("VP", (), {"name": "eye_in_hand"})()
        ok, conf = g.verify_grasp(h, FakeEnv())
        assert ok is False
