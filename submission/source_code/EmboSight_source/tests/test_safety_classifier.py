"""SafetyClassifier (LLM-based) 单元测试。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import math

import numpy as np
import pytest

from tests._mocks import MockLLM


def _make_hyp(label="apple"):
    from src.world_belief import Hypothesis
    return Hypothesis(
        object_id="o0", label=label,
        label_alternatives=[(label, 0.9)], label_entropy=0.1,
        position_3d=np.zeros(3), position_std_m=0.05,
    )


class TestSafetyClassifier:
    def test_classify_returns_dist(self):
        from src.safety_gate import SafetyClassifier
        llm = MockLLM(responses=[json.dumps({
            "dist": {"safe": 0.8, "fragile": 0.1, "sharp": 0.05,
                     "hot": 0.0, "chemical": 0.05},
            "reasoning": "看起来像水果",
        })])
        sc = SafetyClassifier(llm=llm)
        ev = sc.classify(_make_hyp(label="apple"))
        assert ev.source == "llm_safety"
        assert ev.raw_payload["dist"]["safe"] == pytest.approx(0.8)

    def test_entropy_computed(self):
        from src.safety_gate import SafetyClassifier
        llm = MockLLM(responses=[json.dumps({
            "dist": {"safe": 0.5, "fragile": 0.5},
            "reasoning": "?",
        })])
        sc = SafetyClassifier(llm=llm)
        ev = sc.classify(_make_hyp())
        assert ev.raw_payload["entropy"] == pytest.approx(math.log(2), abs=0.01)

    def test_malformed_json_returns_unknown(self):
        from src.safety_gate import SafetyClassifier
        llm = MockLLM(responses=["not json at all"])
        sc = SafetyClassifier(llm=llm)
        ev = sc.classify(_make_hyp())
        assert ev.raw_payload["entropy"] == pytest.approx(0.0)
        assert "parse_failed" in ev.raw_payload.get("reasoning", "")

    def test_dist_normalized(self):
        """LLM 输出 dist 不归一时, 自动归一。"""
        from src.safety_gate import SafetyClassifier
        llm = MockLLM(responses=[json.dumps({
            "dist": {"safe": 0.4, "fragile": 0.4},   # sum=0.8
            "reasoning": "?",
        })])
        sc = SafetyClassifier(llm=llm)
        ev = sc.classify(_make_hyp())
        assert sum(ev.raw_payload["dist"].values()) == pytest.approx(1.0, abs=1e-3)

    def test_open_key_dist_accepted(self):
        """LLM 自定义 key (e.g. weight) 不报错 (F4)。"""
        from src.safety_gate import SafetyClassifier
        llm = MockLLM(responses=[json.dumps({
            "dist": {"safe": 0.5, "weight": 0.5},
            "reasoning": "very heavy",
        })])
        sc = SafetyClassifier(llm=llm)
        ev = sc.classify(_make_hyp())
        assert "weight" in ev.raw_payload["dist"]


class TestSafetyPriorHint:
    def test_no_hint_default_prompt(self):
        from src.safety_gate import SafetyClassifier
        llm = MockLLM(responses=[json.dumps({
            "dist": {"safe": 1.0}, "reasoning": "x",
        })])
        sc = SafetyClassifier(llm=llm)
        sc.classify(_make_hyp(label="apple"))
        prompt = llm.calls[0][0]
        assert "Historical" not in prompt
        assert "prior" not in prompt.lower()

    def test_with_prior_hint_injects_into_prompt(self):
        from src.safety_gate import SafetyClassifier
        llm = MockLLM(responses=[json.dumps({
            "dist": {"sharp": 0.9, "safe": 0.1}, "reasoning": "x",
        })])
        sc = SafetyClassifier(llm=llm)
        sc.classify(
            _make_hyp(label="knife"),
            prior_hint="Historical: 'knife' previously classified as sharp (0.85, n=4 obs).",
        )
        prompt = llm.calls[0][0]
        assert "Historical" in prompt
        assert "knife" in prompt
        assert "sharp" in prompt

    def test_prior_hint_does_not_break_parsing(self):
        """Even with prior, classifier still parses dist correctly."""
        from src.safety_gate import SafetyClassifier
        llm = MockLLM(responses=[json.dumps({
            "dist": {"sharp": 0.95, "safe": 0.05},
            "reasoning": "y",
        })])
        sc = SafetyClassifier(llm=llm)
        ev = sc.classify(
            _make_hyp(label="knife"),
            prior_hint="anything",
        )
        assert ev.raw_payload["dist"]["sharp"] == pytest.approx(0.95)
