"""Integration tests: memory system influences strategy selection."""
from __future__ import annotations

import pytest
import numpy as np


class MockLLM:
    """Returns a fixed strategy JSON response."""
    def __init__(self, response: str):
        self._response = response
        self._last_prompt = ""

    def generate(self, prompt, system=""):
        self._last_prompt = prompt
        return self._response


class TestStrategyPromptInjection:
    def test_memory_advice_appears_in_prompt(self):
        """When memory_advice is passed, it appears in the strategy prompt."""
        from src.grasp_planner import GraspPlanner
        from src.world_belief import Hypothesis

        llm = MockLLM('{"strategy": "top_down", "reasoning": "test", "speech": "test"}')
        planner = GraspPlanner(vlm=None, env=None, llm=llm)

        hyp = Hypothesis(
            object_id="o1", label="tupperware",
            label_alternatives=[("tupperware", 0.9)],
            label_entropy=0.3,
            position_3d=np.array([0.5, 0, 0.9]),
            position_std_m=0.02,
        )
        planner.select_strategy(
            hyp,
            memory_advice="tupperware: avoid geometric_centroid (ik_unreachable x2)",
        )
        assert "geometric_centroid" in llm._last_prompt
        assert "ik_unreachable" in llm._last_prompt

    def test_no_memory_advice_still_works(self):
        """Without memory_advice, prompt still works (backward compat)."""
        from src.grasp_planner import GraspPlanner
        from src.world_belief import Hypothesis

        llm = MockLLM('{"strategy": "top_down", "reasoning": "ok", "speech": "ok"}')
        planner = GraspPlanner(vlm=None, env=None, llm=llm)
        hyp = Hypothesis(
            object_id="o1", label="apple",
            label_alternatives=[("apple", 0.9)],
            label_entropy=0.3,
            position_3d=np.array([0.5, 0, 0.9]),
            position_std_m=0.02,
        )
        result = planner.select_strategy(hyp)
        assert result.strategy == "top_down"
        assert "No prior experience" in llm._last_prompt
