"""Integration tests: memory system influences strategy selection."""
from __future__ import annotations

import pytest
import numpy as np


import tempfile
from pathlib import Path


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


class TestEndToEndMemoryFlow:
    def test_failed_strategy_recorded_in_working_memory(self):
        """After grasp failure, working memory contains the failure entry."""
        from src.memory_manager import MemoryEntry, MemoryManager

        mm = MemoryManager(memory_dir=Path(tempfile.mkdtemp()))
        mm.record_event(MemoryEntry(
            step=5, domain="grasp", event="strategy_failed",
            context={"strategy": "geometric_centroid", "failure": "ik_unreachable", "object": "tupperware"},
            lesson="tupperware: geometric_centroid failed (ik_unreachable), avoid this strategy",
        ))
        summary = mm.get_working_summary(domain="grasp")
        assert "geometric_centroid" in summary
        assert "ik_unreachable" in summary

    def test_consolidate_then_load_round_trip(self):
        """Consolidate working memory -> reload -> advice available."""
        import yaml
        from src.memory_manager import MemoryEntry, MemoryManager

        d = Path(tempfile.mkdtemp()) / "memory"
        d.mkdir()
        grasp_path = d / "grasp_experience.yaml"
        (d / "index.yaml").write_text(yaml.dump({
            "version": 1,
            "domains": {"grasp": str(grasp_path)},
        }), encoding="utf-8")
        grasp_path.write_text(yaml.dump({"entries": []}), encoding="utf-8")

        # Episode 1: fail then succeed
        mm = MemoryManager(memory_dir=d)
        mm.record_event(MemoryEntry(
            step=5, domain="grasp", event="strategy_failed",
            context={"strategy": "geometric_centroid", "failure": "ik_unreachable"},
            lesson="tupperware: geometric_centroid failed",
        ))
        mm.record_event(MemoryEntry(
            step=7, domain="grasp", event="strategy_succeeded",
            context={"strategy": "top_down"},
            lesson="tupperware: top_down succeeded",
        ))
        mm.consolidate(success=True, object_type="tupperware")

        # Episode 2: load and check
        mm2 = MemoryManager(memory_dir=d)
        advice = mm2.get_grasp_advice("tupperware")
        assert advice is not None
        assert "top_down" in advice
        assert "geometric_centroid" in advice
        assert "ik_unreachable" in advice
