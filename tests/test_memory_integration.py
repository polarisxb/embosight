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


class TestRecognitionSynonymInjection:
    def _make_memory_with_synonym(self, target: str, synonym: str):
        import yaml
        d = Path(tempfile.mkdtemp()) / "memory"
        d.mkdir()
        (d / "index.yaml").write_text(yaml.dump({
            "version": 1,
            "domains": {"recognition": str(d / "recognition_hints.yaml")},
        }), encoding="utf-8")
        (d / "recognition_hints.yaml").write_text(yaml.dump({
            "entries": [{
                "target": target,
                "vlm_common_labels": [synonym],
                "effective_synonyms": [
                    {"name": synonym, "count": 2, "last_method": "clip"},
                ],
                "clip_helpful": True,
                "notes": "",
                "last_updated": "2026-05-13",
            }],
        }), encoding="utf-8")
        from src.memory_manager import MemoryManager
        return MemoryManager(memory_dir=d)

    def test_run_start_injects_persisted_synonym(self):
        from src.agent import EmboSightAgent
        from src.world_belief import DecomposedTask, WorldBelief

        mm = self._make_memory_with_synonym("tangerine", "orange")
        agent = EmboSightAgent.__new__(EmboSightAgent)
        agent.memory = mm

        belief = WorldBelief(user_query="pick tangerine")
        belief.decomposed = DecomposedTask(
            primary_target="tangerine",
            primary_target_synonyms=["mandarin"],
        )

        agent._apply_recognition_hints(belief)

        syns = belief.decomposed.primary_target_synonyms
        assert "mandarin" in syns
        assert "orange" in syns

    def test_run_start_dedupes_synonyms(self):
        from src.agent import EmboSightAgent
        from src.world_belief import DecomposedTask, WorldBelief

        mm = self._make_memory_with_synonym("tangerine", "orange")
        agent = EmboSightAgent.__new__(EmboSightAgent)
        agent.memory = mm

        belief = WorldBelief(user_query="pick tangerine")
        belief.decomposed = DecomposedTask(
            primary_target="tangerine",
            primary_target_synonyms=["orange"],
        )

        agent._apply_recognition_hints(belief)
        syns = belief.decomposed.primary_target_synonyms
        assert syns.count("orange") == 1

    def test_run_start_noop_without_decomposed(self):
        from src.agent import EmboSightAgent
        from src.world_belief import WorldBelief

        mm = self._make_memory_with_synonym("tangerine", "orange")
        agent = EmboSightAgent.__new__(EmboSightAgent)
        agent.memory = mm

        belief = WorldBelief(user_query="")
        belief.decomposed = None
        # should not raise
        agent._apply_recognition_hints(belief)

    def test_run_start_noop_without_hints(self):
        """target with no persisted hints → list unchanged."""
        import tempfile
        from src.agent import EmboSightAgent
        from src.memory_manager import MemoryManager
        from src.world_belief import DecomposedTask, WorldBelief

        agent = EmboSightAgent.__new__(EmboSightAgent)
        agent.memory = MemoryManager(memory_dir=Path(tempfile.mkdtemp()))

        belief = WorldBelief(user_query="pick apple")
        belief.decomposed = DecomposedTask(
            primary_target="apple",
            primary_target_synonyms=["fruit"],
        )
        agent._apply_recognition_hints(belief)
        assert belief.decomposed.primary_target_synonyms == ["fruit"]


class TestRecognitionRoundTrip:
    def test_clip_episode_persists_and_injects_next_episode(self, tmp_path):
        """Episode 1: simulate CLIP-injected evidence + grasp success.
        Episode 2: fresh MemoryManager+agent loads the persisted synonym.
        """
        import time
        import yaml
        from src.agent import EmboSightAgent
        from src.memory_manager import MemoryManager
        from src.world_belief import DecomposedTask, Evidence, WorldBelief

        # Setup persistent memory dir
        d = tmp_path / "memory"
        d.mkdir()
        (d / "index.yaml").write_text(yaml.dump({
            "version": 1,
            "domains": {"recognition": str(d / "recognition_hints.yaml")},
        }), encoding="utf-8")
        (d / "recognition_hints.yaml").write_text(yaml.dump({"entries": []}), encoding="utf-8")

        # ===== Episode 1 =====
        mm1 = MemoryManager(memory_dir=d)
        agent1 = EmboSightAgent.__new__(EmboSightAgent)
        agent1.memory = mm1

        belief1 = WorldBelief(user_query="pick tangerine")
        belief1.decomposed = DecomposedTask(primary_target="tangerine")

        ev = Evidence(
            source="vlm_ground", timestamp=time.time(),
            raw_payload={
                "hypotheses": [{
                    "object_id": "obj_0",
                    "label": "citrus",
                    "label_alternatives": [("citrus", 0.5), ("tangerine", 0.5)],
                    "label_entropy": 0.69,
                    "position_3d": [0.5, 0.0, 0.9],
                    "position_std_m": 0.02,
                    "bbox_per_view": {"v0": [0, 0, 50, 50]},
                    "observed_in_views": ["v0"],
                }],
                "clip_injected": {
                    "target": "tangerine",
                    "synonym": "orange",
                    "sim": 0.30,
                    "vlm_label": "citrus",
                },
            },
        )
        agent1._merge_hypotheses_from_evidence(belief1, ev)
        # Working memory should now contain the synonym_effective event
        assert any(
            e.domain == "recognition" and e.event == "synonym_effective"
            for e in mm1.working_memory
        )
        # Simulate successful grasp → consolidate
        mm1.consolidate(success=True, object_type="tangerine")

        # ===== Episode 2 =====
        mm2 = MemoryManager(memory_dir=d)
        agent2 = EmboSightAgent.__new__(EmboSightAgent)
        agent2.memory = mm2

        belief2 = WorldBelief(user_query="pick tangerine")
        belief2.decomposed = DecomposedTask(
            primary_target="tangerine",
            primary_target_synonyms=[],
        )
        agent2._apply_recognition_hints(belief2)

        assert "orange" in belief2.decomposed.primary_target_synonyms

    def test_llm_fallback_episode_persists_and_injects(self, tmp_path):
        """End-to-end for label_corrected path: LLM fallback → consolidate → next episode hint."""
        import yaml
        from src.agent import EmboSightAgent
        from src.memory_manager import MemoryManager
        from src.world_belief import DecomposedTask, WorldBelief

        d = tmp_path / "memory"
        d.mkdir()
        (d / "index.yaml").write_text(yaml.dump({
            "version": 1,
            "domains": {"recognition": str(d / "recognition_hints.yaml")},
        }), encoding="utf-8")
        (d / "recognition_hints.yaml").write_text(yaml.dump({"entries": []}), encoding="utf-8")

        # ===== Episode 1: LLM fallback corrects label =====
        mm1 = MemoryManager(memory_dir=d)
        agent1 = EmboSightAgent.__new__(EmboSightAgent)
        agent1.memory = mm1

        agent1._record_label_corrected("yogurt", "container", method="llm")
        mm1.consolidate(success=True, object_type="yogurt")

        # ===== Episode 2 =====
        mm2 = MemoryManager(memory_dir=d)
        agent2 = EmboSightAgent.__new__(EmboSightAgent)
        agent2.memory = mm2

        belief2 = WorldBelief(user_query="pick yogurt")
        belief2.decomposed = DecomposedTask(
            primary_target="yogurt",
            primary_target_synonyms=[],
        )
        agent2._apply_recognition_hints(belief2)

        assert "container" in belief2.decomposed.primary_target_synonyms


class TestSafetyRoundTrip:
    def _make_memory_dir(self, tmp_path):
        import yaml
        d = tmp_path / "memory"
        d.mkdir()
        (d / "index.yaml").write_text(yaml.dump({
            "version": 1,
            "domains": {
                "grasp": str(d / "grasp_experience.yaml"),
                "safety": str(d / "safety_knowledge.yaml"),
            },
        }), encoding="utf-8")
        (d / "grasp_experience.yaml").write_text(yaml.dump({"entries": []}), encoding="utf-8")
        (d / "safety_knowledge.yaml").write_text(yaml.dump({"entries": []}), encoding="utf-8")
        return d

    def test_safety_classify_persists_and_injects_next_episode(self, tmp_path):
        """Episode 1: agent records safety_classified → consolidate.
        Episode 2: fresh agent retrieves prior, formats hint.
        """
        from src.agent import EmboSightAgent
        from src.memory_manager import MemoryManager
        from src.world_belief import Evidence

        d = self._make_memory_dir(tmp_path)

        # ===== Episode 1 =====
        mm1 = MemoryManager(memory_dir=d)
        agent1 = EmboSightAgent.__new__(EmboSightAgent)
        agent1.memory = mm1

        ev = Evidence(
            source="llm_safety", timestamp=0.0,
            raw_payload={
                "dist": {"sharp": 0.80, "safe": 0.20},
                "entropy": 0.50,
                "reasoning": "metal blade visible",
            },
        )
        agent1._record_safety_classified("knife", ev)
        assert any(
            e.domain == "safety" and e.event == "safety_classified"
            for e in mm1.working_memory
        )
        mm1.consolidate(success=True, object_type="knife")

        # ===== Episode 2 =====
        mm2 = MemoryManager(memory_dir=d)
        agent2 = EmboSightAgent.__new__(EmboSightAgent)
        agent2.memory = mm2

        prior = mm2.get_safety_prior("knife")
        assert prior is not None
        assert prior["top_class"] == "sharp"
        assert prior["observations"] == 1
        assert abs(prior["dist"]["sharp"] - 0.80) < 1e-6

        hint = agent2._format_safety_prior_hint("knife")
        assert hint is not None
        assert "knife" in hint
        assert "sharp" in hint

    def test_safety_dist_converges_across_episodes(self, tmp_path):
        """Multiple successful episodes → running average sharpens belief."""
        from src.agent import EmboSightAgent
        from src.memory_manager import MemoryManager
        from src.world_belief import Evidence

        d = self._make_memory_dir(tmp_path)

        # 3 successful episodes, each classifies knife with high sharp prob
        for sharp_p in (0.80, 0.85, 0.90):
            mm = MemoryManager(memory_dir=d)
            agent = EmboSightAgent.__new__(EmboSightAgent)
            agent.memory = mm
            ev = Evidence(
                source="llm_safety", timestamp=0.0,
                raw_payload={
                    "dist": {"sharp": sharp_p, "safe": 1.0 - sharp_p},
                    "entropy": 0.4,
                },
            )
            agent._record_safety_classified("knife", ev)
            mm.consolidate(success=True, object_type="knife")

        mm_final = MemoryManager(memory_dir=d)
        prior = mm_final.get_safety_prior("knife")
        assert prior["observations"] == 3
        # average should be (0.80 + 0.85 + 0.90) / 3 = 0.85
        assert abs(prior["dist"]["sharp"] - 0.85) < 1e-6
        assert prior["top_class"] == "sharp"

    def test_failed_episode_does_not_persist_safety(self, tmp_path):
        """Phase 2 contract: only success consolidates safety."""
        from src.agent import EmboSightAgent
        from src.memory_manager import MemoryManager
        from src.world_belief import Evidence

        d = self._make_memory_dir(tmp_path)
        mm = MemoryManager(memory_dir=d)
        agent = EmboSightAgent.__new__(EmboSightAgent)
        agent.memory = mm

        ev = Evidence(
            source="llm_safety", timestamp=0.0,
            raw_payload={"dist": {"sharp": 0.9, "safe": 0.1}, "entropy": 0.3},
        )
        agent._record_safety_classified("knife", ev)
        mm.consolidate(success=False, object_type="knife")  # FAILED

        mm2 = MemoryManager(memory_dir=d)
        assert mm2.get_safety_prior("knife") is None
