"""Tests for dual-store episodic memory."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest


class TestMemoryEntry:
    def test_create_entry(self):
        from src.memory_manager import MemoryEntry

        e = MemoryEntry(
            step=5, domain="grasp", event="strategy_failed",
            context={"strategy": "geometric_centroid", "failure": "ik_unreachable", "object": "tupperware"},
            lesson="tupperware: geometric_centroid failed (ik_unreachable)",
        )
        assert e.step == 5
        assert e.domain == "grasp"
        assert e.event == "strategy_failed"
        assert e.context["strategy"] == "geometric_centroid"


class TestMemoryManagerInit:
    def test_init_creates_empty_working_memory(self):
        from src.memory_manager import MemoryManager

        mm = MemoryManager(memory_dir=Path(tempfile.mkdtemp()))
        assert mm.working_memory == []

    def test_record_event_appends(self):
        from src.memory_manager import MemoryEntry, MemoryManager

        mm = MemoryManager(memory_dir=Path(tempfile.mkdtemp()))
        e = MemoryEntry(step=1, domain="grasp", event="strategy_failed",
                        context={"strategy": "top_down"}, lesson="test")
        mm.record_event(e)
        assert len(mm.working_memory) == 1
        assert mm.working_memory[0] is e

    def test_get_working_summary_empty(self):
        from src.memory_manager import MemoryManager

        mm = MemoryManager(memory_dir=Path(tempfile.mkdtemp()))
        assert mm.get_working_summary() == ""

    def test_get_working_summary_with_entries(self):
        from src.memory_manager import MemoryEntry, MemoryManager

        mm = MemoryManager(memory_dir=Path(tempfile.mkdtemp()))
        mm.record_event(MemoryEntry(
            step=3, domain="grasp", event="strategy_failed",
            context={}, lesson="geometric_centroid failed",
        ))
        summary = mm.get_working_summary()
        assert "geometric_centroid failed" in summary

    def test_get_working_summary_filters_domain(self):
        from src.memory_manager import MemoryEntry, MemoryManager

        mm = MemoryManager(memory_dir=Path(tempfile.mkdtemp()))
        mm.record_event(MemoryEntry(step=1, domain="grasp", event="x",
                                     context={}, lesson="grasp lesson"))
        mm.record_event(MemoryEntry(step=2, domain="recognition", event="y",
                                     context={}, lesson="recog lesson"))
        grasp_only = mm.get_working_summary(domain="grasp")
        assert "grasp lesson" in grasp_only
        assert "recog lesson" not in grasp_only


class TestLongTermMemory:
    def _make_memory_dir(self) -> Path:
        """Create a temp memory dir with index + grasp file."""
        import yaml
        d = Path(tempfile.mkdtemp()) / "memory"
        d.mkdir()
        (d / "index.yaml").write_text(yaml.dump({
            "version": 1,
            "domains": {
                "grasp": str(d / "grasp_experience.yaml"),
                "recognition": str(d / "recognition_hints.yaml"),
            },
        }), encoding="utf-8")
        (d / "grasp_experience.yaml").write_text(yaml.dump({
            "entries": [
                {
                    "object_type": "tupperware",
                    "best_strategy": "top_down",
                    "failed": [{"strategy": "geometric_centroid", "reason": "ik_unreachable", "count": 2}],
                    "total_attempts": 5,
                    "success_count": 3,
                    "last_updated": "2026-05-10",
                },
            ],
        }), encoding="utf-8")
        (d / "recognition_hints.yaml").write_text(yaml.dump({
            "entries": [],
        }), encoding="utf-8")
        return d

    def test_get_grasp_advice_found(self):
        from src.memory_manager import MemoryManager
        d = self._make_memory_dir()
        mm = MemoryManager(memory_dir=d)
        advice = mm.get_grasp_advice("tupperware")
        assert advice is not None
        assert "top_down" in advice
        assert "geometric_centroid" in advice

    def test_get_grasp_advice_not_found(self):
        from src.memory_manager import MemoryManager
        d = self._make_memory_dir()
        mm = MemoryManager(memory_dir=d)
        assert mm.get_grasp_advice("banana") is None

    def test_load_for_task(self):
        from src.memory_manager import MemoryManager
        d = self._make_memory_dir()
        mm = MemoryManager(memory_dir=d)
        summary = mm.load_for_task("tupperware")
        assert "top_down" in summary

    def test_consolidate_creates_new_entry(self):
        from src.memory_manager import MemoryEntry, MemoryManager
        d = self._make_memory_dir()
        mm = MemoryManager(memory_dir=d)
        mm.record_event(MemoryEntry(
            step=5, domain="grasp", event="strategy_failed",
            context={"strategy": "vlm_top_grasp", "failure": "hit_z_floor"},
            lesson="apple: vlm_top_grasp failed",
        ))
        mm.record_event(MemoryEntry(
            step=7, domain="grasp", event="strategy_succeeded",
            context={"strategy": "top_down"},
            lesson="apple: top_down succeeded",
        ))
        mm.consolidate(success=True, object_type="apple")
        # Reload and check
        mm2 = MemoryManager(memory_dir=d)
        advice = mm2.get_grasp_advice("apple")
        assert advice is not None
        assert "top_down" in advice
        assert "vlm_top_grasp" in advice

    def test_consolidate_merges_existing(self):
        from src.memory_manager import MemoryEntry, MemoryManager
        d = self._make_memory_dir()
        mm = MemoryManager(memory_dir=d)
        mm.record_event(MemoryEntry(
            step=5, domain="grasp", event="strategy_failed",
            context={"strategy": "geometric_centroid", "failure": "ik_unreachable"},
            lesson="tupperware: geometric_centroid failed again",
        ))
        mm.consolidate(success=False, object_type="tupperware")
        mm2 = MemoryManager(memory_dir=d)
        advice = mm2.get_grasp_advice("tupperware")
        # count should be 3 (was 2, +1)
        assert "3" in advice

    def test_graceful_on_missing_dir(self):
        from src.memory_manager import MemoryManager
        mm = MemoryManager(memory_dir=Path(tempfile.mkdtemp()) / "nonexistent")
        assert mm.get_grasp_advice("anything") is None
        assert mm.load_for_task("anything") == ""


class TestRecognitionConsolidation:
    def _make_dir(self) -> Path:
        import yaml
        d = Path(tempfile.mkdtemp()) / "memory"
        d.mkdir()
        (d / "index.yaml").write_text(yaml.dump({
            "version": 1,
            "domains": {
                "grasp": str(d / "grasp_experience.yaml"),
                "recognition": str(d / "recognition_hints.yaml"),
            },
        }), encoding="utf-8")
        (d / "grasp_experience.yaml").write_text(yaml.dump({"entries": []}), encoding="utf-8")
        (d / "recognition_hints.yaml").write_text(yaml.dump({"entries": []}), encoding="utf-8")
        return d

    def test_consolidate_recognition_creates_entry(self):
        from src.memory_manager import MemoryEntry, MemoryManager
        d = self._make_dir()
        mm = MemoryManager(memory_dir=d)
        mm.record_event(MemoryEntry(
            step=3, domain="recognition", event="synonym_effective",
            context={"target": "tangerine", "synonym": "orange",
                     "sim": 0.31, "vlm_label": "orange"},
            lesson="tangerine: CLIP via 'orange'",
        ))
        mm.consolidate(success=True, object_type="tangerine")

        mm2 = MemoryManager(memory_dir=d)
        entries = mm2._load_domain("recognition")
        assert len(entries) == 1
        e = entries[0]
        assert e["target"] == "tangerine"
        assert "orange" in e["vlm_common_labels"]
        assert any(s["name"] == "orange" and s["count"] == 1 for s in e["effective_synonyms"])
        assert e["clip_helpful"] is True

    def test_consolidate_recognition_skips_on_failure(self):
        from src.memory_manager import MemoryEntry, MemoryManager
        d = self._make_dir()
        mm = MemoryManager(memory_dir=d)
        mm.record_event(MemoryEntry(
            step=3, domain="recognition", event="synonym_effective",
            context={"target": "tangerine", "synonym": "orange",
                     "sim": 0.31, "vlm_label": "orange"},
            lesson="x",
        ))
        mm.consolidate(success=False, object_type="tangerine")

        mm2 = MemoryManager(memory_dir=d)
        entries = mm2._load_domain("recognition")
        assert entries == []

    def test_consolidate_recognition_merges_count(self):
        from src.memory_manager import MemoryEntry, MemoryManager
        d = self._make_dir()
        # Episode 1
        mm = MemoryManager(memory_dir=d)
        mm.record_event(MemoryEntry(
            step=3, domain="recognition", event="synonym_effective",
            context={"target": "tangerine", "synonym": "orange",
                     "sim": 0.31, "vlm_label": "orange"},
            lesson="x",
        ))
        mm.consolidate(success=True, object_type="tangerine")

        # Episode 2: same (target, synonym)
        mm2 = MemoryManager(memory_dir=d)
        mm2.record_event(MemoryEntry(
            step=4, domain="recognition", event="synonym_effective",
            context={"target": "tangerine", "synonym": "orange",
                     "sim": 0.40, "vlm_label": "citrus"},
            lesson="x",
        ))
        mm2.consolidate(success=True, object_type="tangerine")

        mm3 = MemoryManager(memory_dir=d)
        entries = mm3._load_domain("recognition")
        e = entries[0]
        # count merged to 2
        syn = next(s for s in e["effective_synonyms"] if s["name"] == "orange")
        assert syn["count"] == 2
        # vlm_common_labels accumulated (set semantics)
        assert "orange" in e["vlm_common_labels"]
        assert "citrus" in e["vlm_common_labels"]

    def test_consolidate_recognition_llm_method(self):
        from src.memory_manager import MemoryEntry, MemoryManager
        d = self._make_dir()
        mm = MemoryManager(memory_dir=d)
        mm.record_event(MemoryEntry(
            step=3, domain="recognition", event="label_corrected",
            context={"target": "yogurt", "detected_label": "container",
                     "method": "llm"},
            lesson="x",
        ))
        mm.consolidate(success=True, object_type="yogurt")

        mm2 = MemoryManager(memory_dir=d)
        entries = mm2._load_domain("recognition")
        e = entries[0]
        assert "container" in e["vlm_common_labels"]
        syn = next(s for s in e["effective_synonyms"] if s["name"] == "container")
        assert syn["last_method"] == "llm"
        assert e["clip_helpful"] is False

    def test_consolidate_recognition_synonyms_capped_at_5(self):
        from src.memory_manager import MemoryEntry, MemoryManager
        d = self._make_dir()
        mm = MemoryManager(memory_dir=d)
        for i in range(7):
            mm.record_event(MemoryEntry(
                step=i, domain="recognition", event="synonym_effective",
                context={"target": "tangerine", "synonym": f"syn_{i}",
                         "sim": 0.30, "vlm_label": f"syn_{i}"},
                lesson="x",
            ))
        mm.consolidate(success=True, object_type="tangerine")

        mm2 = MemoryManager(memory_dir=d)
        entries = mm2._load_domain("recognition")
        assert len(entries[0]["effective_synonyms"]) == 5

    def test_grasp_consolidation_still_runs_on_failure(self):
        """Phase 1 contract: grasp consolidation writes regardless of success."""
        from src.memory_manager import MemoryEntry, MemoryManager
        d = self._make_dir()
        mm = MemoryManager(memory_dir=d)
        mm.record_event(MemoryEntry(
            step=5, domain="grasp", event="strategy_failed",
            context={"strategy": "top_down", "failure": "hit_z_floor"},
            lesson="x",
        ))
        mm.consolidate(success=False, object_type="apple")

        mm2 = MemoryManager(memory_dir=d)
        advice = mm2.get_grasp_advice("apple")
        assert advice is not None
        assert "top_down" in advice


class TestRecognitionHints:
    def _make_dir_with_hints(self) -> Path:
        import yaml
        d = Path(tempfile.mkdtemp()) / "memory"
        d.mkdir()
        (d / "index.yaml").write_text(yaml.dump({
            "version": 1,
            "domains": {"recognition": str(d / "recognition_hints.yaml")},
        }), encoding="utf-8")
        (d / "recognition_hints.yaml").write_text(yaml.dump({
            "entries": [
                {
                    "target": "tangerine",
                    "vlm_common_labels": ["orange", "citrus"],
                    "effective_synonyms": [
                        {"name": "orange", "count": 3, "last_method": "clip"},
                        {"name": "citrus", "count": 1, "last_method": "clip"},
                    ],
                    "clip_helpful": True,
                    "notes": "",
                    "last_updated": "2026-05-13",
                },
            ],
        }), encoding="utf-8")
        return d

    def test_get_synonyms_returns_count_desc(self):
        from src.memory_manager import MemoryManager
        d = self._make_dir_with_hints()
        mm = MemoryManager(memory_dir=d)
        syns = mm.get_recognition_hints_synonyms("tangerine")
        assert syns == ["orange", "citrus"]

    def test_get_synonyms_case_insensitive(self):
        from src.memory_manager import MemoryManager
        d = self._make_dir_with_hints()
        mm = MemoryManager(memory_dir=d)
        assert mm.get_recognition_hints_synonyms("TANGERINE") == ["orange", "citrus"]

    def test_get_synonyms_unknown_returns_empty(self):
        from src.memory_manager import MemoryManager
        d = self._make_dir_with_hints()
        mm = MemoryManager(memory_dir=d)
        assert mm.get_recognition_hints_synonyms("banana") == []

    def test_get_synonyms_missing_file_returns_empty(self):
        from src.memory_manager import MemoryManager
        mm = MemoryManager(memory_dir=Path(tempfile.mkdtemp()))
        assert mm.get_recognition_hints_synonyms("tangerine") == []
