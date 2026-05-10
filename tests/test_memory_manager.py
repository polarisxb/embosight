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
