"""Tests for dual-store episodic memory."""
from __future__ import annotations

import tempfile
from pathlib import Path


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


class TestSafetyConsolidation:
    def _make_dir(self) -> Path:
        import yaml
        d = Path(tempfile.mkdtemp()) / "memory"
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

    def test_consolidate_safety_creates_entry(self):
        from src.memory_manager import MemoryEntry, MemoryManager
        d = self._make_dir()
        mm = MemoryManager(memory_dir=d)
        mm.record_event(MemoryEntry(
            step=2, domain="safety", event="safety_classified",
            context={
                "label": "knife",
                "dist": {"safe": 0.05, "sharp": 0.85, "fragile": 0.10},
                "entropy": 0.40,
            },
            lesson="knife: sharp(0.85)",
        ))
        mm.consolidate(success=True, object_type="knife")

        mm2 = MemoryManager(memory_dir=d)
        entries = mm2._load_domain("safety")
        assert len(entries) == 1
        e = entries[0]
        assert e["label"] == "knife"
        assert e["observations"] == 1
        assert e["top_class"] == "sharp"
        assert abs(e["dist"]["sharp"] - 0.85) < 1e-6

    def test_consolidate_safety_skips_on_failure(self):
        from src.memory_manager import MemoryEntry, MemoryManager
        d = self._make_dir()
        mm = MemoryManager(memory_dir=d)
        mm.record_event(MemoryEntry(
            step=2, domain="safety", event="safety_classified",
            context={
                "label": "knife",
                "dist": {"sharp": 0.9, "safe": 0.1},
                "entropy": 0.32,
            },
            lesson="x",
        ))
        mm.consolidate(success=False, object_type="knife")

        mm2 = MemoryManager(memory_dir=d)
        assert mm2._load_domain("safety") == []

    def test_consolidate_safety_running_average(self):
        """Episode 2 observation merges as running average."""
        from src.memory_manager import MemoryEntry, MemoryManager
        d = self._make_dir()

        mm = MemoryManager(memory_dir=d)
        mm.record_event(MemoryEntry(
            step=1, domain="safety", event="safety_classified",
            context={
                "label": "knife",
                "dist": {"safe": 0.20, "sharp": 0.80},
                "entropy": 0.50,
            },
            lesson="x",
        ))
        mm.consolidate(success=True, object_type="knife")

        mm2 = MemoryManager(memory_dir=d)
        mm2.record_event(MemoryEntry(
            step=1, domain="safety", event="safety_classified",
            context={
                "label": "knife",
                "dist": {"safe": 0.40, "sharp": 0.60},
                "entropy": 0.67,
            },
            lesson="x",
        ))
        mm2.consolidate(success=True, object_type="knife")

        mm3 = MemoryManager(memory_dir=d)
        e = mm3._load_domain("safety")[0]
        assert e["observations"] == 2
        # running average: (0.80 + 0.60)/2 = 0.70
        assert abs(e["dist"]["sharp"] - 0.70) < 1e-6
        assert abs(e["dist"]["safe"] - 0.30) < 1e-6
        assert e["top_class"] == "sharp"

    def test_consolidate_safety_dedup_within_episode(self):
        """Same label observed twice in one episode → only latest dist counted once."""
        from src.memory_manager import MemoryEntry, MemoryManager
        d = self._make_dir()
        mm = MemoryManager(memory_dir=d)
        # First observation
        mm.record_event(MemoryEntry(
            step=1, domain="safety", event="safety_classified",
            context={"label": "knife", "dist": {"sharp": 0.5, "safe": 0.5},
                     "entropy": 0.69},
            lesson="x",
        ))
        # Second observation of the same label later in the episode
        mm.record_event(MemoryEntry(
            step=3, domain="safety", event="safety_classified",
            context={"label": "knife", "dist": {"sharp": 0.9, "safe": 0.1},
                     "entropy": 0.32},
            lesson="x",
        ))
        mm.consolidate(success=True, object_type="knife")

        mm2 = MemoryManager(memory_dir=d)
        e = mm2._load_domain("safety")[0]
        assert e["observations"] == 1  # 同 episode 同 label 视为一次
        # 应该用最新（最后一次）dist
        assert abs(e["dist"]["sharp"] - 0.9) < 1e-6


class TestSafetyPriorReader:
    def _make_dir_with_safety(self, label: str = "knife") -> Path:
        import yaml
        d = Path(tempfile.mkdtemp()) / "memory"
        d.mkdir()
        (d / "index.yaml").write_text(yaml.dump({
            "version": 1,
            "domains": {"safety": str(d / "safety_knowledge.yaml")},
        }), encoding="utf-8")
        (d / "safety_knowledge.yaml").write_text(yaml.dump({
            "entries": [
                {
                    "label": label,
                    "dist": {"safe": 0.10, "sharp": 0.85, "fragile": 0.05},
                    "top_class": "sharp",
                    "observations": 4,
                    "last_updated": "2026-05-13",
                },
            ],
        }), encoding="utf-8")
        return d

    def test_get_safety_prior_returns_entry(self):
        from src.memory_manager import MemoryManager
        d = self._make_dir_with_safety()
        mm = MemoryManager(memory_dir=d)
        prior = mm.get_safety_prior("knife")
        assert prior is not None
        assert prior["top_class"] == "sharp"
        assert prior["observations"] == 4
        assert "dist" in prior

    def test_get_safety_prior_case_insensitive(self):
        from src.memory_manager import MemoryManager
        d = self._make_dir_with_safety()
        mm = MemoryManager(memory_dir=d)
        assert mm.get_safety_prior("KNIFE") is not None

    def test_get_safety_prior_unknown_returns_none(self):
        from src.memory_manager import MemoryManager
        d = self._make_dir_with_safety()
        mm = MemoryManager(memory_dir=d)
        assert mm.get_safety_prior("banana") is None

    def test_get_safety_prior_missing_file_returns_none(self):
        from src.memory_manager import MemoryManager
        mm = MemoryManager(memory_dir=Path(tempfile.mkdtemp()))
        assert mm.get_safety_prior("knife") is None


# ──────────────────────────────────────────────────────────────────────
# Schema v2 / code_version / per-reason ban (added 2026-05-17)
# ──────────────────────────────────────────────────────────────────────

class TestSchemaV2:
    """v1 entries auto-migrate to v2 dict shape on load."""

    def _make_v1_dir(self) -> Path:
        import yaml
        d = Path(tempfile.mkdtemp()) / "memory"
        d.mkdir()
        (d / "index.yaml").write_text(yaml.dump({
            "version": 1,
            "domains": {"grasp": str(d / "grasp_experience.yaml")},
        }), encoding="utf-8")
        (d / "grasp_experience.yaml").write_text(yaml.dump({
            "entries": [{
                "object_type": "wooden_spoon",
                "best_strategy": "handle_grasp",
                "failed": [
                    {"strategy": "top_down", "reason": "slipped", "count": 3},
                ],
                "total_attempts": 4,
                "success_count": 1,
            }],
        }), encoding="utf-8")
        return d

    def test_v1_entry_migrates_strategies_dict(self):
        from src.memory_manager import MemoryManager
        d = self._make_v1_dir()
        mm = MemoryManager(memory_dir=d)
        entries = mm._load_domain("grasp")
        e = entries[0]
        # v1 fields preserved
        assert e["best_strategy"] == "handle_grasp"
        assert len(e["failed"]) == 1
        # v2 dict generated
        assert "strategies" in e
        assert e["strategies"]["top_down"]["failures"] == 3
        assert e["strategies"]["top_down"]["failures_by_reason"]["slipped"] == 3
        assert e["strategies"]["handle_grasp"]["successes"] == 1

    def test_v2_entry_passes_through(self):
        """Already-v2 entries are not re-migrated (idempotent)."""
        import yaml
        from src.memory_manager import MemoryManager
        d = Path(tempfile.mkdtemp()) / "memory"
        d.mkdir()
        (d / "index.yaml").write_text(yaml.dump({
            "domains": {"grasp": str(d / "grasp_experience.yaml")},
        }), encoding="utf-8")
        (d / "grasp_experience.yaml").write_text(yaml.dump({
            "schema_version": 2,
            "entries": [{
                "object_type": "spoon",
                "strategies": {
                    "top_down": {"successes": 0, "failures": 2,
                                 "failures_by_reason": {"slipped_lift": 2}},
                },
            }],
        }), encoding="utf-8")
        mm = MemoryManager(memory_dir=d)
        e = mm._load_domain("grasp")[0]
        assert e["strategies"]["top_down"]["failures_by_reason"]["slipped_lift"] == 2


class TestCodeVersionInvalidation:
    def _make_dir_with_code_version(self, cv: str) -> Path:
        import yaml
        d = Path(tempfile.mkdtemp()) / "memory"
        d.mkdir()
        (d / "index.yaml").write_text(yaml.dump({
            "domains": {"grasp": str(d / "grasp_experience.yaml")},
        }), encoding="utf-8")
        (d / "grasp_experience.yaml").write_text(yaml.dump({
            "code_version": cv,
            "entries": [{
                "object_type": "wooden_spoon",
                "strategies": {
                    "top_down": {"successes": 0, "failures": 5,
                                 "failures_by_reason": {"slipped_lift": 5}},
                },
            }],
        }), encoding="utf-8")
        return d

    def test_matching_code_version_keeps_data_authoritative(self):
        from src.memory_manager import MemoryManager, GRASP_CODE_VERSION
        d = self._make_dir_with_code_version(GRASP_CODE_VERSION)
        mm = MemoryManager(memory_dir=d)
        assert mm.is_grasp_memory_stale() is False
        assert mm.is_strategy_banned("wooden_spoon", "top_down") is True

    def test_mismatched_code_version_flags_stale_disables_ban(self):
        from src.memory_manager import MemoryManager
        d = self._make_dir_with_code_version("v0.0-bug-era")
        mm = MemoryManager(memory_dir=d)
        assert mm.is_grasp_memory_stale() is True
        # ban logic disabled -- previous bugs shouldn't poison fresh runs
        assert mm.is_strategy_banned("wooden_spoon", "top_down") is False
        assert mm.get_banned_strategies("wooden_spoon") == set()

    def test_v62_memory_is_stale_after_executed_strategy_attribution_change(self):
        from src.memory_manager import MemoryManager
        d = self._make_dir_with_code_version("v6.2")
        mm = MemoryManager(memory_dir=d)
        assert mm.is_grasp_memory_stale() is True
        assert mm.is_strategy_banned("wooden_spoon", "top_down") is False
        assert mm.get_proven_strategy("wooden_spoon") is None

    def test_save_resets_stale_flag(self):
        """Saving with current code stamps the file with current version,
        clearing stale on subsequent reads."""
        import yaml
        from src.memory_manager import (
            GRASP_CODE_VERSION, MemoryEntry, MemoryManager,
        )
        d = self._make_dir_with_code_version("v0.0-bug-era")
        mm = MemoryManager(memory_dir=d)
        assert mm.is_grasp_memory_stale() is True

        mm.record_event(MemoryEntry(
            step=1, domain="grasp", event="strategy_succeeded",
            context={"strategy": "handle_grasp"}, lesson="x",
        ))
        mm.consolidate(success=True, object_type="wooden_spoon")

        # Verify file now has current code_version
        with open(d / "grasp_experience.yaml", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["code_version"] == GRASP_CODE_VERSION

        # Fresh load: stale flag should be cleared
        mm2 = MemoryManager(memory_dir=d)
        assert mm2.is_grasp_memory_stale() is False


class TestPerReasonBan:
    def _make_dir_with_strategies(self, strategies: dict) -> Path:
        import yaml
        from src.memory_manager import GRASP_CODE_VERSION
        d = Path(tempfile.mkdtemp()) / "memory"
        d.mkdir()
        (d / "index.yaml").write_text(yaml.dump({
            "domains": {"grasp": str(d / "grasp_experience.yaml")},
        }), encoding="utf-8")
        (d / "grasp_experience.yaml").write_text(yaml.dump({
            "code_version": GRASP_CODE_VERSION,
            "entries": [{
                "object_type": "wooden_spoon",
                "strategies": strategies,
            }],
        }), encoding="utf-8")
        return d

    def test_three_failures_one_reason_bans(self):
        from src.memory_manager import MemoryManager
        d = self._make_dir_with_strategies({
            "top_down": {"successes": 0, "failures": 3,
                         "failures_by_reason": {"slipped_lift": 3}},
        })
        mm = MemoryManager(memory_dir=d)
        assert mm.is_strategy_banned("wooden_spoon", "top_down") is True

    def test_three_distinct_reasons_each_one_does_not_ban(self):
        """Per-reason threshold: 3 different fail modes (each x1) should NOT ban.
        Prevents conflating distinct failure modes.
        """
        from src.memory_manager import MemoryManager
        d = self._make_dir_with_strategies({
            "top_down": {"successes": 0, "failures": 3,
                         "failures_by_reason": {
                             "slipped_lift": 1,
                             "slipped_descend": 1,
                             "gripper_empty": 1,
                         }},
        })
        mm = MemoryManager(memory_dir=d)
        assert mm.is_strategy_banned("wooden_spoon", "top_down") is False

    def test_retired_entry_disables_ban(self):
        from src.memory_manager import MemoryManager
        d = self._make_dir_with_strategies({
            "top_down": {"successes": 0, "failures": 5,
                         "failures_by_reason": {"slipped_lift": 5}},
        })
        # patch entry to add retired flag
        import yaml
        with open(d / "grasp_experience.yaml", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        data["entries"][0]["retired"] = True
        with open(d / "grasp_experience.yaml", "w", encoding="utf-8") as f:
            yaml.dump(data, f)
        mm = MemoryManager(memory_dir=d)
        assert mm.is_strategy_banned("wooden_spoon", "top_down") is False
        assert mm.get_banned_strategies("wooden_spoon") == set()

    def test_get_banned_strategies_collects_all(self):
        from src.memory_manager import MemoryManager
        d = self._make_dir_with_strategies({
            "top_down": {"successes": 0, "failures": 3,
                         "failures_by_reason": {"slipped_lift": 3}},
            "handle_grasp": {"successes": 1, "failures": 1,
                             "failures_by_reason": {"slipped_lift": 1}},
            "tilted_grasp": {"successes": 0, "failures": 4,
                             "failures_by_reason": {"unreachable": 4}},
        })
        mm = MemoryManager(memory_dir=d)
        banned = mm.get_banned_strategies("wooden_spoon")
        assert banned == {"top_down", "tilted_grasp"}
        assert "handle_grasp" not in banned

    def test_consolidate_writes_failures_by_reason(self):
        from src.memory_manager import (
            GRASP_CODE_VERSION, MemoryEntry, MemoryManager,
        )
        import yaml
        d = Path(tempfile.mkdtemp()) / "memory"
        d.mkdir()
        (d / "index.yaml").write_text(yaml.dump({
            "domains": {"grasp": str(d / "grasp_experience.yaml")},
        }), encoding="utf-8")
        (d / "grasp_experience.yaml").write_text(
            yaml.dump({"entries": []}), encoding="utf-8",
        )
        mm = MemoryManager(memory_dir=d)
        for reason in ["slipped_lift", "slipped_lift", "gripper_empty"]:
            mm.record_event(MemoryEntry(
                step=1, domain="grasp", event="strategy_failed",
                context={"strategy": "top_down", "failure": reason},
                lesson="x",
            ))
        mm.consolidate(success=False, object_type="wooden_spoon")

        # File should have both v2 strategies dict + v1 failed list
        with open(d / "grasp_experience.yaml", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["code_version"] == GRASP_CODE_VERSION
        e = data["entries"][0]
        assert e["strategies"]["top_down"]["failures"] == 3
        fbr = e["strategies"]["top_down"]["failures_by_reason"]
        assert fbr["slipped_lift"] == 2
        assert fbr["gripper_empty"] == 1


class TestProvenStrategy:
    """get_proven_strategy returns the best zero-failure strategy."""

    def _make_dir(self, strategies: dict, code_version=None) -> Path:
        import yaml
        from src.memory_manager import GRASP_CODE_VERSION
        d = Path(tempfile.mkdtemp()) / "memory"
        d.mkdir()
        (d / "index.yaml").write_text(yaml.dump({
            "domains": {"grasp": str(d / "grasp_experience.yaml")},
        }), encoding="utf-8")
        (d / "grasp_experience.yaml").write_text(yaml.dump({
            "code_version": code_version or GRASP_CODE_VERSION,
            "entries": [{
                "object_type": "wooden_spoon",
                "strategies": strategies,
            }],
        }), encoding="utf-8")
        return d

    def test_returns_zero_failure_strategy(self):
        from src.memory_manager import MemoryManager
        d = self._make_dir({
            "top_down": {"successes": 0, "failures": 2,
                         "failures_by_reason": {"slipped_lift": 2}},
            "handle_grasp": {"successes": 3, "failures": 0,
                             "failures_by_reason": {}},
        })
        mm = MemoryManager(memory_dir=d)
        assert mm.get_proven_strategy("wooden_spoon") == "handle_grasp"

    def test_picks_highest_success_count(self):
        from src.memory_manager import MemoryManager
        d = self._make_dir({
            "handle_grasp": {"successes": 1, "failures": 0,
                             "failures_by_reason": {}},
            "tilted_grasp": {"successes": 5, "failures": 0,
                             "failures_by_reason": {}},
        })
        mm = MemoryManager(memory_dir=d)
        assert mm.get_proven_strategy("wooden_spoon") == "tilted_grasp"

    def test_none_when_all_have_failures(self):
        from src.memory_manager import MemoryManager
        d = self._make_dir({
            "top_down": {"successes": 10, "failures": 1,
                         "failures_by_reason": {"slipped_lift": 1}},
        })
        mm = MemoryManager(memory_dir=d)
        assert mm.get_proven_strategy("wooden_spoon") is None

    def test_none_when_stale(self):
        from src.memory_manager import MemoryManager
        d = self._make_dir(
            {"handle_grasp": {"successes": 5, "failures": 0,
                              "failures_by_reason": {}}},
            code_version="v0.0-old",
        )
        mm = MemoryManager(memory_dir=d)
        assert mm.get_proven_strategy("wooden_spoon") is None

    def test_none_when_no_entry(self):
        from src.memory_manager import MemoryManager
        d = self._make_dir({})
        mm = MemoryManager(memory_dir=d)
        assert mm.get_proven_strategy("apple") is None


class TestGraspPlannerWithMemory:
    """grasp_planner uses memory.get_banned_strategies when memory= passed."""

    def test_stale_memory_does_not_ban(self):
        """Bug-era data with mismatched code_version should not influence ban."""
        import yaml
        from src.grasp_planner import GraspPlanner
        from src.memory_manager import MemoryManager
        from src.world_belief import Hypothesis, Pose
        d = Path(tempfile.mkdtemp()) / "memory"
        d.mkdir()
        (d / "index.yaml").write_text(yaml.dump({
            "domains": {"grasp": str(d / "grasp_experience.yaml")},
        }), encoding="utf-8")
        (d / "grasp_experience.yaml").write_text(yaml.dump({
            "code_version": "v0.0-bug-era",
            "entries": [{
                "object_type": "wooden_spoon",
                "strategies": {
                    "top_down": {"successes": 0, "failures": 5,
                                 "failures_by_reason": {"slipped_lift": 5}},
                },
            }],
        }), encoding="utf-8")
        mm = MemoryManager(memory_dir=d)

        class _FakeEnv:
            def is_reachable(self, *a, **kw):
                return True
            def get_eef_pos(self):
                import numpy as np
                return np.array([0, 0, 1])
            def get_base_pose(self):
                import numpy as np
                return np.array([0, 0, 0]), np.eye(3)

        import numpy as np
        h = Hypothesis(
            object_id="obj_0",
            label="wooden_spoon",
            label_alternatives=[("wooden_spoon", 1.0)],
            label_entropy=0.0,
            position_3d=np.array([0.5, 0.0, 0.1]),
            position_std_m=0.01,
            pose_estimate=Pose(
                position=np.array([0.5, 0.0, 0.1]),
                rotation_quat=np.array([0, 0, 0, 1]),
                upright=True,
            ),
            safety_dist={"safe": 1.0},
        )
        planner = GraspPlanner(vlm=None, env=_FakeEnv(), llm=None)
        # legacy regex fallback would ban top_down, but structured API knows
        # the data is stale -> top_down available again
        strat = planner.select_strategy(
            h,
            memory_advice="wooden_spoon: avoid top_down (slipped_lift x5)",
            memory=mm,
        )
        assert strat.strategy == "top_down"

    def test_proven_strategy_skips_llm(self):
        """When memory has a proven winner, planner uses it directly."""
        import yaml
        from src.grasp_planner import GraspPlanner
        from src.memory_manager import GRASP_CODE_VERSION, MemoryManager
        from src.world_belief import Hypothesis, Pose
        d = Path(tempfile.mkdtemp()) / "memory"
        d.mkdir()
        (d / "index.yaml").write_text(yaml.dump({
            "domains": {"grasp": str(d / "grasp_experience.yaml")},
        }), encoding="utf-8")
        (d / "grasp_experience.yaml").write_text(yaml.dump({
            "code_version": GRASP_CODE_VERSION,
            "entries": [{
                "object_type": "wooden_spoon",
                "strategies": {
                    "top_down": {"successes": 0, "failures": 2,
                                 "failures_by_reason": {"slipped_lift": 2}},
                    "handle_grasp": {"successes": 1, "failures": 0,
                                     "failures_by_reason": {}},
                },
            }],
        }), encoding="utf-8")
        mm = MemoryManager(memory_dir=d)

        class _FakeEnv:
            def is_reachable(self, *a, **kw):
                return True
            def get_eef_pos(self):
                import numpy as np
                return np.array([0, 0, 1])
            def get_base_pose(self):
                import numpy as np
                return np.array([0, 0, 0]), np.eye(3)

        import numpy as np
        h = Hypothesis(
            object_id="obj_0",
            label="wooden_spoon",
            label_alternatives=[("wooden_spoon", 1.0)],
            label_entropy=0.0,
            position_3d=np.array([0.5, 0.0, 0.1]),
            position_std_m=0.01,
            pose_estimate=Pose(
                position=np.array([0.5, 0.0, 0.1]),
                rotation_quat=np.array([0, 0, 0, 1]),
                upright=True,
            ),
            safety_dist={"safe": 1.0},
        )
        planner = GraspPlanner(vlm=None, env=_FakeEnv(), llm=None)
        strat = planner.select_strategy(h, memory_advice="", memory=mm)
        assert strat.strategy == "handle_grasp"
        assert "proven" in strat.reasoning
