# Dual-Store Episodic Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a dual-store memory system (working + long-term) so the agent learns from failures within and across episodes, directly fixing the grasp strategy regression in scenarios 001/007/009.

**Architecture:** `MemoryManager` owns both stores. Working memory is a list of `MemoryEntry` dataclasses held in-memory during an episode. Long-term memory is YAML files in `memory/` with a pointer-index (`index.yaml`). At episode start, relevant long-term entries are loaded and formatted as prompt text. During the episode, failures are recorded in real-time to working memory, which is injected into strategy-selection prompts. At episode end, working memory is consolidated into long-term files via read-before-write merge.

**Tech Stack:** Python 3.10+, dataclasses, PyYAML (already in deps), pytest

**Spec:** `docs/superpowers/specs/2026-05-11-dual-store-memory-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/memory_manager.py` | Create | MemoryEntry dataclass + MemoryManager class |
| `memory/index.yaml` | Create | Pointer-index (domains → file paths) |
| `memory/grasp_experience.yaml` | Create | Empty initial grasp experience store |
| `memory/recognition_hints.yaml` | Create | Empty initial recognition hints store |
| `prompts/grasp/select_strategy.txt` | Modify | Add `{past_experience}` slot |
| `src/grasp_planner.py` | Modify | Pass memory_advice to prompt |
| `src/agent.py` | Modify | Init MemoryManager, record events, consolidate |
| `src/world_belief.py` | Modify | Add working_memory field to WorldBelief |
| `tests/test_memory_manager.py` | Create | Unit tests for MemoryManager |
| `tests/test_memory_integration.py` | Create | Integration: strategy prompt gets memory |

---

### Task 1: MemoryEntry dataclass + MemoryManager skeleton

**Files:**
- Create: `src/memory_manager.py`
- Test: `tests/test_memory_manager.py`

- [ ] **Step 1: Write failing tests for MemoryEntry and basic MemoryManager**

```python
# tests/test_memory_manager.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_memory_manager.py -v --tb=short`
Expected: ImportError — `src.memory_manager` does not exist

- [ ] **Step 3: Implement MemoryEntry and MemoryManager skeleton**

```python
# src/memory_manager.py
"""Dual-store episodic memory for EmboSight agent.

Working memory: in-process list, written in real-time during episode.
Long-term memory: YAML files in memory/ dir, consolidated after episode.

Design: docs/superpowers/specs/2026-05-11-dual-store-memory-design.md
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    step: int
    domain: str          # "grasp" | "recognition" | "safety"
    event: str           # "strategy_failed" | "strategy_succeeded" | "label_corrected" | ...
    context: dict        # structured payload
    lesson: str          # one-line summary for LLM prompt injection


class MemoryManager:
    """Dual-store episodic memory manager."""

    def __init__(self, memory_dir: Path = Path("memory")):
        self.memory_dir = memory_dir
        self.working_memory: list[MemoryEntry] = []
        self._long_term: dict[str, list[dict]] = {}
        self._load_index()

    # ── Load ──

    def _load_index(self) -> None:
        """Load index.yaml and parse domain pointers."""
        idx_path = self.memory_dir / "index.yaml"
        if not idx_path.exists():
            self._domain_files: dict[str, Path] = {}
            return
        try:
            import yaml
            with open(idx_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            domains = data.get("domains", {})
            self._domain_files = {
                k: Path(v) for k, v in domains.items()
            }
        except Exception as e:
            logger.warning("[memory] failed to load index: %s", e)
            self._domain_files = {}

    def _load_domain(self, domain: str) -> list[dict]:
        """Load a domain file, return list of entries."""
        if domain in self._long_term:
            return self._long_term[domain]
        fpath = self._domain_files.get(domain)
        if not fpath or not fpath.exists():
            self._long_term[domain] = []
            return []
        try:
            import yaml
            with open(fpath, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            entries = data.get("entries", [])
            self._long_term[domain] = entries
            return entries
        except Exception as e:
            logger.warning("[memory] failed to load %s: %s", domain, e)
            self._long_term[domain] = []
            return []

    # ── Record (real-time) ──

    def record_event(self, entry: MemoryEntry) -> None:
        """Append to working memory (in-process, immediate)."""
        self.working_memory.append(entry)
        logger.debug("[memory] recorded: step=%d domain=%s event=%s",
                     entry.step, entry.domain, entry.event)

    # ── Read (prompt injection) ──

    def get_working_summary(self, domain: Optional[str] = None) -> str:
        """Return working memory as text for LLM prompt injection."""
        entries = self.working_memory
        if domain:
            entries = [e for e in entries if e.domain == domain]
        if not entries:
            return ""
        lines = [f"- Step {e.step}: {e.lesson}" for e in entries]
        return "\n".join(lines)

    def get_grasp_advice(self, object_type: str) -> Optional[str]:
        """Get long-term grasp advice for a specific object type."""
        entries = self._load_domain("grasp")
        obj_key = object_type.lower().strip()
        for entry in entries:
            if entry.get("object_type", "").lower().strip() == obj_key:
                parts = []
                best = entry.get("best_strategy")
                if best:
                    parts.append(f"prefer {best}")
                for f in entry.get("failed", []):
                    parts.append(
                        f"avoid {f['strategy']} ({f['reason']} x{f.get('count', 1)})"
                    )
                if parts:
                    return f"{object_type}: {', '.join(parts)}"
        return None

    def get_recognition_hints(self, target: str) -> Optional[str]:
        """Get long-term recognition hints for a target."""
        entries = self._load_domain("recognition")
        tgt_key = target.lower().strip()
        for entry in entries:
            if entry.get("target", "").lower().strip() == tgt_key:
                parts = []
                labels = entry.get("vlm_common_labels", [])
                if labels:
                    parts.append(f"VLM often labels as: {', '.join(labels)}")
                syns = entry.get("effective_synonyms", [])
                if syns:
                    parts.append(f"effective synonyms: {', '.join(syns)}")
                if parts:
                    return f"{target}: {'; '.join(parts)}"
        return None

    def load_for_task(self, primary_target: str, object_type: str = "") -> str:
        """Load relevant long-term experience, return summary text."""
        parts: list[str] = []
        obj = object_type or primary_target
        grasp = self.get_grasp_advice(obj)
        if grasp:
            parts.append(grasp)
        recog = self.get_recognition_hints(primary_target)
        if recog:
            parts.append(recog)
        return "\n".join(parts)

    # ── Consolidate (episode end) ──

    def consolidate(self, success: bool, object_type: str = "") -> None:
        """Merge working memory into long-term YAML files (read-before-write)."""
        grasp_events = [e for e in self.working_memory if e.domain == "grasp"]
        if grasp_events and object_type:
            self._consolidate_grasp(grasp_events, object_type, success)
        recog_events = [e for e in self.working_memory if e.domain == "recognition"]
        if recog_events:
            self._consolidate_recognition(recog_events)

    def _consolidate_grasp(
        self, events: list[MemoryEntry], object_type: str, success: bool,
    ) -> None:
        """Merge grasp events into grasp_experience.yaml."""
        entries = self._load_domain("grasp")
        obj_key = object_type.lower().strip()

        # Find or create entry
        target_entry = None
        for entry in entries:
            if entry.get("object_type", "").lower().strip() == obj_key:
                target_entry = entry
                break
        if target_entry is None:
            target_entry = {
                "object_type": object_type,
                "best_strategy": None,
                "failed": [],
                "total_attempts": 0,
                "success_count": 0,
                "notes": "",
                "last_updated": "",
            }
            entries.append(target_entry)

        # Merge events
        for e in events:
            target_entry["total_attempts"] = target_entry.get("total_attempts", 0) + 1
            if e.event == "strategy_succeeded":
                target_entry["success_count"] = target_entry.get("success_count", 0) + 1
                target_entry["best_strategy"] = e.context.get("strategy")
            elif e.event == "strategy_failed":
                strat = e.context.get("strategy", "")
                reason = e.context.get("failure", "unknown")
                # Find existing failed entry or create new
                existing_fail = None
                for f in target_entry.get("failed", []):
                    if f.get("strategy") == strat and f.get("reason") == reason:
                        existing_fail = f
                        break
                if existing_fail:
                    existing_fail["count"] = existing_fail.get("count", 1) + 1
                else:
                    target_entry.setdefault("failed", []).append(
                        {"strategy": strat, "reason": reason, "count": 1}
                    )

        target_entry["last_updated"] = datetime.now().strftime("%Y-%m-%d")
        self._long_term["grasp"] = entries
        self._save_domain("grasp")

    def _consolidate_recognition(self, events: list[MemoryEntry]) -> None:
        """Merge recognition events into recognition_hints.yaml (placeholder)."""
        # Phase 2 — not implemented yet
        pass

    def _save_domain(self, domain: str) -> None:
        """Write domain entries back to YAML."""
        fpath = self._domain_files.get(domain)
        if not fpath:
            return
        try:
            import yaml
            self.memory_dir.mkdir(parents=True, exist_ok=True)
            fpath.parent.mkdir(parents=True, exist_ok=True)
            with open(fpath, "w", encoding="utf-8") as f:
                yaml.dump(
                    {"entries": self._long_term.get(domain, [])},
                    f, allow_unicode=True, default_flow_style=False,
                    sort_keys=False,
                )
        except Exception as e:
            logger.warning("[memory] failed to save %s: %s", domain, e)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_memory_manager.py -v --tb=short`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/memory_manager.py tests/test_memory_manager.py
git commit -m "feat(memory): MemoryEntry + MemoryManager skeleton with working memory"
```

---

### Task 2: Long-term memory YAML files + load/save tests

**Files:**
- Create: `memory/index.yaml`
- Create: `memory/grasp_experience.yaml`
- Create: `memory/recognition_hints.yaml`
- Test: `tests/test_memory_manager.py` (append)

- [ ] **Step 1: Write failing tests for long-term load/save**

Append to `tests/test_memory_manager.py`:

```python
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
        mm = MemoryManager(memory_dir=Path("/nonexistent/path"))
        assert mm.get_grasp_advice("anything") is None
        assert mm.load_for_task("anything") == ""
```

- [ ] **Step 2: Create initial memory YAML files**

`memory/index.yaml`:
```yaml
version: 1
last_updated: "2026-05-11"
domains:
  grasp: memory/grasp_experience.yaml
  recognition: memory/recognition_hints.yaml
```

`memory/grasp_experience.yaml`:
```yaml
entries: []
```

`memory/recognition_hints.yaml`:
```yaml
entries: []
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/test_memory_manager.py -v --tb=short`
Expected: All 12 tests PASS

- [ ] **Step 4: Commit**

```bash
git add memory/ tests/test_memory_manager.py
git commit -m "feat(memory): long-term YAML files + load/save/consolidate tests"
```

---

### Task 3: Inject memory into grasp strategy prompt

**Files:**
- Modify: `prompts/grasp/select_strategy.txt` (add `{past_experience}` slot)
- Modify: `src/grasp_planner.py:52-93` (pass memory_advice into prompt)
- Test: `tests/test_memory_integration.py`

- [ ] **Step 1: Write failing integration test**

```python
# tests/test_memory_integration.py
"""Integration tests: memory system influences strategy selection."""
from __future__ import annotations

import pytest


class MockLLM:
    """Returns a fixed strategy JSON response."""
    def __init__(self, response: str):
        self._response = response
    def generate(self, prompt, system=""):
        self._last_prompt = prompt
        return self._response


class TestStrategyPromptInjection:
    def test_memory_advice_appears_in_prompt(self):
        """When memory_advice is passed, it appears in the strategy prompt."""
        from src.grasp_planner import GraspPlanner
        from src.world_belief import GraspStrategy, Hypothesis
        import numpy as np

        llm = MockLLM('{"strategy": "top_down", "reasoning": "test", "speech": "test"}')
        planner = GraspPlanner(vlm=None, env=None, llm=llm)

        hyp = Hypothesis(
            object_id="o1", label="tupperware",
            label_alternatives=[("tupperware", 0.9)],
            label_entropy=0.3,
            position_3d=np.array([0.5, 0, 0.9]),
            position_std_m=0.02,
        )
        planner.select_strategy(hyp, memory_advice="tupperware: avoid geometric_centroid (ik_unreachable x2)")
        assert "geometric_centroid" in llm._last_prompt
        assert "ik_unreachable" in llm._last_prompt

    def test_no_memory_advice_still_works(self):
        """Without memory_advice, prompt still works (backward compat)."""
        from src.grasp_planner import GraspPlanner
        from src.world_belief import Hypothesis
        import numpy as np

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
        # No crash, no memory text in prompt
        assert "Past experience" not in llm._last_prompt or "No prior experience" in llm._last_prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_memory_integration.py -v --tb=short`
Expected: FAIL — `select_strategy()` doesn't accept `memory_advice`

- [ ] **Step 3: Modify `prompts/grasp/select_strategy.txt`**

Add before the "Reply with ONLY raw JSON" line:

```
Past experience with this object type: {past_experience}
If past experience indicates a strategy failure, do NOT repeat that strategy.
```

- [ ] **Step 4: Modify `src/grasp_planner.py` — add `memory_advice` parameter**

In `select_strategy()`, add `memory_advice: str = ""` parameter. In the prompt construction, add:

```python
.replace("{past_experience}", memory_advice or "No prior experience with this object.")
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_memory_integration.py tests/test_memory_manager.py -v --tb=short`
Expected: All PASS

- [ ] **Step 6: Run existing grasp strategy tests to check backward compat**

Run: `python -m pytest tests/test_grasp_strategy.py -v --tb=short`
Expected: All PASS (no regressions)

- [ ] **Step 7: Commit**

```bash
git add prompts/grasp/select_strategy.txt src/grasp_planner.py tests/test_memory_integration.py
git commit -m "feat(memory): inject past experience into grasp strategy prompt"
```

---

### Task 4: Wire MemoryManager into Agent

**Files:**
- Modify: `src/world_belief.py:204-211` (add working_memory field)
- Modify: `src/agent.py:27-45` (init MemoryManager)
- Modify: `src/agent.py:367-401` (load at start, consolidate at end)
- Modify: `src/agent.py:436-512` (record events on grasp success/failure, pass memory_advice to select_strategy)

- [ ] **Step 1: Add working_memory to WorldBelief**

In `src/world_belief.py`, in the `WorldBelief` class `__init__` or dataclass fields, add:

```python
from src.memory_manager import MemoryEntry
# In WorldBelief.__init__ or as field:
self.working_memory: list[MemoryEntry] = []
```

(Exact insertion point depends on whether WorldBelief is a dataclass or regular class — check the file. It uses `__init__`, so add `self.working_memory = []` in `__init__`.)

- [ ] **Step 2: Modify agent `__init__` to accept optional MemoryManager**

In `src/agent.py`, add to `__init__`:

```python
from src.memory_manager import MemoryManager
# Add parameter:
def __init__(self, ..., memory_manager: Optional[MemoryManager] = None):
    ...
    self.memory = memory_manager or MemoryManager()
```

- [ ] **Step 3: Modify `agent.run()` — load memory at episode start**

After `belief.decomposed = ...`, add:

```python
prior_knowledge = self.memory.load_for_task(
    belief.decomposed.primary_target if belief.decomposed else "",
)
if prior_knowledge:
    logger.info("[agent] loaded prior knowledge:\n%s", prior_knowledge)
```

- [ ] **Step 4: Modify `agent.run()` — consolidate at episode end**

Before every `return` in `run()` (success, giveup, MAX_STEPS), add:

```python
target_hyp = belief.target()
self.memory.consolidate(
    success=<True or False>,
    object_type=target_hyp.label if target_hyp else "",
)
```

- [ ] **Step 5: Modify `_execute_action` grasp branch — record events**

In the `elif action.kind == "grasp":` block of `_execute_action`, after `action.target_hypothesis.grasp_attempts.append(result.attempt)`, add:

```python
hyp = action.target_hypothesis
strategy_name = (hyp.grasp_strategy.strategy if hyp.grasp_strategy else "unknown")
from src.memory_manager import MemoryEntry
if result.attempt.failure_mode == "success":
    self.memory.record_event(MemoryEntry(
        step=len(belief.action_history),
        domain="grasp", event="strategy_succeeded",
        context={"strategy": strategy_name, "object": hyp.label},
        lesson=f"{hyp.label}: {strategy_name} succeeded",
    ))
else:
    self.memory.record_event(MemoryEntry(
        step=len(belief.action_history),
        domain="grasp", event="strategy_failed",
        context={
            "strategy": strategy_name,
            "failure": result.attempt.failure_mode,
            "object": hyp.label,
        },
        lesson=f"{hyp.label}: {strategy_name} failed ({result.attempt.failure_mode}), avoid this strategy",
    ))
```

- [ ] **Step 6: Modify `_execute_action` plan_grasp_candidates branch — pass memory_advice**

In the `elif action.kind == "plan_grasp_candidates":` block, change `select_strategy` call:

```python
# Build memory advice
grasp_advice = self.memory.get_grasp_advice(hyp.label) or ""
working_advice = self.memory.get_working_summary(domain="grasp")
memory_advice = "\n".join(filter(None, [grasp_advice, working_advice]))

strategy = self.grasp_planner.select_strategy(hyp, memory_advice=memory_advice)
```

- [ ] **Step 7: Run full test suite**

Run: `python -m pytest tests/ --tb=short 2>&1 | Select-String "passed|failed"`
Expected: All tests pass, no regressions

- [ ] **Step 8: Commit**

```bash
git add src/agent.py src/world_belief.py
git commit -m "feat(memory): wire MemoryManager into agent lifecycle (load/record/consolidate)"
```

---

### Task 5: End-to-end test + push

**Files:**
- Test: `tests/test_memory_integration.py` (append)

- [ ] **Step 1: Write end-to-end memory test**

Append to `tests/test_memory_integration.py`:

```python
class TestEndToEndMemoryFlow:
    def test_failed_strategy_recorded_in_working_memory(self):
        """After grasp failure, working memory contains the failure entry."""
        from src.memory_manager import MemoryEntry, MemoryManager
        import tempfile
        from pathlib import Path

        mm = MemoryManager(memory_dir=Path(tempfile.mkdtemp()))
        # Simulate agent recording a failure
        mm.record_event(MemoryEntry(
            step=5, domain="grasp", event="strategy_failed",
            context={"strategy": "geometric_centroid", "failure": "ik_unreachable", "object": "tupperware"},
            lesson="tupperware: geometric_centroid failed (ik_unreachable), avoid this strategy",
        ))
        # Working summary should contain the lesson
        summary = mm.get_working_summary(domain="grasp")
        assert "geometric_centroid" in summary
        assert "ik_unreachable" in summary

    def test_consolidate_then_load_round_trip(self):
        """Consolidate working memory → reload → advice available."""
        from src.memory_manager import MemoryEntry, MemoryManager
        import tempfile, yaml
        from pathlib import Path

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
```

- [ ] **Step 2: Run full test suite**

Run: `python -m pytest tests/ --tb=short 2>&1 | Select-String "passed|failed"`
Expected: All tests pass

- [ ] **Step 3: Commit and push**

```bash
git add -A
git commit -m "feat(memory): dual-store episodic memory system (Phase 1)

- MemoryManager with working memory (in-episode) + long-term (YAML)
- Grasp failures/successes recorded in real-time
- Past experience injected into strategy selection prompt
- Consolidation: working → long-term at episode end
- Self-healing: read-before-write merge

Fixes: 001/007/009 grasp strategy regression"
git push
```

- [ ] **Step 4: Run batch eval on server**

```bash
cd ~/embodied-AI-one && git pull
nohup python eval/run_batch.py --parallel 4 > batch_out4.txt 2>&1 &
```

Expected: scenarios 001, 007, 009 should now pass (agent retries with different strategy after failure), bringing success rate to ≥ 8/11 (73%).

---

## Summary

| Task | What | Est. Time |
|------|------|-----------|
| 1 | MemoryEntry + MemoryManager skeleton + working memory | 10 min |
| 2 | Long-term YAML files + load/save/consolidate | 10 min |
| 3 | Inject memory into strategy prompt | 10 min |
| 4 | Wire into agent lifecycle | 15 min |
| 5 | E2E test + push + batch eval | 10 min |
| **Total** | | **~55 min** |
