# Memory Phase 2 — Recognition Hints Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the write-side of `recognition_hints` memory so that `synonym_effective` and `label_corrected` events recorded during an episode are consolidated (on grasp success) into `memory/recognition_hints.yaml` and re-injected as `primary_target_synonyms` at the start of the next episode targeting the same primary.

**Architecture:** Extend Phase 1's `MemoryManager` with a `_consolidate_recognition` branch and a `get_recognition_hints_synonyms()` reader. `perception.observe()` writes a `clip_injected` block into Evidence `raw_payload` when CLIP injection fires with a non-primary best query. `agent._merge_hypotheses_from_evidence` detects that block and records a `synonym_effective` working-memory entry. `agent._llm_semantic_fallback` records a `label_corrected` entry directly. `agent.run()` merges the persisted synonyms into `belief.decomposed.primary_target_synonyms` at episode start.

**Tech Stack:** Python stdlib (`dataclasses`, `datetime`, `pathlib`, `yaml`), pytest, existing `MemoryManager` / `MemoryEntry` / `Evidence` / `WorldBelief` types. No new dependencies.

---

## File Structure

- Modify: `src/memory_manager.py`
  - Adds `_consolidate_recognition(events: list[MemoryEntry]) -> None`
  - Adds `get_recognition_hints_synonyms(target: str) -> list[str]`
  - Extends `consolidate()` to dispatch recognition (only when `success=True`)
- Modify: `src/perception.py`
  - `_inject_clip_scores()` returns the chosen injection info (or `None`) instead of always returning `None`
  - `observe()` puts that info into `Evidence.raw_payload["clip_injected"]` when present and `best_q != primary_target`
- Modify: `src/agent.py`
  - `_merge_hypotheses_from_evidence` detects `clip_injected` and records a `synonym_effective` event
  - `_llm_semantic_fallback` records a `label_corrected` event when it injects a match
  - `run()` calls `get_recognition_hints_synonyms()` after `load_for_task()` and merges the result into `belief.decomposed.primary_target_synonyms`
- Modify: `tests/test_memory_manager.py`
  - New `TestRecognitionConsolidation` class
  - New `TestRecognitionHints` class for `get_recognition_hints_synonyms`
- Modify: `tests/test_perception.py`
  - New `TestClipInjectedEvidence` class covering the `raw_payload["clip_injected"]` block
- Modify: `tests/test_semantic_fallback.py`
  - New tests asserting working-memory writes for both CLIP and LLM-fallback paths
- Modify: `tests/test_memory_integration.py`
  - New `TestRecognitionRoundTrip` class for episode-N → episode-N+1 synonym persistence

---

## Task 1: Add `_consolidate_recognition` to `MemoryManager`

**Files:**
- Modify: `src/memory_manager.py`
- Modify: `tests/test_memory_manager.py`

- [ ] **Step 1: Write failing tests for recognition consolidation**

Append this class to the end of `tests/test_memory_manager.py`:

```python
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
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_memory_manager.py::TestRecognitionConsolidation -v
```

Expected: all 6 tests FAIL (KeyError / empty entries / etc.) — recognition branch does not exist yet.

- [ ] **Step 3: Implement `_consolidate_recognition` and extend `consolidate()` in `src/memory_manager.py`**

Replace the `consolidate` method body (currently lines 142-146):

```python
    def consolidate(self, success: bool, object_type: str = "") -> None:
        # grasp: always (Phase 1 contract — records failures too)
        grasp_events = [e for e in self.working_memory if e.domain == "grasp"]
        if grasp_events and object_type:
            self._consolidate_grasp(grasp_events, object_type, success)
        # recognition: only on success (avoids固化误命中)
        if success:
            recognition_events = [
                e for e in self.working_memory if e.domain == "recognition"
            ]
            if recognition_events:
                self._consolidate_recognition(recognition_events)
```

Add a class-level constant near the top of the `MemoryManager` class (right before `__init__`, around line 35):

```python
    _MAX_SYNONYMS_PER_TARGET = 5
```

Then add `_consolidate_recognition` right after `_consolidate_grasp` (before `_save_domain`):

```python
    def _consolidate_recognition(self, events: list[MemoryEntry]) -> None:
        entries = self._load_domain("recognition")

        # group events by target
        from collections import defaultdict
        by_target: dict[str, list[MemoryEntry]] = defaultdict(list)
        for e in events:
            tgt = str(e.context.get("target", "")).strip().lower()
            if tgt:
                by_target[tgt].append(e)

        for target, evts in by_target.items():
            entry = None
            for ex in entries:
                if str(ex.get("target", "")).strip().lower() == target:
                    entry = ex
                    break
            if entry is None:
                entry = {
                    "target": target,
                    "vlm_common_labels": [],
                    "effective_synonyms": [],
                    "clip_helpful": False,
                    "notes": "",
                    "last_updated": "",
                }
                entries.append(entry)

            # dedupe within this episode by (synonym | detected_label)
            seen_in_episode: set[tuple[str, str]] = set()
            for e in evts:
                ctx = e.context
                if e.event == "synonym_effective":
                    syn = str(ctx.get("synonym", "")).strip().lower()
                    method = "clip"
                    vlm_label = str(ctx.get("vlm_label", "")).strip().lower()
                elif e.event == "label_corrected":
                    syn = str(ctx.get("detected_label", "")).strip().lower()
                    method = str(ctx.get("method", "llm"))
                    vlm_label = syn
                else:
                    continue
                if not syn or syn == target:
                    continue
                key = (e.event, syn)
                if key in seen_in_episode:
                    continue
                seen_in_episode.add(key)

                # vlm_common_labels set semantics
                if vlm_label and vlm_label not in entry["vlm_common_labels"]:
                    entry["vlm_common_labels"].append(vlm_label)

                # effective_synonyms count merge
                existing_syn = next(
                    (s for s in entry["effective_synonyms"] if s["name"] == syn),
                    None,
                )
                if existing_syn:
                    existing_syn["count"] = existing_syn.get("count", 1) + 1
                    existing_syn["last_method"] = method
                else:
                    entry["effective_synonyms"].append(
                        {"name": syn, "count": 1, "last_method": method},
                    )

                if method == "clip":
                    entry["clip_helpful"] = True

            # sort and cap
            entry["effective_synonyms"].sort(
                key=lambda s: s["count"], reverse=True,
            )
            entry["effective_synonyms"] = (
                entry["effective_synonyms"][:self._MAX_SYNONYMS_PER_TARGET]
            )
            entry["last_updated"] = datetime.now().strftime("%Y-%m-%d")

        self._long_term["recognition"] = entries
        self._save_domain("recognition")
```

- [ ] **Step 4: Run tests to verify pass**

Run:

```powershell
python -m pytest tests/test_memory_manager.py::TestRecognitionConsolidation -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Run full memory test file to catch regressions**

Run:

```powershell
python -m pytest tests/test_memory_manager.py -v
```

Expected: all tests PASS (Phase 1 tests still green).

- [ ] **Step 6: Commit**

```powershell
git add src/memory_manager.py tests/test_memory_manager.py
git commit -m "feat(memory): add recognition consolidation (synonym/label events)"
```

---

## Task 2: Add `get_recognition_hints_synonyms` reader

**Files:**
- Modify: `src/memory_manager.py`
- Modify: `tests/test_memory_manager.py`

- [ ] **Step 1: Write failing tests**

Append this class to `tests/test_memory_manager.py`:

```python
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
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_memory_manager.py::TestRecognitionHints -v
```

Expected: 4 tests FAIL with `AttributeError: 'MemoryManager' object has no attribute 'get_recognition_hints_synonyms'`.

- [ ] **Step 3: Add `get_recognition_hints_synonyms` to `src/memory_manager.py`**

Insert after `get_recognition_hints` (current line 127):

```python
    def get_recognition_hints_synonyms(self, target: str) -> list[str]:
        """Return historical effective synonyms for target (sorted by count desc).

        Returns [] if no entry or load fails.
        """
        entries = self._load_domain("recognition")
        tgt_key = target.lower().strip()
        for entry in entries:
            if str(entry.get("target", "")).lower().strip() == tgt_key:
                syns = entry.get("effective_synonyms", []) or []
                # already stored sorted by count desc; defensive re-sort
                ordered = sorted(
                    syns, key=lambda s: s.get("count", 0), reverse=True,
                )
                return [str(s["name"]) for s in ordered if s.get("name")]
        return []
```

- [ ] **Step 4: Run tests to verify pass**

Run:

```powershell
python -m pytest tests/test_memory_manager.py::TestRecognitionHints -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/memory_manager.py tests/test_memory_manager.py
git commit -m "feat(memory): add get_recognition_hints_synonyms reader"
```

---

## Task 3: Surface CLIP injection info in Evidence

**Files:**
- Modify: `src/perception.py`
- Modify: `tests/test_perception.py`

- [ ] **Step 1: Write failing test**

Append this class to `tests/test_perception.py`:

```python
class TestClipInjectedEvidence:
    def _make_grounder_with_clip(self, scores: dict):
        from src.perception import QueryAwareGrounder
        from src.vlm_cache import VLMCache
        from tests.test_semantic_fallback import MockCLIPScorer

        return QueryAwareGrounder(
            vlm=MockVLM([]), llm=MockLLM([]),
            cache=VLMCache(),
            label_temperature=1.0,
            clip_scorer=MockCLIPScorer(scores=scores),
        )

    def _make_belief(self, primary: str, synonyms=None):
        from src.world_belief import DecomposedTask, WorldBelief
        b = WorldBelief(user_query=f"pick the {primary}")
        b.decomposed = DecomposedTask(
            primary_target=primary,
            primary_target_synonyms=list(synonyms or []),
        )
        return b

    def test_observe_emits_clip_injected_when_synonym_hits(self, tmp_image):
        # VLM returns 'orange' but user wants 'tangerine'; CLIP synonym 'orange' hits
        vlm = MockVLM([_make_vlm_json([
            {"bbox_2d": [0, 0, 50, 50], "label": "orange",
             "alternatives": [["orange", 1.0]],
             "confidence": 0.9, "visible_features": "round"},
        ])])
        from src.perception import QueryAwareGrounder
        from src.vlm_cache import VLMCache
        from tests.test_semantic_fallback import MockCLIPScorer

        g = QueryAwareGrounder(
            vlm=vlm, llm=MockLLM([]),
            cache=VLMCache(),
            label_temperature=1.0,
            clip_scorer=MockCLIPScorer(scores={
                "tangerine": [0.10],   # primary below strict 0.23
                "orange":    [0.30],   # synonym above relaxed 0.20
            }),
        )
        belief = self._make_belief("tangerine", synonyms=["orange"])

        class _FakeEnv:
            def observe(self, vp):
                class _Obs:
                    image_path = tmp_image
                return _Obs()

        ev = g.observe(viewpoint=None, env=_FakeEnv(), belief=belief)
        assert "clip_injected" in ev.raw_payload
        info = ev.raw_payload["clip_injected"]
        assert info["target"] == "tangerine"
        assert info["synonym"] == "orange"
        assert info["sim"] > 0.20
        assert info["vlm_label"] == "orange"

    def test_observe_omits_clip_injected_when_primary_self_hits(self, tmp_image):
        """best_q == primary → no synonym knowledge → field absent."""
        vlm = MockVLM([_make_vlm_json([
            {"bbox_2d": [0, 0, 50, 50], "label": "fruit",
             "alternatives": [["fruit", 1.0]],
             "confidence": 0.9, "visible_features": "round"},
        ])])
        from src.perception import QueryAwareGrounder
        from src.vlm_cache import VLMCache
        from tests.test_semantic_fallback import MockCLIPScorer

        g = QueryAwareGrounder(
            vlm=vlm, llm=MockLLM([]),
            cache=VLMCache(),
            label_temperature=1.0,
            clip_scorer=MockCLIPScorer(scores={"cake": [0.30]}),
        )
        belief = self._make_belief("cake")

        class _FakeEnv:
            def observe(self, vp):
                class _Obs:
                    image_path = tmp_image
                return _Obs()

        ev = g.observe(viewpoint=None, env=_FakeEnv(), belief=belief)
        assert "clip_injected" not in ev.raw_payload

    def test_observe_omits_clip_injected_when_no_injection(self, tmp_image):
        """All CLIP scores below threshold → no field."""
        vlm = MockVLM([_make_vlm_json([
            {"bbox_2d": [0, 0, 50, 50], "label": "fruit",
             "alternatives": [["fruit", 1.0]],
             "confidence": 0.9, "visible_features": "round"},
        ])])
        from src.perception import QueryAwareGrounder
        from src.vlm_cache import VLMCache
        from tests.test_semantic_fallback import MockCLIPScorer

        g = QueryAwareGrounder(
            vlm=vlm, llm=MockLLM([]),
            cache=VLMCache(),
            label_temperature=1.0,
            clip_scorer=MockCLIPScorer(scores={
                "tangerine": [0.05], "orange": [0.05],
            }),
        )
        belief = self._make_belief("tangerine", synonyms=["orange"])

        class _FakeEnv:
            def observe(self, vp):
                class _Obs:
                    image_path = tmp_image
                return _Obs()

        ev = g.observe(viewpoint=None, env=_FakeEnv(), belief=belief)
        assert "clip_injected" not in ev.raw_payload
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_perception.py::TestClipInjectedEvidence -v
```

Expected: 3 tests FAIL — `"clip_injected" in ev.raw_payload` is False because the field is never set.

- [ ] **Step 3: Change `_inject_clip_scores` signature to return info, update `observe`**

In `src/perception.py`, change `_inject_clip_scores` return type from `None` to `Optional[dict]`. Locate the method (lines 236-309). Replace the method's tail (`return` statement at the end after `logger.info(...)`) so the **full revised method** is:

```python
    def _inject_clip_scores(
        self, hyps: list[Hypothesis], image_path: str, primary_target: str,
        synonyms: Optional[list[str]] = None,
    ) -> Optional[dict]:
        """用 CLIP 视觉相似度将 primary_target 注入 VLM 未识别的 hypothesis。

        Returns: 命中时 {"target","synonym","sim","vlm_label"}; 否则 None。
                 注: 当 best_q == primary_target 时仍返回 None
                 (synonym 字段无知识可记)。
        """
        synonyms = synonyms or []
        target_key = _label_key(primary_target)
        synonym_keys = [_label_key(s) for s in synonyms if _label_key(s)]
        all_keys = [target_key] + synonym_keys
        already_found = any(
            any(k and k in _label_key(h.label) for k in all_keys)
            or any(
                any(k and k in _label_key(lbl) for k in all_keys)
                for lbl, _ in h.label_alternatives
            )
            for h in hyps
        )
        if already_found:
            return None

        vp_name = hyps[0].observed_in_views[0] if hyps[0].observed_in_views else "v0"
        bboxes = [h.bbox_per_view.get(vp_name, (0, 0, 0, 0)) for h in hyps]

        queries = [primary_target] + list(synonyms)
        per_bbox_max: list[tuple[float, str]] = [
            (0.0, primary_target) for _ in bboxes
        ]
        try:
            all_scores = self._clip_scorer.score_crops_multi(
                image_path, bboxes, queries,
            )
        except Exception as e:
            logger.debug("[clip] multi-query failed: %s", e)
            return None
        for qi, q in enumerate(queries):
            for i, s in enumerate(all_scores[qi]):
                if s > per_bbox_max[i][0]:
                    per_bbox_max[i] = (float(s), q)

        from src.clip_scorer import CLIPScorer
        strict_thr = CLIPScorer.INJECT_THRESHOLD
        relaxed_thr = max(0.20, strict_thr - 0.03)

        best_idx, best_score, best_q = -1, 0.0, primary_target
        for i, (score, q) in enumerate(per_bbox_max):
            thr = strict_thr if q == primary_target else relaxed_thr
            if score >= thr and score > best_score:
                best_idx, best_score, best_q = i, score, q
        if best_idx < 0:
            return None

        h = hyps[best_idx]
        inject_prob = max(0.35, float(best_score))
        h.label_alternatives.append((primary_target, inject_prob))
        total = sum(p for _, p in h.label_alternatives) or 1.0
        h.label_alternatives = sorted(
            ((lbl, p / total) for lbl, p in h.label_alternatives),
            key=lambda x: x[1], reverse=True,
        )
        h.label_entropy = _shannon([p for _, p in h.label_alternatives])
        logger.info(
            "[clip] injected '%s' via query='%s' (sim=%.3f, prob=%.2f) "
            "into %s (label='%s')",
            primary_target, best_q, best_score, inject_prob, h.object_id, h.label,
        )
        # Only surface synonym knowledge when best_q != primary
        if best_q == primary_target:
            return None
        return {
            "target": primary_target,
            "synonym": best_q,
            "sim": float(best_score),
            "vlm_label": h.label,
        }
```

Then in `observe()` (around lines 142-154), change the call site to capture and stash the info:

Old:

```python
        hyps = self._parse_to_hypotheses(raw, viewpoint, env)
        # CLIP semantic injection: 当 VLM 标签不含 target 时，用视觉相似度补救
        if self._clip_scorer and primary and hyps:
            self._inject_clip_scores(hyps, image_path, primary, synonyms)
        return Evidence(
            source="vlm_ground", timestamp=time.time(),
            raw_payload={
                "viewpoint": getattr(viewpoint, "name", str(viewpoint)),
                "hypotheses": [self._hyp_to_dict(h) for h in hyps],
                "image_path": image_path,
                "raw_vlm_text": raw[:1000],
            },
        )
```

New:

```python
        hyps = self._parse_to_hypotheses(raw, viewpoint, env)
        clip_info: Optional[dict] = None
        # CLIP semantic injection: 当 VLM 标签不含 target 时，用视觉相似度补救
        if self._clip_scorer and primary and hyps:
            clip_info = self._inject_clip_scores(
                hyps, image_path, primary, synonyms,
            )
        payload: dict[str, Any] = {
            "viewpoint": getattr(viewpoint, "name", str(viewpoint)),
            "hypotheses": [self._hyp_to_dict(h) for h in hyps],
            "image_path": image_path,
            "raw_vlm_text": raw[:1000],
        }
        if clip_info is not None:
            payload["clip_injected"] = clip_info
        return Evidence(
            source="vlm_ground", timestamp=time.time(),
            raw_payload=payload,
        )
```

- [ ] **Step 4: Run new tests to verify pass**

Run:

```powershell
python -m pytest tests/test_perception.py::TestClipInjectedEvidence -v
```

Expected: 3 tests PASS.

- [ ] **Step 5: Run full perception + semantic fallback suites to catch regressions**

Run:

```powershell
python -m pytest tests/test_perception.py tests/test_semantic_fallback.py -v
```

Expected: all tests PASS (existing CLIP-injection tests still green — they don't assert on return value).

- [ ] **Step 6: Commit**

```powershell
git add src/perception.py tests/test_perception.py
git commit -m "feat(perception): surface clip_injected info in evidence payload"
```

---

## Task 4: Agent records `synonym_effective` from CLIP evidence

**Files:**
- Modify: `src/agent.py`
- Modify: `tests/test_semantic_fallback.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_semantic_fallback.py`:

```python
class TestAgentRecordsCLIPEvent:
    def test_merge_evidence_records_synonym_effective(self):
        import tempfile
        from pathlib import Path
        import time
        from src.agent import EmboSightAgent
        from src.memory_manager import MemoryManager
        from src.world_belief import (
            DecomposedTask, Evidence, Hypothesis, WorldBelief,
        )

        mm = MemoryManager(memory_dir=Path(tempfile.mkdtemp()))
        agent = EmboSightAgent.__new__(EmboSightAgent)
        agent.memory = mm
        agent.logger = None

        belief = WorldBelief(user_query="pick tangerine")
        belief.decomposed = DecomposedTask(primary_target="tangerine")

        # Forge an evidence with clip_injected payload + 1 hypothesis dict
        ev = Evidence(
            source="vlm_ground", timestamp=time.time(),
            raw_payload={
                "hypotheses": [{
                    "object_id": "obj_0",
                    "label": "orange",
                    "label_alternatives": [["orange", 0.6], ["tangerine", 0.4]],
                    "label_entropy": 0.5,
                    "position_3d": [0.5, 0.0, 0.9],
                    "position_std_m": 0.02,
                    "bbox_per_view": {"v0": [0, 0, 50, 50]},
                    "observed_in_views": ["v0"],
                }],
                "clip_injected": {
                    "target": "tangerine",
                    "synonym": "orange",
                    "sim": 0.30,
                    "vlm_label": "orange",
                },
            },
        )

        agent._merge_hypotheses_from_evidence(belief, ev)

        recog = [e for e in mm.working_memory if e.domain == "recognition"]
        assert len(recog) == 1
        assert recog[0].event == "synonym_effective"
        assert recog[0].context["target"] == "tangerine"
        assert recog[0].context["synonym"] == "orange"

    def test_merge_evidence_dedups_same_episode_synonym(self):
        import tempfile
        from pathlib import Path
        import time
        from src.agent import EmboSightAgent
        from src.memory_manager import MemoryManager
        from src.world_belief import (
            DecomposedTask, Evidence, Hypothesis, WorldBelief,
        )

        mm = MemoryManager(memory_dir=Path(tempfile.mkdtemp()))
        agent = EmboSightAgent.__new__(EmboSightAgent)
        agent.memory = mm
        agent.logger = None

        belief = WorldBelief(user_query="pick tangerine")
        belief.decomposed = DecomposedTask(primary_target="tangerine")

        clip_info = {"target": "tangerine", "synonym": "orange",
                     "sim": 0.30, "vlm_label": "orange"}
        hyp_dict = {
            "object_id": "obj_0", "label": "orange",
            "label_alternatives": [["orange", 1.0]],
            "label_entropy": 0.0,
            "position_3d": [0.5, 0.0, 0.9],
            "position_std_m": 0.02,
            "bbox_per_view": {"v0": [0, 0, 50, 50]},
            "observed_in_views": ["v0"],
        }
        ev1 = Evidence(source="vlm_ground", timestamp=time.time(),
                        raw_payload={"hypotheses": [hyp_dict], "clip_injected": clip_info})
        ev2 = Evidence(source="vlm_ground", timestamp=time.time(),
                        raw_payload={"hypotheses": [hyp_dict], "clip_injected": clip_info})

        agent._merge_hypotheses_from_evidence(belief, ev1)
        agent._merge_hypotheses_from_evidence(belief, ev2)

        recog = [e for e in mm.working_memory if e.domain == "recognition"]
        assert len(recog) == 1  # second call deduped
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_semantic_fallback.py::TestAgentRecordsCLIPEvent -v
```

Expected: both tests FAIL — `recog` is empty because `_merge_hypotheses_from_evidence` does not look at `clip_injected`.

- [ ] **Step 3: Extend `_merge_hypotheses_from_evidence` in `src/agent.py`**

Locate the method (currently lines 605-619). Replace it with:

```python
    def _merge_hypotheses_from_evidence(
        self, belief: WorldBelief, ev: Evidence,
    ) -> None:
        if ev.source != "vlm_ground":
            return
        new_hyps_data = ev.raw_payload.get("hypotheses", [])
        for h_dict in new_hyps_data:
            new_h = self._dict_to_hypothesis(h_dict)
            merged = False
            for existing in belief.hypotheses:
                if belief.merge_hypothesis(existing, new_h):
                    merged = True
                    break
            if not merged:
                belief.add_hypothesis(new_h)

        # Recognition memory: CLIP synonym hit
        clip_info = ev.raw_payload.get("clip_injected")
        if clip_info:
            self._record_recognition_synonym(clip_info)

    def _record_recognition_synonym(self, info: dict) -> None:
        target = str(info.get("target", "")).strip().lower()
        synonym = str(info.get("synonym", "")).strip().lower()
        if not target or not synonym or synonym == target:
            return
        # dedupe within episode by (target, synonym)
        for e in self.memory.working_memory:
            if (e.domain == "recognition"
                    and e.event == "synonym_effective"
                    and e.context.get("target") == target
                    and e.context.get("synonym") == synonym):
                return
        self.memory.record_event(MemoryEntry(
            step=len(self.memory.working_memory),
            domain="recognition",
            event="synonym_effective",
            context={
                "target": target,
                "synonym": synonym,
                "sim": float(info.get("sim", 0.0)),
                "vlm_label": str(info.get("vlm_label", "")).strip().lower(),
            },
            lesson=f"{target}: CLIP hit via '{synonym}' (sim={info.get('sim', 0.0):.2f})",
        ))
```

- [ ] **Step 4: Run new tests to verify pass**

Run:

```powershell
python -m pytest tests/test_semantic_fallback.py::TestAgentRecordsCLIPEvent -v
```

Expected: both tests PASS.

- [ ] **Step 5: Run full agent test files to catch regressions**

Run:

```powershell
python -m pytest tests/test_agent_decide_next.py tests/test_agent_run.py tests/test_semantic_fallback.py tests/test_memory_integration.py -v
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```powershell
git add src/agent.py tests/test_semantic_fallback.py
git commit -m "feat(agent): record recognition synonym_effective from CLIP evidence"
```

---

## Task 5: Agent records `label_corrected` in LLM semantic fallback

**Files:**
- Modify: `src/agent.py`
- Modify: `tests/test_semantic_fallback.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_semantic_fallback.py`:

```python
class TestAgentRecordsLLMFallbackEvent:
    def _make_minimal_agent(self, llm_answer: str):
        import tempfile
        from pathlib import Path
        from src.agent import EmboSightAgent
        from src.memory_manager import MemoryManager

        class _StaticLLM:
            def __init__(self, ans):
                self._ans = ans
            def generate(self, prompt, system=""):
                return self._ans

        agent = EmboSightAgent.__new__(EmboSightAgent)
        agent.memory = MemoryManager(memory_dir=Path(tempfile.mkdtemp()))
        agent.logger = None
        agent.llm = _StaticLLM(llm_answer)
        return agent

    def test_llm_fallback_records_label_corrected(self):
        from src.world_belief import (
            DecomposedTask, Hypothesis, WorldBelief,
        )
        import numpy as np

        agent = self._make_minimal_agent("container")
        belief = WorldBelief(user_query="pick yogurt")
        belief.decomposed = DecomposedTask(primary_target="yogurt")
        belief.hypotheses = [Hypothesis(
            object_id="o1", label="container",
            label_alternatives=[("container", 1.0)],
            label_entropy=0.0,
            position_3d=np.array([0.5, 0.0, 0.9]),
            position_std_m=0.02,
        )]

        ok = agent._llm_semantic_fallback(belief)
        assert ok is True

        recog = [e for e in agent.memory.working_memory if e.domain == "recognition"]
        assert len(recog) == 1
        assert recog[0].event == "label_corrected"
        assert recog[0].context["target"] == "yogurt"
        assert recog[0].context["detected_label"] == "container"
        assert recog[0].context["method"] == "llm"

    def test_llm_fallback_no_record_on_none(self):
        from src.world_belief import (
            DecomposedTask, Hypothesis, WorldBelief,
        )
        import numpy as np

        agent = self._make_minimal_agent("none")
        belief = WorldBelief(user_query="pick yogurt")
        belief.decomposed = DecomposedTask(primary_target="yogurt")
        belief.hypotheses = [Hypothesis(
            object_id="o1", label="apple",
            label_alternatives=[("apple", 1.0)],
            label_entropy=0.0,
            position_3d=np.array([0.5, 0.0, 0.9]),
            position_std_m=0.02,
        )]

        ok = agent._llm_semantic_fallback(belief)
        assert ok is False
        recog = [e for e in agent.memory.working_memory if e.domain == "recognition"]
        assert recog == []
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_semantic_fallback.py::TestAgentRecordsLLMFallbackEvent -v
```

Expected: `test_llm_fallback_records_label_corrected` FAILS (no record); the other may pass coincidentally — the failing one is what matters.

- [ ] **Step 3: Add record call inside `_llm_semantic_fallback`**

Locate `_llm_semantic_fallback` in `src/agent.py` (lines 288-364). Just before the final `return True` (the one after the `matched_h.label_alternatives.append(...)` block and logger.info — around line 364), insert:

```python
        # Recognition memory: LLM corrected label
        self.memory.record_event(MemoryEntry(
            step=len(self.memory.working_memory),
            domain="recognition",
            event="label_corrected",
            context={
                "target": primary.strip().lower(),
                "detected_label": answer.strip().lower(),
                "method": "llm",
            },
            lesson=f"{primary}: LLM matched detected '{answer}'",
        ))
```

Also add the same record in the early-return branch that handles `already`. Locate the `if already:` block (lines 332-351). After `return True` at line 351, change the control flow so the record is added before any `return True` path. The simplest patch: extract a small helper or add the record before each `return True`.

Concrete edit: at line 351 (inside the `if already:` block), replace the bare `return True` with:

```python
            self.memory.record_event(MemoryEntry(
                step=len(self.memory.working_memory),
                domain="recognition",
                event="label_corrected",
                context={
                    "target": primary.strip().lower(),
                    "detected_label": answer.strip().lower(),
                    "method": "llm",
                },
                lesson=f"{primary}: LLM matched detected '{answer}' (already present)",
            ))
            return True
```

Note: agent.py already imports `MemoryEntry` (line 23), so no new import needed.

- [ ] **Step 4: Run tests to verify pass**

Run:

```powershell
python -m pytest tests/test_semantic_fallback.py::TestAgentRecordsLLMFallbackEvent -v
```

Expected: both tests PASS.

- [ ] **Step 5: Run full semantic fallback suite**

Run:

```powershell
python -m pytest tests/test_semantic_fallback.py -v
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```powershell
git add src/agent.py tests/test_semantic_fallback.py
git commit -m "feat(agent): record recognition label_corrected from llm fallback"
```

---

## Task 6: Inject persisted synonyms into `belief.decomposed.primary_target_synonyms` at episode start

**Files:**
- Modify: `src/agent.py`
- Modify: `tests/test_memory_integration.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_memory_integration.py`:

```python
class TestRecognitionSynonymInjection:
    def _make_memory_with_synonym(self, target: str, synonym: str):
        import tempfile
        import yaml
        from pathlib import Path
        from src.memory_manager import MemoryManager

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
                "effective_synonyms": [{"name": synonym, "count": 2, "last_method": "clip"}],
                "clip_helpful": True,
                "notes": "",
                "last_updated": "2026-05-13",
            }],
        }), encoding="utf-8")
        return MemoryManager(memory_dir=d)

    def test_run_start_injects_persisted_synonym(self):
        from src.agent import EmboSightAgent
        from src.world_belief import DecomposedTask, WorldBelief

        mm = self._make_memory_with_synonym("tangerine", "orange")
        agent = EmboSightAgent.__new__(EmboSightAgent)
        agent.memory = mm
        agent.logger = None

        belief = WorldBelief(user_query="pick tangerine")
        belief.decomposed = DecomposedTask(
            primary_target="tangerine",
            primary_target_synonyms=["mandarin"],   # LLM gave one
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
        agent.logger = None

        belief = WorldBelief(user_query="pick tangerine")
        belief.decomposed = DecomposedTask(
            primary_target="tangerine",
            primary_target_synonyms=["orange"],   # already has it
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
        agent.logger = None

        belief = WorldBelief(user_query="")
        belief.decomposed = None
        agent._apply_recognition_hints(belief)  # should not raise
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_memory_integration.py::TestRecognitionSynonymInjection -v
```

Expected: 3 tests FAIL — `AttributeError: 'EmboSightAgent' object has no attribute '_apply_recognition_hints'`.

- [ ] **Step 3: Add helper and wire it into `run()`**

In `src/agent.py`, add this helper right above `_consolidate_memory` (around line 661):

```python
    def _apply_recognition_hints(self, belief: WorldBelief) -> None:
        """Merge persisted recognition synonyms into belief.decomposed.primary_target_synonyms.

        Idempotent: dedupes case-insensitively against existing entries.
        No-op when decomposed is absent.
        """
        if not belief.decomposed or not belief.decomposed.primary_target:
            return
        hint_syns = self.memory.get_recognition_hints_synonyms(
            belief.decomposed.primary_target,
        )
        if not hint_syns:
            return
        existing = {s.lower() for s in belief.decomposed.primary_target_synonyms}
        added = [s for s in hint_syns if s.lower() not in existing]
        if added:
            belief.decomposed.primary_target_synonyms.extend(added)
            logger.info(
                "[memory] injected %d recognition synonym(s) for '%s': %s",
                len(added), belief.decomposed.primary_target, added,
            )
```

Then in `run()`, after `prior = self.memory.load_for_task(...)` block (currently lines 378-383), add a call. The patched block:

```python
        # Load long-term memory for this task
        self.memory.working_memory.clear()
        prior = self.memory.load_for_task(
            belief.decomposed.primary_target if belief.decomposed else "",
        )
        if prior:
            logger.info("[agent] loaded prior knowledge:\n%s", prior)
        self._apply_recognition_hints(belief)
```

- [ ] **Step 4: Run tests to verify pass**

Run:

```powershell
python -m pytest tests/test_memory_integration.py::TestRecognitionSynonymInjection -v
```

Expected: all 3 PASS.

- [ ] **Step 5: Run agent + memory test files to catch regressions**

Run:

```powershell
python -m pytest tests/test_agent_decide_next.py tests/test_agent_run.py tests/test_memory_integration.py tests/test_memory_manager.py -v
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```powershell
git add src/agent.py tests/test_memory_integration.py
git commit -m "feat(agent): inject persisted recognition synonyms at episode start"
```

---

## Task 7: End-to-end round-trip integration test

**Files:**
- Modify: `tests/test_memory_integration.py`

- [ ] **Step 1: Write the round-trip test**

Append to `tests/test_memory_integration.py`:

```python
class TestRecognitionRoundTrip:
    def test_clip_episode_persists_and_injects_next_episode(self, tmp_path):
        """Episode 1: simulate CLIP-injected evidence + grasp success.
        Episode 2: a fresh MemoryManager+agent loads the persisted synonym.
        """
        import time
        import yaml
        from pathlib import Path
        from src.agent import EmboSightAgent
        from src.memory_manager import MemoryManager
        from src.world_belief import (
            DecomposedTask, Evidence, WorldBelief,
        )

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
        agent1.logger = None

        belief1 = WorldBelief(user_query="pick tangerine")
        belief1.decomposed = DecomposedTask(primary_target="tangerine")

        ev = Evidence(
            source="vlm_ground", timestamp=time.time(),
            raw_payload={
                "hypotheses": [{
                    "object_id": "obj_0",
                    "label": "orange",
                    "label_alternatives": [["orange", 0.5], ["tangerine", 0.5]],
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
                    "vlm_label": "orange",
                },
            },
        )
        agent1._merge_hypotheses_from_evidence(belief1, ev)
        # Simulate successful grasp → consolidate
        mm1.consolidate(success=True, object_type="tangerine")

        # ===== Episode 2 =====
        mm2 = MemoryManager(memory_dir=d)
        agent2 = EmboSightAgent.__new__(EmboSightAgent)
        agent2.memory = mm2
        agent2.logger = None

        belief2 = WorldBelief(user_query="pick tangerine")
        belief2.decomposed = DecomposedTask(
            primary_target="tangerine",
            primary_target_synonyms=[],   # LLM gave none this time
        )
        agent2._apply_recognition_hints(belief2)

        assert "orange" in belief2.decomposed.primary_target_synonyms
```

- [ ] **Step 2: Run the new test**

Run:

```powershell
python -m pytest tests/test_memory_integration.py::TestRecognitionRoundTrip -v
```

Expected: PASS (all prerequisite tasks already integrated).

- [ ] **Step 3: Run the full test suite**

Run:

```powershell
python -m pytest -q
```

Expected: all tests PASS, no regressions in any other module.

If failures occur, fix them before committing. Common likely sources:
- Existing `test_memory_integration.py::test_consolidate_then_load_round_trip` may now persist recognition entries; check its memory dir does not bleed recognition.
- `test_perception.py` tests that didn't pass `belief` may need updating if they hit the modified `observe()` path — verify no change in observe signature.

- [ ] **Step 4: Commit**

```powershell
git add tests/test_memory_integration.py
git commit -m "test(memory): end-to-end recognition synonym round-trip"
```

---

## Task 8: Update Phase 1 design doc status

**Files:**
- Modify: `docs/superpowers/specs/2026-05-11-dual-store-memory-design.md`

- [ ] **Step 1: Mark Phase 2 recognition items as done**

In the spec, locate the Phase 2 checklist (around line 387). Update:

```markdown
### Phase 2 (后续 — 扩展领域)

- [x] recognition_hints.yaml 写入 (CLIP/LLM 纠正时)
- [x] recognition hints 注入 perception 流程 (via primary_target_synonyms)
- [ ] safety_knowledge.yaml (安全重分类)
- [ ] 消融实验 (w/ vs w/o memory)
```

- [ ] **Step 2: Commit**

```powershell
git add docs/superpowers/specs/2026-05-11-dual-store-memory-design.md
git commit -m "docs(memory): mark Phase 2 recognition items complete"
```

---

## Final Verification

Run after every task is committed:

```powershell
python -m pytest -q
```

Expected: all tests pass. Then inspect `memory/recognition_hints.yaml` will still be empty (no real episode has run yet) — that is correct; the file fills only on real grasp success.

Optionally, run one fixed scenario to confirm end-to-end behavior:

```powershell
python eval/run_fixed.py --scenario fixed_seed_discover_001
```

(This requires the Linux RoboCasa environment; on Windows it will not run.)
