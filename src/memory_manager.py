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
    """Dual-store episodic memory manager.

    Working memory: in-process list (real-time, episode-scoped).
    Long-term memory: YAML files with pointer-index (persistent, cross-episode).
    """

    _MAX_SYNONYMS_PER_TARGET = 5

    def __init__(self, memory_dir: Path = Path("memory")):
        self.memory_dir = memory_dir
        self.working_memory: list[MemoryEntry] = []
        self._long_term: dict[str, list[dict]] = {}
        self._domain_files: dict[str, Path] = {}
        self._load_index()

    # ── Load (index + domain files) ──

    def _load_index(self) -> None:
        idx_path = self.memory_dir / "index.yaml"
        if not idx_path.exists():
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

    def _load_domain(self, domain: str) -> list[dict]:
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
            entries = data.get("entries", []) or []
            self._long_term[domain] = entries
            return entries
        except Exception as e:
            logger.warning("[memory] failed to load %s: %s", domain, e)
            self._long_term[domain] = []
            return []

    # ── Record (real-time, episode 内) ──

    def record_event(self, entry: MemoryEntry) -> None:
        self.working_memory.append(entry)
        logger.debug("[memory] recorded: step=%d domain=%s event=%s",
                     entry.step, entry.domain, entry.event)

    # ── Read (prompt injection) ──

    def get_working_summary(self, domain: Optional[str] = None) -> str:
        entries = self.working_memory
        if domain:
            entries = [e for e in entries if e.domain == domain]
        if not entries:
            return ""
        lines = [f"- Step {e.step}: {e.lesson}" for e in entries]
        return "\n".join(lines)

    def get_grasp_advice(self, object_type: str) -> Optional[str]:
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
        entries = self._load_domain("recognition")
        tgt_key = target.lower().strip()
        for entry in entries:
            if entry.get("target", "").lower().strip() == tgt_key:
                parts = []
                labels = entry.get("vlm_common_labels", [])
                if labels:
                    parts.append(f"VLM often labels as: {', '.join(labels)}")
                syns = entry.get("effective_synonyms", [])
                # syns may be list[str] (legacy) or list[dict] (Phase 2)
                syn_names = [
                    s["name"] if isinstance(s, dict) else str(s)
                    for s in syns
                ]
                if syn_names:
                    parts.append(f"effective synonyms: {', '.join(syn_names)}")
                if parts:
                    return f"{target}: {'; '.join(parts)}"
        return None

    def get_recognition_hints_synonyms(self, target: str) -> list[str]:
        """Return historical effective synonyms for target (sorted by count desc).

        Returns [] if no entry or load fails.
        """
        entries = self._load_domain("recognition")
        tgt_key = target.lower().strip()
        for entry in entries:
            if str(entry.get("target", "")).lower().strip() == tgt_key:
                syns = entry.get("effective_synonyms", []) or []
                # defensive re-sort by count desc
                ordered = sorted(
                    syns, key=lambda s: s.get("count", 0) if isinstance(s, dict) else 0,
                    reverse=True,
                )
                return [
                    str(s["name"]) if isinstance(s, dict) else str(s)
                    for s in ordered
                    if (s.get("name") if isinstance(s, dict) else s)
                ]
        return []

    def load_for_task(self, primary_target: str, object_type: str = "") -> str:
        parts: list[str] = []
        obj = object_type or primary_target
        grasp = self.get_grasp_advice(obj)
        if grasp:
            parts.append(grasp)
        recog = self.get_recognition_hints(primary_target)
        if recog:
            parts.append(recog)
        return "\n".join(parts)

    # ── Consolidate (episode 结束, working → long-term) ──

    def consolidate(self, success: bool, object_type: str = "") -> None:
        # grasp: always (Phase 1 contract — records failures too)
        grasp_events = [e for e in self.working_memory if e.domain == "grasp"]
        if grasp_events and object_type:
            self._consolidate_grasp(grasp_events, object_type, success)
        # recognition: only on success (avoids fixing误命中 into long-term)
        if success:
            recognition_events = [
                e for e in self.working_memory if e.domain == "recognition"
            ]
            if recognition_events:
                self._consolidate_recognition(recognition_events)
            safety_events = [
                e for e in self.working_memory if e.domain == "safety"
            ]
            if safety_events:
                self._consolidate_safety(safety_events)

    def _consolidate_grasp(
        self, events: list[MemoryEntry], object_type: str, success: bool,
    ) -> None:
        entries = self._load_domain("grasp")
        obj_key = object_type.lower().strip()

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

        for e in events:
            target_entry["total_attempts"] = target_entry.get("total_attempts", 0) + 1
            if e.event == "strategy_succeeded":
                target_entry["success_count"] = target_entry.get("success_count", 0) + 1
                target_entry["best_strategy"] = e.context.get("strategy")
            elif e.event == "strategy_failed":
                strat = e.context.get("strategy", "")
                reason = e.context.get("failure", "unknown")
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
        """Merge recognition working events into long-term YAML.

        - vlm_common_labels: set semantics (append unique)
        - effective_synonyms: count merged per (target, synonym); sorted desc; capped
        - clip_helpful: True iff any event has method='clip'
        Only called when episode succeeded (caller enforces).
        """
        from collections import defaultdict

        entries = self._load_domain("recognition")

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

                if vlm_label and vlm_label not in entry["vlm_common_labels"]:
                    entry["vlm_common_labels"].append(vlm_label)

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

            entry["effective_synonyms"].sort(
                key=lambda s: s.get("count", 0), reverse=True,
            )
            entry["effective_synonyms"] = (
                entry["effective_synonyms"][:self._MAX_SYNONYMS_PER_TARGET]
            )
            entry["last_updated"] = datetime.now().strftime("%Y-%m-%d")

        self._long_term["recognition"] = entries
        self._save_domain("recognition")

    def _consolidate_safety(self, events: list[MemoryEntry]) -> None:
        """Merge safety_classified events into long-term YAML as running average.

        - Same label observed multiple times in one episode → only latest dist
          counted as ONE observation (dedup).
        - Cross-episode: running average over `dist`, observations += 1.
        - top_class = argmax of merged dist.
        Only called when episode succeeded (caller enforces).
        """
        entries = self._load_domain("safety")

        # episode-scoped dedup: keep latest dist per label
        latest_by_label: dict[str, dict] = {}
        for e in events:
            if e.event != "safety_classified":
                continue
            label = str(e.context.get("label", "")).strip().lower()
            dist = e.context.get("dist", {})
            if not label or not isinstance(dist, dict) or not dist:
                continue
            # normalize defensively
            total = sum(float(v) for v in dist.values() if v is not None) or 1.0
            norm = {
                str(k): float(v) / total
                for k, v in dist.items() if v is not None
            }
            latest_by_label[label] = norm

        for label, new_dist in latest_by_label.items():
            entry = next(
                (x for x in entries
                 if str(x.get("label", "")).strip().lower() == label),
                None,
            )
            if entry is None:
                entry = {
                    "label": label,
                    "dist": dict(new_dist),
                    "top_class": max(new_dist, key=new_dist.get),
                    "observations": 1,
                    "last_updated": "",
                }
                entries.append(entry)
            else:
                old_n = int(entry.get("observations", 0)) or 0
                old_dist = entry.get("dist", {}) or {}
                merged_keys = set(old_dist) | set(new_dist)
                new_n = old_n + 1
                merged: dict[str, float] = {}
                for k in merged_keys:
                    old_v = float(old_dist.get(k, 0.0))
                    new_v = float(new_dist.get(k, 0.0))
                    merged[k] = (old_v * old_n + new_v) / new_n
                # renormalize (numerical safety)
                tot = sum(merged.values()) or 1.0
                merged = {k: v / tot for k, v in merged.items()}
                entry["dist"] = merged
                entry["observations"] = new_n
                entry["top_class"] = max(merged, key=merged.get)
            entry["last_updated"] = datetime.now().strftime("%Y-%m-%d")

        self._long_term["safety"] = entries
        self._save_domain("safety")

    def get_safety_prior(self, label: str) -> Optional[dict]:
        """Return historical safety classification prior for label, or None.

        Returns dict with keys: dist, top_class, observations.
        """
        if not label:
            return None
        entries = self._load_domain("safety")
        key = label.lower().strip()
        for entry in entries:
            if str(entry.get("label", "")).lower().strip() == key:
                dist = entry.get("dist", {}) or {}
                return {
                    "dist": dict(dist),
                    "top_class": entry.get("top_class")
                    or (max(dist, key=dist.get) if dist else None),
                    "observations": int(entry.get("observations", 0)),
                }
        return None

    def _save_domain(self, domain: str) -> None:
        fpath = self._domain_files.get(domain)
        if not fpath:
            return
        try:
            import yaml
            fpath.parent.mkdir(parents=True, exist_ok=True)
            with open(fpath, "w", encoding="utf-8") as f:
                yaml.dump(
                    {"entries": self._long_term.get(domain, [])},
                    f, allow_unicode=True, default_flow_style=False,
                    sort_keys=False,
                )
        except Exception as e:
            logger.warning("[memory] failed to save %s: %s", domain, e)
