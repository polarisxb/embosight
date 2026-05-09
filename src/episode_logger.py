"""EpisodeLogger: 记录 belief / action / evidence 流到 JSON, 支持 replay 反序列化。

设计参考: docs/superpowers/specs/2026-05-08-emboSight-belief-driven-agent-design.md §6.10
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.world_belief import (
    Action, BeliefSnapshot, EpisodeResult, Evidence,
)

logger = logging.getLogger(__name__)


@dataclass
class EpisodeRecord:
    query: str
    start_time: float
    snapshots: list[BeliefSnapshot]
    actions: list[Action]
    evidence: list[Evidence]
    user_qa: list[tuple[str, str]]
    final_result: Optional[EpisodeResult] = None


def _to_jsonable(obj: Any) -> Any:
    """递归把 dataclass / numpy / Path 转成 JSON 可序列化对象。"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if is_dataclass(obj) and not isinstance(obj, type):
        return _to_jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    return obj


class EpisodeLogger:
    
    def __init__(self, log_dir: str = "logs/episodes"):
        self._dir = Path(log_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._reset()
    
    def _reset(self) -> None:
        self._query: str = ""
        self._start: float = 0.0
        self._snaps: list[BeliefSnapshot] = []
        self._actions: list[Action] = []
        self._evidence: list[Evidence] = []
        self._user_qa: list[tuple[str, str]] = []
    
    def start_episode(self, query: str) -> None:
        self._reset()
        self._query = query
        self._start = time.time()
        logger.info(f"[episode] START: query={query!r}")
    
    def log_snapshot(self, snap: BeliefSnapshot) -> None:
        self._snaps.append(snap)
    
    def log_action_start(self, action: Action, snap: BeliefSnapshot) -> None:
        self._actions.append(action)
        self._snaps.append(snap)
    
    def log_action_end(self, action: Action, snap: BeliefSnapshot) -> None:
        self._snaps.append(snap)
    
    def log_user_qa(self, q: str, a: str) -> None:
        self._user_qa.append((q, a))
    
    def log_evidence(self, ev: Evidence) -> None:
        self._evidence.append(ev)
    
    def end_episode(self, result: EpisodeResult) -> str:
        ts = int(self._start)
        safe_q = "".join(c if c.isalnum() else "_" for c in self._query)[:30]
        path = self._dir / f"episode_{ts}_{safe_q}.json"
        payload = {
            "query": self._query,
            "start_time": self._start,
            "snapshots": _to_jsonable(self._snaps),
            "actions": _to_jsonable(self._actions),
            "evidence": _to_jsonable(self._evidence),
            "user_qa": _to_jsonable(self._user_qa),
            "final_result": _to_jsonable(result),
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"[episode] END: {path}")
        return str(path)
    
    @classmethod
    def load(cls, json_path: str) -> EpisodeRecord:
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
        # 注意: 这里仅做"浅"反序列化, dataclass 字段作为 dict 保留;
        # 完整反序列化在 Phase 14 EpisodeReplay 里做
        snaps = [BeliefSnapshot(**s) for s in data["snapshots"]]
        actions = [Action(kind=a["kind"], strategy=a.get("strategy"),
                          question=a.get("question"),
                          metadata=a.get("metadata", {}))
                   for a in data["actions"]]
        evidence = [Evidence(source=e["source"], timestamp=e["timestamp"],
                             raw_payload=e["raw_payload"],
                             consumed_by=e.get("consumed_by", []))
                    for e in data["evidence"]]
        final = None
        if data.get("final_result") is not None:
            fr = data["final_result"]
            final = EpisodeResult(
                success=fr["success"], target=None, speech=fr.get("speech", ""),
                belief_trace=[], action_history=[],
                n_steps=fr.get("n_steps", 0),
                elapsed_seconds=fr.get("elapsed_seconds", 0.0),
                failure_reason=fr.get("failure_reason"),
            )
        return EpisodeRecord(
            query=data["query"],
            start_time=data["start_time"],
            snapshots=snaps,
            actions=actions,
            evidence=evidence,
            user_qa=[tuple(x) for x in data.get("user_qa", [])],
            final_result=final,
        )
    
    @classmethod
    def replay(cls, json_path: str, agent_factory) -> EpisodeResult:
        """从 golden episode 回放 → 跑 agent.run, 返回新 result。

        agent_factory: callable(mocks: dict) -> (EmboSightAgent, env)。
        mocks 字典含 'vlm_ground' / 'vlm_zoom' / 'vlm_verify' /
        'llm_safety' / 'llm_decompose' / 'user_answer' 6 个 record-driven mock。
        """
        record = cls.load(json_path)
        mocks = {
            "vlm_ground": _MockFromRecord(record, "vlm_ground"),
            "vlm_zoom":   _MockFromRecord(record, "vlm_zoom"),
            "vlm_verify": _MockFromRecord(record, "vlm_verify"),
            "llm_safety": _MockFromRecord(record, "llm_safety"),
            "llm_decompose": _MockFromRecord(record, "llm_decompose"),
            "user_answer": _MockFromRecord(record, "user_answer"),
        }
        agent, env = agent_factory(mocks)
        return agent.run(record.query, env)


class _MockFromRecord:
    """从 EpisodeRecord 的 evidence 序列回放某 source 的输出。"""

    def __init__(self, record: EpisodeRecord, source: str) -> None:
        self._responses: list[str] = []
        for ev in record.evidence:
            if ev.source == source:
                self._responses.append(
                    json.dumps(ev.raw_payload, ensure_ascii=False)
                )
        # user_answer 特殊: 直接取 raw_payload['a']
        if source == "user_answer":
            self._responses = []
            for ev in record.evidence:
                if ev.source == "user_answer":
                    self._responses.append(str(ev.raw_payload.get("a", "")))

    def describe(self, image_path: str, prompt: str = "") -> str:
        if not self._responses:
            return '{"objects": []}'
        return self._responses.pop(0)

    def generate(self, prompt: str, system: str = "", **kw) -> str:
        if not self._responses:
            return "{}"
        return self._responses.pop(0)
