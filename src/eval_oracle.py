from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class OracleSummary:
    scenario_id: str
    query: str
    expected_object: str | None
    actual_object: str | None
    object_match: bool | None
    success: bool | None
    failure_reason: str | None
    action_sequence: list[str]
    vlm_labels: list[str]
    selected_target_label: str | None
    selected_target_position: list[float] | None
    grasp_failure_mode: str | None
    grasp_candidate_source: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_episode(
    episode_path: str | Path,
    scenario_id: str = "",
    expected_object: str | None = None,
    actual_object: str | None = None,
) -> OracleSummary:
    path = Path(episode_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    final_result = data.get("final_result") or {}
    target = _latest_target_summary(data.get("snapshots", []))
    grasp_attempt = _latest_grasp_attempt(data.get("evidence", []))
    object_match = None
    if expected_object is not None and actual_object is not None:
        object_match = _label_key(expected_object) == _label_key(actual_object)
    return OracleSummary(
        scenario_id=scenario_id,
        query=str(data.get("query", "")),
        expected_object=expected_object,
        actual_object=actual_object,
        object_match=object_match,
        success=final_result.get("success"),
        failure_reason=final_result.get("failure_reason"),
        action_sequence=[str(a.get("kind", "")) for a in data.get("actions", [])],
        vlm_labels=_vlm_labels(data.get("evidence", [])),
        selected_target_label=target.get("label") if target else None,
        selected_target_position=target.get("position_3d") if target else None,
        grasp_failure_mode=grasp_attempt.get("failure_mode") if grasp_attempt else None,
        grasp_candidate_source=grasp_attempt.get("candidate_source") if grasp_attempt else None,
    )


def _latest_target_summary(snapshots: list[dict[str, Any]]) -> dict[str, Any] | None:
    for snap in reversed(snapshots):
        target = snap.get("target_summary")
        if target:
            return target
    return None


def _latest_grasp_attempt(evidence: list[dict[str, Any]]) -> dict[str, Any] | None:
    for ev in reversed(evidence):
        if ev.get("source") == "grasp_attempt":
            payload = ev.get("raw_payload") or {}
            attempt = payload.get("attempt") or {}
            if attempt:
                return attempt
    return None


def _vlm_labels(evidence: list[dict[str, Any]]) -> list[str]:
    labels: list[str] = []
    for ev in evidence:
        if ev.get("source") != "vlm_ground":
            continue
        payload = ev.get("raw_payload") or {}
        for hyp in payload.get("hypotheses", []) or []:
            label = hyp.get("label")
            if label is not None:
                labels.append(str(label))
    return labels


def _label_key(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())
