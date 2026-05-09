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
    selected_target_label_entropy: float | None
    selected_target_position_std_m: float | None
    selected_target_safety_entropy: float | None
    selected_target_grasp_uncertainty: float | None
    dominant_uncertainty_axis: str | None
    planning_blockers: list[str]
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
    target_snapshot = _latest_target_snapshot(data.get("snapshots", []))
    target = target_snapshot.get("target_summary") if target_snapshot else None
    grasp_attempt = _latest_grasp_attempt(data.get("evidence", []))
    action_sequence = [str(a.get("kind", "")) for a in data.get("actions", [])]
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
        action_sequence=action_sequence,
        vlm_labels=_vlm_labels(data.get("evidence", [])),
        selected_target_label=target.get("label") if target else None,
        selected_target_position=target.get("position_3d") if target else None,
        selected_target_label_entropy=_float_or_none(
            target.get("label_entropy") if target else None,
        ),
        selected_target_position_std_m=_float_or_none(
            target.get("position_std_m") if target else None,
        ),
        selected_target_safety_entropy=_float_or_none(
            target.get("safety_entropy") if target else None,
        ),
        selected_target_grasp_uncertainty=_float_or_none(
            target.get("grasp_uncertainty") if target else None,
        ),
        dominant_uncertainty_axis=(
            str(target_snapshot.get("most_uncertain_axis"))
            if target_snapshot and target_snapshot.get("most_uncertain_axis") is not None
            else None
        ),
        planning_blockers=_planning_blockers(target, action_sequence),
        grasp_failure_mode=grasp_attempt.get("failure_mode") if grasp_attempt else None,
        grasp_candidate_source=grasp_attempt.get("candidate_source") if grasp_attempt else None,
    )


def _latest_target_snapshot(snapshots: list[dict[str, Any]]) -> dict[str, Any] | None:
    for snap in reversed(snapshots):
        target = snap.get("target_summary")
        if target:
            return snap
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


def _planning_blockers(
    target: dict[str, Any] | None,
    action_sequence: list[str],
) -> list[str]:
    if target is None or "plan_grasp_candidates" in action_sequence:
        return []
    blockers: list[str] = []
    label_entropy = _float_or_none(target.get("label_entropy"))
    position_std = _float_or_none(target.get("position_std_m"))
    safety_entropy = _float_or_none(target.get("safety_entropy"))
    if label_entropy is not None and label_entropy >= 0.80:
        blockers.append("label_entropy>=0.80")
    if position_std is not None and position_std >= 0.10:
        blockers.append("position_std_m>=0.10")
    if safety_entropy is not None and safety_entropy >= 0.50 and not _is_low_hazard(target):
        blockers.append("safety_entropy>=0.50")
    return blockers


def _is_low_hazard(target: dict[str, Any]) -> bool:
    safety_dist = target.get("safety_dist") or {}
    if not safety_dist:
        return False
    hazard_prob = (
        float(safety_dist.get("sharp", 0.0))
        + float(safety_dist.get("hot", 0.0))
        + float(safety_dist.get("chemical", 0.0))
    )
    return hazard_prob < 0.05


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _label_key(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())
