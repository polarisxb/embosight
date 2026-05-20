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
    post_lift_obj_pos: list[float] | None = None
    post_lift_obj_delta_z: float | None = None
    post_lift_eef_pos: list[float] | None = None
    selected_strategy: str | None = None
    executed_strategy: str | None = None
    depth_margin_m: float | None = None
    squeeze_extra_steps: int | None = None
    finger_width_m: float | None = None
    grasp_profile: str | None = None
    grasp_profile_confidence: float | None = None
    grasp_profile_reasons: list[str] | None = None
    grasp_policy_mode: str | None = None
    grasp_policy_applied: bool | None = None
    grasp_policy_profile: str | None = None
    legacy_depth_margin_m: float | None = None
    legacy_squeeze_extra_steps: int | None = None
    candidate_source_policy: str | None = None
    candidate_source_policy_applied: bool | None = None
    legacy_first_candidate_source: str | None = None
    final_first_candidate_source: str | None = None
    target_resolution_status: str | None = None
    target_body: str | None = None
    target_body_category: str | None = None
    resolved_body_name: str | None = None
    resolved_body_category: str | None = None
    target_resolution_source: str | None = None
    target_resolution_used_fallback: bool | None = None
    candidate_actionability_policy: str | None = None
    candidate_actionability_actionable: bool | None = None
    candidate_actionability_hard_reject: bool | None = None
    candidate_actionability_reason: str | None = None
    actionability_status: str | None = None
    actionability_reason: str | None = None
    actionability_stage: str | None = None
    actionability_gate_enabled: bool | None = None
    actionability_gate_applied: bool | None = None
    actionability_skip_reason: str | None = None
    legacy_first_candidate_actionable: bool | None = None
    final_first_candidate_actionable: bool | None = None
    no_actionable_candidate: bool | None = None
    attempts_count: int = 0
    post_lift_verified: bool | None = None

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
    grasp_attempts = _grasp_attempts(data.get("evidence", []))
    grasp_attempt = grasp_attempts[-1] if grasp_attempts else None
    diagnostic = _attempt_diagnostic(grasp_attempt)
    obj_z_before = _float_or_none(diagnostic.get("obj_z_before"))
    obj_z_after = _float_or_none(diagnostic.get("obj_z_after"))
    obj_delta_z = (
        obj_z_after - obj_z_before
        if obj_z_before is not None and obj_z_after is not None
        else None
    )
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
        post_lift_obj_pos=_float_list_or_none(diagnostic.get("post_lift_obj_pos")),
        post_lift_obj_delta_z=obj_delta_z,
        post_lift_eef_pos=_float_list_or_none(diagnostic.get("post_lift_eef_pos")),
        selected_strategy=_str_or_none(diagnostic.get("selected_strategy")),
        executed_strategy=_str_or_none(diagnostic.get("executed_strategy")),
        depth_margin_m=_float_or_none(diagnostic.get("depth_margin_m")),
        squeeze_extra_steps=_int_or_none(diagnostic.get("squeeze_extra_steps")),
        finger_width_m=_float_or_none(diagnostic.get("finger_width_m")),
        grasp_profile=_str_or_none(diagnostic.get("grasp_profile")),
        grasp_profile_confidence=_float_or_none(
            diagnostic.get("grasp_profile_confidence"),
        ),
        grasp_profile_reasons=_str_list_or_none(
            diagnostic.get("grasp_profile_reasons"),
        ),
        grasp_policy_mode=_str_or_none(diagnostic.get("grasp_policy_mode")),
        grasp_policy_applied=_bool_or_none(diagnostic.get("grasp_policy_applied")),
        grasp_policy_profile=_str_or_none(diagnostic.get("grasp_policy_profile")),
        legacy_depth_margin_m=_float_or_none(
            diagnostic.get("legacy_depth_margin_m"),
        ),
        legacy_squeeze_extra_steps=_int_or_none(
            diagnostic.get("legacy_squeeze_extra_steps"),
        ),
        candidate_source_policy=_str_or_none(
            diagnostic.get("candidate_source_policy"),
        ),
        candidate_source_policy_applied=_bool_or_none(
            diagnostic.get("candidate_source_policy_applied"),
        ),
        legacy_first_candidate_source=_str_or_none(
            diagnostic.get("legacy_first_candidate_source"),
        ),
        final_first_candidate_source=_str_or_none(
            diagnostic.get("final_first_candidate_source"),
        ),
        target_resolution_status=_str_or_none(
            diagnostic.get("target_resolution_status"),
        ),
        target_body=_str_or_none(diagnostic.get("target_body")),
        target_body_category=_str_or_none(diagnostic.get("target_body_category")),
        resolved_body_name=_str_or_none(diagnostic.get("resolved_body_name")),
        resolved_body_category=_str_or_none(
            diagnostic.get("resolved_body_category"),
        ),
        target_resolution_source=_str_or_none(
            diagnostic.get("target_resolution_source"),
        ),
        target_resolution_used_fallback=_bool_or_none(
            diagnostic.get("target_resolution_used_fallback"),
        ),
        candidate_actionability_policy=_str_or_none(
            diagnostic.get("candidate_actionability_policy"),
        ),
        candidate_actionability_actionable=_bool_or_none(
            diagnostic.get("candidate_actionability_actionable"),
        ),
        candidate_actionability_hard_reject=_bool_or_none(
            diagnostic.get("candidate_actionability_hard_reject"),
        ),
        candidate_actionability_reason=_str_or_none(
            diagnostic.get("candidate_actionability_reason"),
        ),
        actionability_status=_str_or_none(diagnostic.get("actionability_status")),
        actionability_reason=_str_or_none(diagnostic.get("actionability_reason")),
        actionability_stage=_str_or_none(diagnostic.get("actionability_stage")),
        actionability_gate_enabled=_bool_or_none(
            diagnostic.get("actionability_gate_enabled"),
        ),
        actionability_gate_applied=_bool_or_none(
            diagnostic.get("actionability_gate_applied"),
        ),
        actionability_skip_reason=_str_or_none(
            diagnostic.get("actionability_skip_reason"),
        ),
        legacy_first_candidate_actionable=_bool_or_none(
            diagnostic.get("legacy_first_candidate_actionable"),
        ),
        final_first_candidate_actionable=_bool_or_none(
            diagnostic.get("final_first_candidate_actionable"),
        ),
        no_actionable_candidate=_bool_or_none(
            diagnostic.get("no_actionable_candidate"),
        ),
        attempts_count=len(grasp_attempts),
        post_lift_verified=_post_lift_verified(
            grasp_attempt,
            diagnostic,
            obj_delta_z,
        ),
    )


def _latest_target_snapshot(snapshots: list[dict[str, Any]]) -> dict[str, Any] | None:
    for snap in reversed(snapshots):
        target = snap.get("target_summary")
        if target:
            return snap
    return None


def _grasp_attempts(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    for ev in evidence:
        if ev.get("source") == "grasp_attempt":
            payload = ev.get("raw_payload") or {}
            attempt = payload.get("attempt") or {}
            if attempt:
                attempts.append(attempt)
    return attempts


def _attempt_diagnostic(attempt: dict[str, Any] | None) -> dict[str, Any]:
    if not attempt:
        return {}
    diagnostic = attempt.get("diagnostic") or {}
    return diagnostic if isinstance(diagnostic, dict) else {}


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


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _bool_or_none(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _float_list_or_none(value: Any) -> list[float] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        return None
    return [float(item) for item in value]


def _str_list_or_none(value: Any) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        return None
    return [str(item) for item in value]


def _post_lift_verified(
    attempt: dict[str, Any] | None,
    diagnostic: dict[str, Any],
    obj_delta_z: float | None,
) -> bool | None:
    explicit = diagnostic.get("post_lift_verified")
    if explicit is not None:
        return bool(explicit)
    if not attempt or attempt.get("failure_mode") != "success":
        return False if attempt else None
    if obj_delta_z is None:
        return None
    return obj_delta_z >= 0.02


def _label_key(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())
