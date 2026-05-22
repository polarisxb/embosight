from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.grasp_execution import (
    PRE_GRASP_AXIS_GAP_TOO_LARGE,
    PRE_GRASP_AXIS_GAP_TOO_SMALL,
    PRE_GRASP_BELOW_GRASP_POINT,
    PRE_GRASP_LATERAL_MISALIGNED,
    PRE_GRASP_SAFE_HANDOFF,
    PRE_GRASP_STRICT_OK,
)


@dataclass(frozen=True)
class CandidateActionability:
    source: str
    selected_strategy: str | None
    executed_strategy: str
    target_body: str | None
    actionable: bool
    hard_reject: bool
    reason: str
    total_error_m: float | None = None
    lateral_error_m: float | None = None
    axis_error_m: float | None = None
    approach_gap_m: float | None = None
    lateral_limit_m: float | None = None
    object_size_m: tuple[float, float, float] | None = None
    score_modifier: float = 0.0
    stage: str = "planner"

    @property
    def status(self) -> str:
        if self.hard_reject:
            return "hard_reject"
        return "actionable" if self.actionable else "not_actionable"

    def to_diagnostic(self, prefix: str = "candidate_actionability") -> dict[str, Any]:
        return {
            f"{prefix}_source": self.source,
            f"{prefix}_selected_strategy": self.selected_strategy,
            f"{prefix}_executed_strategy": self.executed_strategy,
            f"{prefix}_target_body": self.target_body,
            f"{prefix}_actionable": bool(self.actionable),
            f"{prefix}_hard_reject": bool(self.hard_reject),
            f"{prefix}_reason": self.reason,
            f"{prefix}_total_error_m": self.total_error_m,
            f"{prefix}_lateral_error_m": self.lateral_error_m,
            f"{prefix}_axis_error_m": self.axis_error_m,
            f"{prefix}_approach_gap_m": self.approach_gap_m,
            f"{prefix}_lateral_limit_m": self.lateral_limit_m,
            f"{prefix}_object_size_m": (
                list(self.object_size_m) if self.object_size_m is not None else None
            ),
            f"{prefix}_score_modifier": float(self.score_modifier),
            f"{prefix}_stage": self.stage,
            "actionability_status": self.status,
            "actionability_reason": self.reason,
            "actionability_stage": self.stage,
        }


def executed_strategy_name(candidate: Any, selected_strategy: str | None = None) -> str:
    source = str(getattr(candidate, "source", "") or "")
    if source.startswith("strategy_"):
        return source[len("strategy_"):]
    if source == "vlm_top_grasp":
        return "top_down"
    if source == "geometric_centroid":
        return "geometric_centroid"
    return selected_strategy or "unknown"


def unknown_actionability(
    candidate: Any,
    *,
    selected_strategy: str | None,
    target_body: str | None,
    reason: str,
) -> CandidateActionability:
    return CandidateActionability(
        source=str(getattr(candidate, "source", "unknown")),
        selected_strategy=selected_strategy,
        executed_strategy=executed_strategy_name(candidate, selected_strategy),
        target_body=target_body,
        actionable=True,
        hard_reject=False,
        reason=reason,
        stage="planner",
    )


def actionability_from_pre_grasp_result(
    candidate: Any,
    result: Any,
    *,
    selected_strategy: str | None,
    target_body: str | None,
) -> CandidateActionability:
    reason = str(getattr(result, "reason", "unknown"))
    actionable = bool(getattr(result, "ok", False) or getattr(result, "handoff_ok", False))
    recoverable = bool(getattr(result, "needs_recovery", False))
    hard_reject = (
        not actionable
        and not recoverable
        and reason in {
            PRE_GRASP_AXIS_GAP_TOO_SMALL,
            PRE_GRASP_AXIS_GAP_TOO_LARGE,
            PRE_GRASP_BELOW_GRASP_POINT,
        }
    )
    if reason in {PRE_GRASP_STRICT_OK, PRE_GRASP_SAFE_HANDOFF}:
        score_modifier = 0.05
    elif reason == PRE_GRASP_LATERAL_MISALIGNED:
        score_modifier = -0.05
    elif hard_reject:
        score_modifier = -1.0
    else:
        score_modifier = 0.0

    return CandidateActionability(
        source=str(getattr(candidate, "source", "unknown")),
        selected_strategy=selected_strategy,
        executed_strategy=executed_strategy_name(candidate, selected_strategy),
        target_body=target_body,
        actionable=actionable,
        hard_reject=hard_reject,
        reason=reason,
        total_error_m=_float_or_none(getattr(result, "total_error_m", None)),
        lateral_error_m=_float_or_none(getattr(result, "lateral_error_m", None)),
        axis_error_m=_float_or_none(getattr(result, "axis_error_m", None)),
        approach_gap_m=_float_or_none(getattr(result, "approach_gap_m", None)),
        lateral_limit_m=_float_or_none(getattr(result, "lateral_limit_m", None)),
        score_modifier=score_modifier,
        stage="pre_grasp",
    )


def actionability_from_preview(
    candidate: Any,
    preview: dict[str, Any],
    *,
    selected_strategy: str | None,
    target_body: str | None,
) -> CandidateActionability:
    return CandidateActionability(
        source=str(getattr(candidate, "source", "unknown")),
        selected_strategy=selected_strategy,
        executed_strategy=executed_strategy_name(candidate, selected_strategy),
        target_body=target_body,
        actionable=bool(preview.get("actionable", True)),
        hard_reject=bool(preview.get("hard_reject", False)),
        reason=str(preview.get("reason", "preview_unknown")),
        total_error_m=_float_or_none(preview.get("total_error_m")),
        lateral_error_m=_float_or_none(preview.get("lateral_error_m")),
        axis_error_m=_float_or_none(preview.get("axis_error_m")),
        approach_gap_m=_float_or_none(preview.get("approach_gap_m")),
        lateral_limit_m=_float_or_none(preview.get("lateral_limit_m")),
        score_modifier=_float_or_none(preview.get("score_modifier")) or 0.0,
        stage="planner_preview",
    )


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
