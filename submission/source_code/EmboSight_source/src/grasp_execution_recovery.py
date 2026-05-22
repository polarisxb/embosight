from __future__ import annotations

from dataclasses import dataclass
from typing import Any


_RECOVERABLE_FAILURES = {
    ("slipped_descend", "pre_close_alignment", "object_displaced_before_close"),
    ("slipped_lift", "micro_lift_verify", "object_not_following"),
}


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _list_or_none(value: Any) -> list[float] | None:
    if value is None:
        return None
    try:
        return [float(v) for v in value[:3]]
    except (TypeError, ValueError, IndexError):
        return None


def _str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _bool_or_none(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


@dataclass(frozen=True)
class ExecutionFailureDiagnostic:
    failure_mode: str
    stage: str
    reason: str
    recoverable: bool
    candidate_source: str | None = None
    executed_strategy: str | None = None
    branch: str | None = None
    pre_close_lateral_error_m: float | None = None
    pre_close_lateral_limit_m: float | None = None
    pre_close_z_diff_m: float | None = None
    pre_close_eef_pos: list[float] | None = None
    pre_close_obj_pos: list[float] | None = None
    pre_close_candidate_xy: list[float] | None = None
    micro_lift_eef_delta_m: float | None = None
    micro_lift_obj_delta_m: float | None = None
    micro_lift_required_m: float | None = None
    micro_lift_follows: bool | None = None
    obj_z_before: float | None = None
    obj_z_after: float | None = None
    obj_z_delta_m: float | None = None

    def to_diagnostic(self) -> dict[str, Any]:
        return {
            "execution_failure_mode": self.failure_mode,
            "execution_failure_stage": self.stage,
            "execution_failure_reason": self.reason,
            "execution_failure_recoverable": bool(self.recoverable),
            "executed_candidate_source": self.candidate_source,
            "executed_strategy": self.executed_strategy,
            "execution_branch": self.branch,
            "pre_close_lateral_error_m": self.pre_close_lateral_error_m,
            "pre_close_lateral_limit_m": self.pre_close_lateral_limit_m,
            "pre_close_z_diff_m": self.pre_close_z_diff_m,
            "pre_close_eef_pos": self.pre_close_eef_pos,
            "pre_close_obj_pos": self.pre_close_obj_pos,
            "pre_close_candidate_xy": self.pre_close_candidate_xy,
            "micro_lift_eef_delta_m": self.micro_lift_eef_delta_m,
            "micro_lift_obj_delta_m": self.micro_lift_obj_delta_m,
            "micro_lift_required_m": self.micro_lift_required_m,
            "micro_lift_follows": self.micro_lift_follows,
            "obj_z_before": self.obj_z_before,
            "obj_z_after": self.obj_z_after,
            "obj_z_delta_m": self.obj_z_delta_m,
        }


def execution_failure_from_pre_close_alignment(
    *,
    candidate_source: str | None,
    executed_strategy: str | None,
    lateral_error_m: Any,
    lateral_limit_m: Any,
    z_diff_m: Any,
    eef_pos: Any,
    obj_pos: Any,
    candidate_xy: Any,
) -> ExecutionFailureDiagnostic:
    return ExecutionFailureDiagnostic(
        failure_mode="slipped_descend",
        stage="pre_close_alignment",
        reason="object_displaced_before_close",
        recoverable=True,
        candidate_source=candidate_source,
        executed_strategy=executed_strategy,
        pre_close_lateral_error_m=_float_or_none(lateral_error_m),
        pre_close_lateral_limit_m=_float_or_none(lateral_limit_m),
        pre_close_z_diff_m=_float_or_none(z_diff_m),
        pre_close_eef_pos=_list_or_none(eef_pos),
        pre_close_obj_pos=_list_or_none(obj_pos),
        pre_close_candidate_xy=_list_or_none(candidate_xy),
    )


def execution_failure_from_micro_lift(
    *,
    candidate_source: str | None,
    executed_strategy: str | None,
    branch: str,
    follows: bool,
    eef_delta_m: Any,
    obj_delta_m: Any,
    required_m: Any,
) -> ExecutionFailureDiagnostic:
    return ExecutionFailureDiagnostic(
        failure_mode="slipped_lift",
        stage="micro_lift_verify",
        reason="object_not_following" if not follows else "object_following",
        recoverable=not follows,
        candidate_source=candidate_source,
        executed_strategy=executed_strategy,
        branch=branch,
        micro_lift_eef_delta_m=_float_or_none(eef_delta_m),
        micro_lift_obj_delta_m=_float_or_none(obj_delta_m),
        micro_lift_required_m=_float_or_none(required_m),
        micro_lift_follows=bool(follows),
    )


def execution_failure_from_attempt_diagnostic(
    diagnostic: dict[str, Any],
) -> ExecutionFailureDiagnostic:
    return ExecutionFailureDiagnostic(
        failure_mode=str(
            diagnostic.get("execution_failure_mode")
            or diagnostic.get("failure_mode")
            or "unknown",
        ),
        stage=str(
            diagnostic.get("execution_failure_stage")
            or diagnostic.get("stage")
            or "unknown",
        ),
        reason=str(
            diagnostic.get("execution_failure_reason")
            or diagnostic.get("reason")
            or "unknown",
        ),
        recoverable=bool(diagnostic.get("execution_failure_recoverable")),
        candidate_source=_str_or_none(diagnostic.get("executed_candidate_source")),
        executed_strategy=_str_or_none(diagnostic.get("executed_strategy")),
        branch=_str_or_none(
            diagnostic.get("execution_branch") or diagnostic.get("branch"),
        ),
        pre_close_lateral_error_m=_float_or_none(
            diagnostic.get("pre_close_lateral_error_m"),
        ),
        pre_close_lateral_limit_m=_float_or_none(
            diagnostic.get("pre_close_lateral_limit_m"),
        ),
        pre_close_z_diff_m=_float_or_none(diagnostic.get("pre_close_z_diff_m")),
        pre_close_eef_pos=_list_or_none(diagnostic.get("pre_close_eef_pos")),
        pre_close_obj_pos=_list_or_none(diagnostic.get("pre_close_obj_pos")),
        pre_close_candidate_xy=_list_or_none(
            diagnostic.get("pre_close_candidate_xy"),
        ),
        micro_lift_eef_delta_m=_float_or_none(
            diagnostic.get("micro_lift_eef_delta_m"),
        ),
        micro_lift_obj_delta_m=_float_or_none(
            diagnostic.get("micro_lift_obj_delta_m"),
        ),
        micro_lift_required_m=_float_or_none(
            diagnostic.get("micro_lift_required_m"),
        ),
        micro_lift_follows=_bool_or_none(diagnostic.get("micro_lift_follows")),
        obj_z_before=_float_or_none(diagnostic.get("obj_z_before")),
        obj_z_after=_float_or_none(diagnostic.get("obj_z_after")),
        obj_z_delta_m=_float_or_none(diagnostic.get("obj_z_delta_m")),
    )


def should_recover_execution_failure(
    diagnostic: ExecutionFailureDiagnostic,
    *,
    gate_enabled: bool,
    attempts_used: int,
    max_attempts: int,
) -> bool:
    failure_key = (diagnostic.failure_mode, diagnostic.stage, diagnostic.reason)
    return bool(
        gate_enabled
        and diagnostic.recoverable
        and failure_key in _RECOVERABLE_FAILURES
        and int(attempts_used) < max(0, int(max_attempts))
    )
