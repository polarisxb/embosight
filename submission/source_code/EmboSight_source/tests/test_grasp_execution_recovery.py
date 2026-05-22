from src.grasp_execution_recovery import (
    ExecutionFailureDiagnostic,
    execution_failure_from_attempt_diagnostic,
    execution_failure_from_micro_lift,
    execution_failure_from_pre_close_alignment,
    should_recover_execution_failure,
)


def test_pre_close_alignment_diagnostic_serializes_recovery_fields():
    diagnostic = execution_failure_from_pre_close_alignment(
        candidate_source="vlm_top_grasp",
        executed_strategy="top_down",
        lateral_error_m=0.089,
        lateral_limit_m=0.020,
        z_diff_m=0.012,
        eef_pos=[0.50, 0.00, 0.94],
        obj_pos=[0.58, 0.03, 0.90],
        candidate_xy=[0.50, 0.00],
    )

    data = diagnostic.to_diagnostic()

    assert data["execution_failure_stage"] == "pre_close_alignment"
    assert data["execution_failure_reason"] == "object_displaced_before_close"
    assert data["execution_failure_recoverable"] is True
    assert data["executed_candidate_source"] == "vlm_top_grasp"
    assert data["pre_close_lateral_error_m"] == 0.089
    assert data["pre_close_lateral_limit_m"] == 0.020
    assert data["pre_close_z_diff_m"] == 0.012


def test_micro_lift_diagnostic_serializes_object_not_following():
    diagnostic = execution_failure_from_micro_lift(
        candidate_source="strategy_top_down",
        executed_strategy="top_down",
        branch="lift",
        follows=False,
        eef_delta_m=0.020,
        obj_delta_m=0.0,
        required_m=0.010,
    )

    data = diagnostic.to_diagnostic()

    assert data["execution_failure_stage"] == "micro_lift_verify"
    assert data["execution_failure_reason"] == "object_not_following"
    assert data["execution_branch"] == "lift"
    assert data["execution_failure_recoverable"] is True
    assert data["micro_lift_eef_delta_m"] == 0.020
    assert data["micro_lift_obj_delta_m"] == 0.0
    assert data["micro_lift_required_m"] == 0.010
    assert data["micro_lift_follows"] is False


def test_execution_failure_can_be_rebuilt_from_attempt_diagnostic():
    original = execution_failure_from_pre_close_alignment(
        candidate_source="vlm_top_grasp",
        executed_strategy="top_down",
        lateral_error_m=0.089,
        lateral_limit_m=0.020,
        z_diff_m=0.012,
        eef_pos=[0.50, 0.00, 0.94],
        obj_pos=[0.58, 0.03, 0.90],
        candidate_xy=[0.50, 0.00],
    )

    rebuilt = execution_failure_from_attempt_diagnostic(original.to_diagnostic())

    assert rebuilt.failure_mode == "slipped_descend"
    assert rebuilt.stage == "pre_close_alignment"
    assert rebuilt.reason == "object_displaced_before_close"
    assert rebuilt.recoverable is True
    assert rebuilt.candidate_source == "vlm_top_grasp"
    assert rebuilt.executed_strategy == "top_down"


def test_post_lift_failure_is_diagnostic_only_not_recoverable():
    diagnostic = ExecutionFailureDiagnostic(
        failure_mode="slipped_lift",
        stage="post_lift_verify",
        reason="object_not_lifted",
        recoverable=False,
        candidate_source="vlm_top_grasp",
        executed_strategy="top_down",
        obj_z_before=0.94,
        obj_z_after=0.94,
        obj_z_delta_m=0.0,
    )

    assert diagnostic.to_diagnostic()["execution_failure_recoverable"] is False
    assert should_recover_execution_failure(
        diagnostic,
        gate_enabled=True,
        attempts_used=0,
        max_attempts=1,
    ) is False


def test_recovery_decision_requires_gate_and_budget():
    diagnostic = execution_failure_from_pre_close_alignment(
        candidate_source="vlm_top_grasp",
        executed_strategy="top_down",
        lateral_error_m=0.050,
        lateral_limit_m=0.020,
        z_diff_m=0.0,
        eef_pos=[0.0, 0.0, 1.0],
        obj_pos=[0.05, 0.0, 0.9],
        candidate_xy=[0.0, 0.0],
    )

    assert should_recover_execution_failure(
        diagnostic,
        gate_enabled=False,
        attempts_used=0,
        max_attempts=1,
    ) is False
    assert should_recover_execution_failure(
        diagnostic,
        gate_enabled=True,
        attempts_used=1,
        max_attempts=1,
    ) is False
    assert should_recover_execution_failure(
        diagnostic,
        gate_enabled=True,
        attempts_used=0,
        max_attempts=1,
    ) is True
