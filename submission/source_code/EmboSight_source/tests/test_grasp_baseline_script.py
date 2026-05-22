from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "run_grasp_baseline.sh"


def test_grasp_baseline_script_records_reproducible_baseline_outputs() -> None:
    assert SCRIPT.exists(), "baseline wrapper script is missing"
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'LEMON_RUNS="${LEMON_RUNS:-5}"' in text
    assert 'GEN_COUNT="${GEN_COUNT:-10}"' in text
    assert 'GEN_PARALLEL="${GEN_PARALLEL:-4}"' in text
    assert "scripts/validate_lemon_grasp_multi.sh" in text
    assert 'AGENT_CONFIG="${AGENT_CONFIG:-configs/agent.yaml}"' in text
    assert "eval/run_long_generalization.py" in text
    assert '--agent-config "${AGENT_CONFIG}"' in text
    assert "summary.csv" in text
    assert "summary.txt" in text
    assert "report.md" in text
    assert "--dry-run" in text
    assert "failure_mode_by_object" in text
    assert "failure_mode_by_candidate_source" in text
    assert "failure_mode_by_executed_strategy" in text
    assert "success_rate_by_profile" in text
    assert "grasp_policy_usage" in text
    assert "candidate_source_policy_usage" in text
    assert "candidate_source_transition_usage" in text
    assert "failure_family_breakdown" in text
    assert "failure_mode_by_actionability_reason" in text
    assert "candidate_actionability_usage" in text
    assert "target_resolution_source_usage" in text
    assert "execution_failure_stage_usage" in text
    assert "failure_mode_by_execution_stage" in text
    assert "failure_mode_by_execution_reason" in text
    assert "execution_recovery_usage" in text


def test_lemon_multi_script_exports_final_grasp_columns() -> None:
    script = REPO_ROOT / "scripts" / "validate_lemon_grasp_multi.sh"
    assert script.exists(), "lemon validation script is missing"
    text = script.read_text(encoding="utf-8")

    assert 'AGENT_CONFIG="${AGENT_CONFIG:-configs/agent.yaml}"' in text
    assert '--agent-config "${AGENT_CONFIG}"' in text
    assert "post_lift_obj_pos" in text
    assert "post_lift_obj_delta_z" in text
    assert "selected_strategy" in text
    assert "executed_strategy" in text
    assert "depth_margin_m" in text
    assert "squeeze_extra_steps" in text
    assert "grasp_profile" in text
    assert "grasp_policy_mode" in text
    assert "grasp_policy_applied" in text
    assert "candidate_source_policy" in text
    assert "candidate_source_policy_applied" in text
    assert "legacy_first_candidate_source" in text
    assert "final_first_candidate_source" in text
    assert "target_resolution_source" in text
    assert "target_body" in text
    assert "candidate_actionability_policy" in text
    assert "candidate_actionability_reason" in text
    assert "candidate_actionability_actionable" in text
    assert "candidate_actionability_hard_reject" in text
    assert "no_actionable_candidate" in text
    assert "execution_failure_stage" in text
    assert "execution_failure_reason" in text
    assert "execution_recovery_applied" in text
    assert "execution_recovery_skip_count" in text
