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
    assert "eval/run_long_generalization.py" in text
    assert "summary.csv" in text
    assert "summary.txt" in text
    assert "report.md" in text
    assert "--dry-run" in text
