#!/usr/bin/env bash
# Baseline runner for grasp robustness/generalization.
#
# It runs two layers:
#   1. fixed lemon regression via scripts/validate_lemon_grasp_multi.sh
#   2. random-seed generalization via eval/run_long_generalization.py
#
# Outputs are grouped under runs/grasp_baseline/<run-id>/, with a
# report.md that can be pasted into a review thread.

set -euo pipefail

LEMON_RUNS="${LEMON_RUNS:-5}"
GEN_SEED_START="${GEN_SEED_START:-0}"
GEN_COUNT="${GEN_COUNT:-10}"
GEN_PARALLEL="${GEN_PARALLEL:-4}"
TIMEOUT_S="${TIMEOUT_S:-900}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
MEMORY_MODE="${MEMORY_MODE:-isolated}"
RUN_ID="${RUN_ID:-grasp-baseline-$(date +%Y%m%d_%H%M%S)}"
GEN_RUN_ID="${GEN_RUN_ID:-}"
BASE_DIR="${BASE_DIR:-}"
LEMON_OUT_DIR="${LEMON_OUT_DIR:-}"
GEN_LOG_DIR="${GEN_LOG_DIR:-}"
GEN_RUNNER_LOG="${GEN_RUNNER_LOG:-}"
REPORT_PATH="${REPORT_PATH:-}"
DRY_RUN=0
SKIP_LEMON=0
SKIP_GENERALIZATION=0

usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_grasp_baseline.sh [options]

Options:
  --dry-run              Print commands without running simulation.
  --run-id ID            Baseline run id.
  --lemon-runs N         Fixed lemon repetitions. Default: 5.
  --seed-start N         First random seed. Default: 0.
  --gen-count N          Number of random seeds. Default: 10.
  --parallel N           Parallel subprocesses / GPUs. Default: 4.
  --timeout-s N          Per-seed timeout. Default: 900.
  --skip-lemon           Skip fixed lemon regression.
  --skip-generalization  Skip random-seed generalization.
  -h, --help             Show this help.

Environment defaults:
  LEMON_RUNS=5 GEN_SEED_START=0 GEN_COUNT=10 GEN_PARALLEL=4 TIMEOUT_S=900
  MEMORY_MODE=isolated LOG_LEVEL=INFO RUN_ID=<timestamp>
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --run-id)
            RUN_ID="$2"
            shift 2
            ;;
        --lemon-runs)
            LEMON_RUNS="$2"
            shift 2
            ;;
        --seed-start)
            GEN_SEED_START="$2"
            shift 2
            ;;
        --gen-count)
            GEN_COUNT="$2"
            shift 2
            ;;
        --parallel)
            GEN_PARALLEL="$2"
            shift 2
            ;;
        --timeout-s)
            TIMEOUT_S="$2"
            shift 2
            ;;
        --skip-lemon)
            SKIP_LEMON=1
            shift
            ;;
        --skip-generalization)
            SKIP_GENERALIZATION=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

require_positive_int() {
    local name="$1"
    local value="$2"
    if ! [[ "${value}" =~ ^[0-9]+$ ]] || [[ "${value}" -lt 1 ]]; then
        echo "ERROR: ${name} must be a positive integer, got '${value}'." >&2
        exit 2
    fi
}

require_nonnegative_int() {
    local name="$1"
    local value="$2"
    if ! [[ "${value}" =~ ^[0-9]+$ ]]; then
        echo "ERROR: ${name} must be a non-negative integer, got '${value}'." >&2
        exit 2
    fi
}

require_positive_int "LEMON_RUNS" "${LEMON_RUNS}"
require_nonnegative_int "GEN_SEED_START" "${GEN_SEED_START}"
require_positive_int "GEN_COUNT" "${GEN_COUNT}"
require_positive_int "GEN_PARALLEL" "${GEN_PARALLEL}"
require_positive_int "TIMEOUT_S" "${TIMEOUT_S}"

GEN_RUN_ID="${GEN_RUN_ID:-${RUN_ID}-gen}"
BASE_DIR="${BASE_DIR:-runs/grasp_baseline/${RUN_ID}}"
LEMON_OUT_DIR="${LEMON_OUT_DIR:-${BASE_DIR}/lemon}"
GEN_LOG_DIR="${GEN_LOG_DIR:-logs/long_generalization/${GEN_RUN_ID}}"
GEN_RUNNER_LOG="${GEN_RUNNER_LOG:-${BASE_DIR}/generalization_runner.log}"
REPORT_PATH="${REPORT_PATH:-${BASE_DIR}/report.md}"

lemon_command_display() {
    printf 'OUT_DIR="%s" MEMORY_MODE="%s" LOG_LEVEL="%s" bash scripts/validate_lemon_grasp_multi.sh "%s"\n' \
        "${LEMON_OUT_DIR}" "${MEMORY_MODE}" "${LOG_LEVEL}" "${LEMON_RUNS}"
}

generalization_command_display() {
    printf 'MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python eval/run_long_generalization.py --seed-start %s --count %s --parallel %s --timeout-s %s --run-id "%s" --log-level "%s" 2>&1 | tee "%s"\n' \
        "${GEN_SEED_START}" "${GEN_COUNT}" "${GEN_PARALLEL}" "${TIMEOUT_S}" \
        "${GEN_RUN_ID}" "${LOG_LEVEL}" "${GEN_RUNNER_LOG}"
}

print_config() {
    echo "Grasp baseline run"
    echo "  RUN_ID        : ${RUN_ID}"
    echo "  BASE_DIR      : ${BASE_DIR}"
    echo "  LEMON_RUNS    : ${LEMON_RUNS}"
    echo "  GEN_SEEDS     : ${GEN_SEED_START}..$((GEN_SEED_START + GEN_COUNT - 1))"
    echo "  GEN_PARALLEL  : ${GEN_PARALLEL}"
    echo "  TIMEOUT_S     : ${TIMEOUT_S}"
    echo "  MEMORY_MODE   : ${MEMORY_MODE}"
    echo "  REPORT        : ${REPORT_PATH}"
}

append_file_or_missing() {
    local title="$1"
    local path="$2"
    {
        echo
        echo "## ${title}"
        echo
        echo "Path: \`${path}\`"
        echo
        if [[ -f "${path}" ]]; then
            echo '```text'
            cat "${path}"
            echo '```'
        else
            echo "_Missing._"
        fi
    } >> "${REPORT_PATH}"
}

write_report() {
    local lemon_status="$1"
    local gen_status="$2"
    local head_line
    head_line="$(git log --oneline -1 2>/dev/null || echo unknown)"

    mkdir -p "${BASE_DIR}"
    {
        echo "# Grasp Baseline Report"
        echo
        echo "- Run ID: \`${RUN_ID}\`"
        echo "- Generated: \`$(date -Iseconds)\`"
        echo "- HEAD: \`${head_line}\`"
        echo "- Lemon status: \`${lemon_status}\`"
        echo "- Generalization status: \`${gen_status}\`"
        echo
        echo "## Commands"
        echo
        echo "Fixed lemon regression:"
        echo
        echo '```bash'
        lemon_command_display
        echo '```'
        echo
        echo "Random-seed generalization:"
        echo
        echo '```bash'
        generalization_command_display
        echo '```'
        echo
        echo "## Output Paths"
        echo
        echo "- Baseline dir: \`${BASE_DIR}\`"
        echo "- Report: \`${REPORT_PATH}\`"
        echo "- Lemon summary CSV: \`${LEMON_OUT_DIR}/summary.csv\`"
        echo "- Generalization summary: \`${GEN_LOG_DIR}/summary.txt\`"
        echo "- Generalization JSON: \`${GEN_LOG_DIR}/summary.json\`"
        echo "- Generalization results: \`${GEN_LOG_DIR}/results.jsonl\`"
    } > "${REPORT_PATH}"

    append_file_or_missing "Fixed Lemon Summary CSV" "${LEMON_OUT_DIR}/summary.csv"
    append_file_or_missing "Generalization Summary" "${GEN_LOG_DIR}/summary.txt"
}

if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "DRY RUN"
    print_config
    echo
    echo "Fixed lemon command:"
    lemon_command_display
    echo
    echo "Generalization command:"
    generalization_command_display
    echo
    echo "Report path: ${REPORT_PATH}"
    exit 0
fi

if [[ -f ".env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
    echo "ERROR: DEEPSEEK_API_KEY not set (checked env and .env)." >&2
    exit 1
fi

mkdir -p "${BASE_DIR}" "${GEN_LOG_DIR}"
print_config | tee "${BASE_DIR}/config.txt"
git log --oneline -1 > "${BASE_DIR}/head.txt" 2>/dev/null || true

lemon_status=0
gen_status=0

if [[ "${SKIP_LEMON}" -eq 0 ]]; then
    echo
    echo "================================================================"
    echo "Fixed lemon regression"
    echo "================================================================"
    set +e
    OUT_DIR="${LEMON_OUT_DIR}" MEMORY_MODE="${MEMORY_MODE}" LOG_LEVEL="${LOG_LEVEL}" \
        bash scripts/validate_lemon_grasp_multi.sh "${LEMON_RUNS}" \
        2>&1 | tee "${BASE_DIR}/lemon_runner.log"
    lemon_status=${PIPESTATUS[0]}
    set -e
else
    echo "Skipping fixed lemon regression."
fi

if [[ "${SKIP_GENERALIZATION}" -eq 0 ]]; then
    echo
    echo "================================================================"
    echo "Random-seed generalization"
    echo "================================================================"
    set +e
    python eval/run_long_generalization.py \
        --seed-start "${GEN_SEED_START}" \
        --count "${GEN_COUNT}" \
        --parallel "${GEN_PARALLEL}" \
        --timeout-s "${TIMEOUT_S}" \
        --run-id "${GEN_RUN_ID}" \
        --log-level "${LOG_LEVEL}" \
        2>&1 | tee "${GEN_RUNNER_LOG}"
    gen_status=${PIPESTATUS[0]}
    set -e
else
    echo "Skipping random-seed generalization."
fi

write_report "${lemon_status}" "${gen_status}"

echo
echo "================================================================"
echo "Baseline report"
echo "================================================================"
echo "Report: ${REPORT_PATH}"
echo "Paste this file back for review:"
echo "  ${REPORT_PATH}"

if [[ "${lemon_status}" -ne 0 || "${gen_status}" -ne 0 ]]; then
    echo "ERROR: baseline completed with failures (lemon=${lemon_status}, generalization=${gen_status})." >&2
    exit 1
fi
