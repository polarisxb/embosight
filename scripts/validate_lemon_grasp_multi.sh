#!/usr/bin/env bash
# Repeated validation for the fixed seed=42 lemon grasp scenario.
#
# Usage:
#   bash scripts/validate_lemon_grasp_multi.sh
#   bash scripts/validate_lemon_grasp_multi.sh 10
#
# Optional environment variables:
#   RUNS=5                 number of runs when no positional arg is given
#   OUT_DIR=...            directory for logs and summary.csv
#   MEMORY_MODE=isolated   isolated | shared
#   LOG_LEVEL=INFO         eval log level
#   SLEEP_BETWEEN=0        seconds to sleep between runs

set -euo pipefail

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

RUNS="${1:-${RUNS:-5}}"
SCENARIO="${SCENARIO:-fixed_lemon_001}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
MEMORY_MODE="${MEMORY_MODE:-isolated}"
SLEEP_BETWEEN="${SLEEP_BETWEEN:-0}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-runs/lemon_validation_multi/${STAMP}}"
SUMMARY_CSV="${OUT_DIR}/summary.csv"

if ! [[ "${RUNS}" =~ ^[0-9]+$ ]] || [[ "${RUNS}" -lt 1 ]]; then
    echo "ERROR: RUNS must be a positive integer, got '${RUNS}'." >&2
    exit 1
fi

if [[ "${MEMORY_MODE}" != "isolated" && "${MEMORY_MODE}" != "shared" ]]; then
    echo "ERROR: MEMORY_MODE must be 'isolated' or 'shared'." >&2
    exit 1
fi

mkdir -p "${OUT_DIR}"

count_matches() {
    local pattern="$1"
    local file="$2"
    grep -cE "${pattern}" "${file}" || true
}

last_field() {
    local pattern="$1"
    local file="$2"
    grep -E "${pattern}" "${file}" | tail -1 | sed -E 's/.*:[[:space:]]*//' || true
}

extract_failure_mode() {
    local file="$1"
    grep -E '"grasp_failure_mode":' "${file}" \
        | tail -1 \
        | sed -E 's/.*"grasp_failure_mode":[[:space:]]*"?([^",}]*)"?[,}].*/\1/' \
        || true
}

echo "Current HEAD: $(git log --oneline -1)"
echo "Scenario: ${SCENARIO}"
echo "Runs: ${RUNS}"
echo "Memory mode: ${MEMORY_MODE}"
echo "Output dir: ${OUT_DIR}"
echo

echo "run,exit_code,success,pre_close_abort,grasp_confirmed,micro_lift_ok,post_lift_verified,no_grasp,object_not_lifted,steps,time,grasp_failure_mode,log" \
    > "${SUMMARY_CSV}"

successes=0
python_failures=0

for run_idx in $(seq 1 "${RUNS}"); do
    log_path="${OUT_DIR}/run_${run_idx}.log"
    if [[ "${MEMORY_MODE}" == "shared" ]]; then
        memory_dir="${OUT_DIR}/memory_shared"
    else
        memory_dir="${OUT_DIR}/memory_run_${run_idx}"
    fi
    mkdir -p "${memory_dir}"

    echo
    echo "================================================================"
    echo "Run ${run_idx}/${RUNS}: ${SCENARIO}"
    echo "Log: ${log_path}"
    echo "Memory: ${memory_dir}"
    echo "================================================================"

    set +e
    python -m eval.run_fixed \
        --scenario "${SCENARIO}" \
        --memory-dir "${memory_dir}" \
        --log-level "${LOG_LEVEL}" \
        2>&1 | tee "${log_path}"
    exit_code=${PIPESTATUS[0]}
    set -e

    if [[ "${exit_code}" -ne 0 ]]; then
        python_failures=$((python_failures + 1))
    fi

    if grep -qE '^success[[:space:]]*:[[:space:]]*True' "${log_path}"; then
        success="True"
        successes=$((successes + 1))
    else
        success="False"
    fi

    pre_close_abort="$(count_matches '\[pre_close_align\] abort' "${log_path}")"
    grasp_confirmed="$(count_matches '\[close_gripper\] grasp confirmed' "${log_path}")"
    micro_lift_ok="$(count_matches '\[micro_lift\].*follows=True' "${log_path}")"
    post_lift_verified="$(count_matches '\[act\] post-lift verified' "${log_path}")"
    no_grasp="$(count_matches '\[close_gripper\] no grasp' "${log_path}")"
    object_not_lifted="$(count_matches 'object NOT lifted' "${log_path}")"
    steps="$(last_field '^steps[[:space:]]*:' "${log_path}")"
    time_s="$(last_field '^time[[:space:]]*:' "${log_path}")"
    failure_mode="$(extract_failure_mode "${log_path}")"

    echo "${run_idx},${exit_code},${success},${pre_close_abort},${grasp_confirmed},${micro_lift_ok},${post_lift_verified},${no_grasp},${object_not_lifted},${steps},${time_s},${failure_mode},${log_path}" \
        >> "${SUMMARY_CSV}"

    echo
    echo "Run ${run_idx} summary:"
    echo "  success: ${success}"
    echo "  pre_close_abort: ${pre_close_abort}"
    echo "  grasp_confirmed: ${grasp_confirmed}"
    echo "  micro_lift_ok: ${micro_lift_ok}"
    echo "  post_lift_verified: ${post_lift_verified}"
    echo "  no_grasp: ${no_grasp}"
    echo "  object_not_lifted: ${object_not_lifted}"
    echo "  grasp_failure_mode: ${failure_mode:-unknown}"

    if [[ "${run_idx}" -lt "${RUNS}" && "${SLEEP_BETWEEN}" != "0" ]]; then
        sleep "${SLEEP_BETWEEN}"
    fi
done

echo
echo "================================================================"
echo "Multi-run summary"
echo "================================================================"
cat "${SUMMARY_CSV}"
echo
echo "Successes: ${successes}/${RUNS}"
echo "Python process failures: ${python_failures}"
echo "Logs: ${OUT_DIR}"
echo "Summary CSV: ${SUMMARY_CSV}"

if [[ "${successes}" -ne "${RUNS}" || "${python_failures}" -ne 0 ]]; then
    exit 1
fi
