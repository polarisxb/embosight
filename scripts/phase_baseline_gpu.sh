#!/usr/bin/env bash
# Phase 0.5 + Phase 1: GPU baseline measurement and joint probe.
#
# Run after Phase 0 (commit 04fe0d8) and Phase 3 (commit dcf9031) have
# been pushed to GPU server, before implementing Phase 2 (navigate primitive).
#
# Usage (either):
#   1) put DEEPSEEK_API_KEY=sk-... in .env at repo root, then:
#        bash scripts/phase_baseline_gpu.sh
#   2) explicit export:
#        export DEEPSEEK_API_KEY=sk-...
#        bash scripts/phase_baseline_gpu.sh
#
# Outputs:
#   runs/baseline_phase0/  - Episode logs and memory
#   /tmp/baseline.log      - Full stdout/stderr
#   /tmp/probe.log         - Joint probe output

set -euo pipefail

# Auto-load .env from repo root if present (matches scripts/run_agent.py behavior)
if [[ -f ".env" ]]; then
    echo "Loading environment from .env ..."
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
    echo "ERROR: DEEPSEEK_API_KEY not set (checked env and .env)." >&2
    echo "Either add DEEPSEEK_API_KEY=sk-... to .env or 'export DEEPSEEK_API_KEY=...'" >&2
    exit 1
fi

# Show current HEAD for record
echo "Current HEAD: $(git log --oneline -1)"
if ! git log --oneline -10 | grep -q "Phase 3"; then
    echo "WARNING: Phase 3 commit not detected in last 10 commits."
    echo "Make sure you have pulled latest from main (commit dcf9031 or later)."
fi

echo
echo "================================================================"
echo "Phase 1: Joint probe (~30s)"
echo "================================================================"
python scripts/probe_mobilebase_joints.py 2>&1 | tee /tmp/probe.log

echo
echo "================================================================"
echo "Phase 0.5: Baseline measurement on fixed_seed_discover_001"
echo "================================================================"
mkdir -p runs/baseline_phase0/memory
python -m eval.run_fixed \
    --scenario fixed_seed_discover_001 \
    --memory-dir runs/baseline_phase0/memory \
    --log-level INFO \
    2>&1 | tee /tmp/baseline.log

echo
echo "================================================================"
echo "Baseline summary"
echo "================================================================"
echo
echo "max_steps reached count:  $(grep -c 'max_steps reached' /tmp/baseline.log || true)"
echo "stalled count:            $(grep -c '\[move_arm_to\] stalled' /tmp/baseline.log || true)"
echo "ik_unreachable count:     $(grep -c 'ik_unreachable' /tmp/baseline.log || true)"
echo "slipped_lift count:       $(grep -c 'slipped_lift' /tmp/baseline.log || true)"
echo "hit_z_floor count:        $(grep -c 'hit_z_floor' /tmp/baseline.log || true)"
echo "gripper_empty count:      $(grep -c 'gripper_empty' /tmp/baseline.log || true)"
echo
echo "Episode result lines:"
grep -E '(Episode result|final|success)' /tmp/baseline.log | tail -10 || true
echo
echo "Logs saved:"
echo "  /tmp/probe.log     - joint structure"
echo "  /tmp/baseline.log  - full episode log"
echo "  runs/baseline_phase0/  - memory + episode dump"
echo
echo "Next: paste /tmp/probe.log into chat so Phase 2 can be implemented."
