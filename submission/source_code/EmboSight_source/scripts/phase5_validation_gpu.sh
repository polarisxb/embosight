#!/usr/bin/env bash
# Phase 5: GPU validation of the navigation refactor (Phases 0/2/3/4).
#
# Compares after-refactor behavior against the Phase 0.5 baseline
# (see docs/07_navigation_refactor_design.md appendix D.1).
#
# Run on GPU after pulling the Phase 4 commit (or later) to main.
#
# Usage (.env is auto-loaded):
#   bash scripts/phase5_validation_gpu.sh
#
# Outputs (all kept on disk for archival):
#   runs/after_refactor/tupperware/      - single-scenario reproducer
#   runs/after_refactor/memory/          - fresh v6.2 memory store
#   /tmp/after_tupperware.log            - episode log (single scenario)
#
# Optional second pass (uncomment near bottom):
#   /tmp/after_seeds.log                 - 5-seed generalization sweep

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

echo "Current HEAD: $(git log --oneline -1)"
if ! git log --oneline -5 | grep -q "Phase 4"; then
    echo "WARNING: Phase 4 commit not detected in last 5 commits."
    echo "Make sure you have pulled latest from main (commit 34e53f1 or later)."
fi

mkdir -p runs/after_refactor/memory
mkdir -p runs/after_refactor/tupperware

echo
echo "================================================================"
echo "Phase 5.1: Single-scenario reproducer (tupperware, seed=42)"
echo "================================================================"
python -m eval.run_fixed \
    --scenario fixed_seed_discover_001 \
    --memory-dir runs/after_refactor/memory \
    --log-level INFO \
    2>&1 | tee /tmp/after_tupperware.log

echo
echo "================================================================"
echo "Phase 5.1 summary (after vs baseline from D.1)"
echo "================================================================"

count() { grep -c "$1" /tmp/after_tupperware.log || true; }

NAV=$(count "\[navigate\]")
STALL=$(count "max_steps reached")
IK=$(count "ik_unreachable")
SLIP=$(count "slipped_lift")
HITZ=$(count "hit_z_floor")
GE=$(count "gripper_empty")

cat <<SUMMARY

Metric                 baseline (D.1)   after_refactor
---------------------  ---------------  --------------
[navigate] triggers    0                ${NAV}
max_steps reached      3                ${STALL}
ik_unreachable         3                ${IK}
slipped_lift           0                ${SLIP}
hit_z_floor            0                ${HITZ}
gripper_empty          0                ${GE}

Episode result lines:
SUMMARY
grep -E '(EPISODE RESULT|success |steps |time |reason )' /tmp/after_tupperware.log | tail -20 || true

echo
echo "================================================================"
echo "Detailed navigate / move_arm_to traces"
echo "================================================================"
grep -E "\[navigate\]|max_steps reached|\[base\] detected" /tmp/after_tupperware.log | head -30 || true

echo
echo "================================================================"
echo "Outputs:"
echo "  /tmp/after_tupperware.log       - full single-scenario log"
echo "  runs/after_refactor/memory/     - fresh v6.2 memory entries"
echo "================================================================"

# -------------------------------------------------------------------
# OPTIONAL: 5-seed generalization sweep (uncomment to run, ~20 min).
# Disabled by default to keep the primary validation fast.
# -------------------------------------------------------------------
# echo
# echo "Phase 5.2 (optional): 5-seed sweep (~20 min)"
# python -m eval.run_long_generalization \
#     --seed-start 200 --count 5 --parallel 1 \
#     --run-id after_refactor --timeout-s 600 \
#     2>&1 | tee /tmp/after_seeds.log
#
# echo
# echo "5-seed summary:"
# echo "  success cases:    $(grep -c 'success.*True' /tmp/after_seeds.log || true)"
# echo "  failure cases:    $(grep -c 'success.*False' /tmp/after_seeds.log || true)"
# echo "  navigate calls:   $(grep -c '\[navigate\]' /tmp/after_seeds.log || true)"
# echo "  stalls:           $(grep -c 'max_steps reached' /tmp/after_seeds.log || true)"
