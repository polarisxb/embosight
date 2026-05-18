#!/usr/bin/env bash
# Validation: grasp the lemon (geometrically graspable by Panda)
# at the SAME seed=42 scene where tupperware was tested.
#
# Run 9 confirmed tupperware is physically ungraspable (10-15cm wide
# > 8cm gripper opening). Lemon is in the same scene as
# distr_counter_main with a 5-6cm round shape that fits the gripper.
#
# This script validates that the navigation + grasp pipeline (Phase 4
# + Phase 7 IK regression detection + Phase 8b adaptive offset) works
# end-to-end on a graspable target.
#
# Expected outcome: success=True on lemon. Failure means a real
# pipeline bug remains; success means tupperware failure was purely
# a hardware/geometry limit.
#
# Usage (.env auto-loaded):
#   bash scripts/validate_lemon_grasp.sh

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

mkdir -p runs/lemon_validation/memory

echo
echo "================================================================"
echo "Lemon grasp validation (seed=42, query='pick up the lemon')"
echo "================================================================"
python -m eval.run_fixed \
    --scenario fixed_lemon_001 \
    --memory-dir runs/lemon_validation/memory \
    --log-level INFO \
    2>&1 | tee /tmp/lemon_validation.log

echo
echo "================================================================"
echo "Result summary"
echo "================================================================"
grep -E '(EPISODE RESULT|success |steps |time |reason |grasp_failure_mode)' \
    /tmp/lemon_validation.log | tail -20 || true

echo
echo "================================================================"
echo "Key pipeline signals (Phase 4 + 7 + 8b validation)"
echo "================================================================"
echo "[navigate] teleports:"
grep -E "\[navigate\] teleported" /tmp/lemon_validation.log || true
echo
echo "[act] adaptive offset usage:"
grep -E "\[act\] adaptive offset" /tmp/lemon_validation.log || true
echo
echo "IK-regression detections (Phase 7 step 3):"
grep -E "IK-unreachable regression" /tmp/lemon_validation.log || true
echo
echo "Post-nudge lateral re-align:"
grep -E "post-nudge lateral re-align" /tmp/lemon_validation.log || true
echo
echo "Pre-grasp alignment:"
grep -E "\[pre_grasp_align\]" /tmp/lemon_validation.log || true
echo
echo "Descend outcomes:"
grep -E "\[descend\]" /tmp/lemon_validation.log || true
echo
echo "Gripper outcomes:"
grep -E "\[close_gripper\]|\[gripper\]|object NOT lifted|object lifted" \
    /tmp/lemon_validation.log || true
