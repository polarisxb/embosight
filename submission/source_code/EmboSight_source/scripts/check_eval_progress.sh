#!/bin/bash
# ============================================================
# 快速检查长跑评估进度 (不用 attach tmux)
# 用法: bash scripts/check_eval_progress.sh [run-id]
# ============================================================

RUN_ID="${1:-overnight-gen-50}"
LOG_DIR="logs/long_generalization/${RUN_ID}"
RESULTS="${LOG_DIR}/results.jsonl"

if [ ! -f "${RESULTS}" ]; then
  echo "No results yet: ${RESULTS}"
  exit 0
fi

TOTAL=$(grep -c "scenario_id" "${LOG_DIR}/scenarios.yaml" 2>/dev/null || echo "?")
DONE=$(wc -l < "${RESULTS}")
OK=$(grep -c '"success": true' "${RESULTS}" 2>/dev/null || echo 0)
FAIL=$(( DONE - OK ))
ERRORS=$(grep -c '"error":' "${RESULTS}" 2>/dev/null || echo 0)
TIMEOUTS=$(grep -c '"timeout"' "${RESULTS}" 2>/dev/null || echo 0)

echo "============================================"
echo " EmboSight Long-Run Progress: ${RUN_ID}"
echo "============================================"
echo " Completed : ${DONE} / ${TOTAL}"
echo " Success   : ${OK}"
echo " Failed    : ${FAIL}"
echo " Errors    : ${ERRORS}"
echo " Timeouts  : ${TIMEOUTS}"

if [ "${DONE}" -gt 0 ]; then
  RATE=$(echo "scale=1; ${OK} * 100 / ${DONE}" | bc)
  echo " Rate      : ${RATE}%"
fi

echo ""
echo "--- Last 5 Results ---"
tail -5 "${RESULTS}" | python3 -c "
import sys, json
for line in sys.stdin:
    line = line.strip()
    if not line: continue
    try:
        r = json.loads(line)
        status = 'OK' if r.get('success') else 'FAIL'
        sid = r.get('scenario_id', '?')
        t = r.get('time_s', 0)
        obj = r.get('actual_object', '-')
        print(f'  {sid:25s} {status:5s} {t:6.1f}s  obj={obj}')
    except: pass
"

echo ""
echo "--- Timing ---"
if [ -f "${LOG_DIR}/manifest.json" ]; then
  START=$(python3 -c "import json; print(json.load(open('${LOG_DIR}/manifest.json'))['started_at'])" 2>/dev/null || echo "?")
  echo " Started: ${START}"
fi
echo " Now:     $(date -Iseconds)"
echo "============================================"
