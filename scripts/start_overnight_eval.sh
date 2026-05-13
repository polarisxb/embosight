#!/bin/bash
# ============================================================
# EmboSight 长跑纯泛化测试 — tmux 一键启动脚本
# 用法: bash scripts/start_overnight_eval.sh [run-id]
# 恢复: bash scripts/start_overnight_eval.sh [run-id] --resume
# ============================================================

set -euo pipefail

RUN_ID="${1:-overnight-gen-50}"
RESUME_FLAG="${2:-}"
SESSION="embosight-eval"
COUNT=50
PARALLEL=4
SEED_START=0
TIMEOUT_S=900
LOG_DIR="logs/long_generalization/${RUN_ID}"

# --- 颜色 ---
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== EmboSight Overnight Eval ===${NC}"
echo "  Run ID    : ${RUN_ID}"
echo "  Seeds     : ${SEED_START}..$(( SEED_START + COUNT - 1 ))"
echo "  Parallel  : ${PARALLEL}"
echo "  Timeout/s : ${TIMEOUT_S}s"
echo "  Log dir   : ${LOG_DIR}"
echo "  Resume    : ${RESUME_FLAG:-no}"
echo ""

# --- 构建命令 ---
CMD="python eval/run_long_generalization.py \
  --seed-start ${SEED_START} \
  --count ${COUNT} \
  --parallel ${PARALLEL} \
  --timeout-s ${TIMEOUT_S} \
  --run-id ${RUN_ID}"

if [ "${RESUME_FLAG}" = "--resume" ]; then
  CMD="${CMD} --resume"
fi

# --- 检查已有 session ---
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo -e "${YELLOW}tmux session '${SESSION}' already exists!${NC}"
  echo "  Attach:  tmux attach -t ${SESSION}"
  echo "  Kill:    tmux kill-session -t ${SESSION}"
  exit 1
fi

# --- 创建 tmux session (3 个窗格) ---
# 窗格布局:
#   ┌──────────────────────────────────┐
#   │         0: main (runner)         │
#   ├──────────────────┬───────────────┤
#   │  1: progress     │ 2: tail log   │
#   └──────────────────┴───────────────┘

tmux new-session -d -s "${SESSION}" -n eval

# 窗格 0: 主进程
tmux send-keys -t "${SESSION}:eval" "cd $(pwd) && ${CMD} 2>&1 | tee ${LOG_DIR}/runner.log" C-m

# 水平分割下半部分
tmux split-window -v -t "${SESSION}:eval"

# 窗格 1: 实时进度监控
tmux send-keys -t "${SESSION}:eval.1" "cd $(pwd) && watch -n 5 '\
  echo \"===== PROGRESS =====\"; \
  if [ -f ${LOG_DIR}/results.jsonl ]; then \
    DONE=\$(wc -l < ${LOG_DIR}/results.jsonl); \
    OK=\$(grep -c \"\\\"success\\\": true\" ${LOG_DIR}/results.jsonl 2>/dev/null || echo 0); \
    FAIL=\$(( DONE - OK )); \
    echo \"Completed: \${DONE} / ${COUNT}\"; \
    echo \"Success:   \${OK}\"; \
    echo \"Failed:    \${FAIL}\"; \
    if [ \${DONE} -gt 0 ]; then \
      RATE=\$(echo \"scale=1; \${OK} * 100 / \${DONE}\" | bc); \
      echo \"Rate:      \${RATE}%\"; \
    fi; \
    echo \"\"; \
    echo \"===== LATEST 5 =====\"; \
    tail -5 ${LOG_DIR}/results.jsonl | python3 -c \"import sys,json; \
      [print(f\\\"  {json.loads(l).get(chr(39)+'scenario_id'+chr(39),chr(39)+'?'+chr(39)):25s} {chr(39)+'OK'+chr(39) if json.loads(l).get(chr(39)+'success'+chr(39)) else chr(39)+'FAIL'+chr(39):5s} {json.loads(l).get(chr(39)+'time_s'+chr(39),0):6.1f}s\\\") for l in sys.stdin if l.strip()]\" 2>/dev/null; \
  else \
    echo \"Waiting for first result...\"; \
  fi'" C-m

# 垂直分割窗格 1
tmux split-window -h -t "${SESSION}:eval.1"

# 窗格 2: 实时 tail 最新日志
tmux send-keys -t "${SESSION}:eval.2" "cd $(pwd) && sleep 3 && tail -f ${LOG_DIR}/runner.log 2>/dev/null || echo 'Waiting for log file...'" C-m

# 聚焦到主窗格
tmux select-pane -t "${SESSION}:eval.0"

echo -e "${GREEN}tmux session '${SESSION}' created!${NC}"
echo ""
echo "  Attach (查看):   tmux attach -t ${SESSION}"
echo "  Detach (退出):   Ctrl+B, D"
echo "  切换窗格:        Ctrl+B, 方向键"
echo "  Kill (停止):     tmux kill-session -t ${SESSION}"
echo ""
echo -e "${YELLOW}现在 attach 进去看:${NC}"
echo "  tmux attach -t ${SESSION}"
