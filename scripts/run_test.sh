#!/usr/bin/env bash
# 서버 기동 → 헬스 대기 → test 실행(시드 2) → 서버 종료. 진행 로그는 results/test_chain.log
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=results/test_chain.log
mkdir -p results
echo "[$(date +%T)] server start" >> "$LOG"
bash scripts/serve.sh > results/vllm_server.log 2>&1 &
SRV=$!
for i in $(seq 1 180); do
  curl -sf localhost:8000/health > /dev/null && break
  sleep 2
done
echo "[$(date +%T)] server healthy (wait ${i}x2s)" >> "$LOG"
~/vllm-venv/bin/python rca/run_experiment.py --run test --seed 2 --reps 4 --gap-sec 60 >> "$LOG" 2>&1
echo "[$(date +%T)] experiment exit $?" >> "$LOG"
kill $SRV; wait $SRV 2>/dev/null
touch results/test/_DONE
echo "[$(date +%T)] done" >> "$LOG"
