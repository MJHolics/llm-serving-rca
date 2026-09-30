#!/usr/bin/env bash
# vLLM 서버 기동. KV 캐시 블록 수를 고정해 "더 큰 모델을 작은 GPU에 올린 상황"의 용량 제약을 만든다.
# (1.5B 모델에 16GB를 다 주면 KV 여유가 너무 커서 긴 컨텍스트 장애가 재현되지 않는다)
set -euo pipefail
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
export VLLM_USE_FLASHINFER_SAMPLER=0
BLOCKS="${BLOCKS:-2048}"   # 2048 블록 × 16 토큰 = 32,768 토큰
exec ~/vllm-venv/bin/vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --port 8000 \
  --gpu-memory-utilization 0.55 \
  --max-model-len 4096 \
  --num-gpu-blocks-override "$BLOCKS" \
  --max-num-seqs 64
