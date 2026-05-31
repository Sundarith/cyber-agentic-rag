#!/bin/bash
# Smoke test the context-prepended CTI-Bench prompt on the known-sensitive
# Phi-4-mini mapped-903 slice.

set -uo pipefail

LOG_DIR=logs/reeval_context_prompt
mkdir -p "$LOG_DIR"

wait_endpoint() {
  local port=$1
  for _ in $(seq 1 180); do
    if curl -s --max-time 2 "http://localhost:$port/v1/models" 2>/dev/null | grep -q "Qwen/Qwen2.5-7B-Instruct"; then
      return 0
    fi
    sleep 5
  done
  return 1
}

kill_vllm() {
  for pid in $(ps -eo pid,args | awk '/[v]llm serve/ {print $1}'); do
    kill -9 "$pid" 2>/dev/null || true
  done
  sleep 2
  for pid in $(ps -eo pid,comm | awk '$2 ~ /^VLLM::Engine/ {print $1}'); do
    kill -9 "$pid" 2>/dev/null || true
  done
  sleep 15
}

stamp=$(date +%Y%m%d_%H%M%S)
serve_a="$LOG_DIR/vllm_phi4_mapped_context_gpu0_${stamp}.log"
serve_b="$LOG_DIR/vllm_phi4_mapped_context_gpu1_${stamp}.log"
eval_log="$LOG_DIR/eval_phi4_mapped903_context_prompt_${stamp}.log"
fail_dst="$LOG_DIR/eval_failures_phi4_mapped903_context_prompt_${stamp}.jsonl"

echo "[$(date -Iseconds)] === Phi mapped-903 context-prompt smoke ===" | tee -a "$eval_log"
kill_vllm

CUDA_VISIBLE_DEVICES=0 nohup /home/sheng/miniconda3/bin/vllm serve "microsoft/Phi-4-mini-reasoning" \
  --served-model-name "Qwen/Qwen2.5-7B-Instruct" \
  --enable-prefix-caching --trust-remote-code --port 8000 \
  --max-model-len 24576 --gpu-memory-utilization 0.80 \
  > "$serve_a" 2>&1 &
disown

CUDA_VISIBLE_DEVICES=1 nohup /home/sheng/miniconda3/bin/vllm serve "microsoft/Phi-4-mini-reasoning" \
  --served-model-name "Qwen/Qwen2.5-7B-Instruct" \
  --enable-prefix-caching --trust-remote-code --port 8001 \
  --max-model-len 24576 --gpu-memory-utilization 0.80 \
  > "$serve_b" 2>&1 &
disown

echo "[$(date -Iseconds)] waiting for endpoints..." | tee -a "$eval_log"
wait_endpoint 8000 || { echo "[ERROR] port 8000 not ready" | tee -a "$eval_log"; exit 1; }
wait_endpoint 8001 || { echo "[ERROR] port 8001 not ready" | tee -a "$eval_log"; exit 1; }
echo "[$(date -Iseconds)] endpoints ready" | tee -a "$eval_log"

rm -f eval_failures_debug.jsonl

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cyber-ft

export CTI_RAG_LLM_ROUTER=1 \
       CTI_RAG_RCM_ONLY=1 \
       CTI_RAG_PROMPT_CONTEXT_ONLY=1 \
       CTI_RAG_CWE_HYDE=0 \
       CTI_RAG_CWE_PHRASE_SELECTOR=0 \
       CTI_RAG_CWE_MAPPED_FAST_CONTEXT=0 \
       CTI_RAG_MAPPED_BRIDGE_PREFER_LAST_NVD=0 \
       CTI_RAG_LLM_MAPPED_RESPONSE_BUDGET=2048 \
       CTI_RAG_LLM_RESPONSE_BUDGET=2048 \
       CTI_RAG_LLM_MAX_MODEL_LEN=24576 \
       CTI_RAG_MAPPED_BRIDGE_SOFT_CONTEXT=0 \
       CTI_RAG_UNMAPPED_AGGREGATE=0 \
       CTI_RAG_LLM_MAPPED_ENDPOINT=http://localhost:8001/v1/chat/completions \
       CTI_RAG_LLM_UNMAPPED_ENDPOINT=http://localhost:8000/v1/chat/completions \
       CTI_RAG_LLM_ENDPOINT=http://localhost:8000/v1/chat/completions \
       CTI_RAG_EMBEDDER_DEVICE=cpu \
       CTI_RAG_EMBEDDER_SECOND_DEVICE=cpu \
       CTI_RAG_CWE_CROSSENCODER_DEVICE=cpu \
       CTI_RAG_EVAL_WORKERS=16

python3 -u eval_rcm.py 1000 --nvd-mapped --debug-failures 2>&1 | tee -a "$eval_log"

if [ -f eval_failures_debug.jsonl ]; then
  cp eval_failures_debug.jsonl "$fail_dst"
  echo "[$(date -Iseconds)] archived failures to $fail_dst" | tee -a "$eval_log"
fi

kill_vllm
echo "[$(date -Iseconds)] done; vLLM cleaned up" | tee -a "$eval_log"
