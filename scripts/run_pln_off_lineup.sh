#!/bin/bash
# Orchestrate prefer-last-NVD=0 mapped-903 eval for the paper lineup.
# Runs models sequentially: DeepSeek (already running) → Phi-4-mini → Granite 4.1-8B.
# Each run archives its eval log + failure jsonl into logs/reeval_no_prefer_last_nvd/.

set -uo pipefail
LOG_DIR=logs/reeval_no_prefer_last_nvd
mkdir -p "$LOG_DIR"

wait_endpoint() {
  local port=$1
  for i in $(seq 1 180); do
    if curl -s --max-time 2 "http://localhost:$port/v1/models" 2>/dev/null | grep -q "Qwen/Qwen2.5-7B-Instruct"; then
      return 0
    fi
    sleep 5
  done
  return 1
}

kill_vllm() {
  pkill -9 -f "vllm serve" 2>/dev/null
  sleep 2
  # Kill orphaned EngineCore by PID. `comm` truncates to 15 chars so the
  # displayed name is "VLLM::EngineCor" — match the prefix.
  for pid in $(ps -eo pid,comm | awk '$2 ~ /^VLLM::Engine/ {print $1}'); do
    kill -9 "$pid" 2>/dev/null
  done
  sleep 15
}

run_one() {
  local tag=$1
  local model=$2
  local extra_serve_args=$3   # e.g. "--max-model-len 16384 --gpu-memory-utilization 0.85"

  local stamp=$(date +%Y%m%d_%H%M%S)
  local serve_a="$LOG_DIR/vllm_${tag}_gpu0_${stamp}.log"
  local serve_b="$LOG_DIR/vllm_${tag}_gpu1_${stamp}.log"
  local eval_log="$LOG_DIR/eval_${tag}_no_pln_${stamp}.log"
  local fail_dst="$LOG_DIR/eval_failures_${tag}_${stamp}.jsonl"

  echo "[$(date -Iseconds)] === MODEL: $tag ($model) ===" | tee -a "$eval_log"

  echo "[$(date -Iseconds)] Killing any existing vLLM servers..." | tee -a "$eval_log"
  kill_vllm

  echo "[$(date -Iseconds)] Launching GPU0 server..." | tee -a "$eval_log"
  CUDA_VISIBLE_DEVICES=0 nohup /home/sheng/miniconda3/bin/vllm serve "$model" \
    --served-model-name "Qwen/Qwen2.5-7B-Instruct" \
    --enable-prefix-caching --trust-remote-code --port 8000 $extra_serve_args \
    > "$serve_a" 2>&1 &
  disown
  echo "[$(date -Iseconds)] Launching GPU1 server..." | tee -a "$eval_log"
  CUDA_VISIBLE_DEVICES=1 nohup /home/sheng/miniconda3/bin/vllm serve "$model" \
    --served-model-name "Qwen/Qwen2.5-7B-Instruct" \
    --enable-prefix-caching --trust-remote-code --port 8001 $extra_serve_args \
    > "$serve_b" 2>&1 &
  disown

  echo "[$(date -Iseconds)] Waiting for endpoints..." | tee -a "$eval_log"
  if ! wait_endpoint 8000; then
    echo "[ERROR] port 8000 never ready for $tag" | tee -a "$eval_log"
    return 1
  fi
  if ! wait_endpoint 8001; then
    echo "[ERROR] port 8001 never ready for $tag" | tee -a "$eval_log"
    return 1
  fi
  echo "[$(date -Iseconds)] Both endpoints ready." | tee -a "$eval_log"

  rm -f eval_failures_debug.jsonl

  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate cyber-ft
  export CTI_RAG_LLM_ROUTER=1 \
         CTI_RAG_RCM_ONLY=1 \
         CTI_RAG_CWE_HYDE=0 \
         CTI_RAG_CWE_PHRASE_SELECTOR=0 \
         CTI_RAG_CWE_MAPPED_FAST_CONTEXT=0 \
         CTI_RAG_MAPPED_BRIDGE_PREFER_LAST_NVD=0 \
         CTI_RAG_LLM_MAPPED_RESPONSE_BUDGET=2048 \
         CTI_RAG_LLM_RESPONSE_BUDGET=2048 \
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
    echo "[$(date -Iseconds)] Archived failures to $fail_dst" | tee -a "$eval_log"
  fi
  echo "[$(date -Iseconds)] === FINISHED: $tag ===" | tee -a "$eval_log"
}

# Skip DeepSeek if its run is already in progress / done; the in-flight launcher
# already handles archival. We just orchestrate Phi-4-mini and Granite.

# Wait for the in-flight DeepSeek eval (mapped-903) to finish.
echo "[$(date -Iseconds)] Waiting for in-flight DeepSeek eval to finish..."
while pgrep -f "eval_rcm.py 1000 --nvd-mapped" >/dev/null 2>&1; do
  sleep 30
done
echo "[$(date -Iseconds)] DeepSeek eval no longer running."

# Phi-4-mini-reasoning (3.8B)
run_one "phi4_mini_reasoning" "microsoft/Phi-4-mini-reasoning" \
  "--max-model-len 24576 --gpu-memory-utilization 0.80"

# IBM Granite 4.1-8B
run_one "granite_4_1_8b" "ibm-granite/granite-4.1-8b" \
  "--max-model-len 16384 --gpu-memory-utilization 0.85"

# Cleanup at the very end
kill_vllm
echo "[$(date -Iseconds)] All lineup runs complete. vLLM cleaned up."
