#!/bin/bash
# Continue context-prepended CTI-Bench prompt evals after the Phi mapped smoke.
# Runs remaining mapped slices, then unmapped slices, then DeepSeek full-1000.

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

run_eval() {
  local tag=$1
  local model=$2
  local subset=$3
  local extra_serve_args=$4
  local enable_thinking=$5
  local max_model_len_env=$6
  local timing_audit=$7

  local stamp serve_a serve_b eval_log fail_dst timing_dst args
  stamp=$(date +%Y%m%d_%H%M%S)
  serve_a="$LOG_DIR/vllm_${tag}_${subset}_gpu0_${stamp}.log"
  serve_b="$LOG_DIR/vllm_${tag}_${subset}_gpu1_${stamp}.log"
  eval_log="$LOG_DIR/eval_${tag}_${subset}_context_prompt_${stamp}.log"
  fail_dst="$LOG_DIR/eval_failures_${tag}_${subset}_context_prompt_${stamp}.jsonl"
  timing_dst="$LOG_DIR/eval_timing_${tag}_${subset}_context_prompt_${stamp}.jsonl"

  echo "[$(date -Iseconds)] === $tag / $subset / context prompt ==="
  kill_vllm

  CUDA_VISIBLE_DEVICES=0 nohup /home/sheng/miniconda3/bin/vllm serve "$model" \
    --served-model-name "Qwen/Qwen2.5-7B-Instruct" \
    --enable-prefix-caching --trust-remote-code --port 8000 $extra_serve_args \
    > "$serve_a" 2>&1 &
  disown

  CUDA_VISIBLE_DEVICES=1 nohup /home/sheng/miniconda3/bin/vllm serve "$model" \
    --served-model-name "Qwen/Qwen2.5-7B-Instruct" \
    --enable-prefix-caching --trust-remote-code --port 8001 $extra_serve_args \
    > "$serve_b" 2>&1 &
  disown

  echo "[$(date -Iseconds)] waiting for $tag endpoints..."
  wait_endpoint 8000 || { echo "[ERROR] $tag port 8000 not ready"; return 1; }
  wait_endpoint 8001 || { echo "[ERROR] $tag port 8001 not ready"; return 1; }
  echo "[$(date -Iseconds)] endpoints ready for $tag"

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
         CTI_RAG_MAPPED_BRIDGE_SOFT_CONTEXT=0 \
         CTI_RAG_UNMAPPED_AGGREGATE=0 \
         CTI_RAG_LLM_MAPPED_ENDPOINT=http://localhost:8001/v1/chat/completions \
         CTI_RAG_LLM_UNMAPPED_ENDPOINT=http://localhost:8000/v1/chat/completions \
         CTI_RAG_LLM_ENDPOINT=http://localhost:8000/v1/chat/completions \
         CTI_RAG_EMBEDDER_DEVICE=cpu \
         CTI_RAG_EMBEDDER_SECOND_DEVICE=cpu \
         CTI_RAG_CWE_CROSSENCODER_DEVICE=cpu \
         CTI_RAG_EVAL_WORKERS=16

  if [ "$enable_thinking" = "1" ]; then
    export CTI_RAG_LLM_ENABLE_THINKING=1
  else
    unset CTI_RAG_LLM_ENABLE_THINKING
  fi

  if [ -n "$max_model_len_env" ]; then
    export CTI_RAG_LLM_MAX_MODEL_LEN="$max_model_len_env"
  else
    unset CTI_RAG_LLM_MAX_MODEL_LEN
  fi

  args=(1000 --debug-failures)
  if [ "$subset" = "mapped903" ]; then
    args+=(--nvd-mapped)
  elif [ "$subset" = "unmapped97" ]; then
    args+=(--nvd-unmapped)
  fi
  if [ "$timing_audit" = "1" ]; then
    args+=(--timing-audit)
  fi

  python3 -u eval_rcm.py "${args[@]}" > "$eval_log" 2>&1

  if [ -f eval_failures_debug.jsonl ]; then
    cp eval_failures_debug.jsonl "$fail_dst"
  fi

  if [ "$timing_audit" = "1" ]; then
    latest_timing=$(ls -t logs/eval_timing_*.jsonl 2>/dev/null | head -1)
    if [ -n "${latest_timing:-}" ]; then
      cp "$latest_timing" "$timing_dst"
    fi
  fi

  grep -n -A8 -B2 "CTI-RCM Results" "$eval_log" || true
  echo "[$(date -Iseconds)] archived log: $eval_log"
  echo "[$(date -Iseconds)] archived failures: $fail_dst"
}

# Remaining mapped-903 after the Phi smoke.
run_eval "deepseek" "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" "mapped903" \
  "--max-model-len 24576 --gpu-memory-utilization 0.80" "0" "24576" "0"
run_eval "granite_4_1_8b" "ibm-granite/granite-4.1-8b" "mapped903" \
  "--max-model-len 16384 --gpu-memory-utilization 0.85" "0" "16384" "0"
run_eval "gemma4" "google/gemma-4-E4B-it" "mapped903" \
  "--max-model-len 24576 --gpu-memory-utilization 0.80" "1" "24576" "0"

# Unmapped-97 for all four models under the same prompt.
run_eval "deepseek" "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" "unmapped97" \
  "--max-model-len 24576 --gpu-memory-utilization 0.80" "0" "24576" "0"
run_eval "phi4_mini_reasoning" "microsoft/Phi-4-mini-reasoning" "unmapped97" \
  "--max-model-len 24576 --gpu-memory-utilization 0.80" "0" "24576" "0"
run_eval "granite_4_1_8b" "ibm-granite/granite-4.1-8b" "unmapped97" \
  "--max-model-len 16384 --gpu-memory-utilization 0.85" "0" "16384" "0"
run_eval "gemma4" "google/gemma-4-E4B-it" "unmapped97" \
  "--max-model-len 24576 --gpu-memory-utilization 0.80" "1" "24576" "0"

# Final headline audit.
run_eval "deepseek" "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" "full1000" \
  "--max-model-len 24576 --gpu-memory-utilization 0.80" "0" "24576" "1"

kill_vllm
echo "[$(date -Iseconds)] context-prompt remaining lineup complete; vLLM cleaned up."
