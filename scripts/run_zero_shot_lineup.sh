#!/bin/bash
# Sequentially launch each of the 7 lineup models on both GPUs (dual-instance
# load-balanced) and run zero-shot CTI-RCM eval on the full 1000-query split.
# Each model is kept in a separate vLLM lifecycle: launch → wait ready → eval → kill.
set -uo pipefail

LOGDIR=/home/sheng/cyber-ft/logs/experiments/zero_shot_lineup
mkdir -p "$LOGDIR"

# Pretty model name | HF repo | extra serve flags | --no-think for eval | max_tokens
# Order: non-thinking first (fast), thinking last (slow).
declare -a MODELS=(
  "granite_4_1_8b|ibm-granite/granite-4.1-8b|--max-model-len 24576 --gpu-memory-utilization 0.85|0|256"
  "phi4_mini_reasoning|microsoft/Phi-4-mini-reasoning|--max-model-len 24576 --gpu-memory-utilization 0.80 --trust-remote-code|0|2048"
  "deepseek_r1_distill|deepseek-ai/DeepSeek-R1-Distill-Llama-8B|--max-model-len 24576 --gpu-memory-utilization 0.80 --trust-remote-code|0|2048"
  "fdtn_sec_8b_r|fdtn-ai/Foundation-Sec-8B-Reasoning|--max-model-len 24576 --gpu-memory-utilization 0.80|0|2048"
  "qwen3_8b_think|Qwen/Qwen3-8B|--max-model-len 24576 --gpu-memory-utilization 0.80|0|2048"
  "gemma4_e4b_think|google/gemma-4-E4B-it|--max-model-len 24576 --gpu-memory-utilization 0.80|0|2048"
  "qwen3_5_9b_think|Qwen/Qwen3.5-9B|--max-model-len 24576 --gpu-memory-utilization 0.92 --enforce-eager|0|2048"
)

kill_vllm() {
  pkill -9 -f "vllm serve" 2>/dev/null || true
  pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
  sleep 15
}

launch_pair() {
  local repo="$1"; local flags="$2"; local tag="$3"
  CUDA_VISIBLE_DEVICES=0 nohup /home/sheng/miniconda3/bin/vllm serve "$repo" \
    --served-model-name "Qwen/Qwen2.5-7B-Instruct" \
    $flags --enable-prefix-caching --port 8000 \
    > "$LOGDIR/vllm_${tag}_gpu0.log" 2>&1 &
  disown
  CUDA_VISIBLE_DEVICES=1 nohup /home/sheng/miniconda3/bin/vllm serve "$repo" \
    --served-model-name "Qwen/Qwen2.5-7B-Instruct" \
    $flags --enable-prefix-caching --port 8001 \
    > "$LOGDIR/vllm_${tag}_gpu1.log" 2>&1 &
  disown
}

wait_ready() {
  for i in $(seq 1 90); do
    S0=$(curl -sf http://localhost:8000/v1/models > /dev/null 2>&1 && echo 1 || echo 0)
    S1=$(curl -sf http://localhost:8001/v1/models > /dev/null 2>&1 && echo 1 || echo 0)
    if [ "$S0" = "1" ] && [ "$S1" = "1" ]; then echo "  ready after ${i}*10s"; return 0; fi
    sleep 10
  done
  echo "  TIMEOUT waiting for servers"; return 1
}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cyber-ft
cd /home/sheng/cyber-ft

OVERALL_LOG="$LOGDIR/run_summary_$(date '+%Y%m%d_%H%M%S').log"
echo "Overall log: $OVERALL_LOG"
echo "Started $(date)" > "$OVERALL_LOG"

for entry in "${MODELS[@]}"; do
  IFS='|' read -r TAG REPO FLAGS NO_THINK MAX_TOKENS <<< "$entry"
  echo "" | tee -a "$OVERALL_LOG"
  echo "=== $TAG ($REPO) ===" | tee -a "$OVERALL_LOG"

  echo "  Killing prior vLLM..." | tee -a "$OVERALL_LOG"
  kill_vllm

  echo "  Launching $REPO on both GPUs..." | tee -a "$OVERALL_LOG"
  launch_pair "$REPO" "$FLAGS" "$TAG"

  if ! wait_ready; then
    echo "  FAILED: $TAG did not come up" | tee -a "$OVERALL_LOG"
    tail -20 "$LOGDIR/vllm_${TAG}_gpu0.log" | tee -a "$OVERALL_LOG"
    continue
  fi

  TS=$(date '+%Y%m%d_%H%M%S')
  EVAL_LOG="$LOGDIR/eval_${TAG}_${TS}.log"
  JSONL_LOG="$LOGDIR/eval_${TAG}_${TS}.jsonl"
  NO_THINK_FLAG=""
  if [ "$NO_THINK" = "1" ]; then NO_THINK_FLAG="--no-think"; fi

  echo "  Running zero-shot eval -> $EVAL_LOG" | tee -a "$OVERALL_LOG"
  python3 -u scripts/zero_shot_eval.py 1000 \
    --endpoint-a http://localhost:8000/v1/chat/completions \
    --endpoint-b http://localhost:8001/v1/chat/completions \
    --workers 16 \
    --max-tokens "$MAX_TOKENS" \
    $NO_THINK_FLAG \
    --log "$JSONL_LOG" \
    > "$EVAL_LOG" 2>&1 || true

  echo "  $TAG result:" | tee -a "$OVERALL_LOG"
  grep -A6 "Zero-shot CTI-RCM Results" "$EVAL_LOG" | tee -a "$OVERALL_LOG"
done

echo "" | tee -a "$OVERALL_LOG"
echo "Killing final vLLM..." | tee -a "$OVERALL_LOG"
kill_vllm

echo "" | tee -a "$OVERALL_LOG"
echo "ALL DONE $(date)" | tee -a "$OVERALL_LOG"
