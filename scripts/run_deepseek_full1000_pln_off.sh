#!/bin/bash
# After the main orchestrator finishes, run DeepSeek-R1-Distill full-1000
# with PLN=0 and --timing-audit to produce the per-row jsonl for Fig. 6.

set -uo pipefail
LOG_DIR=logs/reeval_no_prefer_last_nvd
mkdir -p "$LOG_DIR"
ORCH_PID=${ORCH_PID_OVERRIDE:-2620223}

echo "[$(date -Iseconds)] Waiting for orchestrator PID $ORCH_PID to finish..."
while kill -0 "$ORCH_PID" 2>/dev/null; do
  sleep 60
done
echo "[$(date -Iseconds)] Orchestrator done. Starting DeepSeek full-1000."

# Clean any leftover vLLM
pkill -9 -f "vllm serve" 2>/dev/null
sleep 2
for pid in $(ps -eo pid,comm | awk '$2 ~ /^VLLM::Engine/ {print $1}'); do
  kill -9 "$pid" 2>/dev/null
done
sleep 15

stamp=$(date +%Y%m%d_%H%M%S)
serve_a="$LOG_DIR/vllm_deepseek_full1000_gpu0_${stamp}.log"
serve_b="$LOG_DIR/vllm_deepseek_full1000_gpu1_${stamp}.log"
eval_log="$LOG_DIR/eval_deepseek_full1000_no_pln_${stamp}.log"
fail_dst="$LOG_DIR/eval_failures_deepseek_full1000_${stamp}.jsonl"
timing_dst="$LOG_DIR/eval_timing_deepseek_full1000_${stamp}.jsonl"

CUDA_VISIBLE_DEVICES=0 nohup /home/sheng/miniconda3/bin/vllm serve deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --served-model-name "Qwen/Qwen2.5-7B-Instruct" \
  --max-model-len 24576 --gpu-memory-utilization 0.80 \
  --enable-prefix-caching --trust-remote-code --port 8000 > "$serve_a" 2>&1 &
disown
CUDA_VISIBLE_DEVICES=1 nohup /home/sheng/miniconda3/bin/vllm serve deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --served-model-name "Qwen/Qwen2.5-7B-Instruct" \
  --max-model-len 24576 --gpu-memory-utilization 0.80 \
  --enable-prefix-caching --trust-remote-code --port 8001 > "$serve_b" 2>&1 &
disown

for port in 8000 8001; do
  for i in $(seq 1 180); do
    if curl -s --max-time 2 "http://localhost:$port/v1/models" 2>/dev/null | grep -q "Qwen/Qwen2.5-7B-Instruct"; then
      echo "[$(date -Iseconds)] port $port ready" | tee -a "$eval_log"
      break
    fi
    sleep 5
  done
done

rm -f eval_failures_debug.jsonl

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cyber-ft
export CTI_RAG_LLM_ROUTER=1 CTI_RAG_RCM_ONLY=1 \
       CTI_RAG_CWE_HYDE=0 CTI_RAG_CWE_PHRASE_SELECTOR=0 \
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

python3 -u eval_rcm.py 1000 --timing-audit --debug-failures 2>&1 | tee -a "$eval_log"

# eval_rcm with --timing-audit writes a per-row timing jsonl. Find the most recent one.
latest_timing=$(ls -t logs/eval_timing_*.jsonl 2>/dev/null | head -1)
if [ -n "$latest_timing" ]; then
  cp "$latest_timing" "$timing_dst"
  echo "[$(date -Iseconds)] Timing archived: $timing_dst" | tee -a "$eval_log"
fi
if [ -f eval_failures_debug.jsonl ]; then
  cp eval_failures_debug.jsonl "$fail_dst"
  echo "[$(date -Iseconds)] Failures archived: $fail_dst" | tee -a "$eval_log"
fi

pkill -9 -f "vllm serve" 2>/dev/null
sleep 2
for pid in $(ps -eo pid,comm | awk '$2 ~ /VLLM::EngineCore/ {print $1}'); do
  kill -9 "$pid" 2>/dev/null
done

echo "[$(date -Iseconds)] DeepSeek full-1000 PLN-off complete." | tee -a "$eval_log"
