#!/usr/bin/env bash
set -euo pipefail

# Requires two OpenAI-compatible vLLM servers:
#   - Foundation-Sec-8B-Reasoning on port 8000 for NVD-unmapped CVEs
#   - Granite 4.1 8B on port 8001 for NVD-mapped CVEs
#
# Both servers must expose the fixed alias Qwen/Qwen2.5-7B-Instruct because the
# evaluation code keeps the model name stable while routing by endpoint.

export CTI_RAG_LLM_ROUTER="${CTI_RAG_LLM_ROUTER:-1}"
export CTI_RAG_LLM_MAPPED_ENDPOINT="${CTI_RAG_LLM_MAPPED_ENDPOINT:-http://localhost:8001/v1/chat/completions}"
export CTI_RAG_LLM_UNMAPPED_ENDPOINT="${CTI_RAG_LLM_UNMAPPED_ENDPOINT:-http://localhost:8000/v1/chat/completions}"
export CTI_RAG_LLM_HYDE_ENDPOINT="${CTI_RAG_LLM_HYDE_ENDPOINT:-$CTI_RAG_LLM_UNMAPPED_ENDPOINT}"
export CTI_RAG_LLM_MAX_MODEL_LEN="${CTI_RAG_LLM_MAX_MODEL_LEN:-24576}"

# With both 8B models resident on the two GPUs, keep retrieval-side models on CPU
# unless the vLLM servers were launched with enough spare GPU memory.
export CTI_RAG_EMBEDDER_DEVICE="${CTI_RAG_EMBEDDER_DEVICE:-cpu}"
export CTI_RAG_EMBEDDER_SECOND_DEVICE="${CTI_RAG_EMBEDDER_SECOND_DEVICE:-none}"
export CTI_RAG_CWE_CROSSENCODER_DEVICE="${CTI_RAG_CWE_CROSSENCODER_DEVICE:-cpu}"

export CTI_RAG_CWE_PHRASE_SELECTOR="${CTI_RAG_CWE_PHRASE_SELECTOR:-1}"
export CTI_RAG_CWE_CROSSENCODER="${CTI_RAG_CWE_CROSSENCODER:-1}"
export CTI_RAG_CWE_HYDE="${CTI_RAG_CWE_HYDE:-1}"
export CTI_RAG_CWE_HYDE_CE_FILTER="${CTI_RAG_CWE_HYDE_CE_FILTER:-1}"
export CTI_RAG_LLM_HYDE_ROUTE="${CTI_RAG_LLM_HYDE_ROUTE:-1}"  # route mapped HyDE to Granite to balance GPU load
export CTI_RAG_EVAL_WORKERS="${CTI_RAG_EVAL_WORKERS:-3}"

if (($#)); then
  conda run -n cyber-ft python3 -u eval_rcm.py "$@"
else
  conda run -n cyber-ft python3 -u eval_rcm.py 1000
fi
