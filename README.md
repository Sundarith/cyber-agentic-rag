# CTI-RAG

Local graph-augmented hybrid retrieval-augmented generation (RAG) for cyber threat
intelligence, evaluated on CVE-to-CWE root-cause mapping.

CTI-RAG maps a CVE description to its root-cause CWE identifier by retrieving public
vulnerability knowledge at inference time — the NIST NVD CVE corpus and the official
MITRE CWE corpus — rather than relying on cybersecurity-specialized model weights. The
system name is **Graph-Augmented Hybrid RAG**.

## Result

On the CTI-Bench CTI-RCM benchmark (1,000 CVE-to-CWE prompts, strict CWE-ID match),
the shipped configuration scores **90.9% (909/1000)** using a single 3.8B-parameter
general-purpose model — **Phi-4-mini-reasoning** — with no cybersecurity continued
pre-training, no task-specific fine-tuning, and exactly one LLM call per query.

| Subset | Score |
| --- | ---: |
| All 1,000 queries | **90.9%** (909/1000) |
| NVD-mapped CVEs | 92.7% (837/903) |
| NVD-unmapped CVEs | 74.2% (72/97) |

This surpasses the published GPT-4 no-retrieval baseline by 18.9 pp and the strongest
published open-source baseline by 15.3 pp, using less than half the parameter count of
every other open-source LLM on the leaderboard.

| System | CTI-RCM | Setup |
| --- | ---: | --- |
| **Ours (Phi-4-mini-reasoning + RAG)** | **90.9%** | 3.8B general-purpose model, single RTX 4090, RAG over public NVD/MITRE data |
| Sec-Gemini v1 (Google) | ~86% † | closed, self-reported |
| SecLM (Google) | ~85% † | closed, self-reported |
| RoBERTa-base CVE-to-CWE | 75.6% | open, fine-tuned classifier |
| Foundation-Sec-8B-R (Cisco) | 75.3% | open, cybersecurity continued pre-training |
| GPT-4 | 72.0% | closed frontier, no retrieval |

† self-reported; not independently evaluated on the public split.

### RAG Carries the Result (Model-Independence)

The headline is attributable to the retrieval pipeline, not to any one model. Under the
same benchmark-clean context-only prompt, four open-source compact models from distinct
post-training regimes — none with cybersecurity training or task fine-tuning — all become
competitive once paired with the pipeline:

| Primary model | Zero-shot | + RAG | RAG buys |
| --- | ---: | ---: | ---: |
| Phi-4-mini-reasoning (3.8B, shipped) | 17.9% | **90.9%** | +73.0 pp |
| Gemma 4 E4B (thinking on) | 69.6% | 90.3% | +20.7 pp |
| IBM Granite 4.1-8B (RAG-tuned instruct) | 63.6% | 85.6% | +22.0 pp |
| DeepSeek-R1-Distill-Llama-8B | 25.7% | 82.1% | +56.4 pp |

Every with-RAG row beats GPT-4 (72.0%) and the strongest fine-tuned open-source
specialist (75.6%). The choice of retrieval pipeline shifts the result by tens to
hundreds of cases; the choice of LLM only shifts it by a few.

This is not an apples-to-apples model comparison: CTI-RAG retrieves over public security
data, while the GPT-4 number is a no-retrieval baseline. It is operationally meaningful —
security teams have NVD and MITRE data available, and the benchmark tests whether a system
can use that evidence reliably.

## Why This Exists

CVE-to-CWE mapping determines how analysts group vulnerabilities, select mitigations, and
prioritize remediation. Errors propagate into downstream analysis. Proprietary frontier
models can do the mapping but require paid API access and transmitting vulnerability data
to a third party; security-specialized open models narrow the gap through expensive
domain pre-training. CTI-RAG asks a different question: how far can a compact, off-the-shelf,
locally-served model go when it simply *reads* authoritative public CVE/CWE evidence at
inference time, with no domain or task post-training of the generator?

## Architecture

```text
CTI-Bench prompt (CVE description)
  |
  v
Hybrid retrieval
  - BM25 lexical scoring + BGE-small dense similarity
  - equal weights, top_k = 8
  |
  v
1-hop knowledge-graph expansion
  - NVD bridge edges:   CVE -> NVD-assigned CWE(s)
  - CWE hierarchy edges: child <-> parent (MITRE CWE taxonomy)
  - top 5 neighbours re-scored and added to context
  |
  +-- NVD-mapped CVE --> Bridge injection (fast path)
  |     - inject the structured NVD-assigned CWE chunk(s)
  |     - strip competing CWE chunks
  |
  +-- NVD-unmapped CVE --> Weighted k-NN CWE voting
  |     - 5 nearest mapped-CVE neighbours, similarity-weighted
  |     - authoritative if top share >= 0.90 and margin >= 1.50,
  |       otherwise top-3 added as soft evidence
  |     - BGE-Reranker-Base cross-encoder reranking
  |
  v
Single final-answer LLM call (Phi-4-mini-reasoning via vLLM)
```

Graph edges come only from structured NVD `cwe_ids` and the MITRE CWE hierarchy; CWE
identifiers mentioned in CVE prose are deliberately not treated as edges. The two
execution paths converge on exactly one final-answer call — no second model, no
ensembling.

The repository also implements broader CTI multi-hop traversal (ATT&CK / CAPEC / CWE /
mitigation) used by the interactive chatbot; the paper and the numbers above are scoped
to the CVE-to-CWE (CTI-RCM) task.

## Limits

- The mapped-CVE score is largely retrieval finding the matching NVD record and reading
  its structured CWE assignment (retrieval-as-lookup); the system commits to NVD even
  when CTI-Bench's curator disagreed, so mapped accuracy is capped by NVD/CTI-Bench label
  agreement rather than inflated by it.
- The unmapped-CVE score is the stricter reasoning test: no structured NVD label exists,
  so the system must infer the CWE from description similarity and retrieved definitions.
- Remaining failures are dominated by CWE taxonomy near-misses (sibling / parent / child
  confusion) and NVD-vs-CTI-Bench annotation disagreements.
- Built for local research and reproducibility, not as a hardened production service.

## Data

The system expects public MITRE/NVD data and a local CTI-Bench clone:

```text
data/raw/enterprise-attack.json              # MITRE ATT&CK STIX bundle
data/raw/stix-capec.json                     # CAPEC STIX bundle
data/raw/cwec_latest.xml                     # CWE XML dataset
data/raw/cves/cves/YYYY/                     # NVD CVE JSON files, gitignored
data/cti-bench/                              # CTI-Bench clone

data/processed/attack_chunks.jsonl           # built from ATT&CK
data/processed/capec_chunks.jsonl            # built from CAPEC
data/processed/cwe_chunks.jsonl              # built from CWE
data/processed/cve_chunks.jsonl              # built from NVD, gitignored
data/processed/cve_cwe_index.json            # CVE ID -> NVD CWE IDs
data/processed/entity_relations.json         # Group/tool/malware relations
data/processed/capec_attack_relations.json   # wrapper key: tech_to_capec
data/processed/capec_cwe_relations.json      # wrapper key: capec_to_cwe
data/processed/chunk_embs.npy                # embedding cache, gitignored
```

Large generated files are intentionally not committed.

## Requirements

Tested environment:

- Python 3.11
- conda environment named `cyber-ft`
- CUDA 12.x
- PyTorch 2.10.0+cu128
- sentence-transformers 5.4.1
- vLLM 0.19.x
- Primary model: `microsoft/Phi-4-mini-reasoning` (3.8B)
- Retriever: `BAAI/bge-small-en-v1.5`; cross-encoder: `BAAI/bge-reranker-base`

All experiments run on a single NVIDIA RTX 4090 (24 GB VRAM). The bi-encoder and
cross-encoder run on CPU so the GPU is dedicated to the vLLM LLM endpoint.

## Quick Start

Start the vLLM endpoint in a separate terminal. The eval harness addresses the model
under the fixed alias `Qwen/Qwen2.5-7B-Instruct`, so the served model name is set
accordingly regardless of which checkpoint is loaded:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve microsoft/Phi-4-mini-reasoning \
  --served-model-name "Qwen/Qwen2.5-7B-Instruct" \
  --enable-prefix-caching --trust-remote-code --port 8000 \
  --max-model-len 24576 --gpu-memory-utilization 0.90
```

Run the interactive chatbot:

```bash
conda activate cyber-ft
python3 better_rag.py
```

## Reproduce CTI-RCM

The shipped 90.9% configuration uses the context-only clean prompt with the unchanged
CTI-Bench prompt field, scored by strict CWE-ID match:

```bash
CTI_RAG_RCM_ONLY=1 \
CTI_RAG_PROMPT_CONTEXT_ONLY=1 \
CTI_RAG_LLM_RESPONSE_BUDGET=2048 \
CTI_RAG_LLM_MAX_MODEL_LEN=24576 \
CTI_RAG_EMBEDDER_DEVICE=cpu \
CTI_RAG_CWE_CROSSENCODER_DEVICE=cpu \
CTI_RAG_EVAL_WORKERS=16 \
conda run -n cyber-ft python3 -u eval_rcm.py 1000
```

- The final classification call uses the unchanged CTI-Bench `row["Prompt"]` at default
  temperature with a single sample, so the protocol is apples-to-apples with the other
  models reported by CTI-Bench.
- `CTI_RAG_PROMPT_CONTEXT_ONLY=1` selects the shipped context-only clean prompt. The
  experiment-only `CTI_RAG_PROMPT_INSTRUCTION=1` adds task framing and output-format
  directives (reported as an ablation, not the shipped method).
- k-NN voting defaults (`CTI_RAG_KNN_CONFIDENCE_THRESHOLD=0.90`,
  `CTI_RAG_KNN_CONFIDENCE_MARGIN=1.50`) are the reported values and need not be set.
- A larger response budget is used because the reasoning-distilled primary emits a
  reasoning preamble before the final CWE-ID.

The four-model comparison (Table in the paper) is reproduced by swapping the served
model — `ibm-granite/granite-4.1-8b`, `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`,
`google/gemma-4-E4B-it` (add `CTI_RAG_LLM_ENABLE_THINKING=1` for Gemma) — and re-running
the same eval. Zero-shot rows (no retrieval) are produced with `scripts/zero_shot_eval.py`.
The orchestration used for the paper's lineup lives in `scripts/run_*_lineup.sh` and
`scripts/run_*_pln_off*.sh`.

Useful subset runs:

```bash
conda run -n cyber-ft python3 -u eval_rcm.py 1000 --nvd-mapped
conda run -n cyber-ft python3 -u eval_rcm.py 1000 --nvd-unmapped
```

Notes:

- Start vLLM before running evaluation.
- The full 1,000-query benchmark takes roughly 25-30 minutes with `WORKERS=16`.
- `--max-model-len 24576` is recommended; shorter context windows can fail on large CVE prompts.
- The eval intentionally uses `row["Prompt"]`, not a query containing the CVE ID — a CVE-ID
  query would turn the benchmark into a database lookup.
- Add `--debug-failures` to write per-failure retrieval state, or `--timing-audit` for
  per-query and per-LLM-request timing logs under `logs/`.

## Rebuild Processed Data

Run after changing raw source data or a chunk builder:

```bash
conda run -n cyber-ft python3 build_attack_chunks.py
conda run -n cyber-ft python3 build_capec_chunks.py
conda run -n cyber-ft python3 build_cwe_chunks.py
conda run -n cyber-ft python3 build_cve_chunks.py
```

After rebuilding CVE chunks, rebuild the lightweight CVE-to-CWE index:

```bash
conda run -n cyber-ft python3 -c "
import json
from pathlib import Path
index = {}
with open('data/processed/cve_chunks.jsonl') as f:
    for line in f:
        d = json.loads(line)
        index[d['identifier']] = d.get('cwe_ids', [])
Path('data/processed/cve_cwe_index.json').write_text(json.dumps(index))
print(f'Written {len(index):,} entries')
"
```

If a chunk file changes, remove the embedding cache so it is rebuilt:

```bash
rm -f data/processed/chunk_embs.npy
```

## Implementation Notes

- Hybrid retrieval uses BM25 plus BGE embeddings with `HYBRID_ALPHA = 0.5`.
- Retrieval tuning constants are named at the top of `better_rag.py`.
- Bridge JSON files have wrapper keys; use `tech_to_capec` and `capec_to_cwe`.
- CVE-to-CWE graph edges come only from structured NVD `cwe_ids`, not from scanning CVE prose.
- For explicit MITRE mappings, bridge injection is preferred over semantic retrieval.
- Generated chunk files should be rebuilt with the matching `build_*.py` script, not edited by hand.

## Sources

- MITRE ATT&CK
- MITRE CAPEC
- MITRE CWE
- NVD CVE data, including CNA and ADP/CISA Vulnrichment mappings
- CTI-Bench / CTI-RCM

## License

MIT
