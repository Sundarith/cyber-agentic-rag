# CTI-RAG

Local retrieval-augmented generation for cyber threat intelligence.

CTI-RAG answers questions across MITRE ATT&CK, CAPEC, CWE, and NVD CVE data, including multi-hop paths such as:

```text
Group -> Malware/Tool -> Technique -> CAPEC -> CWE -> Mitigation
```

The system uses hybrid retrieval, explicit MITRE/NVD relationship bridges, graph expansion, and a local 7B instruct model served by vLLM.

## Result

CTI-RAG scores **84.3% (843/1000)** on the CTI-Bench CTI-RCM benchmark for CVE-to-CWE root cause mapping.

| System | CTI-RCM score | Setup |
| --- | ---: | --- |
| CTI-RAG | **84.3%** | Qwen2.5-7B-Instruct, local FP16 inference, RAG over public NVD/MITRE data |
| GPT-4 in CTI-Bench | 72.0% | Frontier model, no retrieval, memory-only answering |

Breakdown:

| Subset | Score |
| --- | ---: |
| All 1000 queries | **84.3%** (843/1000) |
| NVD-mapped CVEs | **86.6%** (782/903) |
| NVD-unmapped CVEs | **62.9%** (61/97) |

This is not an apples-to-apples model comparison. CTI-RAG uses retrieval over public security data, while the GPT-4 number from CTI-Bench is a no-retrieval baseline. The result is still operationally meaningful: security teams usually have NVD and MITRE data available, and the benchmark tests whether a system can use that evidence reliably.

## Why This Exists

Threat intelligence questions often require crossing dataset boundaries. A question about a threat group can require finding malware, then ATT&CK techniques, then CAPEC attack patterns, then CWE weaknesses, then mitigations.

Plain top-k semantic retrieval is brittle on this task. It can retrieve the wrong object because IDs share numbers, stop after one hop, or miss structured relationships that are present in the source data. CTI-RAG makes those relationships explicit and gives the model grounded context instead of asking it to reconstruct the chain from memory.

## Architecture

```text
Query
  |
  v
Hybrid retrieval
  - BM25 lexical scoring
  - BGE embedding similarity
  - alpha = 0.5, top_k = 8
  |
  v
Entity bridge
  - Detect groups, malware, tools, and campaigns
  - Boost the entity chunk and associated techniques
  |
  v
Knowledge graph expansion
  - 1-hop by default
  - 2-hop Deep Search for root-cause, CAPEC, detection, and mitigation questions
  |
  v
Explicit bridge injection
  - ATT&CK Technique -> CAPEC
  - CAPEC -> CWE
  - CVE description match -> NVD cwe_ids
  - Unmapped CVE -> k-NN vote from nearest mapped CVEs
  |
  v
Conflict stripping
  - Remove competing CWE chunks when an authoritative bridge fires
  |
  v
Qwen2.5-7B-Instruct via vLLM
```

## What Works

- Multi-hop ATT&CK/CAPEC/CWE/Mitigation traversal.
- Reverse entity lookups such as group-to-malware and malware-to-techniques.
- Explicit Tech-to-CAPEC and CAPEC-to-CWE mappings from MITRE data.
- CVE-to-CWE mapping from NVD structured `problemTypes.cweId`, including CNA and ADP/CISA Vulnrichment containers.
- k-NN CWE voting for CVEs that do not have an NVD CWE assignment.
- Local FP16 inference with vLLM continuous batching.

Validated examples include:

```text
Sandworm -> Prestige -> T1112 -> CAPEC-203 -> CWE-15 -> M1024
Lazarus -> BLINDINGCAN -> T1566 -> CAPEC-98 -> CWE-451 -> M1017/M1031
```

## Limits

- The mapped-CVE score is mostly retrieval finding the matching NVD record and reading its structured CWE assignment.
- The unmapped-CVE score is lower because the system must infer a CWE from description similarity and local context.
- Remaining failures are dominated by CWE taxonomy near-misses and disagreements between NVD assignments and CTI-Bench ground truth.
- The project is built for local research and reproducibility, not as a hardened production service.

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
- PyTorch 2.6.0+cu124
- sentence-transformers 5.4.1
- vLLM 0.19.1
- `Qwen/Qwen2.5-7B-Instruct`
- `BAAI/bge-small-en-v1.5`

The reference machine uses two RTX 4090 GPUs. vLLM runs on `cuda:0`; the second GPU can be used by a second embedder instance during evaluation.

## Quick Start

Start vLLM first in a separate terminal:

```bash
CUDA_VISIBLE_DEVICES=0 /home/sheng/miniconda3/bin/vllm serve Qwen/Qwen2.5-7B-Instruct \
  --max-model-len 32768 --gpu-memory-utilization 0.90 \
  --enable-prefix-caching --port 8000
```

Run the chatbot:

```bash
conda activate cyber-ft
python3 better_rag.py
```

## Rebuild Processed Data

Run these after changing raw source data or a chunk builder:

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

If a chunk file changes, remove the embedding cache before the next run so it is rebuilt:

```bash
rm -f data/processed/chunk_embs.npy
```

## Reproduce CTI-RCM

Use the same CTI-Bench prompt field used for the published GPT-4 comparison:

```bash
conda run -n cyber-ft python3 -u eval_rcm.py 1000
```

Useful subset runs:

```bash
conda run -n cyber-ft python3 -u eval_rcm.py 100 --nvd-mapped
conda run -n cyber-ft python3 -u eval_rcm.py 100 --nvd-unmapped
conda run -n cyber-ft python3 -u eval_rcm.py 100 --matched
```

Notes:

- Start vLLM before running evaluation.
- The full 1000-query benchmark takes about 25-30 minutes on the reference machine with `WORKERS=16`.
- vLLM should be started with `--max-model-len 32768`; shorter context windows can fail on large CVE prompts.
- The eval script intentionally uses `row["Prompt"]`, not a query containing the CVE ID. CVE-ID queries turn the benchmark into a database lookup.
- Add `--profile` or set `CTI_RAG_PROFILE=1` to print `[R]` retrieval timings and `[T]` end-to-end checkpoints.

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
