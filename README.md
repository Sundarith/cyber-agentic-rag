# CTI-RAG (cyber-ft)

Multi-source Cyber Threat Intelligence RAG system for traversing MITRE and NVD knowledge across:

**Group -> Malware/Tool -> Technique -> CAPEC -> CWE -> Mitigation**

The system combines hybrid retrieval over ATT&CK, CAPEC, CWE, and CVE corpora with a local 7B model.

## Benchmark Result

**84.3% (843/1000)** on CTI-Bench CTI-RCM (exact-match CVE -> CWE classification).

- NVD-mapped CVEs: **86.6% (782/903)**
- NVD-unmapped CVEs: **62.9% (61/97)**

| System | Score | Setup |
|---|---|---|
| **CTI-RAG (this repo)** | **84.3%** | Qwen2.5-7B-Instruct (local, FP16) + RAG over public NVD/MITRE data |
| GPT-4 (CTI-Bench paper) | 72.0% | Frontier model, no retrieval, memory-only |

### Fair framing

This result does not mean a 7B model is inherently smarter than GPT-4.

It means a local 7B model with strong retrieval over authoritative public data can beat memory-only answering on this benchmark format.

The comparison is not apples-to-apples, but it is operationally meaningful for security teams that already use NVD and MITRE data.

## Architecture

```text
User Query
  |
  v
Hybrid Retrieval (BM25 + BGE embeddings, alpha=0.5, top-k=8)
  |
  v
Knowledge Graph Expansion (1-hop / 2-hop Deep Search)
  |
  v
Bridge Injection
  - Technique -> CAPEC (explicit mapping)
  - CAPEC -> CWE (explicit mapping)
  - CVE description match -> NVD cwe_ids (authoritative)
  - Unmapped CVE -> k-NN CWE vote from nearest mapped CVEs
  |
  v
Conflict Stripping
  - Remove competing CWE chunks when bridge is authoritative
  |
  v
vLLM (Qwen2.5-7B-Instruct, FP16) -> final answer
```

## What Works Well

- Reliable multi-hop CTI traversal across ATT&CK, CAPEC, CWE, and CVE.
- Strong mapped CVE performance via structured NVD CWE bridge.
- Useful unmapped CVE fallback via similarity-weighted k-NN voting.
- Local inference (no per-query API cost) with practical throughput.

## Current Limits

- Not a pure reasoning comparison with frontier models; retrieval does most of the heavy lifting.
- NVD vs CTI-Bench CWE disagreements are unavoidable scoring penalties.
- Unmapped CVEs still fail mostly on near-miss CWE taxonomy confusion.

## Reproducibility

### 1. Environment

- Python 3.11
- conda env: `cyber-ft`
- CUDA 12.x
- vLLM 0.19.1

### 2. Start vLLM (first, in a separate terminal)

```bash
CUDA_VISIBLE_DEVICES=0 /home/sheng/miniconda3/bin/vllm serve Qwen/Qwen2.5-7B-Instruct \
  --max-model-len 32768 --gpu-memory-utilization 0.90 \
  --enable-prefix-caching --port 8000
```

### 3. Build / rebuild chunks

Run after changing source data or chunk builders:

```bash
conda run -n cyber-ft python3 build_attack_chunks.py
conda run -n cyber-ft python3 build_capec_chunks.py
conda run -n cyber-ft python3 build_cwe_chunks.py
conda run -n cyber-ft python3 build_cve_chunks.py
```

After rebuilding `cve_chunks.jsonl`, also rebuild `cve_cwe_index.json`:

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

If any `*_chunks.jsonl` file changed, delete embedding cache before running:

```bash
rm -f data/processed/chunk_embs.npy
```

### 4. Run chatbot

```bash
conda activate cyber-ft
python3 better_rag.py
```

### 5. Run CTI-RCM eval

```bash
conda run -n cyber-ft python3 -u eval_rcm.py 1000
```

Useful subsets:

```bash
conda run -n cyber-ft python3 -u eval_rcm.py 100 --nvd-mapped
conda run -n cyber-ft python3 -u eval_rcm.py 100 --nvd-unmapped
conda run -n cyber-ft python3 -u eval_rcm.py 100 --matched
```

## Final Locked Scores

- **All queries (1000): 84.3% (843/1000)**
- **NVD-mapped (903): 86.6% (782/903)**
- **NVD-unmapped (97): 62.9% (61/97)**

## Data Layout

```text
data/raw/enterprise-attack.json
data/raw/stix-capec.json
data/raw/cwec_latest.xml
data/raw/cves/cves/YYYY/                # NVD CVE JSON files (gitignored)
data/cti-bench/                          # third-party benchmark clone

data/processed/attack_chunks.jsonl
data/processed/capec_chunks.jsonl
data/processed/cwe_chunks.jsonl
data/processed/cve_chunks.jsonl          # gitignored

data/processed/cve_cwe_index.json        # CVE ID -> [CWE IDs]
data/processed/entity_relations.json
data/processed/capec_attack_relations.json  # wrapper: tech_to_capec
data/processed/capec_cwe_relations.json     # wrapper: capec_to_cwe
data/processed/chunk_embs.npy            # gitignored cache
```

## Hard Rules (for contributors)

- Do not edit `*_chunks.jsonl` directly; regenerate via `build_*.py`.
- Rebuild `cve_cwe_index.json` after `build_cve_chunks.py`.
- Keep hybrid retrieval (`HYBRID_ALPHA=0.5`) unless explicitly justified.
- Prefer explicit bridge mappings over semantic guesses for Tech<->CAPEC and CAPEC<->CWE.
- CVE->CWE graph edges come only from structured `cwe_ids` (NVD problemTypes), not prose scanning.
- Keep anti-hallucination prompt constraints in place.

## Sources

- MITRE ATT&CK
- MITRE CAPEC
- MITRE CWE
- NVD CVE feeds (including CNA + ADP/CISA Vulnrichment mappings)
- CTI-Bench (Alam et al., NeurIPS 2024)

## Models

- Embedder: `BAAI/bge-small-en-v1.5`
- LLM: `Qwen/Qwen2.5-7B-Instruct` via vLLM (FP16)

## License

MIT
