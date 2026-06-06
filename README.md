# Agentic CTI-RAG

An agentic retrieval-augmented generation system over local cyber threat intelligence
(MITRE ATT&CK, CAPEC, CWE). A reasoning LLM drives a bounded tool loop to gather evidence,
and the core research goal is to measure whether the agent retrieves the **correct evidence
path** (e.g. ATT&CK technique → CAPEC attack pattern → CWE weakness), not merely whether it
produces a plausible final answer.

> **Lineage.** This project descends from **CTI-RAG**, a graph-augmented hybrid RAG system for
> CVE-to-CWE root-cause mapping (Phi-4-mini-reasoning, 90.9% on CTI-Bench CTI-RCM). That is a
> separate, earlier project; its result is summarized under
> [Predecessor](#predecessor-cti-rag-cve-to-cwe) below. The agentic system here is the current
> line of work.

## What it is

A reasoning LLM (IBM Granite-4.1-8B via a local vLLM endpoint) drives a **ReAct-style tool loop**
over the CTI corpus and relation graph. Each step the model *chooses* a tool; the tools execute
deterministically; a validation gate checks the final answer before it is asserted.

Tools the agent chooses among:
- `resolve` — natural name → CTI id ("Process Discovery" → `T1057`)
- `search` — lexical (BM25) search over the corpus
- `open` — read a node by id
- `find` — search inside one node
- `expand` — walk the relation graph (the provable ATT&CK ↔ CAPEC ↔ CWE hop)

Design:
- **Granite orchestrates** (chooses tools each step); the graph traversal and the
  **answer-validation gate** are deterministic tools/guardrails it calls. Cited ids must be
  retrieved and claimed path edges must exist in the graph, so answers stay auditable. A small
  id-repair snaps a model id-typo to the unique retrieved id that forms the claimed edge.
- **Agentic-by-default**: chat and CLI use the LLM loop when the endpoint is reachable and fall
  back to a deterministic regex baseline when it is not.
- A standalone `EntityResolver` enables natural, no-ID questions (technique/CAPEC/CWE name,
  forward or reverse).

Key code: `agentic_rag/orchestrator.py` (the controller), `agentic_rag/resolver.py`,
`agentic_rag/agent.py`; architecture figure under `research/figures/`.

## Quick start

Serve the LLM endpoint (cap the context — Granite's 131072 default OOMs the KV cache on a 24GB GPU):

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve ibm-granite/granite-4.1-8b \
  --served-model-name ibm-granite/granite-4.1-8b \
  --max-model-len 8192 --gpu-memory-utilization 0.92 \
  --enable-prefix-caching --trust-remote-code --port 8000
```

Then chat (agentic-by-default; falls back to deterministic if the endpoint is down):

```bash
python3 chat_agentic_rag.py
# or one-shot:
python3 -m agentic_rag "For Process Discovery, which CWE is reached via its CAPEC?"
```

Run the tests (offline, no endpoint needed):

```bash
python3 -m unittest discover -s tests
```

## Evaluation

- `eval_agentic_multihop.py` — controlled ATT&CK → CAPEC → CWE gold-path diagnostic; the primary
  internal metric is **evidence-path recovery**, not just answer accuracy. Flags: `--hidden-id`
  (name-only questions), `--entry {technique,capec,cwe}` (entry point + forward/reverse direction),
  `--modes ... agentic_granite` (LLM-orchestrated vs the deterministic baseline).
- `eval_agentic_rcm.py`, `eval_seceval.py` — public benchmark harnesses (CTI-RCM, SecEval).

These gold-path numbers are internal diagnostics, not a published benchmark result.

## Status

Works: the agentic loop is live-validated — Granite drives resolve → expand → expand to a
validated `T1057 → CAPEC-573 → CWE-200`; hidden-ID natural QA at technique/CAPEC/CWE entry points,
forward and reverse; deterministic baseline; 79 unit tests passing.

Not yet provable: threat-actor and mitigation questions. The graph for group/software/mitigation
edges is incomplete (built from text mentions), so the agent can *navigate* toward an answer but
the validation gate flags it as ungrounded — these need STIX-derived relation files first. No CVE
corpus is loaded in this repo.

## Requirements

- Python 3.13 (local); legacy/predecessor flows may need a Python 3.11 env.
- PyTorch 2.10.0+cu128, vLLM 0.19.x, NumPy 2.2.6.
- Default LLM: `ibm-granite/granite-4.1-8b`. A single RTX 4090 (24 GB) is sufficient.

## Sources

MITRE ATT&CK, MITRE CAPEC, MITRE CWE. (The predecessor below also used NVD CVE data and
CTI-Bench / CTI-RCM.)

## License

MIT

---

## Predecessor: CTI-RAG (CVE-to-CWE)

The earlier project (the system name is **Graph-Augmented Hybrid RAG**) maps a CVE description to
its root-cause CWE by retrieving public vulnerability knowledge at inference time (NIST NVD + MITRE
CWE) rather than relying on cybersecurity-specialized model weights — a single 3.8B general-purpose
model, no domain pre-training, no task fine-tuning, one LLM call per query.

On CTI-Bench CTI-RCM (1,000 CVE-to-CWE prompts, strict CWE-ID match):

| System | CTI-RCM | Setup |
| --- | ---: | --- |
| **CTI-RAG (Phi-4-mini-reasoning + RAG)** | **90.9%** | 3.8B general model, single RTX 4090, RAG over public NVD/MITRE |
| RoBERTa-base CVE-to-CWE | 75.6% | open, fine-tuned classifier |
| Foundation-Sec-8B-R (Cisco) | 75.3% | open, cybersecurity continued pre-training |
| GPT-4 | 72.0% | closed frontier, no retrieval |

The result is attributable to the retrieval pipeline, not one model — four compact open models all
become competitive once paired with it:

| Primary model | Zero-shot | + RAG |
| --- | ---: | ---: |
| Phi-4-mini-reasoning (3.8B, shipped) | 17.9% | **90.9%** |
| Gemma 4 E4B (thinking) | 69.6% | 90.3% |
| IBM Granite 4.1-8B | 63.6% | 85.6% |
| DeepSeek-R1-Distill-Llama-8B | 25.7% | 82.1% |

Predecessor pipeline (one final-answer LLM call): hybrid BM25 + BGE-small retrieval → 1-hop CWE
graph expansion → NVD bridge injection (mapped CVEs) or weighted k-NN CWE voting with cross-encoder
reranking (unmapped CVEs). Graph edges came only from structured NVD `cwe_ids` and the MITRE CWE
hierarchy, never from CVE prose.

Full predecessor documentation (reproduction commands, data layout, rebuild scripts, the
`better_rag.py` / `eval_rcm.py` workflow, conda `cyber-ft` env) is preserved in git history at the
pre-agentic README (commit `5d931d9`) and in the predecessor repo (`upstream` →
`Sundarith/CTI-RAG`).
