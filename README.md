# Agentic CTI-RAG

An agentic retrieval-augmented generation system over local cyber threat intelligence
(MITRE ATT&CK, CAPEC, CWE). A reasoning LLM drives a bounded tool loop to gather evidence,
and the core research goal is to measure whether the agent retrieves the **correct evidence
path** (e.g. ATT&CK technique → CAPEC attack pattern → CWE weakness), not merely whether it
produces a plausible final answer.

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

- Python 3.13 (local).
- PyTorch 2.10.0+cu128, vLLM 0.19.x, NumPy 2.2.6.
- Default LLM: `ibm-granite/granite-4.1-8b`. A single RTX 4090 (24 GB) is sufficient.

## Sources

MITRE ATT&CK, MITRE CAPEC, MITRE CWE.

## License

MIT

