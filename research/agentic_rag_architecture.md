# Agentic CTI-RAG Architecture Notes

This branch starts the second-paper system as an agentic evidence-gathering layer over
the shipped CTI-RAG corpus.

## First Prototype

The initial implementation is intentionally dependency-light:

- `CorpusIndex`: lazy JSONL loader, lexical retrieval, exact ID lookup, and graph expansion.
- `QueryPlanner`: converts a question into bounded tool actions.
- `AgenticRAG`: executes search/open/expand/find actions, tracks evidence, and stops when verification passes.
- `EvidenceVerifier`: checks whether the evidence covers requested facets such as CWE root cause, CAPEC attack pattern, or detection.
- `ExtractiveSynthesizer`: emits a cited grounded draft before a generative LLM backend is introduced.
- `eval_agentic_rcm.py`: CTI-RCM harness with strict CWE-ID scoring, per-query traces,
  timing, evidence summaries, and NVD-mapped/unmapped filtering.

This is not intended to beat the submitted one-call CTI-RCM result yet. It creates a clean
research harness for studying when multi-step retrieval helps, when it hurts, and how much
extra cost it introduces.

## Target Research System

The planned full system should compare:

- Non-agentic hybrid RAG baseline from the submitted paper.
- Agentic retrieval with explicit tools and stopping criteria.
- Agentic retrieval plus graph traversal over CVE/CWE/CAPEC/ATT&CK relations.
- Agentic retrieval plus verifier/reflection loops.
- Agentic retrieval plus LLM synthesis with citation validation.

Core metrics should include strict task accuracy, evidence recall, citation correctness,
number of tool calls, latency, and failure recovery rate on unmapped or taxonomy-near-miss
CVE cases.
