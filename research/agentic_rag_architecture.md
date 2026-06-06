# Agentic CTI-RAG Architecture Notes

The system is an **LLM-orchestrated agentic RAG** over local CTI (MITRE ATT&CK, CAPEC, CWE).
A reasoning LLM (Granite-4.1-8B via vLLM) drives a bounded tool loop; the deterministic graph
and a validation gate are tools/guardrails it calls. The research goal is to measure whether the
agent recovers the **correct evidence path** (e.g. ATT&CK technique → CAPEC → CWE), not just whether
it produces a plausible answer.

## Control flow (the agent)

`GraniteOrchestrator` (`agentic_rag/orchestrator.py`) runs a ReAct loop: each step the LLM emits a
JSON action — a tool call or a final answer; the tool executes; the observation is appended; repeat
until the answer or the iteration budget. The LLM *chooses* the tools; nothing routes by regex.

Tools the agent chooses among:
- `resolve` — natural name → CTI id (`EntityResolver`); auto-opens the resolved node.
- `search` — lexical (BM25-style) search over the corpus.
- `open` / `find` — read a node / search inside one node.
- `expand` — walk the relation graph (the provable ATT&CK ↔ CAPEC ↔ CWE hop).

## Guardrails (kept deterministic on purpose)

- **Validation gate**: a final answer is accepted only if every cited id was retrieved and every
  claimed path edge exists in the graph. An id-repair snaps a model typo (e.g. "T10") to the unique
  retrieved id that forms the claimed edge, so answers stay auditable without inventing ids.
- **Trace**: every (thought, action, observation) step is recorded, so the evidence path is
  measurable, not just the final answer.

## Modes and defaults

- Agentic-by-default in chat/CLI (`--controller auto`): the LLM loop when the endpoint is reachable,
  a deterministic regex-planner baseline as fallback / for offline reproducibility.
- `EntityResolver` enables hidden-ID natural questions at technique/CAPEC/CWE entry points, forward
  and reverse.

## Evaluation

`eval_agentic_multihop.py` — controlled ATT&CK → CAPEC → CWE gold-path diagnostic; primary metric is
evidence-path recovery (full-path recall, hop success, entity-resolution accuracy/recall@k,
ambiguity, failure reason), alongside answer accuracy. `agentic_granite` mode runs the LLM loop
head-to-head against the deterministic baseline on the same gold paths.

## Open directions

Multi-query reformulation in `search` and a context-window `summarize` tool (per the AgenticRAG
paper, arXiv:2605.05538); a "switcher" to route simple questions to the deterministic baseline; and
STIX-derived relation files to make threat-actor / mitigation questions *provable* (the current
text-mention graph for those is too incomplete).
