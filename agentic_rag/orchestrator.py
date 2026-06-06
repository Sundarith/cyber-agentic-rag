"""LLM-orchestrated agentic RAG loop (hybrid controller).

The default :class:`~agentic_rag.agent.AgenticRAG` runs a *deterministic* planner:
regex routing picks every tool action and Granite only composes the final text.
That makes off-template questions ("what mitigations for X?") fall into the
nearest hardcoded branch.

This module implements the alternative described in Suresh et al.,
"AgenticRAG: Agentic Retrieval for Enterprise Knowledge Bases" (arXiv:2605.05538):
the reasoning LLM *drives* a bounded ReAct-style loop and chooses which tool to
call each step. The crucial difference from that paper -- and the reason this is
a *hybrid* -- is that the tools execute deterministically over our CTI corpus and
graph, and the final answer is passed through a validation gate against the
retrieved evidence and the graph. The LLM gets the flexibility; the graph keeps
the receipts, so evidence paths stay provable.

The loop uses a prompt-based JSON action contract (not native tool-calling), so
it works against any OpenAI-compatible endpoint and is unit-testable offline with
a mock client that returns scripted action strings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .corpus import CorpusIndex, extract_ids
from .resolver import EntityResolver
from .retrieval import LexicalRetriever, SearchBackend
from .schema import Evidence, Verification
from .synthesizer import (
    OpenAIChatClient,
    normalize_id_list,
    normalize_identifier,
    parse_json_object,
)
from .verifier import EvidenceVerifier, dedupe_evidence

TOOL_NAMES = ("search", "open", "find", "expand", "resolve")

SYSTEM_PROMPT = (
    "You are a cyber threat intelligence retrieval agent working over a local corpus of "
    "MITRE ATT&CK techniques (T####), CAPEC attack patterns (CAPEC-#), and CWE weaknesses (CWE-#), "
    "connected by a relation graph.\n"
    "Each turn, respond with EXACTLY ONE JSON object and no other text. Use a tool to gather "
    "evidence, or give the final answer. Available actions:\n"
    '  {"thought": "...", "tool": "resolve", "args": {"name": "Process Discovery"}}  '
    "-> map a natural name to a CTI id\n"
    '  {"thought": "...", "tool": "search", "args": {"query": "..."}}  -> lexical search the corpus\n'
    '  {"thought": "...", "tool": "open", "args": {"id": "T1057"}}  -> read a node by id\n'
    '  {"thought": "...", "tool": "find", "args": {"id": "T1057", "pattern": "discovery"}}  '
    "-> search inside one node\n"
    '  {"thought": "...", "tool": "expand", "args": {"id": "CAPEC-573"}}  '
    "-> list graph neighbors (the provable relation hop)\n"
    '  {"thought": "...", "answer": {"text": "...", "cited_ids": ["T1057","CAPEC-573","CWE-200"], '
    '"path": ["T1057","CAPEC-573","CWE-200"]}}  -> finish\n'
    "Rules:\n"
    "1. Output ONE JSON object and nothing else — no prose, no markdown fences.\n"
    "2. After you `resolve` a name to an id, your NEXT action MUST be `expand` (or `open`) on that "
    "id. Never call `resolve` for the same name twice.\n"
    "3. Never repeat an action you already ran; each step should make progress.\n"
    "4. Use `expand` to prove a relation between two ids rather than asserting it. For a path "
    "question, expand from the entry id until you reach the target (e.g. a CWE), then answer with "
    "the full path.\n"
    "5. Only cite ids you actually retrieved this session. When you have enough, output the answer.\n"
    "6. Copy ids EXACTLY as they appear in observations (write T1057, never T10)."
)


@dataclass
class OrchestratorConfig:
    max_iters: int = 8
    search_k: int = 8
    expand_k: int = 12
    max_observation_items: int = 10
    max_snippet_chars: int = 200
    max_json_retries: int = 3


@dataclass
class _State:
    evidence: dict[str, Evidence] = field(default_factory=dict)
    steps: list[dict[str, Any]] = field(default_factory=list)
    llm_messages: list[dict[str, str]] = field(default_factory=list)
    seen_actions: set = field(default_factory=set)
    parse_errors: int = 0


class GraniteOrchestrator:
    """LLM-driven tool loop over the deterministic CTI corpus + graph."""

    def __init__(
        self,
        corpus: CorpusIndex,
        client: OpenAIChatClient | None = None,
        resolver: EntityResolver | None = None,
        search_backend: SearchBackend | None = None,
        config: OrchestratorConfig | None = None,
    ) -> None:
        self.corpus = corpus
        self.client = client or OpenAIChatClient()
        self.resolver = resolver if resolver is not None else EntityResolver(corpus)
        self.search_backend = search_backend or LexicalRetriever(corpus)
        self.config = config or OrchestratorConfig()
        self.verifier = EvidenceVerifier()

    def answer(self, question: str) -> dict:
        state = _State()
        state.llm_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}\nBegin."},
        ]
        # Seed any explicit ids the question already names so they are citable.
        for ident in extract_ids(question):
            self._remember(state, self.corpus.open(ident))

        final: dict[str, Any] | None = None
        for step_no in range(1, self.config.max_iters + 1):
            raw = self.client.complete(state.llm_messages)
            try:
                action = parse_json_object(raw)
            except Exception:  # noqa: BLE001 - json/value errors from a loose model
                state.parse_errors += 1
                state.steps.append({"step": step_no, "action": "parse_error", "value": "", "new_evidence": [],
                                    "supported": False, "confidence": 0.0, "missing": [], "raw": raw[:300]})
                if state.parse_errors > self.config.max_json_retries:
                    break
                state.llm_messages.append({"role": "assistant", "content": raw})
                state.llm_messages.append({"role": "user", "content":
                    "That was not valid JSON. Reply with exactly ONE JSON object using the schema, "
                    "e.g. {\"tool\": \"expand\", \"args\": {\"id\": \"T1057\"}} or an answer object."})
                continue
            state.llm_messages.append({"role": "assistant", "content": raw})

            if "answer" in action:
                final = self._finalize(state, step_no, action)
                break

            observation, new_ids, thought = self._dispatch(state, action)
            sig = (str(action.get("tool") or "").lower(), _action_value(action).lower())
            if sig in state.seen_actions:
                observation += ("\n(You already ran this exact action. Choose a DIFFERENT action that "
                                "makes progress, or output the answer.)")
            state.seen_actions.add(sig)
            verification = self.verifier.verify(question, list(state.evidence.values()))
            state.steps.append({
                "step": step_no,
                "action": str(action.get("tool") or "unknown"),
                "value": _action_value(action),
                "thought": thought,
                "new_evidence": new_ids,
                "supported": verification.supported,
                "confidence": verification.confidence,
                "missing": verification.missing,
            })
            state.llm_messages.append({"role": "user", "content": observation})

        evidence = dedupe_evidence(list(state.evidence.values()))
        verification = self.verifier.verify(question, evidence)
        if final is None:
            final = self._force_final(state, evidence)
        return {
            "question": question,
            "answer": final["text"],
            "verification": verification,
            "evidence": evidence,
            "trace": state.steps,
            "resolution": None,
            "route": {"direction": "llm", "target_type": ""},
            "reverse_answer": None,
            "llm_final": final,
            "controller": "granite",
        }

    # -- tool dispatch -------------------------------------------------------

    def _dispatch(self, state: _State, action: dict[str, Any]) -> tuple[str, list[str], str]:
        tool = str(action.get("tool") or "").lower()
        args = action.get("args") or {}
        thought = str(action.get("thought") or "")
        if tool == "search":
            evs = self.search_backend.search(str(args.get("query") or ""), k=self.config.search_k)
        elif tool == "open":
            ev = self.corpus.open(str(args.get("id") or ""))
            evs = [ev] if ev else []
        elif tool == "expand":
            evs = self.corpus.expand(str(args.get("id") or ""), k=self.config.expand_k)
        elif tool == "find":
            evs = self._find_in(str(args.get("id") or ""), str(args.get("pattern") or ""))
        elif tool == "resolve":
            return self._do_resolve(state, args, thought)
        else:
            return (f"Unknown tool '{tool}'. Use one of: {', '.join(TOOL_NAMES)}, or answer.", [], thought)

        new_ids = [self._remember(state, ev) for ev in evs if ev]
        new_ids = [i for i in new_ids if i]
        return (self._format_obs(tool, evs), new_ids, thought)

    def _do_resolve(self, state: _State, args: dict[str, Any], thought: str) -> tuple[str, list[str], str]:
        name = str(args.get("name") or "")
        prefer = args.get("prefer_types")
        result = self.resolver.resolve_question(name, prefer_types=prefer) if name else None
        if not result or not result.resolved:
            return (f"resolve('{name}') found no matching id. Try `search` with keywords instead.", [], thought)
        # Auto-open the resolved node so the loop makes progress and the model
        # can cite it, then steer the next hop.
        ident = result.chosen_id
        new_id = self._remember(state, self.corpus.open(ident))
        flag = " [ambiguous]" if result.ambiguous else ""
        obs = (f"resolve('{name}') -> {ident} [{result.type}]{flag}; opened it. "
               f"Next, call expand on {ident} to follow its relations toward the answer.")
        return (obs, [new_id] if new_id else [], thought)

    def _find_in(self, identifier: str, pattern: str) -> list[Evidence]:
        chunk = self.corpus.by_identifier.get(normalize_identifier(identifier))
        if not chunk or pattern.lower() not in chunk.text.lower():
            return []
        offset = chunk.text.lower().find(pattern.lower())
        snippet = chunk.text[max(0, offset - 120): offset + len(pattern) + 200].strip()
        return [Evidence(chunk=chunk, score=1.0, tool="find", reason=pattern, snippet=snippet)]

    def _remember(self, state: _State, ev: Evidence | None) -> str:
        if not ev:
            return ""
        ident = normalize_identifier(ev.identifier)
        state.evidence.setdefault(ident, ev)
        return ident

    def _format_obs(self, tool: str, evs: list[Evidence]) -> str:
        if not evs:
            return f"{tool} returned no results."
        lines = [f"{tool} returned {len(evs)} result(s):"]
        for ev in evs[: self.config.max_observation_items]:
            name = ev.chunk.name or ""
            snippet = (ev.snippet or ev.chunk.text)[: self.config.max_snippet_chars].replace("\n", " ").strip()
            lines.append(f"- {ev.identifier} | {name} | {ev.chunk.type} | {snippet}")
        return "\n".join(lines)

    # -- finalize + validation gate -----------------------------------------

    def _finalize(self, state: _State, step_no: int, action: dict[str, Any]) -> dict[str, Any]:
        payload = action.get("answer") or {}
        text = str(payload.get("text") or "")
        cited = normalize_id_list(payload.get("cited_ids"))
        path = normalize_id_list(payload.get("path"))
        allowed = set(state.evidence)
        # Repair small id slips (e.g. the model wrote "T10" for "T1057"): snap a
        # cited/path id to the UNIQUE retrieved id that both resembles it and
        # forms the claimed graph edge. This stays provable -- we only recover an
        # id the evidence actually supports, never invent one.
        path, repairs = self._repair_path(path, allowed)
        cited = [repairs.get(c, c) for c in cited]
        cited = [c if c in allowed else self._repair_simple(c, allowed, repairs) for c in cited]
        for bad, good in repairs.items():
            text = re.sub(rf"\b{re.escape(bad)}\b", good, text)
        unsupported = [i for i in cited if i not in allowed]
        bad_edges = [f"{a}->{b}" for a, b in zip(path, path[1:]) if not self._edge(a, b)]
        supported = not unsupported and not bad_edges
        validation = {"supported": supported, "unsupported_ids": unsupported, "invalid_edges": bad_edges,
                      "repaired_ids": repairs}
        state.steps.append({
            "step": step_no, "action": "answer", "value": "", "new_evidence": [],
            "thought": str(action.get("thought") or ""), "cited_ids": cited, "path": path,
            "validation": validation, "supported": supported, "confidence": 1.0 if supported else 0.0,
            "missing": [],
        })
        if not supported:
            note = []
            if unsupported:
                note.append("unsupported ids: " + ", ".join(unsupported))
            if bad_edges:
                note.append("unproven relations: " + ", ".join(bad_edges))
            text = (text + "\n\n[validation] answer rejected — " + "; ".join(note)
                    + ". Not asserting an unvalidated path.").strip()
        return {"text": text, "cited_ids": cited, "path": path, "validation": validation}

    def _force_final(self, state: _State, evidence: list[Evidence]) -> dict[str, Any]:
        ids = [normalize_identifier(ev.identifier) for ev in evidence]
        text = ("Iteration budget reached. Collected evidence: " + ", ".join(ids[:12])
                if ids else "No supporting CTI evidence was retrieved.")
        return {"text": text, "cited_ids": ids, "path": [],
                "validation": {"supported": False, "unsupported_ids": [], "invalid_edges": [], "forced": True}}

    def _edge(self, a: str, b: str) -> bool:
        return b in self.corpus.graph.get(a, set()) or a in self.corpus.graph.get(b, set())

    def _repair_path(self, path: list[str], allowed: set[str]) -> tuple[list[str], dict[str, str]]:
        repaired = list(path)
        repairs: dict[str, str] = {}
        for i, tok in enumerate(path):
            if tok in allowed:
                continue
            neighbors = [repaired[i - 1]] if i > 0 else []
            if i + 1 < len(path):
                neighbors.append(path[i + 1])
            cands = [e for e in allowed if _looks_like(tok, e)]
            edge_cands = [e for e in cands if any(self._edge(e, nb) for nb in neighbors)] if neighbors else cands
            pick = edge_cands[0] if len(edge_cands) == 1 else (cands[0] if len(cands) == 1 else None)
            if pick:
                repaired[i] = pick
                repairs[tok] = pick
        return repaired, repairs

    def _repair_simple(self, token: str, allowed: set[str], repairs: dict[str, str]) -> str:
        cands = [e for e in allowed if _looks_like(token, e)]
        if len(cands) == 1:
            repairs[token] = cands[0]
            return cands[0]
        return token


def _action_value(action: dict[str, Any]) -> str:
    args = action.get("args") or {}
    return str(args.get("id") or args.get("query") or args.get("name") or args.get("pattern") or "")


def _id_family(identifier: str) -> str:
    up = identifier.upper()
    for prefix in ("CAPEC-", "CWE-", "CVE-"):
        if up.startswith(prefix):
            return prefix
    return up[:1]  # e.g. "T", "S", "G", "M"


def _looks_like(a: str, b: str) -> bool:
    """A model-typo id `a` plausibly means retrieved id `b` (same family, shared prefix)."""
    if not a or not b:
        return False
    a, b = a.upper(), b.upper()
    if _id_family(a) != _id_family(b):
        return False
    return a == b or a.startswith(b) or b.startswith(a) or (len(a) >= 4 and len(b) >= 4 and a[:4] == b[:4])
