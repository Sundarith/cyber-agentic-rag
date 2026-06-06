from __future__ import annotations

import re
from dataclasses import dataclass

from .corpus import CorpusIndex, extract_ids
from .planner import QueryPlanner
from .resolver import EntityResolver, ResolutionResult
from .retrieval import LexicalRetriever, SearchBackend
from .schema import Action, ActionType, AgentState, Evidence, Verification
from .synthesizer import ExtractiveSynthesizer, Synthesizer
from .verifier import EvidenceVerifier, dedupe_evidence


@dataclass(frozen=True)
class AgenticRAGConfig:
    retrieve_k: int = 8
    max_steps: int = 5
    evidence_budget: int = 14
    resolve_entities: bool = True
    # "deterministic" = the regex planner loop; "granite" = the LLM-driven
    # orchestrator (agentic_rag/orchestrator.py). Default stays deterministic so
    # the system runs offline and existing benchmarks are unaffected.
    controller: str = "deterministic"


class AgenticRAG:
    def __init__(
        self,
        corpus: CorpusIndex,
        config: AgenticRAGConfig | None = None,
        planner: QueryPlanner | None = None,
        verifier: EvidenceVerifier | None = None,
        synthesizer: Synthesizer | None = None,
        search_backend: SearchBackend | None = None,
        resolver: EntityResolver | None = None,
        orchestrator: "GraniteOrchestrator | None" = None,
    ) -> None:
        self.corpus = corpus
        self.config = config or AgenticRAGConfig()
        self.planner = planner or QueryPlanner()
        self.verifier = verifier or EvidenceVerifier(self.planner)
        self.synthesizer = synthesizer or ExtractiveSynthesizer()
        self.search_backend = search_backend or LexicalRetriever(corpus)
        self.resolver = resolver if resolver is not None else EntityResolver(corpus)
        self.orchestrator = orchestrator

    def answer(self, question: str) -> dict:
        if self.config.controller == "granite":
            return self._orchestrator().answer(question)
        state = AgentState(question=question)
        route = self.planner.route(question)
        resolution = self._resolve_entities(question, route.prefer_types)
        queue = self.planner.initial_actions(question, self.config.retrieve_k)
        if resolution is not None and resolution.resolved:
            queue = seed_actions_for_resolution(resolution, route) + queue
        verification = Verification(False, 0.0, [], [])

        for step in range(self.config.max_steps):
            if not queue:
                queue = self.planner.next_actions(state, verification, self.config.retrieve_k)
            if not queue:
                break

            action = queue.pop(0)
            state.actions.append(action)
            new_evidence = self._run_action(action)
            state.evidence.extend(new_evidence)
            for ev in new_evidence:
                if action.kind == ActionType.OPEN:
                    state.opened_ids.add(ev.identifier)
            if action.kind == ActionType.EXPAND:
                state.opened_ids.add(action.value.upper())
            state.evidence = budget_evidence(dedupe_evidence(state.evidence), self.config.evidence_budget)
            verification = self.verifier.verify(question, state.evidence)
            priority_actions = self.planner.priority_actions_after_step(
                state,
                action,
                new_evidence,
                verification,
                self.config.retrieve_k,
            )
            if priority_actions:
                queue = prepend_priority_actions(priority_actions, queue)
            state.step_logs.append(
                {
                    "step": step + 1,
                    "action": action.kind.value,
                    "value": action.value,
                    "expanded_from": action.value if action.kind == ActionType.EXPAND else "",
                    "rationale": action.rationale,
                    "new_evidence": [ev.identifier for ev in new_evidence],
                    "new_evidence_count": len(new_evidence),
                    "new_cwe_ids": [ev.identifier for ev in new_evidence if ev.identifier.startswith("CWE-")],
                    "new_capec_ids": [ev.identifier for ev in new_evidence if ev.identifier.startswith("CAPEC-")],
                    "new_attack_ids": [ev.identifier for ev in new_evidence if is_attack_identifier(ev.identifier)],
                    "relation_hop_completed": relation_hops_completed(action, new_evidence),
                    "frontier_source": frontier_source(action),
                    "supported": verification.supported,
                    "confidence": verification.confidence,
                    "missing": verification.missing,
                }
            )
            if verification.supported and not has_pending_expansion(queue):
                continuation = [
                    action
                    for action in self.planner.next_actions(state, verification, self.config.retrieve_k)
                    if action.kind == ActionType.EXPAND
                ]
                if continuation:
                    queue = continuation
                    continue
                break

        answer = self.synthesizer.synthesize(question, state.evidence, verification)
        reverse_answer = self._reverse_answer_set(route, resolution)
        return {
            "question": question,
            "answer": answer,
            "verification": verification,
            "evidence": state.evidence,
            "trace": state.step_logs,
            "resolution": resolution.as_dict() if resolution is not None else None,
            "route": {"direction": route.direction, "target_type": route.target_type},
            "reverse_answer": reverse_answer,
        }

    def _orchestrator(self) -> "GraniteOrchestrator":
        if self.orchestrator is None:
            from .orchestrator import GraniteOrchestrator

            self.orchestrator = GraniteOrchestrator(
                self.corpus, resolver=self.resolver, search_backend=self.search_backend
            )
        return self.orchestrator

    def _resolve_entities(self, question: str, prefer_types) -> ResolutionResult | None:
        """Resolve a natural entity name only when no explicit ID is present.

        Explicit-ID questions keep their original behavior (resolver does not
        fire), so existing benchmarks and tests are unaffected.
        """
        if not self.config.resolve_entities or self.resolver is None:
            return None
        if extract_ids(question):
            return None
        return self.resolver.resolve_question(question, prefer_types=prefer_types)

    def _reverse_answer_set(self, route, resolution: ResolutionResult | None) -> list[str] | None:
        """For a reverse question, return all graph neighbors of the target type.

        Reverse "which attack patterns reference CWE X?" questions are set
        queries: a popular CWE maps to dozens of CAPECs, far beyond the forward
        flow's evidence budget. We answer them deterministically from the graph
        (fully provable) rather than through the single-answer agentic loop.
        """
        if route.direction != "reverse" or resolution is None or not resolution.resolved:
            return None
        neighbors = self.corpus.expand(resolution.chosen_id, k=10_000)
        return sorted({ev.identifier for ev in neighbors if ev.chunk.type == route.target_type})

    def _run_action(self, action: Action) -> list[Evidence]:
        limit = action.limit or self.config.retrieve_k
        if action.kind == ActionType.SEARCH:
            return self.search_backend.search(action.value, k=limit)
        if action.kind == ActionType.OPEN:
            evidence = self.corpus.open(action.value)
            return [evidence] if evidence else []
        if action.kind == ActionType.EXPAND:
            return self.corpus.expand(action.value, k=limit)
        if action.kind == ActionType.FIND:
            return self.corpus.find(action.value, k=limit)
        raise ValueError(f"Unsupported action: {action.kind}")


ATTACK_ID_RE = re.compile(r"^T\d{4}(?:\.\d{3})?$", re.IGNORECASE)


def seed_actions_for_resolution(resolution: ResolutionResult, route=None) -> list[Action]:
    """Seed graph traversal from a resolved identifier.

    Mirrors ``QueryPlanner.initial_actions`` for an explicit identifier so a
    name-addressed question enters the same ``open -> expand`` flow that an
    ID-addressed question would. Reverse (set) questions use a large expansion
    budget so the trace surfaces the neighbor set, not just the top-k.
    """
    ident = resolution.chosen_id
    rationale = f"resolved entity '{resolution.query}' to {ident} ({resolution.type})"
    expand_limit = 200 if (route is not None and route.direction == "reverse") else 8
    return [
        Action(ActionType.OPEN, ident, rationale),
        Action(ActionType.EXPAND, ident, rationale + "; expand for graph context", expand_limit),
    ]


def is_attack_identifier(identifier: str) -> bool:
    return bool(ATTACK_ID_RE.match(identifier or ""))


def has_pending_expansion(queue: list[Action]) -> bool:
    return any(action.kind == ActionType.EXPAND for action in queue)


def prepend_priority_actions(priority_actions: list[Action], queue: list[Action]) -> list[Action]:
    seen: set[tuple[ActionType, str]] = set()
    merged = []
    for action in priority_actions + queue:
        key = (action.kind, action.value.upper())
        if key not in seen:
            merged.append(action)
            seen.add(key)
    return merged


def relation_hops_completed(action: Action, evidence: list[Evidence]) -> list[str]:
    if action.kind != ActionType.EXPAND:
        return []
    src = action.value.upper()
    if is_attack_identifier(src) and any(ev.identifier.startswith("CAPEC-") for ev in evidence):
        return ["attack_to_capec"]
    if src.startswith("CAPEC-") and any(ev.identifier.startswith("CWE-") for ev in evidence):
        return ["capec_to_cwe"]
    return []


def frontier_source(action: Action) -> str:
    if action.kind == ActionType.EXPAND and "direct ATT&CK relation frontier" in action.rationale:
        return "graph_direct_path"
    if action.kind == ActionType.SEARCH:
        return "lexical_candidate_path"
    return ""


def budget_evidence(evidence: list[Evidence], budget: int) -> list[Evidence]:
    if budget <= 0:
        return []
    if len(evidence) <= budget:
        return evidence

    selected: list[Evidence] = []
    selected_ids: set[str] = set()
    for family in ("attack", "capec", "cwe"):
        ev = next((candidate for candidate in evidence if evidence_family(candidate) == family), None)
        if ev and ev.identifier not in selected_ids:
            selected.append(ev)
            selected_ids.add(ev.identifier)
    for family in ("cwe", "capec", "attack"):
        for ev in evidence:
            if len(selected) >= budget:
                return selected
            if ev.identifier not in selected_ids and evidence_family(ev) == family:
                selected.append(ev)
                selected_ids.add(ev.identifier)
    for ev in evidence:
        if len(selected) >= budget:
            break
        if ev.identifier not in selected_ids:
            selected.append(ev)
            selected_ids.add(ev.identifier)
    return selected


def evidence_family(evidence: Evidence) -> str:
    identifier = evidence.identifier
    if is_attack_identifier(identifier) or "ATT&CK" in evidence.chunk.source:
        return "attack"
    if identifier.startswith("CAPEC-"):
        return "capec"
    if identifier.startswith("CWE-"):
        return "cwe"
    return "other"
