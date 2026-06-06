from __future__ import annotations

import re
from dataclasses import dataclass

from .corpus import extract_ids
from .schema import Action, ActionType, AgentState, Evidence, Verification

ROOT_CAUSE_RE = re.compile(r"\b(cwe|root cause|weakness|underlies|underlying)\b", re.IGNORECASE)
CAPEC_RE = re.compile(r"\bcapec|attack pattern\b", re.IGNORECASE)
DETECTION_RE = re.compile(r"\bdetect|detection|monitor|log|analytics\b", re.IGNORECASE)
ATTACK_ID_RE = re.compile(r"^T\d{4}(?:\.\d{3})?$", re.IGNORECASE)
# First noun after "which/what" names the requested answer type, which tells us
# the question's entry/direction: "which CWE ..." enters at a technique/CAPEC
# (forward); "which attack patterns/techniques ..." enters at a CWE (reverse).
ASK_TARGET_RE = re.compile(r"\b(?:which|what)\b\s+(?:listed\s+)?([a-z]+)", re.IGNORECASE)
_REVERSE_TARGETS = {"attack", "capec", "technique", "techniques", "pattern", "patterns"}
# prefer_types orderings handed to the entity resolver per direction.
FORWARD_PREFER_TYPES = ("technique", "attack_pattern", "weakness")
REVERSE_PREFER_TYPES = ("weakness", "attack_pattern", "technique")
# Map the requested-answer noun to the graph neighbor type a reverse question
# should collect.
_REVERSE_TARGET_TYPE = {
    "attack": "attack_pattern",
    "pattern": "attack_pattern",
    "patterns": "attack_pattern",
    "capec": "attack_pattern",
    "technique": "technique",
    "techniques": "technique",
}


@dataclass(frozen=True)
class Route:
    """How the agent should resolve and traverse a natural question.

    - ``forward``: enter at a technique/CAPEC, traverse toward a single CWE
      (the existing multiple-choice / single-answer flow).
    - ``reverse``: enter at a CWE (or CAPEC) and collect the *set* of neighbors
      of ``target_type`` that reference it (e.g. all CAPECs for a CWE).
    """

    direction: str  # "forward" | "reverse"
    prefer_types: tuple[str, ...]
    target_type: str  # neighbor type collected for reverse set answers


class QueryPlanner:
    """Heuristic planner for the first agentic RAG prototype."""

    def route(self, question: str) -> Route:
        """Classify a question's entry/direction from the first requested noun.

        Soft routing only -- an exact resolver name match always dominates the
        prefer ordering -- so this mainly needs to distinguish forward
        (asks for a CWE) from reverse (asks for attack patterns/techniques).
        """
        match = ASK_TARGET_RE.search(question)
        target = match.group(1).lower() if match else ""
        if target in _REVERSE_TARGETS:
            return Route("reverse", REVERSE_PREFER_TYPES, _REVERSE_TARGET_TYPE.get(target, "attack_pattern"))
        return Route("forward", FORWARD_PREFER_TYPES, "weakness")

    def entry_prefer_types(self, question: str) -> tuple[str, ...]:
        """Backward-compatible accessor for the resolver prefer ordering."""
        return self.route(question).prefer_types

    def initial_actions(self, question: str, retrieve_k: int) -> list[Action]:
        actions: list[Action] = []
        ids = extract_ids(question)
        for ident in ids:
            actions.append(Action(ActionType.OPEN, ident, "question names an exact CTI identifier"))
        for ident in ids:
            actions.append(Action(ActionType.EXPAND, ident, "explicit identifier may need graph context", 8))
        actions.append(Action(ActionType.SEARCH, question, "baseline evidence retrieval", retrieve_k))
        return actions

    def next_actions(self, state: AgentState, verification: Verification, retrieve_k: int) -> list[Action]:
        facets = self.required_facets(state.question)
        cited = verification.cited_ids
        if verification.supported:
            return self._relation_expansions(state, cited, facets)

        actions: list[Action] = []
        actions.extend(self._relation_expansions(state, cited, facets))
        if "root_cause" in verification.missing:
            actions.append(
                Action(
                    ActionType.SEARCH,
                    f"{state.question} CWE weakness root cause vulnerability",
                    "root-cause coverage missing",
                    retrieve_k,
                )
            )
        if "attack_pattern" in verification.missing:
            actions.append(
                Action(ActionType.SEARCH, f"{state.question} CAPEC attack pattern", "CAPEC coverage missing", retrieve_k)
            )
        if "detection" in verification.missing:
            actions.append(
                Action(ActionType.FIND, "## Detection", "detection section requested but not yet covered", retrieve_k)
            )
        for ident in cited[:4]:
            if ident not in state.opened_ids:
                actions.append(Action(ActionType.EXPAND, ident, "expand strongest cited evidence", 6))
        return dedupe_actions(actions)

    def priority_actions_after_step(
        self,
        state: AgentState,
        action: Action,
        new_evidence: list[Evidence],
        verification: Verification,
        retrieve_k: int,
    ) -> list[Action]:
        facets = self.required_facets(state.question)
        if "root_cause" not in facets:
            return []
        if action.kind != ActionType.EXPAND or not is_attack_identifier(action.value):
            return []

        actions = []
        for ev in new_evidence:
            ident = ev.identifier.upper()
            if ident.startswith("CAPEC-") and ident not in state.opened_ids:
                actions.append(
                    Action(
                        ActionType.EXPAND,
                        ident,
                        "direct ATT&CK relation frontier to root-cause CWE",
                        10,
                    )
                )
        return dedupe_actions(actions)

    def required_facets(self, question: str) -> list[str]:
        facets = []
        if ROOT_CAUSE_RE.search(question):
            facets.append("root_cause")
        if CAPEC_RE.search(question):
            facets.append("attack_pattern")
        if DETECTION_RE.search(question):
            facets.append("detection")
        return facets

    def _relation_expansions(self, state: AgentState, cited: list[str], facets: list[str]) -> list[Action]:
        actions: list[Action] = []
        visible = visible_identifiers(state, cited)
        if "root_cause" in facets:
            for ident in visible:
                if ident.startswith("CAPEC-") and ident not in state.opened_ids:
                    actions.append(Action(ActionType.EXPAND, ident, "CAPEC evidence may link to root-cause CWE", 10))
            if actions:
                return dedupe_actions(actions)
        if "attack_pattern" in facets or "root_cause" in facets:
            for ident in visible:
                if is_attack_identifier(ident) and ident not in state.opened_ids:
                    actions.append(Action(ActionType.EXPAND, ident, "ATT&CK evidence may link to CAPEC patterns", 10))
        return dedupe_actions(actions)


def visible_identifiers(state: AgentState, cited: list[str]) -> list[str]:
    seen: set[str] = set()
    identifiers = []
    for ident in cited + [ev.identifier for ev in state.evidence]:
        normalized = ident.upper()
        if normalized and normalized not in seen:
            identifiers.append(normalized)
            seen.add(normalized)
    return identifiers


def is_attack_identifier(identifier: str) -> bool:
    return bool(ATTACK_ID_RE.match(identifier or ""))


def dedupe_actions(actions: list[Action]) -> list[Action]:
    seen: set[tuple[ActionType, str]] = set()
    deduped = []
    for action in actions:
        key = (action.kind, action.value.upper())
        if key not in seen:
            deduped.append(action)
            seen.add(key)
    return deduped
