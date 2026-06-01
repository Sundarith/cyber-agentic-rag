from __future__ import annotations

import re

from .corpus import extract_ids
from .schema import Action, ActionType, AgentState, Verification

ROOT_CAUSE_RE = re.compile(r"\b(cwe|root cause|weakness|underlies|underlying)\b", re.IGNORECASE)
CAPEC_RE = re.compile(r"\bcapec|attack pattern\b", re.IGNORECASE)
DETECTION_RE = re.compile(r"\bdetect|detection|monitor|log|analytics\b", re.IGNORECASE)


class QueryPlanner:
    """Heuristic planner for the first agentic RAG prototype."""

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
        if verification.supported:
            return []

        cited = verification.cited_ids
        actions: list[Action] = []
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
        return actions

    def required_facets(self, question: str) -> list[str]:
        facets = []
        if ROOT_CAUSE_RE.search(question):
            facets.append("root_cause")
        if CAPEC_RE.search(question):
            facets.append("attack_pattern")
        if DETECTION_RE.search(question):
            facets.append("detection")
        return facets
