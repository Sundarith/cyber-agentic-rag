from __future__ import annotations

from dataclasses import dataclass

from .corpus import CorpusIndex
from .planner import QueryPlanner
from .schema import Action, ActionType, AgentState, Evidence, Verification
from .synthesizer import ExtractiveSynthesizer
from .verifier import EvidenceVerifier, dedupe_evidence


@dataclass(frozen=True)
class AgenticRAGConfig:
    retrieve_k: int = 8
    max_steps: int = 5
    evidence_budget: int = 14


class AgenticRAG:
    def __init__(
        self,
        corpus: CorpusIndex,
        config: AgenticRAGConfig | None = None,
        planner: QueryPlanner | None = None,
        verifier: EvidenceVerifier | None = None,
        synthesizer: ExtractiveSynthesizer | None = None,
    ) -> None:
        self.corpus = corpus
        self.config = config or AgenticRAGConfig()
        self.planner = planner or QueryPlanner()
        self.verifier = verifier or EvidenceVerifier(self.planner)
        self.synthesizer = synthesizer or ExtractiveSynthesizer()

    def answer(self, question: str) -> dict:
        state = AgentState(question=question)
        queue = self.planner.initial_actions(question, self.config.retrieve_k)
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
                if action.kind in {ActionType.OPEN, ActionType.EXPAND}:
                    state.opened_ids.add(ev.identifier)
            state.evidence = dedupe_evidence(state.evidence)[: self.config.evidence_budget]
            verification = self.verifier.verify(question, state.evidence)
            state.step_logs.append(
                {
                    "step": step + 1,
                    "action": action.kind.value,
                    "value": action.value,
                    "rationale": action.rationale,
                    "new_evidence": [ev.identifier for ev in new_evidence],
                    "confidence": verification.confidence,
                    "missing": verification.missing,
                }
            )
            if verification.supported:
                break

        answer = self.synthesizer.synthesize(question, state.evidence, verification)
        return {
            "question": question,
            "answer": answer,
            "verification": verification,
            "evidence": state.evidence,
            "trace": state.step_logs,
        }

    def _run_action(self, action: Action) -> list[Evidence]:
        limit = action.limit or self.config.retrieve_k
        if action.kind == ActionType.SEARCH:
            return self.corpus.search(action.value, k=limit)
        if action.kind == ActionType.OPEN:
            evidence = self.corpus.open(action.value)
            return [evidence] if evidence else []
        if action.kind == ActionType.EXPAND:
            return self.corpus.expand(action.value, k=limit)
        if action.kind == ActionType.FIND:
            return self.corpus.find(action.value, k=limit)
        raise ValueError(f"Unsupported action: {action.kind}")
