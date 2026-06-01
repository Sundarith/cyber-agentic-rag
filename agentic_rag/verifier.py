from __future__ import annotations

from .planner import QueryPlanner
from .schema import Evidence, Verification


class EvidenceVerifier:
    def __init__(self, planner: QueryPlanner | None = None) -> None:
        self.planner = planner or QueryPlanner()

    def verify(self, question: str, evidence: list[Evidence]) -> Verification:
        deduped = dedupe_evidence(evidence)
        cited_ids = [ev.identifier for ev in deduped if ev.identifier]
        missing = []
        facets = self.planner.required_facets(question)
        if "root_cause" in facets and not any(ev.identifier.startswith("CWE-") for ev in deduped):
            missing.append("root_cause")
        if "attack_pattern" in facets and not any(ev.identifier.startswith("CAPEC-") for ev in deduped):
            missing.append("attack_pattern")
        if "detection" in facets and not any("## Detection" in ev.chunk.text for ev in deduped):
            missing.append("detection")

        useful = min(len(deduped), 5) / 5.0
        source_diversity = len({ev.chunk.source for ev in deduped if ev.chunk.source}) / 4.0
        coverage = 1.0 if not missing else max(0.0, 1.0 - 0.35 * len(missing))
        confidence = min(1.0, 0.45 * useful + 0.25 * source_diversity + 0.30 * coverage)
        return Verification(
            supported=bool(deduped) and not missing and confidence >= 0.55,
            confidence=confidence,
            missing=missing,
            cited_ids=cited_ids,
        )


def dedupe_evidence(evidence: list[Evidence]) -> list[Evidence]:
    best: dict[str, Evidence] = {}
    for ev in evidence:
        key = ev.identifier
        old = best.get(key)
        if old is None or ev.score > old.score:
            best[key] = ev
    return sorted(best.values(), key=lambda ev: ev.score, reverse=True)
