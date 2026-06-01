from __future__ import annotations

import re

from .schema import Evidence, Verification
from .verifier import dedupe_evidence


class ExtractiveSynthesizer:
    """Grounded placeholder synthesizer before adding a generative LLM backend."""

    def synthesize(self, question: str, evidence: list[Evidence], verification: Verification) -> str:
        ranked = dedupe_evidence(evidence)[:6]
        if not ranked:
            return "I could not find supporting CTI evidence for this question."

        lines = [
            f"Evidence confidence: {verification.confidence:.2f}",
            "",
            "Grounded answer draft:",
        ]
        for ev in ranked[:4]:
            ident = ev.identifier
            name = ev.chunk.name or ev.chunk.type or ev.chunk.source or "evidence"
            snippet = ev.snippet or first_substantive_sentence(ev.chunk.text)
            lines.append(f"- [{ident}] {name}: {snippet}")
        if verification.missing:
            lines.append("")
            lines.append("Missing evidence facets: " + ", ".join(verification.missing))
        lines.append("")
        lines.append("Citations: " + ", ".join(ev.identifier for ev in ranked if ev.identifier))
        return "\n".join(lines)


def first_substantive_sentence(text: str, max_chars: int = 360) -> str:
    clean = re.sub(r"\s+", " ", text).strip()
    clean = re.sub(r"^#\s*", "", clean)
    parts = re.split(r"(?<=[.!?])\s+", clean)
    for part in parts:
        if len(part) > 40:
            return part[:max_chars]
    return clean[:max_chars]
