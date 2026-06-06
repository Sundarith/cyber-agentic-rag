from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

from .schema import Evidence, Verification
from .verifier import dedupe_evidence

DEFAULT_GRANITE_MODEL = "ibm-granite/granite-4.1-8b"
DEFAULT_LLM_ENDPOINT = "http://localhost:8000/v1/chat/completions"


class Synthesizer(Protocol):
    def synthesize(self, question: str, evidence: list[Evidence], verification: Verification) -> str:
        ...


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


@dataclass
class OpenAIChatClient:
    """Tiny OpenAI-compatible chat client for local vLLM/serving endpoints."""

    endpoint: str = DEFAULT_LLM_ENDPOINT
    model: str = DEFAULT_GRANITE_MODEL
    temperature: float = 0.0
    max_tokens: int = 256
    timeout_s: float = 120.0

    def complete(self, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM endpoint unavailable at {self.endpoint}") from exc
        choices = raw.get("choices") or []
        if not choices:
            raise RuntimeError("LLM response did not include choices")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if not isinstance(content, str):
            raise RuntimeError("LLM response did not include message.content")
        return content.strip()


def endpoint_reachable(endpoint: str, timeout_s: float = 2.0) -> bool:
    """Quick liveness check for an OpenAI-compatible endpoint (the /models route)."""
    models_url = endpoint.replace("/chat/completions", "/models")
    try:
        with urllib.request.urlopen(models_url, timeout=timeout_s) as response:
            return getattr(response, "status", 200) == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


class GraniteGroundedSynthesizer:
    """Granite-backed answerer constrained to retrieved CTI evidence."""

    def __init__(
        self,
        client: OpenAIChatClient | None = None,
        max_evidence: int = 8,
        max_evidence_chars: int = 900,
    ) -> None:
        self.client = client or OpenAIChatClient()
        self.max_evidence = max_evidence
        self.max_evidence_chars = max_evidence_chars

    def synthesize(self, question: str, evidence: list[Evidence], verification: Verification) -> str:
        ranked = dedupe_evidence(evidence)[: self.max_evidence]
        if not ranked:
            return "I could not find supporting CTI evidence for this question."
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a cybersecurity RAG answerer. Answer only from the supplied evidence. "
                    "If the task asks for a CWE mapping, give a brief evidence-grounded justification "
                    "and make the final line contain only the CWE ID. Do not invent CWE IDs that are "
                    "not supported by the evidence."
                ),
            },
            {
                "role": "user",
                "content": self._build_prompt(question, ranked, verification),
            },
        ]
        return self.client.complete(messages)

    def _build_prompt(self, question: str, evidence: list[Evidence], verification: Verification) -> str:
        lines = [
            f"Question:\n{question.strip()}",
            "",
            f"Verifier confidence: {verification.confidence:.2f}",
        ]
        if verification.missing:
            lines.append("Missing evidence facets: " + ", ".join(verification.missing))
        lines += ["", "Evidence:"]
        for idx, ev in enumerate(evidence, start=1):
            ident = ev.identifier
            title = ev.chunk.name or ev.chunk.type or ev.chunk.source or "evidence"
            text = re.sub(r"\s+", " ", ev.snippet or ev.chunk.text).strip()
            lines.append(
                f"[{idx}] {ident} | {ev.chunk.source} | {title}\n"
                f"{_truncate(text, self.max_evidence_chars)}"
            )
        lines += [
            "",
            "Instructions:",
            "- Use only the evidence above.",
            "- Cite evidence IDs in the justification.",
            "- If mapping to CWE, final line must contain only one CWE ID.",
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class StructuredSynthesisValidation:
    """Post-generation validation for a grounded structured LLM answer."""

    supported: bool
    selected_cwe_valid: bool
    selected_path_valid: bool
    unsupported_ids: list[str]
    unsupported_citations: list[str]
    warnings: list[str]
    candidate_paths: list[list[str]]
    allowed_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "supported": self.supported,
            "selected_cwe_valid": self.selected_cwe_valid,
            "selected_path_valid": self.selected_path_valid,
            "unsupported_ids": self.unsupported_ids,
            "unsupported_citations": self.unsupported_citations,
            "warnings": self.warnings,
            "candidate_paths": self.candidate_paths,
            "allowed_ids": self.allowed_ids,
        }


@dataclass(frozen=True)
class StructuredSynthesisResult:
    """Granite JSON answer plus local validation metadata."""

    raw: str
    parsed: dict[str, Any] | None
    answerable: bool
    selected_cwe: str
    selected_path: list[str]
    cited_evidence_ids: list[str]
    confidence: str
    reason: str
    validation: StructuredSynthesisValidation
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw": self.raw,
            "parsed": self.parsed,
            "answerable": self.answerable,
            "selected_cwe": self.selected_cwe,
            "selected_path": self.selected_path,
            "cited_evidence_ids": self.cited_evidence_ids,
            "confidence": self.confidence,
            "reason": self.reason,
            "validation": self.validation.to_dict(),
            "error": self.error,
        }


class GraniteStructuredSynthesizer:
    """Granite-backed selector that must answer with validated grounded JSON."""

    def __init__(
        self,
        client: OpenAIChatClient | None = None,
        max_evidence: int = 14,
        max_evidence_chars: int = 600,
    ) -> None:
        self.client = client or OpenAIChatClient(max_tokens=384)
        self.max_evidence = max_evidence
        self.max_evidence_chars = max_evidence_chars

    def synthesize_structured(
        self,
        question: str,
        evidence: list[Evidence],
        verification: Verification,
        trace: list[dict[str, Any]] | None = None,
        choices: list[str] | None = None,
    ) -> StructuredSynthesisResult:
        candidate_paths = candidate_cwe_paths(trace or [])
        ranked = select_structured_evidence(evidence, candidate_paths, self.max_evidence)
        answer_cwe_ids = choice_cwe_ids(choices or [])
        raw = ""
        try:
            raw = self.client.complete(
                [
                    {
                        "role": "system",
                        "content": (
                            "You are a cybersecurity RAG selector. Return valid JSON only. "
                            "You may select only IDs listed in the evidence packet or candidate paths. "
                            "If no supported path answers the question, set answerable=false."
                        ),
                    },
                    {
                        "role": "user",
                        "content": self._build_structured_prompt(
                            question,
                            ranked,
                            verification,
                            candidate_paths,
                            choices or [],
                            answer_cwe_ids,
                        ),
                    },
                ]
            )
            try:
                parsed = parse_json_object(raw)
            except Exception:
                raw = self.client.complete(
                    [
                        {
                            "role": "system",
                            "content": "Return valid JSON only. No markdown, no prose, no reasoning outside JSON.",
                        },
                        {
                            "role": "user",
                            "content": self._build_retry_prompt(raw, ranked, candidate_paths),
                        },
                    ]
                )
                parsed = parse_json_object(raw)
            return normalize_structured_result(raw, parsed, ranked, candidate_paths, answer_cwe_ids)
        except Exception as exc:
            validation = validate_structured_payload({}, ranked, candidate_paths)
            return StructuredSynthesisResult(
                raw=raw,
                parsed=None,
                answerable=False,
                selected_cwe="",
                selected_path=[],
                cited_evidence_ids=[],
                confidence="low",
                reason="",
                validation=validation,
                error=str(exc),
            )

    def compose_grounded_answer(
        self,
        question: str,
        evidence: list[Evidence],
        selected_path: list[str],
    ) -> str:
        normalized_path = normalize_id_list(selected_path)
        if not normalized_path:
            raise ValueError("cannot compose answer without a selected path")
        id_names = evidence_name_map(evidence)
        payload = {
            "question": question.strip(),
            "selected_path": normalized_path,
            "path_entities": [
                {
                    "id": identifier,
                    "name": id_names.get(identifier, identifier),
                }
                for identifier in normalized_path
            ],
        }
        raw = self.client.complete(
            [
                {
                    "role": "system",
                    "content": (
                        "You write concise cybersecurity RAG final answers. Use only the supplied path. "
                        "Do not mention IDs outside the supplied path. Do not add unsupported caveats."
                    ),
                },
                {
                    "role": "user",
                    "content": "\n".join(
                        [
                            "Write a short final answer in 2-4 sentences.",
                            "Mention the CWE answer first.",
                            "Do not include the ATT&CK -> CAPEC -> CWE path as a separate line; it is displayed elsewhere.",
                            "Use only these IDs and names:",
                            json.dumps(payload, indent=2, sort_keys=True),
                        ]
                    ),
                },
            ]
        )
        validate_composed_answer(raw, normalized_path)
        return raw.strip()

    def _build_structured_prompt(
        self,
        question: str,
        evidence: list[Evidence],
        verification: Verification,
        candidate_paths: list[list[str]],
        choices: list[str],
        answer_cwe_ids: list[str],
    ) -> str:
        possible_cwe_ids = answer_cwe_ids or candidate_cwe_ids(evidence, candidate_paths)
        packet = {
            "question": question.strip(),
            "choices": choices,
            "verifier": {
                "supported": verification.supported,
                "confidence": round(float(verification.confidence), 3),
                "missing": verification.missing,
            },
            "allowed_evidence_ids": [normalize_identifier(ev.identifier) for ev in evidence],
            "candidate_paths": candidate_paths,
            "candidate_cwe_ids": possible_cwe_ids,
            "evidence": [serialize_evidence_for_prompt(ev, self.max_evidence_chars) for ev in evidence],
        }
        return "\n".join(
            [
                "Use this evidence packet to answer the question.",
                "Return exactly one JSON object with these fields:",
                (
                    '{"answerable": true, "selected_cwe": "CWE-000", '
                    '"selected_path": ["T0000", "CAPEC-000", "CWE-000"], '
                    '"cited_evidence_ids": ["T0000", "CAPEC-000", "CWE-000"], '
                    '"confidence": "low|medium|high", "reason": "short reason"}'
                ),
                "Rules:",
                "- Do not invent IDs.",
                "- selected_cwe must be one of candidate_cwe_ids.",
                "- If choices are provided, selected_cwe must be one of the CWE IDs in those choices.",
                "- selected_path must exactly match one candidate_paths entry, or be [] when answerable=false.",
                "- cited_evidence_ids must come from allowed_evidence_ids or candidate_paths.",
                "- If multiple paths are possible and the evidence cannot disambiguate, set answerable=false.",
                "",
                "Evidence packet JSON:",
                json.dumps(packet, indent=2, sort_keys=True),
            ]
        )

    def _build_retry_prompt(
        self,
        invalid_output: str,
        evidence: list[Evidence],
        candidate_paths: list[list[str]],
    ) -> str:
        return "\n".join(
            [
                "The previous response was invalid because it was not a single JSON object.",
                "Rewrite it as one JSON object with exactly these fields:",
                "answerable, selected_cwe, selected_path, cited_evidence_ids, confidence, reason",
                "Use only these allowed evidence IDs:",
                json.dumps([normalize_identifier(ev.identifier) for ev in evidence]),
                "Use only these candidate paths:",
                json.dumps(candidate_paths),
                "Previous invalid response:",
                _truncate(str(invalid_output or ""), 1200),
            ]
        )


def serialize_evidence_for_prompt(evidence: Evidence, max_chars: int) -> dict[str, Any]:
    text = re.sub(r"\s+", " ", evidence.snippet or evidence.chunk.text).strip()
    return {
        "id": normalize_identifier(evidence.identifier),
        "source": evidence.chunk.source,
        "type": evidence.chunk.type,
        "name": evidence.chunk.name,
        "tool": evidence.tool,
        "snippet": _truncate(text, max_chars),
    }


def parse_json_object(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        raise ValueError("empty structured LLM response")
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.IGNORECASE | re.DOTALL)
    if fenced:
        raw = fenced.group(1).strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        if start < 0:
            raise
        decoder = json.JSONDecoder()
        parsed, _end = decoder.raw_decode(raw[start:])
    if not isinstance(parsed, dict):
        raise ValueError("structured LLM response was not a JSON object")
    return parsed


def normalize_structured_result(
    raw: str,
    parsed: dict[str, Any],
    evidence: list[Evidence],
    candidate_paths: list[list[str]],
    answer_cwe_ids: list[str] | None = None,
) -> StructuredSynthesisResult:
    answerable = bool(parsed.get("answerable"))
    selected_cwe = normalize_identifier(str(parsed.get("selected_cwe") or ""))
    selected_path = normalize_id_list(parsed.get("selected_path"))
    cited_evidence_ids = normalize_id_list(parsed.get("cited_evidence_ids"))
    confidence = normalize_confidence(parsed.get("confidence"))
    reason = str(parsed.get("reason") or "").strip()
    normalized = {
        "answerable": answerable,
        "selected_cwe": selected_cwe,
        "selected_path": selected_path,
        "cited_evidence_ids": cited_evidence_ids,
        "confidence": confidence,
        "reason": reason,
    }
    validation = validate_structured_payload(normalized, evidence, candidate_paths, answer_cwe_ids or [])
    return StructuredSynthesisResult(
        raw=raw,
        parsed=parsed,
        answerable=answerable,
        selected_cwe=selected_cwe,
        selected_path=selected_path,
        cited_evidence_ids=cited_evidence_ids,
        confidence=confidence,
        reason=reason,
        validation=validation,
    )


def validate_structured_payload(
    payload: dict[str, Any],
    evidence: list[Evidence],
    candidate_paths: list[list[str]],
    answer_cwe_ids: list[str] | None = None,
) -> StructuredSynthesisValidation:
    allowed_ids = sorted({normalize_identifier(ev.identifier) for ev in evidence if ev.identifier})
    allowed = set(allowed_ids)
    normalized_paths = [normalize_id_list(path) for path in candidate_paths]
    path_tuples = {tuple(path) for path in normalized_paths}
    path_ids = {identifier for path in normalized_paths for identifier in path}
    candidate_cwes = set(candidate_cwe_ids(evidence, normalized_paths))
    answer_cwes = set(normalize_id_list(answer_cwe_ids or []))
    selected_cwe = normalize_identifier(str(payload.get("selected_cwe") or ""))
    selected_path = normalize_id_list(payload.get("selected_path"))
    cited_evidence_ids = normalize_id_list(payload.get("cited_evidence_ids"))
    answerable = bool(payload.get("answerable"))

    referenced_ids = set(selected_path) | set(cited_evidence_ids)
    if selected_cwe:
        referenced_ids.add(selected_cwe)
    supported_ids = allowed | path_ids
    unsupported_ids = sorted(identifier for identifier in referenced_ids if identifier and identifier not in supported_ids)
    unsupported_citations = sorted(identifier for identifier in cited_evidence_ids if identifier and identifier not in supported_ids)
    selected_cwe_valid = (not answerable and not selected_cwe) or (
        bool(selected_cwe)
        and selected_cwe in candidate_cwes
        and (not answer_cwes or selected_cwe in answer_cwes)
    )
    selected_path_valid = (not answerable and not selected_path) or (bool(selected_path) and tuple(selected_path) in path_tuples)

    warnings = []
    if unsupported_ids:
        warnings.append("unsupported_ids")
    if unsupported_citations:
        warnings.append("unsupported_citations")
    if answerable and not selected_cwe_valid:
        warnings.append("selected_cwe_not_in_candidates_or_choices")
    if answerable and not selected_path_valid:
        warnings.append("selected_path_not_in_trace")
    if answerable and not candidate_paths:
        warnings.append("no_candidate_paths")

    return StructuredSynthesisValidation(
        supported=not warnings,
        selected_cwe_valid=selected_cwe_valid,
        selected_path_valid=selected_path_valid,
        unsupported_ids=unsupported_ids,
        unsupported_citations=unsupported_citations,
        warnings=warnings,
        candidate_paths=normalized_paths,
        allowed_ids=allowed_ids,
    )


def candidate_cwe_paths(trace: list[dict[str, Any]]) -> list[list[str]]:
    attack_to_capecs: dict[str, list[str]] = {}
    capec_to_cwes: dict[str, list[str]] = {}
    for step in trace:
        if str(step.get("action") or "") != "expand":
            continue
        src = normalize_identifier(str(step.get("expanded_from") or step.get("value") or ""))
        new_ids = normalize_id_list(step.get("new_evidence") or [])
        if is_attack_identifier(src):
            attack_to_capecs[src] = [identifier for identifier in new_ids if identifier.startswith("CAPEC-")]
        if src.startswith("CAPEC-"):
            cwes = [identifier for identifier in new_ids if identifier.startswith("CWE-")]
            if cwes:
                capec_to_cwes[src] = cwes

    paths: list[list[str]] = []
    for attack_id, capec_ids in attack_to_capecs.items():
        for capec_id in capec_ids:
            for cwe_id in capec_to_cwes.get(capec_id, []):
                paths.append([attack_id, capec_id, cwe_id])
    return dedupe_paths(paths)


def select_structured_evidence(
    evidence: list[Evidence],
    candidate_paths: list[list[str]],
    max_evidence: int,
) -> list[Evidence]:
    deduped = dedupe_evidence(evidence)
    if max_evidence <= 0:
        return []
    by_id = {normalize_identifier(ev.identifier): ev for ev in deduped}
    selected: list[Evidence] = []
    selected_ids: set[str] = set()

    def add(ev: Evidence | None) -> None:
        if not ev or len(selected) >= max_evidence:
            return
        identifier = normalize_identifier(ev.identifier)
        if identifier and identifier not in selected_ids:
            selected.append(ev)
            selected_ids.add(identifier)

    for path in candidate_paths:
        for identifier in path:
            add(by_id.get(normalize_identifier(identifier)))
    for family in ("attack", "cwe", "capec"):
        for ev in deduped:
            if evidence_family(ev) == family:
                add(ev)
    for ev in deduped:
        add(ev)
    return selected


def evidence_family(evidence: Evidence) -> str:
    identifier = normalize_identifier(evidence.identifier)
    if is_attack_identifier(identifier) or "ATT&CK" in evidence.chunk.source:
        return "attack"
    if identifier.startswith("CAPEC-"):
        return "capec"
    if identifier.startswith("CWE-"):
        return "cwe"
    return "other"


def evidence_name_map(evidence: list[Evidence]) -> dict[str, str]:
    names = {}
    for ev in evidence:
        identifier = normalize_identifier(ev.identifier)
        if identifier and identifier not in names:
            names[identifier] = ev.chunk.name or identifier
    return names


def validate_composed_answer(answer: str, allowed_ids: list[str]) -> None:
    allowed = set(normalize_id_list(allowed_ids))
    mentioned = set(re.findall(r"\b(?:T\d{4}(?:\.\d{3})?|CAPEC-\d+|CWE-\d+)\b", answer or "", flags=re.IGNORECASE))
    unsupported = sorted(normalize_identifier(identifier) for identifier in mentioned if normalize_identifier(identifier) not in allowed)
    if unsupported:
        raise ValueError("composed answer mentioned unsupported IDs: " + ", ".join(unsupported))
    missing = [identifier for identifier in allowed if identifier not in {normalize_identifier(item) for item in mentioned}]
    if missing:
        raise ValueError("composed answer omitted required IDs: " + ", ".join(missing))


def candidate_cwe_ids(evidence: list[Evidence], candidate_paths: list[list[str]]) -> list[str]:
    ids = {identifier for path in candidate_paths for identifier in path if identifier.startswith("CWE-")}
    ids.update(normalize_identifier(ev.identifier) for ev in evidence if normalize_identifier(ev.identifier).startswith("CWE-"))
    return sorted(ids)


def choice_cwe_ids(choices: list[str]) -> list[str]:
    ids = []
    for choice in choices:
        ids.extend(re.findall(r"\bCWE-\d+\b", str(choice), flags=re.IGNORECASE))
    return sorted({normalize_identifier(identifier) for identifier in ids})


def normalize_id_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [normalize_identifier(str(item)) for item in value if normalize_identifier(str(item))]


def normalize_identifier(identifier: str) -> str:
    return str(identifier or "").strip().upper()


def normalize_confidence(value: Any) -> str:
    if isinstance(value, (int, float)):
        if value >= 0.8:
            return "high"
        if value >= 0.5:
            return "medium"
        return "low"
    normalized = str(value or "").strip().lower()
    return normalized if normalized in {"low", "medium", "high"} else "low"


def is_attack_identifier(identifier: str) -> bool:
    return bool(re.match(r"^T\d{4}(?:\.\d{3})?$", identifier or "", re.IGNORECASE))


def dedupe_paths(paths: list[list[str]]) -> list[list[str]]:
    seen: set[tuple[str, ...]] = set()
    deduped = []
    for path in paths:
        key = tuple(path)
        if key not in seen:
            deduped.append(path)
            seen.add(key)
    return deduped


def first_substantive_sentence(text: str, max_chars: int = 360) -> str:
    clean = re.sub(r"\s+", " ", text).strip()
    clean = re.sub(r"^#\s*", "", clean)
    parts = re.split(r"(?<=[.!?])\s+", clean)
    for part in parts:
        if len(part) > 40:
            return part[:max_chars]
    return clean[:max_chars]


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 16)].rstrip() + " ...[truncated]"
