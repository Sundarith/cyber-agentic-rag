"""Natural-name -> CTI identifier resolution for hidden-ID multihop QA.

The planner seeds graph traversal from explicit identifiers found in the
question (``extract_ids``). Natural questions such as
``"For Process Discovery, which CWE weakness ..."`` contain no identifier, so
the agent never enters the ``open -> expand -> expand`` graph flow. This module
maps a natural entity name to a CTI identifier (e.g. ``"Process Discovery" ->
T1057``) so resolution can feed the existing planner.

The resolver is deliberately lexical/deterministic for v1: it indexes the
``name`` field already carried by every :class:`~agentic_rag.schema.Chunk` and
ranks candidates with three tiers (exact name, alias/substring, token overlap),
breaking ties by a caller-supplied type priority. Type priority removes
"wrong-type" collisions -- e.g. a malware or group chunk whose name matches a
technique phrase -- before they can register as ambiguity. Genuine *same-type*
near-ties are resolved to the top candidate but flagged with alternatives so an
eval can measure an ambiguity rate without losing a scoreable path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .corpus import CorpusIndex

# Relative score margin below which two same-type candidates are treated as a
# genuine ambiguous tie. Named for easy tuning from evals.
AMBIGUITY_MARGIN = 0.1

# Type priority used when the caller does not override it. The hidden-ID v1
# target enters the graph at an ATT&CK technique, so techniques win ties; CAPEC
# attack patterns and CWE weaknesses follow for name-addressed questions.
DEFAULT_PREFER_TYPES = ("technique", "attack_pattern", "weakness")

# Leading/trailing filler stripped from candidate spans before resolution so a
# sentence-initial span like "For Process Discovery" reduces to "Process
# Discovery".
_STOPWORDS = {
    "a", "an", "the", "for", "of", "to", "in", "on", "and", "or",
    "what", "which", "when", "where", "who", "whose", "that", "this",
    "is", "are", "by", "with", "from", "about",
}

_PUNCT_RE = re.compile(r"[^a-z0-9\s]+")
_WS_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[a-z0-9]+")
# Maximal runs of Capitalized/ALLCAPS words, the candidate entity spans in a
# question (e.g. "Process Discovery", "Command and Scripting Interpreter").
_CAP_SPAN_RE = re.compile(r"[A-Z][A-Za-z0-9]*(?:[\s-]+[A-Za-z0-9]+)*")


def normalize_name(name: str) -> str:
    """Lowercase, drop punctuation, and collapse whitespace."""
    lowered = name.lower()
    lowered = _PUNCT_RE.sub(" ", lowered)
    return _WS_RE.sub(" ", lowered).strip()


def _tokens(normalized: str) -> frozenset[str]:
    return frozenset(_TOKEN_RE.findall(normalized))


@dataclass(frozen=True)
class Candidate:
    identifier: str
    name: str
    source: str
    type: str
    score: float
    mode: str  # exact | alias | fallback

    def as_dict(self) -> dict[str, object]:
        return {
            "identifier": self.identifier,
            "name": self.name,
            "source": self.source,
            "type": self.type,
            "score": round(self.score, 4),
            "mode": self.mode,
        }


@dataclass(frozen=True)
class ResolutionResult:
    query: str
    chosen_id: str
    score: float
    source: str
    type: str
    name_matched: str
    mode: str  # exact | alias | fallback | none
    ambiguous: bool
    margin: float
    candidates: list[Candidate] = field(default_factory=list)

    @property
    def resolved(self) -> bool:
        return bool(self.chosen_id)

    def as_dict(self) -> dict[str, object]:
        return {
            "query": self.query,
            "chosen_id": self.chosen_id,
            "score": round(self.score, 4),
            "source": self.source,
            "type": self.type,
            "name_matched": self.name_matched,
            "mode": self.mode,
            "ambiguous": self.ambiguous,
            "margin": round(self.margin, 4),
            "candidates": [c.as_dict() for c in self.candidates],
        }


_NONE_RESULT_FIELDS = dict(
    chosen_id="", score=0.0, source="", type="", name_matched="", mode="none",
    ambiguous=False, margin=0.0,
)


class EntityResolver:
    """Maps natural entity names to CTI identifiers from a :class:`CorpusIndex`."""

    # n-gram span lengths probed for lowercase questions (no capitalization cue).
    MAX_SPAN_WORDS = 7

    def __init__(self, corpus: CorpusIndex, max_candidates: int = 5) -> None:
        self.max_candidates = max_candidates
        # normalized name -> list of (identifier, name, source, type)
        self._by_norm: dict[str, list[tuple[str, str, str, str]]] = {}
        # entries (identifier, name, source, type, norm, token set) for the
        # substring/overlap fallback tiers.
        self._entries: list[tuple[str, str, str, str, str, frozenset[str]]] = []
        # identifier -> (name, source, type) for alias-tier candidate metadata.
        self._meta: dict[str, tuple[str, str, str]] = {}
        seen_ids: set[str] = set()
        for chunk in corpus.chunks:
            if not chunk.name or not chunk.identifier:
                continue
            # One entry per identifier; the first (technique vs detection share
            # an id) is enough for name resolution.
            key = (chunk.identifier.upper(), chunk.type)
            if key in seen_ids:
                continue
            seen_ids.add(key)
            norm = normalize_name(chunk.name)
            if not norm:
                continue
            ident = chunk.identifier.upper()
            row = (ident, chunk.name, chunk.source, chunk.type)
            self._by_norm.setdefault(norm, []).append(row)
            self._entries.append((*row, norm, _tokens(norm)))
            self._meta.setdefault(ident, (chunk.name, chunk.source, chunk.type))
        # normalized alias phrase -> list of identifiers (e.g. CWE alt terms).
        self._aliases: dict[str, list[str]] = {}
        for phrase, ids in getattr(corpus, "aliases", {}).items():
            norm_phrase = normalize_name(phrase)
            if not norm_phrase:
                continue
            bucket = self._aliases.setdefault(norm_phrase, [])
            for ident in ids:
                up = ident.upper()
                if up in self._meta and up not in bucket:
                    bucket.append(up)

    def resolve(self, name: str, prefer_types: tuple[str, ...] | list[str] | None = None) -> ResolutionResult:
        prefer = tuple(prefer_types) if prefer_types is not None else DEFAULT_PREFER_TYPES
        query = name.strip()
        norm = normalize_name(query)
        if not norm:
            return ResolutionResult(query=query, candidates=[], **_NONE_RESULT_FIELDS)

        return self._result_from_candidates(query, self._collect_candidates(norm), prefer)

    def resolve_question(
        self, question: str, prefer_types: tuple[str, ...] | list[str] | None = None
    ) -> ResolutionResult:
        """Extract candidate entity spans from a question and resolve the best.

        Two passes keep this cheap: first probe every span with the O(1)
        exact-name/alias dictionaries (handles capitalized and lowercase
        phrasings); only if nothing matches exactly do we run the expensive
        substring/token-overlap tiers, and then only on the longest few spans.
        """
        prefer = tuple(prefer_types) if prefer_types is not None else DEFAULT_PREFER_TYPES
        spans = self._candidate_spans(question)

        # Pass A: exact name + alias dictionary lookups only.
        cheap: list[Candidate] = []
        for span in spans:
            norm = normalize_name(span)
            if norm:
                cheap.extend(self._lookup_exact_alias(norm))
        if cheap:
            return self._result_from_candidates(question.strip(), cheap, prefer)

        # Pass B: fuzzy fallback. Pool candidates from the most entity-like
        # spans (longest first) and rank once, so prefer_types is applied
        # globally rather than picked per span by a type-agnostic key.
        pool: list[Candidate] = []
        for span in sorted(set(spans), key=len, reverse=True)[:8]:
            norm = normalize_name(span)
            if norm:
                pool.extend(self._collect_candidates(norm))
        return self._result_from_candidates(question.strip(), pool, prefer)

    def _result_from_candidates(
        self, query: str, candidates: list[Candidate], prefer: tuple[str, ...]
    ) -> ResolutionResult:
        if not candidates:
            return ResolutionResult(query=query, candidates=[], **_NONE_RESULT_FIELDS)
        # Deduplicate by identifier, keeping the highest-scoring candidate.
        best_by_id: dict[str, Candidate] = {}
        for cand in candidates:
            existing = best_by_id.get(cand.identifier)
            if existing is None or cand.score > existing.score:
                best_by_id[cand.identifier] = cand
        ranked = sorted(best_by_id.values(), key=lambda c: self._rank_key(c, prefer), reverse=True)
        ranked = ranked[: self.max_candidates]
        top = ranked[0]
        margin, ambiguous = self._ambiguity(ranked)
        return ResolutionResult(
            query=query,
            chosen_id=top.identifier,
            score=top.score,
            source=top.source,
            type=top.type,
            name_matched=top.name,
            mode=top.mode,
            ambiguous=ambiguous,
            margin=margin,
            candidates=ranked,
        )

    def _lookup_exact_alias(self, norm: str) -> list[Candidate]:
        candidates: list[Candidate] = []
        for (i, n, s, t) in self._by_norm.get(norm, []):
            candidates.append(Candidate(i, n, s, t, 1.0, "exact"))
        for ident in self._aliases.get(norm, []):
            name, source, ctype = self._meta[ident]
            candidates.append(Candidate(ident, name, source, ctype, 0.9, "alias"))
        return candidates

    # -- internals -----------------------------------------------------------

    def _collect_candidates(self, norm: str) -> list[Candidate]:
        # Tier 1 + 1.5: exact normalized-name match and exact alias phrase.
        exact = self._lookup_exact_alias(norm)
        if exact:
            return exact

        query_tokens = _tokens(norm)
        candidates: list[Candidate] = []
        # Tier 2: substring containment.
        for (i, n, s, t, cand_norm, toks) in self._entries:
            if norm and (norm in cand_norm or cand_norm in norm):
                shorter, longer = sorted((len(norm), len(cand_norm)))
                score = (shorter / longer) if longer else 0.0
                candidates.append(Candidate(i, n, s, t, 0.5 + 0.49 * score, "alias"))
        if candidates:
            return candidates

        # Tier 3: token-overlap (Jaccard) fallback.
        if not query_tokens:
            return []
        for (i, n, s, t, cand_norm, toks) in self._entries:
            if not toks:
                continue
            inter = len(query_tokens & toks)
            if not inter:
                continue
            jaccard = inter / len(query_tokens | toks)
            if jaccard > 0.0:
                candidates.append(Candidate(i, n, s, t, 0.49 * jaccard, "fallback"))
        return candidates

    def _rank_key(self, candidate: Candidate, prefer: tuple[str, ...]) -> tuple[float, float, str]:
        # Higher is better: type-priority first, then score, then a stable id
        # tiebreak so ordering is deterministic.
        if candidate.type in prefer:
            type_rank = float(len(prefer) - prefer.index(candidate.type))
        else:
            type_rank = 0.0
        return (type_rank, candidate.score, _id_sort_value(candidate.identifier))

    def _ambiguity(self, candidates: list[Candidate]) -> tuple[float, bool]:
        if len(candidates) < 2:
            return 0.0, False
        top, second = candidates[0], candidates[1]
        margin = (top.score - second.score) / top.score if top.score else 0.0
        # Cross-type ties were already broken deterministically by type priority.
        same_type = top.type == second.type
        exact_vs_not = top.mode == "exact" and second.mode != "exact"
        ambiguous = same_type and (not exact_vs_not) and margin < AMBIGUITY_MARGIN
        return margin, ambiguous

    def _candidate_spans(self, question: str) -> list[str]:
        spans: list[str] = []
        seen: set[str] = set()

        def add(span: str) -> None:
            key = span.lower().strip()
            if span and key not in seen:
                spans.append(span)
                seen.add(key)

        # Fast path: capitalized/ALLCAPS spans (high precision).
        for match in _CAP_SPAN_RE.finditer(question):
            for span in _strip_stopwords(match.group(0)):
                add(span)

        # Lowercase-tolerant path: contiguous word n-grams over the question,
        # edge stopwords trimmed. Resolution stays cheap because resolve_question
        # probes these with O(1) exact/alias lookups before any fuzzy scan.
        words = [w for w in re.split(r"[\s,]+", question.strip()) if w]
        n = len(words)
        for length in range(min(self.MAX_SPAN_WORDS, n), 1, -1):
            for start in range(0, n - length + 1):
                trimmed = _trim_edge_stopwords(words[start : start + length])
                if len(trimmed) >= 2:
                    add(" ".join(trimmed))
        return spans


def _strip_stopwords(span: str) -> list[str]:
    """Return the span and its leading/trailing-stopword-trimmed variant."""
    words = span.split()
    variants = [span]
    trimmed = " ".join(_trim_edge_stopwords(words))
    if trimmed and trimmed != span:
        variants.append(trimmed)
    return variants


def _trim_edge_stopwords(words: list[str]) -> list[str]:
    start, end = 0, len(words)
    while start < end and words[start].lower() in _STOPWORDS:
        start += 1
    while end > start and words[end - 1].lower() in _STOPWORDS:
        end -= 1
    return words[start:end]


def _id_sort_value(identifier: str) -> str:
    # Stable, deterministic tiebreak; negate via reverse sort means smaller ids
    # win, which keeps parent techniques (T1059) ahead of sub-techniques only
    # incidentally -- ordering here is purely for reproducibility.
    return identifier
