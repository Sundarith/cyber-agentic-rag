from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from .schema import Chunk, Evidence

ID_RE = re.compile(
    r"\b(T\d{4}(?:\.\d{3})?|M\d{4}|G\d{4}|S\d{4}|C\d{4}|DS\d{4}|"
    r"CAPEC-\d+|CWE-\d+|CVE-\d{4}-\d+)\b",
    re.IGNORECASE,
)
TOKEN_RE = re.compile(r"[a-zA-Z0-9_.-]+")

# Neighbor types that carry ATT&CK -> CAPEC -> CWE path information. When
# expanding a node, these must be returned ahead of high-degree but
# path-irrelevant neighbors (campaigns, malware, detections, groups) so a fixed
# expansion budget never drops the gold CAPEC/CWE. Lower rank = higher priority.
EXPAND_TYPE_PRIORITY = {
    "attack_pattern": 0,  # CAPEC: the ATT&CK -> CAPEC frontier
    "weakness": 1,        # CWE: the CAPEC -> CWE frontier
    "technique": 2,       # ATT&CK technique path node
}
_EXPAND_DEFAULT_PRIORITY = 9


def tokenize(text: str) -> list[str]:
    return [tok.lower() for tok in TOKEN_RE.findall(text)]


class CorpusIndex:
    """Small dependency-free lexical index over CTI JSONL chunks.

    The shipped CTI-RAG pipeline uses dense retrieval and cross-encoders. This
    scaffold intentionally starts with a transparent lexical index so the
    agentic controller can be tested without loading GPU models at import time.
    """

    def __init__(
        self,
        chunks: Iterable[Chunk],
        capec_to_cwe: dict[str, list[str]] | None = None,
        attack_to_capec: dict[str, list[str]] | None = None,
        aliases: dict[str, list[str]] | None = None,
    ) -> None:
        self.chunks = list(chunks)
        self.capec_to_cwe = {k.upper(): [v.upper() for v in vals] for k, vals in (capec_to_cwe or {}).items()}
        self.attack_to_capec = {k.upper(): [v.upper() for v in vals] for k, vals in (attack_to_capec or {}).items()}
        self.by_identifier: dict[str, Chunk] = {
            c.identifier.upper(): c for c in self.chunks if c.identifier
        }
        # Normalized alias phrase -> list of identifiers (e.g. CWE alternate
        # terms). Filtered to identifiers actually present in the corpus.
        self.aliases: dict[str, list[str]] = {}
        for phrase, ids in (aliases or {}).items():
            kept = [i.upper() for i in ids if i.upper() in self.by_identifier]
            if kept:
                self.aliases.setdefault(phrase, [])
                self.aliases[phrase].extend(i for i in kept if i not in self.aliases[phrase])
        self._doc_tokens: list[list[str]] = [tokenize(c.text) for c in self.chunks]
        self._doc_lens = [len(toks) for toks in self._doc_tokens]
        self._avgdl = sum(self._doc_lens) / len(self._doc_lens) if self._doc_lens else 0.0
        self._df: Counter[str] = Counter()
        self._tf: list[Counter[str]] = []
        for toks in self._doc_tokens:
            counts = Counter(toks)
            self._tf.append(counts)
            self._df.update(counts.keys())
        self.graph = self._build_graph()

    @classmethod
    def from_paths(
        cls,
        paths: Iterable[Path],
        capec_to_cwe: dict[str, list[str]] | None = None,
        attack_to_capec: dict[str, list[str]] | None = None,
        aliases: dict[str, list[str]] | None = None,
    ) -> "CorpusIndex":
        chunks: list[Chunk] = []
        for path in paths:
            if not path.exists():
                continue
            with path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        chunks.append(Chunk.from_json(json.loads(line)))
        return cls(chunks, capec_to_cwe=capec_to_cwe, attack_to_capec=attack_to_capec, aliases=aliases)

    @classmethod
    def default(cls, root: Path = Path(".")) -> "CorpusIndex":
        processed = root / "data" / "processed"
        capec_to_cwe = _load_relation_map(processed / "capec_cwe_relations.json", "capec_to_cwe")
        attack_to_capec = _load_relation_map(processed / "capec_attack_relations.json", "tech_to_capec")
        aliases = _load_cwe_aliases(processed / "cwe_phrase_index.json")
        return cls.from_paths(
            [
                processed / "attack_chunks.jsonl",
                processed / "capec_chunks.jsonl",
                processed / "cwe_chunks.jsonl",
                processed / "cve_chunks.jsonl",
            ],
            capec_to_cwe=capec_to_cwe,
            attack_to_capec=attack_to_capec,
            aliases=aliases,
        )

    def _build_graph(self) -> dict[str, set[str]]:
        graph: dict[str, set[str]] = defaultdict(set)
        known = set(self.by_identifier)
        for chunk in self.chunks:
            src = chunk.identifier.upper()
            if not src:
                continue
            for raw in ID_RE.findall(chunk.text):
                dst = raw.upper()
                if dst != src and dst in known:
                    graph[src].add(dst)
                    graph[dst].add(src)
            for cwe_id in chunk.metadata.get("cwe_ids", []) or []:
                dst = str(cwe_id).upper()
                if dst in known:
                    graph[src].add(dst)
                    graph[dst].add(src)
            for parent_id in chunk.metadata.get("parent_cwe_ids", []) or []:
                dst = str(parent_id).upper()
                if dst in known:
                    graph[src].add(dst)
                    graph[dst].add(src)
        for capec_id, cwe_ids in self.capec_to_cwe.items():
            if capec_id not in known:
                continue
            for cwe_id in cwe_ids:
                if cwe_id in known:
                    graph[capec_id].add(cwe_id)
                    graph[cwe_id].add(capec_id)
        for attack_id, capec_ids in self.attack_to_capec.items():
            if attack_id not in known:
                continue
            for capec_id in capec_ids:
                if capec_id in known:
                    graph[attack_id].add(capec_id)
                    graph[capec_id].add(attack_id)
        return graph

    def open(self, identifier: str) -> Evidence | None:
        chunk = self.by_identifier.get(identifier.upper())
        if not chunk:
            return None
        return Evidence(chunk=chunk, score=1.0, tool="open", reason="exact identifier")

    def search(self, query: str, k: int = 8, source_filter: str | None = None) -> list[Evidence]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        scores: list[tuple[int, float]] = []
        n_docs = max(1, len(self.chunks))
        k1 = 1.5
        b = 0.75
        for idx, chunk in enumerate(self.chunks):
            if source_filter and chunk.source.lower() != source_filter.lower():
                continue
            score = 0.0
            counts = self._tf[idx]
            dl = self._doc_lens[idx] or 1
            for token in set(query_tokens):
                tf = counts.get(token, 0)
                if not tf:
                    continue
                df = self._df[token]
                idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
                denom = tf + k1 * (1.0 - b + b * dl / (self._avgdl or 1.0))
                score += idf * (tf * (k1 + 1.0)) / denom
            if chunk.identifier and chunk.identifier.lower() in query.lower():
                score += 10.0
            if chunk.name and chunk.name.lower() in query.lower():
                score += 3.0
            if score > 0:
                scores.append((idx, score))
        scores.sort(key=lambda row: row[1], reverse=True)
        return [
            Evidence(
                chunk=self.chunks[idx],
                score=score,
                tool="search",
                reason=query,
                snippet=make_snippet(self.chunks[idx].text, query_tokens),
            )
            for idx, score in scores[:k]
        ]

    def expand(self, identifier: str, k: int = 6) -> list[Evidence]:
        src = identifier.upper()
        # Order neighbors by path relevance first, then alphabetically for
        # determinism, before applying the budget. This prevents high-degree
        # campaign/malware/detection neighbors from crowding the gold CAPEC/CWE
        # out of the top-k window.
        neighbors = sorted(self.graph.get(src, set()), key=self._expand_sort_key)
        evidence = []
        for dst in neighbors[:k]:
            chunk = self.by_identifier.get(dst)
            if chunk:
                evidence.append(
                    Evidence(chunk=chunk, score=1.0, tool="expand", reason=f"graph neighbor of {src}")
                )
        return evidence

    def _expand_sort_key(self, dst: str) -> tuple[int, str]:
        chunk = self.by_identifier.get(dst)
        priority = EXPAND_TYPE_PRIORITY.get(chunk.type, _EXPAND_DEFAULT_PRIORITY) if chunk else _EXPAND_DEFAULT_PRIORITY
        return (priority, dst)

    def find(self, pattern: str, k: int = 8) -> list[Evidence]:
        needle = pattern.lower()
        results = []
        for chunk in self.chunks:
            offset = chunk.text.lower().find(needle)
            if offset == -1:
                continue
            snippet = chunk.text[max(0, offset - 160): offset + len(pattern) + 220]
            results.append(
                Evidence(chunk=chunk, score=1.0, tool="find", reason=pattern, snippet=snippet.strip())
            )
            if len(results) >= k:
                break
        return results


def extract_ids(text: str) -> list[str]:
    seen: set[str] = set()
    ids = []
    for match in ID_RE.findall(text):
        ident = match.upper()
        if ident not in seen:
            ids.append(ident)
            seen.add(ident)
    return ids


def _load_cwe_aliases(path: Path) -> dict[str, list[str]]:
    """Load CWE alternate-term phrases as normalized alias -> [CWE id].

    Maps phrases such as "information disclosure" -> CWE-200 so descriptive
    natural questions resolve via the resolver's alias tier. The phrase index
    shape is {cwe_id: {"phrases": [{"phrase": str, "sources": [...]}]}}.
    """
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    aliases: dict[str, list[str]] = {}
    for cwe_id, entry in raw.items():
        if not isinstance(entry, dict):
            continue
        for item in entry.get("phrases", []) or []:
            phrase = str(item.get("phrase") or "").strip().lower()
            phrase = re.sub(r"\s+", " ", phrase)
            if not phrase:
                continue
            aliases.setdefault(phrase, [])
            if cwe_id.upper() not in aliases[phrase]:
                aliases[phrase].append(cwe_id.upper())
    return aliases


def _load_relation_map(path: Path, key: str) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    rows = raw.get(key, {})
    if not isinstance(rows, dict):
        return {}
    return {str(src): [str(dst) for dst in dsts] for src, dsts in rows.items()}


def make_snippet(text: str, query_tokens: list[str], max_chars: int = 420) -> str:
    lowered = text.lower()
    first_hit = min((lowered.find(tok) for tok in query_tokens if lowered.find(tok) >= 0), default=0)
    start = max(0, first_hit - 120)
    snippet = text[start:start + max_chars].strip()
    return re.sub(r"\s+", " ", snippet)
