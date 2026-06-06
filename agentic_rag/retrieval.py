from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import ModuleType
from typing import Protocol

from .corpus import CorpusIndex, make_snippet, tokenize
from .schema import Chunk, Evidence


class SearchBackend(Protocol):
    """Search-only retrieval interface used by the agent controller."""

    name: str

    def search(self, query: str, k: int = 8) -> list[Evidence]:
        ...


@dataclass
class LexicalRetriever:
    """Dependency-light lexical backend over the local CorpusIndex."""

    corpus: CorpusIndex
    name: str = "lexical"

    def search(self, query: str, k: int = 8) -> list[Evidence]:
        return self.corpus.search(query, k=k)


class LegacyHybridRetriever:
    """Lazy adapter for the legacy BM25+dense retrieval stack.

    Importing `better_rag.py` builds indexes and loads embedding models at module
    import time, so this adapter deliberately imports it only on the first
    hybrid search call. The new agentic package can therefore remain lightweight
    unless a run explicitly selects hybrid retrieval.
    """

    name = "legacy_hybrid"

    def __init__(self, corpus: CorpusIndex, module_name: str = "better_rag") -> None:
        self.corpus = corpus
        self.module_name = module_name
        self._module: ModuleType | None = None

    def search(self, query: str, k: int = 8) -> list[Evidence]:
        module = self._load_module()
        retrieve = getattr(module, "retrieve", None)
        if retrieve is None:
            raise RuntimeError(f"{self.module_name} does not expose retrieve(query, k)")
        rows = retrieve(query, k=k)
        return [self._to_evidence(row, query) for row in rows]

    def _load_module(self) -> ModuleType:
        if self._module is None:
            try:
                self._module = importlib.import_module(self.module_name)
            except ImportError as exc:
                raise RuntimeError(
                    f"Legacy hybrid retrieval requires importing {self.module_name}; "
                    "run in the legacy CTI-RAG environment with dense retrieval dependencies installed."
                ) from exc
        return self._module

    def _to_evidence(self, row: object, query: str) -> Evidence:
        if isinstance(row, tuple) and len(row) >= 2:
            raw_chunk, score = row[0], row[1]
        else:
            raw_chunk, score = row, 0.0
        if not isinstance(raw_chunk, dict):
            raise TypeError(f"Expected legacy retriever chunk dict, got {type(raw_chunk).__name__}")
        chunk = Chunk.from_json(raw_chunk)
        return Evidence(
            chunk=chunk,
            score=float(score),
            tool="hybrid_search",
            reason=query,
            snippet=make_snippet(chunk.text, tokenize(query)),
        )


def build_search_backend(corpus: CorpusIndex, backend: str) -> SearchBackend:
    if backend == "lexical":
        return LexicalRetriever(corpus)
    if backend == "legacy_hybrid":
        return LegacyHybridRetriever(corpus)
    raise ValueError(f"Unknown retrieval backend: {backend}")
