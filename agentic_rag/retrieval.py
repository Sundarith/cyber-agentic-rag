from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .corpus import CorpusIndex
from .schema import Evidence


class SearchBackend(Protocol):
    """Search-only retrieval interface used by the agent controller."""

    name: str

    def search(self, query: str, k: int = 8) -> list[Evidence]:
        ...


@dataclass
class LexicalRetriever:
    """Dependency-light lexical (BM25-style) backend over the local CorpusIndex."""

    corpus: CorpusIndex
    name: str = "lexical"

    def search(self, query: str, k: int = 8) -> list[Evidence]:
        return self.corpus.search(query, k=k)


def build_search_backend(corpus: CorpusIndex, backend: str = "lexical") -> SearchBackend:
    if backend == "lexical":
        return LexicalRetriever(corpus)
    raise ValueError(f"Unknown retrieval backend: {backend}")
