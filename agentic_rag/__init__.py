"""Agentic RAG research scaffold for CTI tasks."""

from .agent import AgenticRAG, AgenticRAGConfig
from .corpus import CorpusIndex
from .orchestrator import GraniteOrchestrator, OrchestratorConfig
from .resolver import EntityResolver, ResolutionResult
from .retrieval import LexicalRetriever, build_search_backend
from .synthesizer import (
    DEFAULT_GRANITE_MODEL,
    GraniteGroundedSynthesizer,
    GraniteStructuredSynthesizer,
    OpenAIChatClient,
)

__all__ = [
    "AgenticRAG",
    "AgenticRAGConfig",
    "CorpusIndex",
    "EntityResolver",
    "GraniteOrchestrator",
    "OrchestratorConfig",
    "ResolutionResult",
    "DEFAULT_GRANITE_MODEL",
    "GraniteGroundedSynthesizer",
    "GraniteStructuredSynthesizer",
    "LexicalRetriever",
    "OpenAIChatClient",
    "build_search_backend",
]
