"""
Cross-encoder re-ranking of CWE chunks for the CTI-RCM unmapped pipeline.

Loads BAAI/bge-reranker-base lazily (single global instance), scores
(query, CWE chunk text) pairs jointly, and returns the chunks sorted by score.

Env gates (handled here for sub-flags; the master on/off is handled in
better_rag.py via CTI_RAG_CWE_CROSSENCODER):
  CTI_RAG_CWE_CROSSENCODER_MODEL      default: BAAI/bge-reranker-base
  CTI_RAG_CWE_CROSSENCODER_MAX_LENGTH default: 512
  CTI_RAG_CWE_CROSSENCODER_TEXT_LIMIT default: 500
"""
import os
import threading

CROSS_ENCODER_MODEL = os.environ.get("CTI_RAG_CWE_CROSSENCODER_MODEL", "BAAI/bge-reranker-base")
CROSS_ENCODER_MAX_LENGTH = int(os.environ.get("CTI_RAG_CWE_CROSSENCODER_MAX_LENGTH", "512"))
CROSS_ENCODER_TEXT_LIMIT = int(os.environ.get("CTI_RAG_CWE_CROSSENCODER_TEXT_LIMIT", "500"))

_cross_encoder = None
_init_lock = threading.Lock()


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is not None:
        return _cross_encoder
    with _init_lock:
        if _cross_encoder is None:
            from sentence_transformers import CrossEncoder
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _cross_encoder = CrossEncoder(
                CROSS_ENCODER_MODEL,
                max_length=CROSS_ENCODER_MAX_LENGTH,
                device=device,
            )
    return _cross_encoder


def _cwe_text_for_scoring(chunk: dict) -> str:
    text = chunk.get("text") or ""
    return text[:CROSS_ENCODER_TEXT_LIMIT]


def score_cwe_chunks(question: str, cwe_chunks: list[dict]) -> list[tuple[dict, float]]:
    """Score each CWE chunk against the question. Returns (chunk, score) sorted descending."""
    if not cwe_chunks:
        return []
    pairs = [(question, _cwe_text_for_scoring(c)) for c in cwe_chunks]
    enc = _get_cross_encoder()
    scores = enc.predict(pairs)
    paired = [(chunk, float(score)) for chunk, score in zip(cwe_chunks, scores)]
    paired.sort(key=lambda kv: -kv[1])
    return paired
