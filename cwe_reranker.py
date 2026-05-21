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
import re
import threading

CROSS_ENCODER_MODEL = os.environ.get("CTI_RAG_CWE_CROSSENCODER_MODEL", "BAAI/bge-reranker-base")
CROSS_ENCODER_MAX_LENGTH = int(os.environ.get("CTI_RAG_CWE_CROSSENCODER_MAX_LENGTH", "512"))
CROSS_ENCODER_TEXT_LIMIT = int(os.environ.get("CTI_RAG_CWE_CROSSENCODER_TEXT_LIMIT", "500"))
CROSS_ENCODER_DEVICE = os.environ.get("CTI_RAG_CWE_CROSSENCODER_DEVICE", "")
# Optional LoRA adapter applied on top of the base CE. If set, must point to a
# directory containing adapter_model.safetensors + adapter_config.json (the
# output of train_cwe_reranker.py).
CROSS_ENCODER_ADAPTER = os.environ.get("CTI_RAG_CWE_CROSSENCODER_ADAPTER", "")

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
            device = CROSS_ENCODER_DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")
            _cross_encoder = CrossEncoder(
                CROSS_ENCODER_MODEL,
                max_length=CROSS_ENCODER_MAX_LENGTH,
                device=device,
            )
            if CROSS_ENCODER_ADAPTER:
                print(f"[cwe_reranker] loading LoRA adapter from {CROSS_ENCODER_ADAPTER}")
                _cross_encoder.load_adapter(CROSS_ENCODER_ADAPTER)
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


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", (text or "").lower()))


def _differential_text(chunk: dict, sibling_chunks: list[dict]) -> str:
    """Return chunk text with words shared with any sibling candidate removed.

    Subtraction is among siblings (CWEs sharing a parent within the candidate
    set), not against the parent itself. Parent descriptions in MITRE CWE are
    often unions of children (`reads from or writes to`), so subtracting the
    parent would destroy the very read-vs-write signal we want to emphasize.
    Subtracting sibling-shared tokens leaves the uniquely-this-child vocabulary
    intact (e.g. `read` survives for CWE-125 because its only candidate sibling
    CWE-787 says `write` instead).

    Preserves the chunk's structural skeleton (the header line) so the
    cross-encoder still sees this is "CWE-XXX — Name". Punctuation and
    whitespace are preserved word-by-word.
    """
    text = chunk.get("text") or ""
    if not text or not sibling_chunks:
        return text
    sibling_vocab: set[str] = set()
    for s in sibling_chunks:
        sibling_vocab |= _tokens(s.get("text") or "")
        sibling_vocab |= _tokens(s.get("name") or "")
    if not sibling_vocab:
        return text
    lines = text.split("\n", 1)
    header = lines[0]
    body = lines[1] if len(lines) > 1 else ""
    out_parts: list[str] = []
    for piece in re.split(r"(\W+)", body):
        if re.fullmatch(r"\w+", piece):
            if piece.lower() not in sibling_vocab:
                out_parts.append(piece)
        else:
            out_parts.append(piece)
    body_diff = "".join(out_parts)
    return f"{header}\n{body_diff}" if body else header


def score_cwe_chunks_hierarchical(
    question: str,
    cwe_chunks: list[dict],
    id_to_chunk: dict,
    alpha: float = 0.5,
) -> list[tuple[dict, float, float, float]]:
    """Hierarchy-aware scoring: blend vanilla cross-encoder score with a
    sibling-aware differential score that emphasizes the vocabulary that
    distinguishes a CWE from its siblings within the candidate set.

    Returns list of (chunk, final_score, vanilla_score, diff_score) sorted by
    final_score descending. Candidates with no sibling among the input
    candidates fall through to vanilla scoring (diff_score == vanilla_score).
    """
    if not cwe_chunks:
        return []
    parent_sets: list[set[str]] = [
        {p.upper() for p in (c.get("parent_cwe_ids") or [])}
        for c in cwe_chunks
    ]
    sibling_groups: list[list[dict]] = []
    for i, parents_i in enumerate(parent_sets):
        if not parents_i:
            sibling_groups.append([])
            continue
        siblings = [
            cwe_chunks[j]
            for j, parents_j in enumerate(parent_sets)
            if j != i and parents_i & parents_j
        ]
        sibling_groups.append(siblings)

    vanilla_pairs = [(question, _cwe_text_for_scoring(c)) for c in cwe_chunks]
    diff_pairs: list[tuple[str, str]] = []
    has_sib_flags: list[bool] = []
    for c, siblings in zip(cwe_chunks, sibling_groups):
        if siblings:
            diff_text = _differential_text(c, siblings)
            diff_pairs.append((question, diff_text[:CROSS_ENCODER_TEXT_LIMIT]))
            has_sib_flags.append(True)
        else:
            diff_pairs.append((question, _cwe_text_for_scoring(c)))
            has_sib_flags.append(False)

    enc = _get_cross_encoder()
    vanilla_scores = [float(s) for s in enc.predict(vanilla_pairs)]
    diff_scores = [float(s) for s in enc.predict(diff_pairs)]
    out: list[tuple[dict, float, float, float]] = []
    for c, v, d, has_sib in zip(cwe_chunks, vanilla_scores, diff_scores, has_sib_flags):
        final = alpha * v + (1.0 - alpha) * d if has_sib else v
        out.append((c, final, v, d))
    out.sort(key=lambda x: -x[1])
    return out
