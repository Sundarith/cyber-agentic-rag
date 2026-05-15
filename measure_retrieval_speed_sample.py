"""
Measure before/after retrieval latency on the same CTI-RCM prompt sample.

This is retrieval-only: no graph expansion and no LLM call. The "before" path
recreates the old Python-loop retrieval style so we can compare it against the
current vectorized retrieve() implementation on the same prompts.

Usage:
    conda run -n cyber-ft python3 -u measure_retrieval_speed_sample.py 10
"""
import csv
import json
import math
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import difflib
import numpy as np

import better_rag

RCM_PATH = Path("data/cti-bench/data/cti-rcm.tsv")
OUT_PATH = Path("presentation/retrieval_speed_sample.json")
SEED = 42


def extract_cve_id(url: str) -> str:
    m = re.search(r"CVE-\d{4}-\d+", url, re.IGNORECASE)
    return m.group(0).upper() if m else ""


def legacy_bm25_scores(query_tokens: list[str]) -> np.ndarray:
    scores = np.zeros(better_rag._N)
    for t in set(query_tokens):
        postings = better_rag._inv_idx.get(t)
        if not postings:
            continue
        df = len(postings)
        idf = math.log((better_rag._N - df + 0.5) / (df + 0.5) + 1)
        for idx, tf in postings.items():
            dl = better_rag._doc_len[idx]
            scores[idx] += idf * (tf * (better_rag.BM25_K1 + 1)) / (
                tf
                + better_rag.BM25_K1
                * (1 - better_rag.BM25_B + better_rag.BM25_B * dl / better_rag._avgdl)
            )
    return scores


def legacy_retrieve(query: str, k: int = better_rag.TOP_K) -> list:
    q_emb = better_rag._get_embedder().encode([query], normalize_embeddings=True)[0]
    emb_sc = better_rag.chunk_embs @ q_emb

    query_words = better_rag._tokenize(query)
    bm25_sc = legacy_bm25_scores(query_words)
    if bm25_sc.max() > 0:
        bm25_sc = bm25_sc / bm25_sc.max()

    combined = better_rag.HYBRID_ALPHA * emb_sc + (1 - better_rag.HYBRID_ALPHA) * bm25_sc

    if better_rag.DETECTION_RE.search(query):
        for i, c in enumerate(better_rag.chunks):
            if c.get("type") == "detection":
                combined[i] += 2.0

    if len(query_words) <= 15:
        for word in query_words:
            if len(word) < 5:
                continue
            names = [tn[0] for tn in better_rag._tech_name_index]
            matches = difflib.get_close_matches(word, names, n=1, cutoff=0.85)
            if matches:
                best_match = matches[0]
                for name, idx in better_rag._tech_name_index:
                    if name == best_match:
                        combined[idx] += 3.0
                        break

    id_matches = re.findall(
        r"\b(?:[TM]\d{4}(?:\.\d{3})?|CAPEC-\d+|CWE-\d+|CVE-\d{4}-\d+)\b",
        query,
        re.IGNORECASE,
    )
    for match in id_matches:
        for i, c in enumerate(better_rag.chunks):
            if c.get("identifier", "").upper() == match.upper():
                combined[i] += 2.0
                break

    if not id_matches:
        q_lower = query.lower()
        for name, idx in better_rag._tech_name_index:
            if re.search(rf"\b{re.escape(name)}\b", q_lower):
                combined[idx] = combined.max() + 0.1
                break

    found_entities = better_rag._find_all_entities_in_query(query)
    for ent_name, ent_type in found_entities:
        idx_dict = {
            "group": better_rag.group_to_techs,
            "campaign": better_rag.campaign_to_techs,
            "malware": better_rag.malware_to_techs,
            "tool": better_rag.tool_to_techs,
        }[ent_type]
        related_chunks = idx_dict.get(ent_name, [])

        for i, c in enumerate(better_rag.chunks):
            if c.get("name", "").lower() == ent_name and c.get("type") in (
                "group",
                "malware",
                "tool",
                "campaign",
            ):
                combined[i] += 2.0
                break

        for rc in related_chunks:
            for i, c in enumerate(better_rag.chunks):
                if c["identifier"] == rc["identifier"]:
                    boost = 1.5
                    if c["identifier"] in better_rag.attack_to_capec:
                        boost += 1.0
                    combined[i] += boost
                    break

        if ent_type == "group":
            for m_name in better_rag.group_to_malware.get(ent_name, []):
                for i, c in enumerate(better_rag.chunks):
                    if c["name"].lower() == m_name.lower():
                        combined[i] += 1.0
                        break

    if better_rag.ROOT_CAUSE_RE.search(query):
        for i, c in enumerate(better_rag.chunks):
            if c.get("source") == "CWE" or c.get("type") in ("weakness", "category", "view"):
                combined[i] += 0.3

    for i, c in enumerate(better_rag.chunks):
        if not c.get("identifier"):
            combined[i] -= 1.0

    if not re.search(r"\bCVE-\d{4}-\d+\b", query, re.IGNORECASE):
        for i, c in enumerate(better_rag.chunks):
            if c.get("source") == "CVE":
                combined[i] -= 0.5
    else:
        for i, c in enumerate(better_rag.chunks):
            if c.get("source") == "CWE":
                combined[i] -= 1.5

    top_idx = np.argsort(-combined)[:k]
    return [(better_rag.chunks[i], float(combined[i])) for i in top_idx]


def timed(callable_obj, *args):
    start = time.perf_counter()
    result = callable_obj(*args)
    return time.perf_counter() - start, result


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    random.seed(SEED)

    with RCM_PATH.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))

    sample = random.sample(rows, min(n, len(rows)))
    results = []

    # Warm the embedder once so the first measured row is not dominated by CUDA setup.
    better_rag._get_embedder().encode([sample[0]["Prompt"]], normalize_embeddings=True)

    for i, row in enumerate(sample, 1):
        prompt = row["Prompt"]
        before_s, _ = timed(legacy_retrieve, prompt)
        after_s, _ = timed(better_rag.retrieve, prompt)
        result = {
            "index": i,
            "cve_id": extract_cve_id(row["URL"]),
            "before_retrieve_s": before_s,
            "after_retrieve_s": after_s,
            "speedup": before_s / after_s if after_s else 0.0,
        }
        results.append(result)
        print(
            f"[{i}/{len(sample)}] {result['cve_id']} "
            f"before={before_s:.2f}s after={after_s:.2f}s "
            f"speedup={result['speedup']:.1f}x"
        )

    before_values = [r["before_retrieve_s"] for r in results]
    after_values = [r["after_retrieve_s"] for r in results]
    avg_before = sum(before_values) / len(before_values) if before_values else 0.0
    avg_after = sum(after_values) / len(after_values) if after_values else 0.0
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "sample_size": len(results),
        "seed": SEED,
        "summary": {
            "avg_before_retrieve_s": avg_before,
            "avg_after_retrieve_s": avg_after,
            "avg_speedup": avg_before / avg_after if avg_after else 0.0,
        },
        "results": results,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
