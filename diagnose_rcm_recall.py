"""
Diagnose CTI-RCM retrieval recall without LLM generation.

For each sampled CTI-RCM row, this records whether the ground-truth CWE appears
in the main retrieval path, a CWE-only retrieval pool, k-NN votes, graph
neighbors, and the final context sent to the LLM.
"""
from __future__ import annotations

import csv
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import better_rag


RCM_PATH = Path("data/cti-bench/data/cti-rcm.tsv")
CVE_CWE_INDEX = Path("data/processed/cve_cwe_index.json")
OUT_JSONL = Path("analysis/rcm_recall_diagnostics.jsonl")
OUT_MD = Path("analysis/rcm_recall_diagnostics.md")
SEED = 42
CWE_ONLY_K = 50


def extract_cve_id(url: str) -> str:
    match = re.search(r"CVE-\d{4}-\d+", url, re.IGNORECASE)
    return match.group(0).upper() if match else ""


def load_cve_cwe_index() -> dict:
    if CVE_CWE_INDEX.exists():
        return json.loads(CVE_CWE_INDEX.read_text())
    return {}


def load_rows(n: int, nvd_mapped: bool, nvd_unmapped: bool) -> list[dict]:
    rows = []
    with RCM_PATH.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))

    cve_cwe_index = load_cve_cwe_index()
    if nvd_mapped:
        rows = [row for row in rows if bool(cve_cwe_index.get(extract_cve_id(row["URL"])))]
    if nvd_unmapped:
        rows = [row for row in rows if not bool(cve_cwe_index.get(extract_cve_id(row["URL"])))]

    random.seed(SEED)
    return random.sample(rows, min(n, len(rows)))


def cwe_ids_from_debug_rows(rows: list[dict]) -> set[str]:
    ids = set()
    for row in rows or []:
        ident = (row.get("identifier") or "").upper()
        if ident.startswith("CWE-"):
            ids.add(ident)
        for cwe_id in row.get("cwe_ids") or []:
            ids.add(cwe_id.upper())
    return ids


def cwe_only_retrieve(query: str, k: int = CWE_ONLY_K) -> list[dict]:
    return [
        {
            "identifier": candidate["identifier"],
            "name": candidate["name"],
            "score": candidate["score"],
        }
        for candidate in better_rag._cwe_only_candidates(query, k)
    ]


def summarize(rows: list[dict]) -> str:
    total = len(rows)
    counts = Counter()
    for row in rows:
        for key in (
            "gold_in_initial",
            "gold_in_cwe_only_top50",
            "gold_in_knn_votes",
            "gold_is_knn_top",
            "gold_in_graph_neighbors",
            "gold_in_final_context",
            "gold_in_cwe_rescue",
        ):
            counts[key] += int(bool(row[key]))

    def pct(key: str) -> str:
        value = counts[key]
        return f"{value}/{total} ({value / total * 100:.1f}%)" if total else "0/0"

    missing_final = [row for row in rows if not row["gold_in_final_context"]]
    present_final = [row for row in rows if row["gold_in_final_context"]]
    lines = [
        "# CTI-RCM Retrieval Recall Diagnostics",
        "",
        f"Rows analyzed: **{total}**",
        "",
        "## Recall By Stage",
        "",
        f"- Initial retrieval contains GT CWE: {pct('gold_in_initial')}",
        f"- CWE-only top 50 contains GT CWE: {pct('gold_in_cwe_only_top50')}",
        f"- k-NN votes contain GT CWE: {pct('gold_in_knn_votes')}",
        f"- k-NN top vote is GT CWE: {pct('gold_is_knn_top')}",
        f"- Graph neighbors contain GT CWE: {pct('gold_in_graph_neighbors')}",
        f"- Final context contains GT CWE: {pct('gold_in_final_context')}",
    ]
    if any(row.get("cwe_rescue_selected") for row in rows):
        lines.append(f"- CWE rescue selected GT CWE: {pct('gold_in_cwe_rescue')}")
        lines.append(
            f"- CWE rescue added candidates: {sum(len(row.get('cwe_rescue_added') or []) for row in rows)}"
        )
    lines += [
        "",
        "## Final Context Split",
        "",
        f"- GT present in final context: {len(present_final)}/{total}",
        f"- GT missing from final context: {len(missing_final)}/{total}",
        "",
        "## Examples Missing From Final Context",
        "",
    ]
    for row in missing_final[:12]:
        lines.append(
            f"- {row['cve_id']} GT={row['gt_cwe']} | "
            f"initial={row['gold_in_initial']} cwe_top50={row['gold_in_cwe_only_top50']} "
            f"knn={row['knn_top_cwe'] or 'none'} final={row['final_context_cwes'][:5]}"
        )
    return "\n".join(lines) + "\n"


def diagnose_row(row: dict) -> dict:
    cve_id = extract_cve_id(row["URL"])
    gt_cwe = row["GT"].strip().upper()
    query = row["Prompt"]

    debug_info: dict = {}
    better_rag.ask(query, [], eval_mode=True, debug_info=debug_info)

    initial_cwes = cwe_ids_from_debug_rows(debug_info.get("initial_retrieved") or [])
    graph_cwes = cwe_ids_from_debug_rows(debug_info.get("graph_neighbors") or [])
    final_cwes = {cwe.upper() for cwe in debug_info.get("final_context_cwes") or []}

    cwe_only = cwe_only_retrieve(query)
    cwe_only_ids = {row["identifier"] for row in cwe_only}

    knn = debug_info.get("knn_cwe") or {}
    knn_weights = {key.upper(): value for key, value in (knn.get("weights") or {}).items()}
    knn_neighbor_cwes = set()
    for neighbor in knn.get("neighbors") or []:
        for cwe_id in neighbor.get("cwe_ids") or []:
            knn_neighbor_cwes.add(cwe_id.upper())

    rescue = debug_info.get("cwe_rescue") or {}
    rescue_selected = rescue.get("selected") or []
    rescue_added = rescue.get("added") or []
    rescue_selected_ids = {item.get("identifier", "").upper() for item in rescue_selected}

    return {
        "cve_id": cve_id,
        "gt_cwe": gt_cwe,
        "gold_in_initial": gt_cwe in initial_cwes,
        "gold_in_cwe_only_top50": gt_cwe in cwe_only_ids,
        "gold_in_knn_votes": gt_cwe in knn_weights or gt_cwe in knn_neighbor_cwes,
        "gold_is_knn_top": (knn.get("top_cwe") or "").upper() == gt_cwe,
        "gold_in_graph_neighbors": gt_cwe in graph_cwes,
        "gold_in_final_context": gt_cwe in final_cwes,
        "gold_in_cwe_rescue": gt_cwe in rescue_selected_ids,
        "initial_cwes": sorted(initial_cwes),
        "cwe_only_top50": cwe_only[:50],
        "knn_top_cwe": (knn.get("top_cwe") or "").upper(),
        "knn_mode": knn.get("mode", ""),
        "knn_top_share": knn.get("top_share"),
        "knn_weights": knn_weights,
        "cwe_rescue_selected": rescue_selected,
        "cwe_rescue_added": rescue_added,
        "graph_cwes": sorted(graph_cwes),
        "final_context_cwes": sorted(final_cwes),
    }


def main() -> int:
    args = sys.argv[1:]
    nvd_mapped = "--nvd-mapped" in args
    nvd_unmapped = "--nvd-unmapped" in args
    args = [arg for arg in args if arg not in ("--nvd-mapped", "--nvd-unmapped")]
    n = int(args[0]) if args else 97

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    # Keep diagnostics independent of LLM generation and vLLM availability.
    better_rag._llm = lambda prompt: ""

    rows = load_rows(n, nvd_mapped=nvd_mapped, nvd_unmapped=nvd_unmapped)
    results = []
    for idx, row in enumerate(rows, start=1):
        result = diagnose_row(row)
        results.append(result)
        print(
            f"[{idx}/{len(rows)}] {result['cve_id']} GT={result['gt_cwe']} "
            f"initial={result['gold_in_initial']} cwe50={result['gold_in_cwe_only_top50']} "
            f"knn_top={result['knn_top_cwe'] or 'none'} final={result['gold_in_final_context']}",
            flush=True,
        )

    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    OUT_MD.write_text(summarize(results), encoding="utf-8")

    print(f"\nWrote {OUT_JSONL}")
    print(f"Wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
