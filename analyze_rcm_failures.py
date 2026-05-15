"""
Analyze CTI-RCM failure debug records.

Input comes from:
    conda run -n cyber-ft python3 eval_rcm.py 1000 --debug-failures

The analyzer does not call the model. It reads the failure JSONL, checks whether
the gold CWE was present in retrieved/context/k-NN evidence, classifies likely
failure modes, and writes a concise report for retrieval tuning.
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

FAILURES_PATH = Path("eval_failures_debug.jsonl")
CWE_CHUNKS_PATH = Path("data/processed/cwe_chunks.jsonl")
OUT_JSON = Path("analysis/rcm_failure_analysis.json")
OUT_MD = Path("analysis/rcm_failure_analysis.md")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"Missing input file: {path}")
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_cwe_meta() -> dict[str, dict]:
    meta = {}
    for line in CWE_CHUNKS_PATH.open(encoding="utf-8"):
        row = json.loads(line)
        cwe_id = row.get("identifier", "").upper()
        if cwe_id:
            meta[cwe_id] = {
                "name": row.get("name", ""),
                "abstraction": row.get("abstraction", ""),
                "parents": [p.upper() for p in row.get("parent_cwe_ids", [])],
            }
    return meta


def build_children(meta: dict[str, dict]) -> dict[str, set[str]]:
    children = defaultdict(set)
    for cwe_id, row in meta.items():
        for parent in row["parents"]:
            children[parent].add(cwe_id)
    return children


def cwe_ids_from_answer(answer: str) -> list[str]:
    seen = []
    for match in re.findall(r"\bCWE-\d+\b", answer or "", re.IGNORECASE):
        cwe_id = match.upper()
        if cwe_id not in seen:
            seen.append(cwe_id)
    return seen


def cwe_ids(rows: list[dict]) -> set[str]:
    return {
        row.get("identifier", "").upper()
        for row in rows
        if row.get("identifier", "").upper().startswith("CWE-")
    }


def relation(pred: str, gold: str, meta: dict[str, dict], children: dict[str, set[str]]) -> str:
    if not pred or pred == gold:
        return "exact" if pred == gold else "none"
    pred_parents = set(meta.get(pred, {}).get("parents", []))
    gold_parents = set(meta.get(gold, {}).get("parents", []))
    if gold in pred_parents:
        return "predicted_child_of_gold"
    if pred in gold_parents:
        return "predicted_parent_of_gold"
    if pred_parents & gold_parents:
        return "sibling_same_parent"
    if pred in children.get(gold, set()):
        return "predicted_child_of_gold"
    if gold in children.get(pred, set()):
        return "predicted_parent_of_gold"
    return "unrelated_or_distant"


def classify(row: dict, meta: dict[str, dict], children: dict[str, set[str]]) -> dict:
    gold = row.get("gt_cwe", "").upper()
    debug = row.get("debug") or {}
    answer_cwes = cwe_ids_from_answer(row.get("answer", ""))
    predicted = answer_cwes[-1] if answer_cwes else ""

    initial_cwes = cwe_ids(debug.get("initial_retrieved") or [])
    graph_cwes = cwe_ids(debug.get("graph_neighbors") or [])
    final_cwes = set(c.upper() for c in debug.get("final_context_cwes") or [])

    knn = debug.get("knn_cwe") or {}
    knn_weights = {k.upper(): v for k, v in (knn.get("weights") or {}).items()}
    knn_top = (knn.get("top_cwe") or "").upper()
    knn_mode = knn.get("mode", "")

    evidence_locations = []
    if gold in initial_cwes:
        evidence_locations.append("initial_retrieval")
    if gold in graph_cwes:
        evidence_locations.append("graph_neighbors")
    if gold in final_cwes:
        evidence_locations.append("final_context")
    if gold in knn_weights:
        evidence_locations.append("knn_votes")

    pred_relation = relation(predicted, gold, meta, children)

    categories = []
    if gold in final_cwes:
        categories.append("llm_selection_error")
    else:
        categories.append("gold_missing_from_final_context")

    if gold not in initial_cwes and gold not in graph_cwes and gold not in knn_weights:
        categories.append("retrieval_miss")

    if knn:
        if knn_top == gold:
            categories.append("knn_had_correct_top")
        elif knn_top:
            categories.append("knn_top_wrong")
        if gold in knn_weights and knn_top != gold:
            categories.append("knn_had_gold_but_ranked_other")
        if knn_mode == "high_confidence" and knn_top != gold:
            categories.append("wrong_high_confidence_knn")

    if pred_relation != "unrelated_or_distant" and pred_relation != "none":
        categories.append(pred_relation)

    if row.get("nvd_mapped"):
        categories.append("nvd_mapped_failure")
    else:
        categories.append("nvd_unmapped_failure")

    return {
        "cve_id": row.get("cve_id", ""),
        "gold": gold,
        "gold_name": meta.get(gold, {}).get("name", ""),
        "gold_abstraction": meta.get(gold, {}).get("abstraction", ""),
        "predicted": predicted,
        "predicted_name": meta.get(predicted, {}).get("name", ""),
        "predicted_abstraction": meta.get(predicted, {}).get("abstraction", ""),
        "predicted_relation_to_gold": pred_relation,
        "nvd_mapped": bool(row.get("nvd_mapped")),
        "evidence_locations": evidence_locations,
        "initial_cwes": sorted(initial_cwes),
        "graph_cwes": sorted(graph_cwes),
        "final_cwes": sorted(final_cwes),
        "knn_top": knn_top,
        "knn_mode": knn_mode,
        "knn_top_share": knn.get("top_share"),
        "knn_gold_weight": knn_weights.get(gold),
        "categories": categories,
    }


def pct(n: int, total: int) -> str:
    return f"{n}/{total} ({n / total * 100:.1f}%)" if total else "0/0"


def write_report(analyses: list[dict], path: Path) -> None:
    total = len(analyses)
    cat_counts = Counter(cat for row in analyses for cat in row["categories"])
    relation_counts = Counter(row["predicted_relation_to_gold"] for row in analyses)
    mapped_counts = Counter("mapped" if row["nvd_mapped"] else "unmapped" for row in analyses)

    lines = [
        "# CTI-RCM Failure Analysis",
        "",
        f"Input failures analyzed: **{total}**",
        "",
        "## High-Level Split",
        "",
        f"- NVD-mapped failures: {pct(mapped_counts['mapped'], total)}",
        f"- NVD-unmapped failures: {pct(mapped_counts['unmapped'], total)}",
        f"- Gold CWE present in final context but answer still wrong: {pct(cat_counts['llm_selection_error'], total)}",
        f"- Gold CWE missing from final context: {pct(cat_counts['gold_missing_from_final_context'], total)}",
        f"- Retrieval miss with no gold CWE in initial, graph, or k-NN evidence: {pct(cat_counts['retrieval_miss'], total)}",
        "",
        "## k-NN CWE Behavior",
        "",
        f"- k-NN top was correct: {pct(cat_counts['knn_had_correct_top'], total)}",
        f"- k-NN top was wrong: {pct(cat_counts['knn_top_wrong'], total)}",
        f"- k-NN contained gold but ranked another CWE higher: {pct(cat_counts['knn_had_gold_but_ranked_other'], total)}",
        f"- High-confidence k-NN was wrong: {pct(cat_counts['wrong_high_confidence_knn'], total)}",
        "",
        "## CWE Relationship Pattern",
        "",
    ]

    for rel, count in relation_counts.most_common():
        lines.append(f"- {rel}: {pct(count, total)}")

    lines.extend([
        "",
        "## Most Actionable Failure Examples",
        "",
    ])

    priority = [
        "llm_selection_error",
        "wrong_high_confidence_knn",
        "knn_had_gold_but_ranked_other",
        "retrieval_miss",
    ]
    shown = set()
    for category in priority:
        matches = [row for row in analyses if category in row["categories"] and row["cve_id"] not in shown]
        if not matches:
            continue
        lines.append(f"### {category}")
        lines.append("")
        for row in matches[:5]:
            shown.add(row["cve_id"])
            evidence = ", ".join(row["evidence_locations"]) or "none"
            lines.append(
                f"- **{row['cve_id']}** GT `{row['gold']}` ({row['gold_name']}) -> "
                f"predicted `{row['predicted'] or 'none'}` ({row['predicted_name'] or 'no CWE detected'}); "
                f"evidence: {evidence}; k-NN top: `{row['knn_top'] or 'none'}` {row['knn_mode']}"
            )
        lines.append("")

    lines.extend([
        "## Recommended Next Tuning Targets",
        "",
        "1. If `llm_selection_error` is high, add a deterministic CWE selector after retrieval instead of relying only on generated text.",
        "2. If `wrong_high_confidence_knn` appears, raise the k-NN confidence threshold or require a wider margin over the second CWE.",
        "3. If `retrieval_miss` is high, improve CWE candidate retrieval for unmapped CVEs before changing prompts.",
        "4. If parent/child/sibling errors dominate, use CWE hierarchy during candidate reranking.",
        "",
    ])

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else FAILURES_PATH
    failures = load_jsonl(input_path)
    meta = load_cwe_meta()
    children = build_children(meta)
    analyses = [classify(row, meta, children) for row in failures]

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(analyses, indent=2), encoding="utf-8")
    write_report(analyses, OUT_MD)

    cat_counts = Counter(cat for row in analyses for cat in row["categories"])
    print(f"Analyzed {len(analyses)} failures from {input_path}")
    print(f"  gold in final context but wrong answer: {cat_counts['llm_selection_error']}")
    print(f"  gold missing from final context: {cat_counts['gold_missing_from_final_context']}")
    print(f"  retrieval miss: {cat_counts['retrieval_miss']}")
    print(f"  wrong high-confidence k-NN: {cat_counts['wrong_high_confidence_knn']}")
    print(f"Wrote {OUT_MD}")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
