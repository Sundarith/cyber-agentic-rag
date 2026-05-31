"""Classify failures from the routed 864 run into a named typology.

Categories (per failure-analysis literature, e.g. Vul-RAG / SecLLMHolmes):
  A. parent_or_ancestor : predicted CWE is an ancestor of gold (too abstract)
  B. child_or_descendant: predicted is descendant of gold (too specific)
  C. sibling_share_ancestor: predicted shares an ancestor within 2 hops
  D. distant_different_subtree: predicted is in an unrelated subtree
  E. no_cwe_in_answer  : model refused or output unparseable
  F. nvd_label_disagreement: NVD's own assignment != CTI-Bench gold

Reads logs/eval_timing_20260521_003937.jsonl (the 864 routed run) and
data/processed/cwe_chunks.jsonl to walk CWE hierarchy when available.
"""
from __future__ import annotations
import json
import re
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path("/home/sheng/cyber-ft")
EVAL = ROOT / "logs/eval_timing_20260521_003937.jsonl"
CWE_CHUNKS = ROOT / "data/processed/cwe_chunks.jsonl"
CVE_CWE_INDEX = ROOT / "data/processed/cve_cwe_index.json"

CWE_RE = re.compile(r"CWE-(\d+)", re.IGNORECASE)


def load_cwe_hierarchy() -> dict[str, dict]:
    """Returns {cwe_id_upper: {"parents": [...], "children": [...], "name": str}}."""
    nodes: dict[str, dict] = {}
    if not CWE_CHUNKS.exists():
        return nodes
    with CWE_CHUNKS.open() as f:
        for line in f:
            row = json.loads(line)
            ident = (row.get("identifier") or "").upper()
            if not ident.startswith("CWE-"):
                continue
            nodes[ident] = {
                "parents":  [p.upper() for p in row.get("parents", [])],
                "children": [c.upper() for c in row.get("children", [])],
                "name":     row.get("name", ""),
            }
    return nodes


def ancestors(nodes, cwe, max_hops=6):
    seen = set()
    frontier = [cwe]
    for _ in range(max_hops):
        nxt = []
        for c in frontier:
            for p in nodes.get(c, {}).get("parents", []):
                if p not in seen:
                    seen.add(p)
                    nxt.append(p)
        frontier = nxt
        if not frontier:
            break
    return seen


def descendants(nodes, cwe, max_hops=6):
    seen = set()
    frontier = [cwe]
    for _ in range(max_hops):
        nxt = []
        for c in frontier:
            for ch in nodes.get(c, {}).get("children", []):
                if ch not in seen:
                    seen.add(ch)
                    nxt.append(ch)
        frontier = nxt
        if not frontier:
            break
    return seen


def relation(nodes, pred, gold):
    if pred == gold:
        return "same"
    if not pred:
        return "no_cwe_in_answer"
    p_anc = ancestors(nodes, pred)
    g_anc = ancestors(nodes, gold)
    p_desc = descendants(nodes, pred)
    g_desc = descendants(nodes, gold)
    if pred in g_anc:
        return "parent_or_ancestor"      # pred is ancestor of gold
    if pred in g_desc:
        return "child_or_descendant"     # pred is descendant of gold
    # share ancestor within 2 hops?
    common = (p_anc & g_anc) | ({pred} & g_anc) | ({gold} & p_anc)
    if common:
        return "sibling_share_ancestor"
    return "distant_different_subtree"


def extract_predicted_cwe(answer_text: str | None) -> str | None:
    if not answer_text:
        return None
    m = CWE_RE.search(answer_text)
    return f"CWE-{m.group(1)}" if m else None


def main():
    nodes = load_cwe_hierarchy()
    print(f"loaded {len(nodes)} CWE nodes with hierarchy info")

    # load NVD index for disagreement flag
    cve_cwe = {}
    if CVE_CWE_INDEX.exists():
        idx = json.loads(CVE_CWE_INDEX.read_text())
        # accept either {cve: [cwes]} or {cve: {"cwe_ids": [...]}}
        for cve, val in idx.items():
            if isinstance(val, dict):
                cve_cwe[cve.upper()] = [c.upper() for c in val.get("cwe_ids", [])]
            else:
                cve_cwe[cve.upper()] = [c.upper() for c in val]

    failures = []
    with EVAL.open() as f:
        for line in f:
            row = json.loads(line)
            if row.get("passed"):
                continue
            failures.append(row)

    print(f"total failures in routed 864 run: {len(failures)}")

    counter = Counter()
    nvd_disagree = 0
    examples_by_cat: dict[str, list] = defaultdict(list)

    # answers aren't saved in eval_timing; we need a separate source
    # for now, classify by gold only and flag NVD-disagreement
    for row in failures:
        cve = row["cve_id"].upper()
        gold = row["gt_cwe"].upper() if row.get("gt_cwe") else None
        nvd_cwes = cve_cwe.get(cve, [])
        # has NVD mapping but gold disagrees with all of NVD's CWEs
        nvd_disagrees = bool(nvd_cwes) and gold not in set(nvd_cwes)
        if nvd_disagrees:
            nvd_disagree += 1
        # we don't have predicted CWE here; need to grep from a separate log
        # leave per-CVE typology to next stage
        cat = "nvd_label_disagreement" if nvd_disagrees else "unclassified_no_prediction_data"
        counter[cat] += 1
        if len(examples_by_cat[cat]) < 3:
            examples_by_cat[cat].append({
                "cve_id": cve,
                "gold":   gold,
                "nvd_cwes": nvd_cwes,
                "nvd_mapped": row.get("nvd_mapped"),
            })

    print(f"failures where gold disagrees with NVD's own CWE: {nvd_disagree}/{len(failures)} ({nvd_disagree/len(failures)*100:.1f}%)")
    print("category counts (preliminary, no predicted-CWE data yet):")
    for cat, n in counter.most_common():
        print(f"  {cat:35s} {n}")


if __name__ == "__main__":
    main()
