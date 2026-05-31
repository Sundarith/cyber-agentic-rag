"""V2: classify failures using predicted CWE from eval_results files.

We use the no_hyde LOO run (870 / 1000, the cleanest of the LOO set) as the
predicted-CWE source, then cross-reference against the 864 routed run's failures.
For each failure CVE, we recover the predicted CWE from any eval_results file
where the CVE also failed (predictions are usually stable across LOO configs).
"""
from __future__ import annotations
import json
import re
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path("/home/sheng/cyber-ft")
ROUTED_864 = ROOT / "logs/eval_timing_20260521_003937.jsonl"
CWE_CHUNKS = ROOT / "data/processed/cwe_chunks.jsonl"
CVE_CWE_INDEX = ROOT / "data/processed/cve_cwe_index.json"

LOO_FILES = sorted((ROOT / "logs").glob("eval_results_2026052*.jsonl"))

CWE_RE = re.compile(r"CWE-(\d+)", re.IGNORECASE)


def load_cwe_hierarchy() -> dict[str, dict]:
    nodes: dict[str, dict] = {}
    if not CWE_CHUNKS.exists():
        return nodes
    with CWE_CHUNKS.open() as f:
        for line in f:
            row = json.loads(line)
            ident = (row.get("identifier") or "").upper()
            if not ident.startswith("CWE-"):
                continue
            parents = [p.upper() for p in row.get("parent_cwe_ids", [])]
            nodes.setdefault(ident, {"parents": [], "children": [], "name": ""})
            nodes[ident]["parents"] = parents
            nodes[ident]["name"] = row.get("name", "")
    # build reverse children index
    for cid, info in list(nodes.items()):
        for p in info["parents"]:
            nodes.setdefault(p, {"parents": [], "children": [], "name": ""})
            nodes[p]["children"].append(cid)
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
    return seen


def relation(nodes, pred, gold):
    if not pred:
        return "no_cwe_in_answer"
    if pred == gold:
        return "same"
    g_anc = ancestors(nodes, gold)
    g_desc = descendants(nodes, gold)
    if pred in g_anc:
        return "parent_or_ancestor"
    if pred in g_desc:
        return "child_or_descendant"
    p_anc = ancestors(nodes, pred)
    if (p_anc & g_anc):
        return "sibling_share_ancestor"
    return "distant_different_subtree"


def extract_predicted_cwe(answer_text: str | None) -> str | None:
    if not answer_text:
        return None
    m = CWE_RE.search(answer_text)
    return f"CWE-{m.group(1)}".upper() if m else None


def main():
    nodes = load_cwe_hierarchy()

    cve_cwe = {}
    if CVE_CWE_INDEX.exists():
        idx = json.loads(CVE_CWE_INDEX.read_text())
        for cve, val in idx.items():
            ids = val.get("cwe_ids", []) if isinstance(val, dict) else val
            cve_cwe[cve.upper()] = [c.upper() for c in ids]

    # routed-864 failures
    failed_864 = []
    with ROUTED_864.open() as f:
        for line in f:
            row = json.loads(line)
            if not row.get("passed"):
                failed_864.append(row)

    # build per-CVE predicted-CWE lookup from LOO runs (any failing prediction is the predicted CWE)
    pred_by_cve: dict[str, str] = {}
    for loo_path in LOO_FILES:
        with loo_path.open() as f:
            for line in f:
                row = json.loads(line)
                if row.get("passed"):
                    continue
                cve = row["cve_id"].upper()
                if cve in pred_by_cve:
                    continue
                pred = extract_predicted_cwe(row.get("answer", ""))
                if pred:
                    pred_by_cve[cve] = pred

    counter = Counter()
    flagged_nvd_disagree = 0
    examples_by_cat: dict[str, list] = defaultdict(list)

    for row in failed_864:
        cve = row["cve_id"].upper()
        gold = row["gt_cwe"].upper() if row.get("gt_cwe") else None
        nvd_cwes = cve_cwe.get(cve, [])
        nvd_disagrees = bool(nvd_cwes) and gold not in set(nvd_cwes)
        pred = pred_by_cve.get(cve)
        rel = relation(nodes, pred, gold) if gold else "no_cwe_in_answer"
        if nvd_disagrees:
            flagged_nvd_disagree += 1
        counter[rel] += 1
        if len(examples_by_cat[rel]) < 4:
            examples_by_cat[rel].append({
                "cve": cve,
                "gold": gold,
                "gold_name": nodes.get(gold, {}).get("name", "?")[:50],
                "predicted": pred,
                "pred_name": nodes.get(pred or "", {}).get("name", "?")[:50],
                "nvd_cwes": nvd_cwes,
                "nvd_disagrees": nvd_disagrees,
                "nvd_mapped": row.get("nvd_mapped"),
            })

    print(f"=== Failure typology: 864 routed run, {len(failed_864)} failures ===\n")
    print(f"NVD disagrees with CTI-Bench gold: {flagged_nvd_disagree}/{len(failed_864)} ({flagged_nvd_disagree/len(failed_864)*100:.1f}%)\n")
    print("Category counts (predicted-CWE relationship to gold):\n")
    print(f"{'category':35s} {'count':>5s}  pct")
    for cat, n in counter.most_common():
        print(f"  {cat:33s} {n:>5d}  {n/len(failed_864)*100:>5.1f}%")
    print()

    print("Example CVEs per category:\n")
    for cat, items in examples_by_cat.items():
        print(f"-- {cat} --")
        for it in items:
            nvd_flag = " [NVD-disagrees]" if it["nvd_disagrees"] else ""
            print(f"  {it['cve']:20s} gold={it['gold']:8s} ({it['gold_name']})  pred={str(it['predicted']):8s} ({it['pred_name']}){nvd_flag}")
        print()


if __name__ == "__main__":
    main()
