#!/usr/bin/env python3
"""Triage mapped-subset failures from a debug-failures jsonl.

Categories (mirror Table IV in the paper):
  A. NVD disagrees with CTI-Bench, model follows NVD
       -- predicted CWE matches some NVD CWE for the CVE, but no NVD CWE
          matches the gold (so NVD itself disagrees with CTI-Bench's label).
  B. LLM ignored bridge-injected CWE
       -- NVD's CWE list contains the gold, but the model still picked a
          different CWE.
  C. NVD disagrees AND model also drifted
       -- NVD CWEs do NOT contain the gold, AND the model's pick also does
          not match any NVD CWE.
  D. Multi-NVD CWE, model selected non-gold candidate
       -- Specialization of B: NVD has >1 CWE, includes the gold, model
          picked one of the non-gold NVD CWEs.
  E. No CWE in answer
       -- Model output did not contain a parseable CWE-NNN token.

Usage:
  python3 scripts/triage_mapped_failures.py <failures.jsonl>
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

CWE_RE = re.compile(r"CWE[-\s_:]*0*(\d+)", re.IGNORECASE)


def extract_cwe(text: str) -> str | None:
    """Return the LAST CWE-NNN mentioned in text, matching eval_rcm scoring."""
    if not text:
        return None
    matches = CWE_RE.findall(text)
    if not matches:
        return None
    return f"CWE-{int(matches[-1])}"


def norm_cwe(c: str) -> str:
    m = CWE_RE.search(c) if c else None
    return f"CWE-{int(m.group(1))}" if m else ""


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        sys.exit(1)

    fail_path = Path(argv[1])
    cve_index_path = Path("data/processed/cve_cwe_index.json")
    if not cve_index_path.exists():
        print(f"ERROR: {cve_index_path} not found", file=sys.stderr)
        sys.exit(2)

    cve_cwe_index = json.loads(cve_index_path.read_text())

    rows = []
    with fail_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    mapped_failures = [r for r in rows if r.get("nvd_mapped") and not r.get("passed")]
    print(f"Total rows: {len(rows)}")
    print(f"Mapped failures: {len(mapped_failures)}")
    print()

    cat_counts = Counter()
    by_cat: dict[str, list] = {k: [] for k in "ABCDE"}

    for r in mapped_failures:
        cve_id = (r.get("cve_id") or "").upper()
        gold = norm_cwe(r.get("gt_cwe") or "")
        ans = r.get("answer") or ""
        pred = extract_cwe(ans)
        nvd_list = [norm_cwe(c) for c in cve_cwe_index.get(cve_id, [])]
        nvd_set = {c for c in nvd_list if c}

        if pred is None:
            cat = "E"
        elif gold in nvd_set:
            # NVD includes the gold; model failed despite the bridge.
            if len(nvd_set) >= 2 and pred in nvd_set and pred != gold:
                cat = "D"
            else:
                cat = "B"
        else:
            # NVD's CWE list does NOT contain the gold (NVD disagrees with CTI-Bench).
            if pred in nvd_set:
                cat = "A"  # model followed NVD
            else:
                cat = "C"  # NVD wrong AND model drifted

        cat_counts[cat] += 1
        by_cat[cat].append({
            "cve_id": cve_id,
            "gold": gold,
            "pred": pred,
            "nvd": sorted(nvd_set),
        })

    labels = {
        "A": "NVD disagrees with CTI-Bench, model follows NVD",
        "B": "LLM ignored bridge-injected CWE",
        "C": "NVD disagrees AND model also drifted",
        "D": "Multi-NVD CWE, model selected non-gold candidate",
        "E": "No CWE-ID parsed from answer",
    }

    total = len(mapped_failures)
    print(f"{'Cat':<3} {'Count':>6} {'%':>7}  Description")
    print("-" * 78)
    for k in "ABCDE":
        n = cat_counts[k]
        pct = (100.0 * n / total) if total else 0.0
        print(f"{k:<3} {n:>6} {pct:>6.1f}%  {labels[k]}")
    print()

    print("Sample rows per category (first 5):")
    for k in "ABCDE":
        if not by_cat[k]:
            continue
        print(f"\n[{k}] {labels[k]}")
        for row in by_cat[k][:5]:
            print(f"  {row['cve_id']:<20} gold={row['gold']:<10} pred={row['pred'] or 'NONE':<10} nvd={row['nvd']}")


if __name__ == "__main__":
    main(sys.argv)
