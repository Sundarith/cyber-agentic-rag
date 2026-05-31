"""Per-CWE accuracy breakdown for the shipped Clean Prompt Phi-4-mini-reasoning run.

Outputs:
- docs/figures/per_cwe_chart.png  (top-20 CWEs by support, accuracy side-by-side)
- docs/ieee_acsac_2026/IEEE_ACSAC_2026__Sun_/figures/per_cwe_chart.png
- analysis/per_cwe_breakdown_2026-05-24.md (table)
"""
from __future__ import annotations
import json
import re
from pathlib import Path
from collections import defaultdict, Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/sheng/cyber-ft")
MAPPED_LOG = ROOT / "logs/reeval_context_prompt/eval_phi4_mapped903_context_prompt_20260524_234345.log"
UNMAPPED_LOG = ROOT / "logs/reeval_context_prompt/eval_phi4_mini_reasoning_unmapped97_context_prompt_20260525_003537.log"
CWE_CHUNKS = ROOT / "data/processed/cwe_chunks.jsonl"

OUT_PNG = ROOT / "docs/figures/per_cwe_chart.png"
OUT_PAPER_PNG = ROOT / "docs/ieee_acsac_2026/IEEE_ACSAC_2026__Sun_/figures/per_cwe_chart.png"
OUT_MD  = ROOT / "analysis/per_cwe_breakdown_2026-05-24.md"


def load_results_from_logs(mapped_path: Path, unmapped_path: Path):
    rows = []
    # Match pattern: [1/903] CVE-2024-22860  GT: CWE-190  PASS | <think> ...
    line_re = re.compile(r'\[\d+/\d+\]\s+CVE-\d+-\d+\s+GT:\s+(CWE-\d+)\s+(PASS|FAIL)')
    
    for path in [mapped_path, unmapped_path]:
        with path.open() as f:
            for line in f:
                match = line_re.search(line)
                if match:
                    gt_cwe = match.group(1).upper()
                    result = match.group(2)
                    rows.append({
                        "gt_cwe": gt_cwe,
                        "passed": result == "PASS"
                    })
    print(f"Parsed {len(rows)} cases (expected 1000)")
    return rows


def load_cwe_names() -> dict[str, str]:
    names = {}
    with CWE_CHUNKS.open() as f:
        for line in f:
            row = json.loads(line)
            ident = (row.get("identifier") or "").upper()
            if ident.startswith("CWE-") and ident not in names:
                names[ident] = row.get("name", "")
    return names


def accuracy_by_cwe(rows):
    correct = Counter()
    total = Counter()
    for row in rows:
        gold = row["gt_cwe"].upper()
        total[gold] += 1
        if row.get("passed"):
            correct[gold] += 1
    return correct, total


def main():
    shipped = load_results_from_logs(MAPPED_LOG, UNMAPPED_LOG)
    assert len(shipped) == 1000, f"Expected 1000 queries, got {len(shipped)}"
    names = load_cwe_names()

    correct, total = accuracy_by_cwe(shipped)

    top_cwes = [cwe for cwe, _ in total.most_common(20)]

    rows_md = []
    rows_md.append("| CWE | Name | n | Shipped % |")
    rows_md.append("|---|---|---:|---:|")
    for cwe in top_cwes:
        n = total[cwe]
        acc = correct[cwe] / n * 100
        name = names.get(cwe, "?")[:60]
        rows_md.append(f"| {cwe} | {name} | {n} | {acc:.1f} |")

    OUT_MD.parent.mkdir(exist_ok=True, parents=True)
    OUT_MD.write_text("# Per-CWE Accuracy Breakdown (Top-20 by support, shipped Clean Prompt run)\n\n" + "\n".join(rows_md) + "\n")
    print(f"wrote {OUT_MD}")

    # plot
    x = list(range(len(top_cwes)))
    vals = [correct[c] / total[c] * 100 for c in top_cwes]
    supports = [total[c] for c in top_cwes]

    fig, ax = plt.subplots(figsize=(11.5, 4.5), dpi=150)
    ax.bar(x, vals, 0.68, label="Phi-4-mini-reasoning + RAG (Clean prompt)",
           color="#047857", edgecolor="white")
    ax.set_xticks(x)
    labels = [f"{c.replace('CWE-','')}\n(n={supports[i]})" for i, c in enumerate(top_cwes)]
    ax.set_xticklabels(labels, fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("Strict accuracy (%)", fontsize=11)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    OUT_PNG.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(OUT_PNG, bbox_inches="tight")
    print(f"wrote {OUT_PNG}")
    OUT_PAPER_PNG.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(OUT_PAPER_PNG, bbox_inches="tight")
    print(f"wrote {OUT_PAPER_PNG}")


if __name__ == "__main__":
    main()
