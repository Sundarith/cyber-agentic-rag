"""
Run a small CTI-RCM latency sample and write dashboard timing data.

Usage:
    CTI_RAG_CAPTURE_TIMING=1 conda run -n cyber-ft python3 -u measure_latency_sample.py 20
"""
import csv
import json
import os
import random
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("CTI_RAG_CAPTURE_TIMING", "1")

import better_rag

RCM_PATH = Path("data/cti-bench/data/cti-rcm.tsv")
OUT_PATH = Path("presentation/performance_sample.json")
SEED = 42


def extract_cve_id(url: str) -> str:
    m = re.search(r"CVE-\d{4}-\d+", url, re.IGNORECASE)
    return m.group(0).upper() if m else ""


def cwe_in_answer(answer: str, gt_cwe: str) -> bool:
    return gt_cwe.upper() in answer.upper()


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((pct / 100) * (len(ordered) - 1))))
    return ordered[idx]


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    random.seed(SEED)

    with RCM_PATH.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))

    sample = random.sample(rows, min(n, len(rows)))
    results = []
    started = time.time()

    for i, row in enumerate(sample, 1):
        debug_info = {}
        cve_id = extract_cve_id(row["URL"])
        gt_cwe = row["GT"].strip()
        answer, _, _ = better_rag.ask(row["Prompt"], [], eval_mode=True, debug_info=debug_info)
        timing = debug_info.get("timing", {})
        passed = cwe_in_answer(answer, gt_cwe)
        result = {
            "index": i,
            "cve_id": cve_id,
            "gt_cwe": gt_cwe,
            "passed": passed,
            "retrieve_s": timing.get("retrieve_s", 0.0),
            "filters_s": timing.get("filters_s", 0.0),
            "graph_s": timing.get("graph_s", 0.0),
            "neighbors_s": timing.get("neighbors_s", 0.0),
            "prompt_s": timing.get("prompt_s", 0.0),
            "llm_s": timing.get("llm_s", 0.0),
            "total_s": timing.get("total_s", 0.0),
            "candidate_count": timing.get("candidate_count", 0),
        }
        results.append(result)
        print(
            f"[{i}/{len(sample)}] {cve_id} {gt_cwe} "
            f"{'PASS' if passed else 'FAIL'} "
            f"retrieve={result['retrieve_s']:.2f}s total={result['total_s']:.2f}s"
        )

    total_values = [r["total_s"] for r in results]
    retrieve_values = [r["retrieve_s"] for r in results]
    llm_values = [r["llm_s"] for r in results]
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "sample_size": len(results),
        "seed": SEED,
        "elapsed_wall_s": time.time() - started,
        "summary": {
            "avg_retrieve_s": sum(retrieve_values) / len(retrieve_values) if retrieve_values else 0.0,
            "avg_llm_s": sum(llm_values) / len(llm_values) if llm_values else 0.0,
            "avg_total_s": sum(total_values) / len(total_values) if total_values else 0.0,
            "p50_total_s": percentile(total_values, 50),
            "p95_total_s": percentile(total_values, 95),
            "pass_count": sum(1 for r in results if r["passed"]),
        },
        "results": results,
    }

    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
