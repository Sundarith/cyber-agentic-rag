"""
CTI-RCM Evaluation Script
Evaluates the RAG system against the CTI-Bench Root Cause Mapping benchmark.

Usage:
    conda run -n cyber-ft python3 eval_rcm.py              # 100-query sample (all)
    conda run -n cyber-ft python3 eval_rcm.py 1000         # full benchmark (all)
    conda run -n cyber-ft python3 eval_rcm.py --matched    # only CVEs where NVD has the CWE (true retrieval test)
    conda run -n cyber-ft python3 eval_rcm.py 100 --matched
"""
import csv
import json
import random
import re
import sys
from pathlib import Path

import better_rag

RCM_PATH  = Path("data/cti-bench/data/cti-rcm.tsv")
CVE_CHUNKS = Path("data/processed/cve_chunks.jsonl")
SAMPLE_N  = 100
SEED      = 42


def extract_cve_id(url: str) -> str:
    m = re.search(r"CVE-\d{4}-\d+", url, re.IGNORECASE)
    return m.group(0).upper() if m else ""


def cwe_in_answer(answer: str, gt_cwe: str) -> bool:
    return gt_cwe.upper() in answer.upper()


def load_cve_map() -> dict:
    cve_map = {}
    for line in CVE_CHUNKS.open():
        d = json.loads(line)
        cve_map[d["identifier"]] = d["text"]
    return cve_map


def run_eval(n: int = SAMPLE_N, seed: int = SEED, matched_only: bool = False):
    random.seed(seed)

    rows = []
    with open(RCM_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            rows.append(row)

    if matched_only:
        print("Loading CVE chunks to filter to matched-only entries...")
        cve_map = load_cve_map()
        rows = [
            r for r in rows
            if r["GT"].strip().upper() in
               cve_map.get(extract_cve_id(r["URL"]), "").upper()
        ]
        print(f"  {len(rows)} CVEs where NVD chunk contains the GT CWE\n")

    sample = random.sample(rows, min(n, len(rows)))
    mode_label = "matched-only (NVD has CWE)" if matched_only else "all entries"
    results = []

    for i, row in enumerate(sample):
        cve_id = extract_cve_id(row["URL"])
        gt_cwe = row["GT"].strip()
        query  = row["Prompt"]  # same prompt format used to test GPT-4 in CTI-Bench paper

        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(sample)}] {cve_id}  GT: {gt_cwe}")

        try:
            answer, _, _ = better_rag.ask(query, [])
            passed = cwe_in_answer(answer, gt_cwe)
            results.append({
                "cve_id": cve_id,
                "gt_cwe": gt_cwe,
                "passed": passed,
                "answer": answer,
            })
            status = "PASS" if passed else "FAIL"
            print(f"  {status} | {answer[:120].replace(chr(10), ' ')}")
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"cve_id": cve_id, "gt_cwe": gt_cwe, "passed": False, "error": str(e)})

    passed_n = sum(1 for r in results if r["passed"])
    total    = len(results)
    acc      = passed_n / total * 100 if total else 0.0

    print(f"\n{'='*60}")
    print(f"CTI-RCM Results  ({len(sample)} queries, {mode_label}, seed={seed})")
    print(f"{'='*60}")
    print(f"  Passed  : {passed_n}/{total}")
    print(f"  Accuracy: {acc:.1f}%")

    failures = [r for r in results if not r["passed"]]
    if failures:
        print(f"\nFailures ({len(failures)}):")
        for r in failures:
            snippet = r.get("answer", r.get("error", ""))[:100].replace("\n", " ")
            print(f"  {r['cve_id']:20s} GT={r['gt_cwe']:10s} | {snippet}")

    return results, acc


if __name__ == "__main__":
    args = sys.argv[1:]
    matched_only = "--matched" in args
    args = [a for a in args if a != "--matched"]
    n = int(args[0]) if args else SAMPLE_N
    run_eval(n, matched_only=matched_only)
