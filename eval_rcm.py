"""
CTI-RCM Evaluation Script
Evaluates the RAG system against the CTI-Bench Root Cause Mapping benchmark.

Usage:
    conda run -n cyber-ft python3 eval_rcm.py              # 100-query sample (all)
    conda run -n cyber-ft python3 eval_rcm.py 1000         # full benchmark (all)
    conda run -n cyber-ft python3 eval_rcm.py --matched    # only CVEs where NVD chunk text contains the GT CWE
    conda run -n cyber-ft python3 eval_rcm.py 100 --matched
    conda run -n cyber-ft python3 eval_rcm.py --nvd-mapped    # only CVEs with an official NVD cweId mapping
    conda run -n cyber-ft python3 eval_rcm.py --nvd-unmapped  # only CVEs with no NVD cweId (must reason from description)
    conda run -n cyber-ft python3 eval_rcm.py --local      # only CVEs in our DB whose GT CWE is also in our CWE chunks
    conda run -n cyber-ft python3 eval_rcm.py 97 --nvd-unmapped --debug-failures
    conda run -n cyber-ft python3 eval_rcm.py 100 --profile  # print retrieval/timing checkpoints
"""
import csv
import itertools
import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import better_rag

RCM_PATH      = Path("data/cti-bench/data/cti-rcm.tsv")
CVE_CHUNKS    = Path("data/processed/cve_chunks.jsonl")
CVE_CWE_INDEX = Path("data/processed/cve_cwe_index.json")
DEBUG_FAILURES_PATH = Path("eval_failures_debug.jsonl")
SAMPLE_N  = 100
SEED      = 42
WORKERS   = int(os.environ.get("CTI_RAG_EVAL_WORKERS", "16"))  # vLLM continuous batching scales until KV cache saturates


def extract_cve_id(url: str) -> str:
    m = re.search(r"CVE-\d{4}-\d+", url, re.IGNORECASE)
    return m.group(0).upper() if m else ""


def cwe_in_answer(answer: str, gt_cwe: str) -> bool:
    return gt_cwe.upper() in answer.upper()


CWE_CHUNKS = Path("data/processed/cwe_chunks.jsonl")


def load_cve_map() -> dict:
    cve_map = {}
    for line in CVE_CHUNKS.open():
        d = json.loads(line)
        cve_map[d["identifier"]] = d["text"]
    return cve_map


def load_cwe_ids() -> set:
    cwe_ids = set()
    for line in CWE_CHUNKS.open():
        d = json.loads(line)
        ident = d.get("identifier", "")
        if ident:
            cwe_ids.add(ident.upper())
    return cwe_ids


def load_cve_cwe_index() -> dict:
    if CVE_CWE_INDEX.exists():
        return json.loads(CVE_CWE_INDEX.read_text())
    return {}


def run_eval(n: int = SAMPLE_N, seed: int = SEED, matched_only: bool = False, local_only: bool = False,
             nvd_mapped: bool = False, nvd_unmapped: bool = False, debug_failures: bool = False):
    random.seed(seed)

    rows = []
    with open(RCM_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            rows.append(row)

    if nvd_mapped or nvd_unmapped:
        cve_cwe_index = load_cve_cwe_index()
        if nvd_mapped:
            rows = [r for r in rows if bool(cve_cwe_index.get(extract_cve_id(r["URL"])))]
            print(f"  {len(rows)} CVEs with official NVD cweId mapping\n")
        else:
            rows = [r for r in rows if not bool(cve_cwe_index.get(extract_cve_id(r["URL"])))]
            print(f"  {len(rows)} CVEs with no NVD cweId (must reason from description)\n")

    if matched_only:
        print("Loading CVE chunks to filter to matched-only entries...")
        cve_map = load_cve_map()
        rows = [
            r for r in rows
            if r["GT"].strip().upper() in
               cve_map.get(extract_cve_id(r["URL"]), "").upper()
        ]
        print(f"  {len(rows)} CVEs where NVD chunk contains the GT CWE\n")

    if local_only:
        print("Loading CVE + CWE chunks to filter to local-data-only entries...")
        cve_map = load_cve_map()
        cwe_ids = load_cwe_ids()
        rows = [
            r for r in rows
            if extract_cve_id(r["URL"]) in cve_map
            and r["GT"].strip().upper() in cwe_ids
        ]
        print(f"  {len(rows)} entries where both CVE chunk and GT CWE chunk exist in our DB\n")

    cve_cwe_index = load_cve_cwe_index()

    sample = random.sample(rows, min(n, len(rows)))
    if matched_only:
        mode_label = "matched-only (NVD chunk contains GT CWE)"
    elif nvd_mapped:
        mode_label = "NVD-mapped only (has official cweId)"
    elif nvd_unmapped:
        mode_label = "NVD-unmapped only (no cweId, reason from description)"
    elif local_only:
        mode_label = "local-only (CVE + CWE in our DB)"
    else:
        mode_label = "all entries"
    results = [None] * len(sample)

    def _run_one(i: int, row: dict) -> tuple[int, dict]:
        cve_id = extract_cve_id(row["URL"])
        gt_cwe = row["GT"].strip()
        query  = row["Prompt"]  # same prompt format used to test GPT-4 in CTI-Bench paper
        nvd_mapped = bool(cve_cwe_index.get(cve_id))
        debug_info = {} if debug_failures else None
        try:
            answer, _, _ = better_rag.ask(query, [], eval_mode=True, debug_info=debug_info)
            passed = cwe_in_answer(answer, gt_cwe)
            result = {"cve_id": cve_id, "gt_cwe": gt_cwe, "passed": passed, "answer": answer, "nvd_mapped": nvd_mapped}
            if debug_failures and not passed:
                result["debug"] = debug_info
                result["prompt"] = query
            return i, result
        except Exception as e:
            result = {"cve_id": cve_id, "gt_cwe": gt_cwe, "passed": False, "error": str(e), "nvd_mapped": nvd_mapped}
            if debug_failures:
                result["debug"] = debug_info
                result["prompt"] = query
            return i, result

    _counter = itertools.count()
    _embedders = [better_rag.embedder,
                  better_rag._embedder_1 if better_rag._embedder_1 else better_rag.embedder]

    def _thread_init():
        idx = next(_counter)
        emb = _embedders[idx % len(_embedders)]
        better_rag._thread_local.embedder = emb
        print(f"  [Thread {idx}] using embedder on {emb.device}")

    completed = 0
    with ThreadPoolExecutor(max_workers=WORKERS, initializer=_thread_init) as executor:
        futures = {executor.submit(_run_one, i, row): i for i, row in enumerate(sample)}
        for future in as_completed(futures):
            i, result = future.result()
            results[i] = result
            completed += 1
            cve_id = result["cve_id"]
            gt_cwe = result["gt_cwe"]
            status = "PASS" if result["passed"] else "FAIL"
            snippet = result.get("answer", result.get("error", ""))[:120].replace("\n", " ")
            print(f"[{completed}/{len(sample)}] {cve_id}  GT: {gt_cwe}  {status} | {snippet}")

    passed_n = sum(1 for r in results if r["passed"])
    total    = len(results)
    acc      = passed_n / total * 100 if total else 0.0

    mapped_results   = [r for r in results if r.get("nvd_mapped")]
    unmapped_results = [r for r in results if not r.get("nvd_mapped")]
    mapped_pass      = sum(1 for r in mapped_results if r["passed"])
    unmapped_pass    = sum(1 for r in unmapped_results if r["passed"])

    print(f"\n{'='*60}")
    print(f"CTI-RCM Results  ({len(sample)} queries, {mode_label}, seed={seed})")
    print(f"{'='*60}")
    print(f"  Passed  : {passed_n}/{total}")
    print(f"  Accuracy: {acc:.1f}%")
    print(f"\n  Breakdown by NVD CWE mapping:")
    if mapped_results:
        print(f"    NVD has CWE mapping  : {mapped_pass}/{len(mapped_results)}  ({mapped_pass/len(mapped_results)*100:.1f}%)")
    if unmapped_results:
        print(f"    NVD has no mapping   : {unmapped_pass}/{len(unmapped_results)}  ({unmapped_pass/len(unmapped_results)*100:.1f}%)")

    failures = [res for res in results if not res["passed"]]
    if failures:
        print(f"\nFailures ({len(failures)}):")
        for res in failures:
            snippet = res.get("answer", res.get("error", ""))[:100].replace("\n", " ")
            print(f"  {res['cve_id']:20s} GT={res['gt_cwe']:10s} | {snippet}")

    if debug_failures:
        with DEBUG_FAILURES_PATH.open("w", encoding="utf-8") as f:
            for res in failures:
                f.write(json.dumps(res) + "\n")
        print(f"\nWrote failure debug report: {DEBUG_FAILURES_PATH}")

    return results, acc


if __name__ == "__main__":
    args = sys.argv[1:]
    matched_only  = "--matched"      in args
    local_only    = "--local"        in args
    nvd_mapped    = "--nvd-mapped"   in args
    nvd_unmapped  = "--nvd-unmapped" in args
    debug_failures = "--debug-failures" in args
    profile       = "--profile"      in args
    args = [a for a in args if a not in ("--matched", "--local", "--nvd-mapped", "--nvd-unmapped", "--debug-failures", "--profile")]
    n = int(args[0]) if args else SAMPLE_N
    if profile:
        print("Profiling enabled: retrieval/timing checkpoints will be printed.")
    run_eval(n, matched_only=matched_only, local_only=local_only, nvd_mapped=nvd_mapped,
             nvd_unmapped=nvd_unmapped, debug_failures=debug_failures)
