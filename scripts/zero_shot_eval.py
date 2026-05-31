#!/usr/bin/env python3
"""Zero-shot LLM eval on CTI-Bench RCM (no RAG).

Sends each CTI-Bench `Prompt` directly to a vLLM /v1/chat/completions endpoint,
extracts the CWE-ID from the response, and scores against the ground-truth CWE.
Uses the same scoring logic as eval_rcm.py (cwe_in_answer = substring match).

Usage:
    python3 scripts/zero_shot_eval.py [N] [--nvd-mapped|--nvd-unmapped]
        [--endpoint-a URL] [--endpoint-b URL] [--model NAME]
        [--workers N] [--max-tokens N] [--no-think] [--log PATH]

If --endpoint-b is given, requests are randomly load-balanced across the two endpoints
(useful for dual-GPU throughput). Otherwise all requests go to --endpoint-a.

Env overrides:
    ZS_ENDPOINT_A   (default http://localhost:8000/v1/chat/completions)
    ZS_ENDPOINT_B   (default unset; single-endpoint mode)
    ZS_MODEL        (default Qwen/Qwen2.5-7B-Instruct — the served-model-name)
    ZS_MAX_TOKENS   (default 256; bump to 2048 for thinking models)
    ZS_NO_THINK     (1 -> append '/no_think' to user message, for Qwen3 family)
    ZS_WORKERS      (default 16)
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path("/home/sheng/cyber-ft")
RCM_PATH = ROOT / "data/cti-bench/data/cti-rcm.tsv"
CVE_CWE_INDEX = ROOT / "data/processed/cve_cwe_index.json"

CVE_RE = re.compile(r"CVE-\d{4}-\d+", re.IGNORECASE)


def extract_cve_id(url: str) -> str:
    m = CVE_RE.search(url)
    return m.group(0).upper() if m else ""


_CWE_NUM_RE = re.compile(r"CWE[-\s_:]*0*(\d+)", re.IGNORECASE)


def cwe_in_answer(answer: str, gt_cwe: str) -> bool:
    """Match the RAG-eval scoring (substring-anywhere) while tolerating zero-padding.

    eval_rcm.py uses ``gt_cwe.upper() in answer.upper()`` -- substring match anywhere.
    Some reasoning models (DeepSeek-R1-Distill, Phi-4-mini-reasoning, FSec-R) emit
    "CWE-089" instead of "CWE-89", so a literal substring check misses correct answers.
    This function expands the RAG scorer with zero-padding tolerance: if any CWE-NNN
    pattern in the answer (any padding) matches the GT number, the answer passes.
    """
    m = re.search(r"\d+", gt_cwe)
    if not m:
        return False
    gt_num = int(m.group(0))
    nums = {int(x) for x in _CWE_NUM_RE.findall(answer)}
    return gt_num in nums


def load_cve_cwe_index() -> dict:
    with open(CVE_CWE_INDEX) as f:
        return json.load(f)


def call_llm(endpoint: str, model: str, user_msg: str, max_tokens: int, timeout: int = 600,
             chat_template_kwargs: dict | None = None) -> tuple[str, str | None]:
    """Returns (answer_text, error_or_None)."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": user_msg}],
        "temperature": 0.1,
        "max_tokens": max_tokens,
    }
    if chat_template_kwargs:
        payload["chat_template_kwargs"] = chat_template_kwargs
    try:
        r = requests.post(endpoint, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        msg = data["choices"][0]["message"]
        # Some thinking-model endpoints separate reasoning_content from content.
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        # Use whatever has the CWE — prefer content (final answer) then fall back.
        return (content if content else reasoning, None)
    except Exception as exc:
        return ("", f"{type(exc).__name__}: {exc}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("n", nargs="?", type=int, default=1000, help="number of queries to sample (default 1000)")
    p.add_argument("--nvd-mapped", action="store_true")
    p.add_argument("--nvd-unmapped", action="store_true")
    p.add_argument("--endpoint-a", default=os.environ.get("ZS_ENDPOINT_A", "http://localhost:8000/v1/chat/completions"))
    p.add_argument("--endpoint-b", default=os.environ.get("ZS_ENDPOINT_B", ""))
    p.add_argument("--model", default=os.environ.get("ZS_MODEL", "Qwen/Qwen2.5-7B-Instruct"))
    p.add_argument("--workers", type=int, default=int(os.environ.get("ZS_WORKERS", "16")))
    p.add_argument("--max-tokens", type=int, default=int(os.environ.get("ZS_MAX_TOKENS", "256")))
    p.add_argument("--no-think", action="store_true", default=os.environ.get("ZS_NO_THINK", "") == "1",
                   help="Append '/no_think' to the user message (Qwen3 family)")
    p.add_argument("--enable-thinking", action="store_true",
                   default=os.environ.get("ZS_ENABLE_THINKING", "") == "1",
                   help="Pass chat_template_kwargs={'enable_thinking': true} per request (Gemma 4)")
    p.add_argument("--log", default="", help="optional path to write a streaming jsonl of per-query results")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)

    endpoints = [args.endpoint_a]
    if args.endpoint_b:
        endpoints.append(args.endpoint_b)

    # Load rows
    rows = []
    with open(RCM_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            rows.append(row)

    if args.nvd_mapped or args.nvd_unmapped:
        idx = load_cve_cwe_index()
        if args.nvd_mapped:
            rows = [r for r in rows if bool(idx.get(extract_cve_id(r["URL"])))]
            subset_label = "NVD-mapped"
        else:
            rows = [r for r in rows if not bool(idx.get(extract_cve_id(r["URL"])))]
            subset_label = "NVD-unmapped"
    else:
        subset_label = "all"

    sample = random.sample(rows, min(args.n, len(rows)))
    n = len(sample)

    print(f"Zero-shot eval ({subset_label}) on {n} queries")
    print(f"  endpoints={endpoints} model={args.model} workers={args.workers} max_tokens={args.max_tokens} no_think={args.no_think}")

    log_f = open(args.log, "w", encoding="utf-8") if args.log else None

    def _run_one(i: int, row: dict) -> tuple[int, dict]:
        cve_id = extract_cve_id(row["URL"])
        gt = row["GT"].strip()
        prompt = row["Prompt"]
        if args.no_think:
            prompt = prompt + " /no_think"
        endpoint = random.choice(endpoints)
        ctk = {"enable_thinking": True} if args.enable_thinking else None
        t0 = time.time()
        answer, err = call_llm(endpoint, args.model, prompt, args.max_tokens, chat_template_kwargs=ctk)
        dt = time.time() - t0
        passed = cwe_in_answer(answer, gt) if not err else False
        rec = {
            "i": i,
            "cve_id": cve_id,
            "gt": gt,
            "passed": passed,
            "error": err,
            "dt": round(dt, 3),
            "endpoint": endpoint.split("//")[-1].split("/")[0],
            "answer_tail": (answer or "")[-600:],
            "answer_len": len(answer or ""),
        }
        return i, rec

    results = [None] * n
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_run_one, i, row) for i, row in enumerate(sample)]
        for fut in futures:
            i, rec = fut.result()
            results[i] = rec
            mark = "PASS" if rec["passed"] else ("ERR " if rec["error"] else "FAIL")
            ans_preview = (rec.get("answer_tail") or "")[-90:].replace("\n", " ")
            print(f"[{i+1}/{n}] {rec['cve_id']:18s} GT={rec['gt']:8s} {mark} | {ans_preview}")
            if log_f:
                log_f.write(json.dumps(rec) + "\n")
                log_f.flush()

    t_total = time.time() - t_start

    passed = sum(1 for r in results if r and r["passed"])
    errors = sum(1 for r in results if r and r["error"])
    nvd_idx = load_cve_cwe_index()
    mapped_pass = sum(1 for r in results if r and r["passed"] and bool(nvd_idx.get(r["cve_id"])))
    mapped_total = sum(1 for r in results if r and bool(nvd_idx.get(r["cve_id"])))
    unmapped_pass = sum(1 for r in results if r and r["passed"] and not bool(nvd_idx.get(r["cve_id"])))
    unmapped_total = sum(1 for r in results if r and not bool(nvd_idx.get(r["cve_id"])))

    print("\n" + "=" * 60)
    print(f"Zero-shot CTI-RCM Results ({subset_label}, n={n}, seed={args.seed})")
    print("=" * 60)
    print(f"  Passed  : {passed}/{n}")
    print(f"  Accuracy: {100.0 * passed / n:.1f}%")
    print(f"  Errors  : {errors}")
    if mapped_total > 0:
        print(f"  Mapped  : {mapped_pass}/{mapped_total} ({100.0 * mapped_pass / mapped_total:.1f}%)")
    if unmapped_total > 0:
        print(f"  Unmapped: {unmapped_pass}/{unmapped_total} ({100.0 * unmapped_pass / unmapped_total:.1f}%)")
    print(f"  Wall    : {t_total:.1f}s")

    if log_f:
        log_f.close()


if __name__ == "__main__":
    main()
