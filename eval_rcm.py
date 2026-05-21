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
    conda run -n cyber-ft python3 eval_rcm.py 100 --timing-audit  # write query/request timing JSONL
    CTI_RAG_WANDB=1 conda run -n cyber-ft python3 eval_rcm.py 100 --timing-audit
"""
import csv
import itertools
import json
import os
import random
import re
import sys
import time
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
TIMING_AUDIT = (
    "--timing-audit" in sys.argv
    or os.environ.get("CTI_RAG_TIMING_AUDIT", "0").lower() in {"1", "true", "yes", "on"}
)
TIMING_RUN_ID = getattr(better_rag, "TIMING_RUN_ID", time.strftime("%Y%m%d_%H%M%S"))
EVAL_TIMING_PATH = Path(os.environ.get("CTI_RAG_EVAL_TIMING_LOG", f"logs/eval_timing_{TIMING_RUN_ID}.jsonl"))
EVAL_TIMING_SUMMARY_PATH = Path(os.environ.get("CTI_RAG_EVAL_TIMING_SUMMARY", f"logs/eval_timing_{TIMING_RUN_ID}.md"))
STREAM_RESULTS = (
    TIMING_AUDIT
    or os.environ.get("CTI_RAG_EVAL_STREAM_RESULTS", "0").lower() in {"1", "true", "yes", "on"}
)
EVAL_RESULT_STREAM_PATH = Path(os.environ.get("CTI_RAG_EVAL_RESULT_LOG", f"logs/eval_results_{TIMING_RUN_ID}.jsonl"))
WANDB_ENABLED = (
    "--wandb" in sys.argv
    or os.environ.get("CTI_RAG_WANDB", "0").lower() in {"1", "true", "yes", "on"}
)
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "cyber-ft-cti-rag")
WANDB_RUN_NAME = os.environ.get("WANDB_RUN_NAME", f"cti-rcm-{TIMING_RUN_ID}")
WANDB_LOG_TABLE = os.environ.get("CTI_RAG_WANDB_TABLE", "0").lower() in {"1", "true", "yes", "on"}
WANDB_RUN = None


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


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((pct / 100) * (len(ordered) - 1))))
    return ordered[idx]


def timing_stats(values: list[float]) -> dict:
    values = [float(v) for v in values if v is not None]
    total = sum(values)
    return {
        "count": len(values),
        "sum": total,
        "avg": total / len(values) if values else 0.0,
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "max": max(values) if values else 0.0,
    }


def _stats_md(label: str, stats: dict) -> str:
    return (
        f"| {label} | {stats['count']} | {stats['sum']:.2f} | "
        f"{stats['avg']:.2f} | {stats['p50']:.2f} | {stats['p95']:.2f} | {stats['max']:.2f} |"
    )


def init_wandb(n: int, mode_label: str, seed: int):
    if not WANDB_ENABLED:
        return None
    try:
        import wandb
    except Exception as exc:
        print(f"wandb requested but unavailable: {exc}")
        return None
    config = {
        "benchmark": "CTI-Bench CTI-RCM",
        "sample_size": n,
        "mode": mode_label,
        "seed": seed,
        "workers": WORKERS,
        "timing_run_id": TIMING_RUN_ID,
        "timing_audit": TIMING_AUDIT,
        "rcm_only": getattr(better_rag, "RCM_ONLY", False),
        "llm_router_enabled": getattr(better_rag, "LLM_ROUTER_ENABLED", False),
        "hyde_route_enabled": getattr(better_rag, "LLM_HYDE_ROUTE_ENABLED", False),
        "mapped_endpoint": getattr(better_rag, "LLM_MAPPED_ENDPOINT", ""),
        "unmapped_endpoint": getattr(better_rag, "LLM_UNMAPPED_ENDPOINT", ""),
        "hyde_endpoint": getattr(better_rag, "LLM_HYDE_ENDPOINT", ""),
        "embedder_device": getattr(better_rag, "EMBEDDER_DEVICE", ""),
        "crossencoder_enabled": getattr(better_rag, "CWE_CROSSENCODER_ENABLED", False),
        "hyde_enabled": getattr(better_rag, "CWE_HYDE_ENABLED", False),
        "hyde_response_budget": getattr(better_rag, "LLM_HYDE_RESPONSE_BUDGET", 0),
        "hyde_skip_mapped_bridge": getattr(better_rag, "CWE_HYDE_SKIP_MAPPED_BRIDGE", False),
        "mapped_fast_context": getattr(better_rag, "CWE_MAPPED_FAST_CONTEXT", False),
        "phrase_selector_enabled": getattr(better_rag, "CWE_PHRASE_SELECTOR_ENABLED", False),
    }
    run = wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config=config,
        tags=["cti-rcm", "routed" if config["llm_router_enabled"] else "single-llm", mode_label.replace(" ", "-")],
    )
    print(f"wandb run: {run.url}")
    return run


def log_wandb_step(run, result: dict, completed: int, total: int, counters: dict) -> None:
    if run is None:
        return
    timing = result.get("timing") or {}
    llm_calls = result.get("llm_calls") or []
    metrics = {
        "eval/completed": completed,
        "eval/total": total,
        "eval/progress": completed / total if total else 0.0,
        "eval/latest_passed": int(bool(result.get("passed"))),
        "eval/latest_nvd_mapped": int(bool(result.get("nvd_mapped"))),
        "accuracy/overall_so_far": counters["passed"] / completed if completed else 0.0,
        "accuracy/mapped_so_far": counters["mapped_pass"] / counters["mapped_seen"] if counters["mapped_seen"] else 0.0,
        "accuracy/unmapped_so_far": counters["unmapped_pass"] / counters["unmapped_seen"] if counters["unmapped_seen"] else 0.0,
    }
    for key, value in timing.items():
        if isinstance(value, (int, float)):
            metrics[f"timing/latest/{key}"] = value
    metrics["llm/latest/request_count"] = len(llm_calls)
    metrics["llm/latest/error_count"] = sum(1 for call in llm_calls if not call.get("success"))
    metrics["llm/latest/duration_sum_s"] = sum(float(call.get("duration_s") or 0.0) for call in llm_calls)
    for call in llm_calls:
        stage = str(call.get("stage") or "unknown")
        route = str(call.get("route") or "unknown")
        prefix = f"llm/latest/{stage}_{route}"
        metrics[f"{prefix}_duration_s"] = metrics.get(f"{prefix}_duration_s", 0.0) + float(call.get("duration_s") or 0.0)
        metrics[f"{prefix}_requests"] = metrics.get(f"{prefix}_requests", 0) + 1
        metrics[f"{prefix}_prompt_chars"] = metrics.get(f"{prefix}_prompt_chars", 0) + int(call.get("prompt_chars_sent") or 0)
        metrics[f"{prefix}_response_chars"] = metrics.get(f"{prefix}_response_chars", 0) + int(call.get("response_chars") or 0)
        if call.get("prompt_tokens") is not None:
            metrics[f"{prefix}_prompt_tokens"] = metrics.get(f"{prefix}_prompt_tokens", 0) + int(call.get("prompt_tokens") or 0)
        if call.get("completion_tokens") is not None:
            metrics[f"{prefix}_completion_tokens"] = metrics.get(f"{prefix}_completion_tokens", 0) + int(call.get("completion_tokens") or 0)
        if call.get("tokens_per_s") is not None:
            metrics[f"{prefix}_tokens_per_s"] = float(call.get("tokens_per_s") or 0.0)
    try:
        run.log(metrics, step=completed)
    except Exception as exc:
        print(f"wandb log failed: {exc}")


def finish_wandb(run, results: list[dict], acc: float, mapped_pass: int, mapped_total: int,
                 unmapped_pass: int, unmapped_total: int) -> None:
    if run is None:
        return
    summary = run.summary
    summary["final/accuracy"] = acc / 100.0
    summary["final/passed"] = sum(1 for r in results if r.get("passed"))
    summary["final/total"] = len(results)
    summary["final/mapped_accuracy"] = mapped_pass / mapped_total if mapped_total else 0.0
    summary["final/mapped_passed"] = mapped_pass
    summary["final/mapped_total"] = mapped_total
    summary["final/unmapped_accuracy"] = unmapped_pass / unmapped_total if unmapped_total else 0.0
    summary["final/unmapped_passed"] = unmapped_pass
    summary["final/unmapped_total"] = unmapped_total
    for key in ("total_s", "initial_retrieve_s", "hyde_total_s", "hyde_llm_s", "llm_s", "cwe_crossencoder_s"):
        values = [(r.get("timing") or {}).get(key, 0.0) for r in results]
        stats = timing_stats(values)
        summary[f"timing/{key}/avg"] = stats["avg"]
        summary[f"timing/{key}/p95"] = stats["p95"]
        summary[f"timing/{key}/max"] = stats["max"]

    if WANDB_LOG_TABLE:
        try:
            import wandb
            table = wandb.Table(columns=[
                "index", "cve_id", "gt_cwe", "passed", "nvd_mapped",
                "total_s", "initial_retrieve_s", "hyde_total_s", "llm_s",
                "llm_request_count", "llm_error_count",
            ])
            for i, result in enumerate(results, 1):
                timing = result.get("timing") or {}
                llm_calls = result.get("llm_calls") or []
                table.add_data(
                    i,
                    result.get("cve_id"),
                    result.get("gt_cwe"),
                    bool(result.get("passed")),
                    bool(result.get("nvd_mapped")),
                    timing.get("total_s", 0.0),
                    timing.get("initial_retrieve_s", 0.0),
                    timing.get("hyde_total_s", 0.0),
                    timing.get("llm_s", 0.0),
                    len(llm_calls),
                    sum(1 for call in llm_calls if not call.get("success")),
                )
            run.log({"eval/results_table": table})
        except Exception as exc:
            print(f"wandb table log failed: {exc}")
    run.finish()


def write_timing_audit(results: list[dict], mode_label: str, seed: int) -> None:
    EVAL_TIMING_PATH.parent.mkdir(parents=True, exist_ok=True)
    audit_rows = []
    for i, res in enumerate(results, 1):
        audit_rows.append({
            "index": i,
            "cve_id": res.get("cve_id"),
            "gt_cwe": res.get("gt_cwe"),
            "passed": res.get("passed"),
            "nvd_mapped": res.get("nvd_mapped"),
            "timing": res.get("timing", {}),
            "llm_route": res.get("llm_route", {}),
            "llm_calls": res.get("llm_calls", []),
            "error": res.get("error"),
        })
    with EVAL_TIMING_PATH.open("w", encoding="utf-8") as f:
        for row in audit_rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    stage_keys = [
        "total_s",
        "initial_retrieve_s",
        "hyde_total_s",
        "hyde_llm_s",
        "hyde_retrieve_s",
        "hyde_filter_s",
        "retrieve_s",
        "filters_s",
        "graph_s",
        "neighbors_s",
        "knn_s",
        "cwe_crossencoder_s",
        "post_neighbors_s",
        "prompt_build_s",
        "llm_s",
    ]
    groups = {
        "all": audit_rows,
        "mapped": [r for r in audit_rows if r.get("nvd_mapped")],
        "unmapped": [r for r in audit_rows if not r.get("nvd_mapped")],
    }
    lines = [
        "# CTI-RCM Timing Audit",
        "",
        f"- run_id: `{TIMING_RUN_ID}`",
        f"- mode: {mode_label}",
        f"- seed: {seed}",
        f"- workers: {WORKERS}",
        f"- query timing JSONL: `{EVAL_TIMING_PATH}`",
    ]
    if STREAM_RESULTS:
        lines.append(f"- streamed result JSONL: `{EVAL_RESULT_STREAM_PATH}`")
    lines.extend([
        f"- LLM request JSONL: `{getattr(better_rag, 'LLM_TIMING_LOG_PATH', '')}`",
        "",
    ])
    for group_name, rows in groups.items():
        if not rows:
            continue
        lines.extend([
            f"## Query Timings: {group_name}",
            "",
            "| stage | n | sum_s | avg_s | p50_s | p95_s | max_s |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for key in stage_keys:
            values = [(row.get("timing") or {}).get(key, 0.0) for row in rows]
            lines.append(_stats_md(key, timing_stats(values)))
        lines.append("")

    calls = [call for row in audit_rows for call in row.get("llm_calls", [])]
    if calls:
        buckets: dict[tuple[str, str, str], list[dict]] = {}
        for call in calls:
            bucket = (
                str(call.get("stage")),
                str(call.get("route")),
                str(call.get("endpoint")),
            )
            buckets.setdefault(bucket, []).append(call)
        lines.extend([
            "## LLM Request Timings",
            "",
            "| stage | route | endpoint | n | errors | sum_s | avg_s | p50_s | p95_s | max_s | avg_prompt_tokens | avg_completion_tokens | avg_completion_tok_s | avg_prompt_chars | avg_response_chars |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for (stage, route, endpoint), bucket_calls in sorted(buckets.items()):
            durations = [float(c.get("duration_s") or 0.0) for c in bucket_calls]
            stats = timing_stats(durations)
            errors = sum(1 for c in bucket_calls if not c.get("success"))
            avg_prompt = sum(int(c.get("prompt_chars_sent") or 0) for c in bucket_calls) / len(bucket_calls)
            avg_response = sum(int(c.get("response_chars") or 0) for c in bucket_calls) / len(bucket_calls)
            prompt_tokens = [int(c.get("prompt_tokens") or 0) for c in bucket_calls if c.get("prompt_tokens") is not None]
            completion_tokens = [int(c.get("completion_tokens") or 0) for c in bucket_calls if c.get("completion_tokens") is not None]
            tok_rates = [float(c.get("tokens_per_s") or 0.0) for c in bucket_calls if c.get("tokens_per_s") is not None]
            avg_prompt_tokens = sum(prompt_tokens) / len(prompt_tokens) if prompt_tokens else 0.0
            avg_completion_tokens = sum(completion_tokens) / len(completion_tokens) if completion_tokens else 0.0
            avg_tok_rate = sum(tok_rates) / len(tok_rates) if tok_rates else 0.0
            lines.append(
                f"| {stage} | {route} | `{endpoint}` | {stats['count']} | {errors} | "
                f"{stats['sum']:.2f} | {stats['avg']:.2f} | {stats['p50']:.2f} | "
                f"{stats['p95']:.2f} | {stats['max']:.2f} | {avg_prompt_tokens:.0f} | "
                f"{avg_completion_tokens:.0f} | {avg_tok_rate:.2f} | {avg_prompt:.0f} | {avg_response:.0f} |"
            )
        lines.append("")

    EVAL_TIMING_SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote timing audit JSONL: {EVAL_TIMING_PATH}")
    print(f"Wrote timing audit summary: {EVAL_TIMING_SUMMARY_PATH}")
    if TIMING_AUDIT:
        print(f"Wrote LLM request JSONL: {getattr(better_rag, 'LLM_TIMING_LOG_PATH', '')}")


def write_result_checkpoint(result: dict, index: int, completed: int) -> None:
    if not STREAM_RESULTS:
        return
    EVAL_RESULT_STREAM_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = dict(result)
    row["index"] = index + 1
    row["completed"] = completed
    with EVAL_RESULT_STREAM_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


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
    if STREAM_RESULTS:
        EVAL_RESULT_STREAM_PATH.parent.mkdir(parents=True, exist_ok=True)
        EVAL_RESULT_STREAM_PATH.write_text("", encoding="utf-8")
        print(f"Streaming per-query results to: {EVAL_RESULT_STREAM_PATH}")
    wandb_run = init_wandb(len(sample), mode_label, seed)

    def _run_one(i: int, row: dict) -> tuple[int, dict]:
        cve_id = extract_cve_id(row["URL"])
        gt_cwe = row["GT"].strip()
        query  = row["Prompt"]  # same prompt format used to test GPT-4 in CTI-Bench paper
        nvd_mapped = bool(cve_cwe_index.get(cve_id))
        debug_info = {} if (debug_failures or TIMING_AUDIT or WANDB_ENABLED) else None
        try:
            route_hint = "mapped" if nvd_mapped else "unmapped"
            answer, _, _ = better_rag.ask(
                query,
                [],
                eval_mode=True,
                debug_info=debug_info,
                llm_route_hint=route_hint,
            )
            passed = cwe_in_answer(answer, gt_cwe)
            result = {"cve_id": cve_id, "gt_cwe": gt_cwe, "passed": passed, "answer": answer, "nvd_mapped": nvd_mapped}
            if (TIMING_AUDIT or WANDB_ENABLED) and debug_info is not None:
                result["timing"] = debug_info.get("timing", {})
                result["llm_calls"] = debug_info.get("llm_calls", [])
                result["llm_route"] = debug_info.get("llm_route", {})
            if debug_failures and not passed:
                result["debug"] = debug_info
                result["prompt"] = query
            return i, result
        except Exception as e:
            result = {"cve_id": cve_id, "gt_cwe": gt_cwe, "passed": False, "error": str(e), "nvd_mapped": nvd_mapped}
            if (TIMING_AUDIT or WANDB_ENABLED) and debug_info is not None:
                result["timing"] = debug_info.get("timing", {})
                result["llm_calls"] = debug_info.get("llm_calls", [])
                result["llm_route"] = debug_info.get("llm_route", {})
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
    counters = {"passed": 0, "mapped_seen": 0, "mapped_pass": 0, "unmapped_seen": 0, "unmapped_pass": 0}
    with ThreadPoolExecutor(max_workers=WORKERS, initializer=_thread_init) as executor:
        futures = {executor.submit(_run_one, i, row): i for i, row in enumerate(sample)}
        for future in as_completed(futures):
            i, result = future.result()
            results[i] = result
            completed += 1
            if result["passed"]:
                counters["passed"] += 1
            if result.get("nvd_mapped"):
                counters["mapped_seen"] += 1
                if result["passed"]:
                    counters["mapped_pass"] += 1
            else:
                counters["unmapped_seen"] += 1
                if result["passed"]:
                    counters["unmapped_pass"] += 1
            cve_id = result["cve_id"]
            gt_cwe = result["gt_cwe"]
            status = "PASS" if result["passed"] else "FAIL"
            snippet = result.get("answer", result.get("error", ""))[:120].replace("\n", " ")
            print(f"[{completed}/{len(sample)}] {cve_id}  GT: {gt_cwe}  {status} | {snippet}")
            write_result_checkpoint(result, i, completed)
            log_wandb_step(wandb_run, result, completed, len(sample), counters)

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

    if TIMING_AUDIT:
        write_timing_audit(results, mode_label, seed)

    finish_wandb(
        wandb_run,
        results,
        acc,
        mapped_pass,
        len(mapped_results),
        unmapped_pass,
        len(unmapped_results),
    )

    return results, acc


if __name__ == "__main__":
    args = sys.argv[1:]
    matched_only  = "--matched"      in args
    local_only    = "--local"        in args
    nvd_mapped    = "--nvd-mapped"   in args
    nvd_unmapped  = "--nvd-unmapped" in args
    debug_failures = "--debug-failures" in args
    profile       = "--profile"      in args
    timing_audit  = "--timing-audit" in args or TIMING_AUDIT
    wandb_enabled = "--wandb" in args or WANDB_ENABLED
    args = [a for a in args if a not in ("--matched", "--local", "--nvd-mapped", "--nvd-unmapped", "--debug-failures", "--profile", "--timing-audit", "--wandb")]
    n = int(args[0]) if args else SAMPLE_N
    if profile:
        print("Profiling enabled: retrieval/timing checkpoints will be printed.")
    if timing_audit:
        print("Timing audit enabled: query and LLM request timings will be written under logs/.")
    if wandb_enabled:
        print(f"wandb enabled: project={WANDB_PROJECT} run={WANDB_RUN_NAME}")
    run_eval(n, matched_only=matched_only, local_only=local_only, nvd_mapped=nvd_mapped,
             nvd_unmapped=nvd_unmapped, debug_failures=debug_failures)
