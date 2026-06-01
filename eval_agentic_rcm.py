"""
Evaluate the agentic RAG prototype on CTI-Bench CTI-RCM.

This harness is deliberately independent from eval_rcm.py and better_rag.py so
the new agent can be measured before it depends on vLLM or the legacy retriever.

Examples:
    python3 eval_agentic_rcm.py 100
    python3 eval_agentic_rcm.py 97 --nvd-unmapped --trace-log logs/agentic_unmapped.jsonl
    python3 eval_agentic_rcm.py --dataset data/cti-bench/data/cti-rcm.tsv --debug-failures
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from agentic_rag import AgenticRAG, AgenticRAGConfig, CorpusIndex

DEFAULT_DATASET = Path("data/cti-bench/data/cti-rcm.tsv")
DEFAULT_CVE_CWE_INDEX = Path("data/processed/cve_cwe_index.json")
DEFAULT_TRACE_LOG = Path("logs/agentic_rcm_eval.jsonl")
SEED = 42


def extract_cve_id(text: str) -> str:
    match = re.search(r"\bCVE-\d{4}-\d+\b", text, re.IGNORECASE)
    return match.group(0).upper() if match else ""


def answer_contains_cwe(answer: str, gt_cwe: str) -> bool:
    gt = gt_cwe.strip().upper()
    if not gt:
        return False
    return bool(re.search(rf"\b{re.escape(gt)}\b", answer.upper()))


def load_rows(dataset: Path) -> list[dict[str, str]]:
    if not dataset.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset}. Clone CTI-Bench under data/cti-bench "
            "or pass --dataset /path/to/cti-rcm.tsv."
        )
    with dataset.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    required = {"URL", "GT", "Prompt"}
    missing = required - set(rows[0].keys() if rows else [])
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    return rows


def load_cve_cwe_index(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        return {}
    return {str(k).upper(): [str(v).upper() for v in vals] for k, vals in raw.items()}


def filter_rows(
    rows: list[dict[str, str]],
    cve_cwe_index: dict[str, list[str]],
    nvd_mapped: bool = False,
    nvd_unmapped: bool = False,
) -> list[dict[str, str]]:
    if nvd_mapped and nvd_unmapped:
        raise ValueError("--nvd-mapped and --nvd-unmapped are mutually exclusive")
    if nvd_mapped:
        return [row for row in rows if cve_cwe_index.get(extract_cve_id(row["URL"]))]
    if nvd_unmapped:
        return [row for row in rows if not cve_cwe_index.get(extract_cve_id(row["URL"]))]
    return rows


def sample_rows(rows: list[dict[str, str]], n: int | None, seed: int) -> list[dict[str, str]]:
    if n is None or n >= len(rows):
        return list(rows)
    rng = random.Random(seed)
    return rng.sample(rows, n)


def serialize_result(result: dict[str, Any]) -> dict[str, Any]:
    verification = result["verification"]
    return {
        "question": result["question"],
        "answer": result["answer"],
        "verification": asdict(verification),
        "trace": result["trace"],
        "evidence": [
            {
                "identifier": ev.identifier,
                "source": ev.chunk.source,
                "type": ev.chunk.type,
                "name": ev.chunk.name,
                "score": ev.score,
                "tool": ev.tool,
                "reason": ev.reason,
            }
            for ev in result["evidence"]
        ],
    }


def run_one(agent: AgenticRAG, row: dict[str, str], cve_cwe_index: dict[str, list[str]]) -> dict[str, Any]:
    cve_id = extract_cve_id(row["URL"])
    gt_cwe = row["GT"].strip().upper()
    start = time.perf_counter()
    result = agent.answer(row["Prompt"])
    elapsed = time.perf_counter() - start
    answer = result["answer"]
    passed = answer_contains_cwe(answer, gt_cwe)
    trace = result["trace"]
    evidence = result["evidence"]
    return {
        "cve_id": cve_id,
        "gt_cwe": gt_cwe,
        "passed": passed,
        "nvd_mapped": bool(cve_cwe_index.get(cve_id)),
        "answer": answer,
        "duration_s": elapsed,
        "tool_calls": len(trace),
        "evidence_count": len(evidence),
        "verification": asdict(result["verification"]),
        "trace": trace,
        "evidence": serialize_result(result)["evidence"],
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    passed = sum(1 for row in results if row["passed"])
    mapped = [row for row in results if row.get("nvd_mapped")]
    unmapped = [row for row in results if not row.get("nvd_mapped")]
    durations = [float(row.get("duration_s") or 0.0) for row in results]
    tool_calls = [int(row.get("tool_calls") or 0) for row in results]
    return {
        "total": total,
        "passed": passed,
        "accuracy": passed / total if total else 0.0,
        "mapped_total": len(mapped),
        "mapped_passed": sum(1 for row in mapped if row["passed"]),
        "unmapped_total": len(unmapped),
        "unmapped_passed": sum(1 for row in unmapped if row["passed"]),
        "avg_duration_s": sum(durations) / len(durations) if durations else 0.0,
        "avg_tool_calls": sum(tool_calls) / len(tool_calls) if tool_calls else 0.0,
    }


def run_eval(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = load_rows(args.dataset)
    cve_cwe_index_path = resolve_data_path(args.root, args.cve_cwe_index, DEFAULT_CVE_CWE_INDEX)
    cve_cwe_index = load_cve_cwe_index(cve_cwe_index_path)
    rows = filter_rows(rows, cve_cwe_index, args.nvd_mapped, args.nvd_unmapped)
    rows = sample_rows(rows, args.n, args.seed)

    corpus = CorpusIndex.default(args.root)
    agent = AgenticRAG(
        corpus,
        AgenticRAGConfig(
            retrieve_k=args.retrieve_k,
            max_steps=args.max_steps,
            evidence_budget=args.evidence_budget,
        ),
    )

    results = []
    for index, row in enumerate(rows, 1):
        result = run_one(agent, row, cve_cwe_index)
        result["index"] = index
        results.append(result)
        status = "PASS" if result["passed"] else "FAIL"
        print(
            f"[{index}/{len(rows)}] {result['cve_id']} GT={result['gt_cwe']} {status} "
            f"tools={result['tool_calls']} time={result['duration_s']:.3f}s"
        )
        if args.debug_failures and not result["passed"]:
            print(result["answer"][:1000].replace("\n", " "))

    summary = summarize(results)
    write_jsonl(args.trace_log, results)
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return results, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate agentic RAG on CTI-RCM.")
    parser.add_argument("n", nargs="?", type=int, default=None, help="Sample size. Defaults to all rows.")
    parser.add_argument("--root", type=Path, default=Path("."), help="Repository/data root.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--cve-cwe-index", type=Path, default=DEFAULT_CVE_CWE_INDEX)
    parser.add_argument("--trace-log", type=Path, default=DEFAULT_TRACE_LOG)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--retrieve-k", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--evidence-budget", type=int, default=14)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--nvd-mapped", action="store_true")
    group.add_argument("--nvd-unmapped", action="store_true")
    parser.add_argument("--debug-failures", action="store_true")
    return parser


def resolve_data_path(root: Path, value: Path, default_value: Path) -> Path:
    if value.is_absolute() or value != default_value:
        return value
    return root / value


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    results, summary = run_eval(args)
    print("\nAgentic CTI-RCM summary")
    print(f"  Passed: {summary['passed']}/{summary['total']} ({summary['accuracy'] * 100:.1f}%)")
    if summary["mapped_total"]:
        mapped_acc = summary["mapped_passed"] / summary["mapped_total"]
        print(f"  NVD-mapped: {summary['mapped_passed']}/{summary['mapped_total']} ({mapped_acc * 100:.1f}%)")
    if summary["unmapped_total"]:
        unmapped_acc = summary["unmapped_passed"] / summary["unmapped_total"]
        print(f"  NVD-unmapped: {summary['unmapped_passed']}/{summary['unmapped_total']} ({unmapped_acc * 100:.1f}%)")
    print(f"  Avg tool calls: {summary['avg_tool_calls']:.2f}")
    print(f"  Avg duration: {summary['avg_duration_s']:.3f}s")
    print(f"  Trace log: {args.trace_log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
