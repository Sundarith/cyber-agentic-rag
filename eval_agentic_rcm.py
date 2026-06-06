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
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from agentic_rag import (
    DEFAULT_GRANITE_MODEL,
    AgenticRAG,
    AgenticRAGConfig,
    CorpusIndex,
    GraniteGroundedSynthesizer,
    OpenAIChatClient,
    build_search_backend,
)
from agentic_rag.planner import QueryPlanner
from agentic_rag.schema import Action, ActionType, AgentState, Evidence, Verification
from agentic_rag.synthesizer import ExtractiveSynthesizer
from agentic_rag.verifier import EvidenceVerifier

DEFAULT_DATASET = Path("data/cti-bench/data/cti-rcm.tsv")
DEFAULT_CVE_CWE_INDEX = Path("data/processed/cve_cwe_index.json")
DEFAULT_TRACE_LOG = Path("logs/agentic_rcm_eval.jsonl")
SEED = 42
EVAL_MODES = (
    "lexical_baseline",
    "agentic_lexical",
    "agentic_lexical_no_verifier",
    "agentic_lexical_no_graph",
)


class NoGraphPlanner(QueryPlanner):
    """Planner variant for ablations that removes graph-expansion actions."""

    def initial_actions(self, question: str, retrieve_k: int) -> list[Action]:
        return [action for action in super().initial_actions(question, retrieve_k) if action.kind != ActionType.EXPAND]

    def next_actions(self, state: AgentState, verification: Verification, retrieve_k: int) -> list[Action]:
        return [action for action in super().next_actions(state, verification, retrieve_k) if action.kind != ActionType.EXPAND]


class NoVerifier(EvidenceVerifier):
    """Verifier ablation that never stops the agent early."""

    def verify(self, question: str, evidence: list[Evidence]) -> Verification:
        cited_ids = []
        seen: set[str] = set()
        for ev in evidence:
            if ev.identifier and ev.identifier not in seen:
                cited_ids.append(ev.identifier)
                seen.add(ev.identifier)
        return Verification(
            supported=False,
            confidence=0.0,
            missing=["verifier_disabled"],
            cited_ids=cited_ids,
        )


def extract_cve_id(text: str) -> str:
    match = re.search(r"\bCVE-\d{4}-\d+\b", text, re.IGNORECASE)
    return match.group(0).upper() if match else ""


def answer_contains_cwe(answer: str, gt_cwe: str) -> bool:
    gt = gt_cwe.strip().upper()
    if not gt:
        return False
    return bool(re.search(rf"\b{re.escape(gt)}\b", answer.upper()))


def extract_cwe_ids(text: str) -> list[str]:
    seen: set[str] = set()
    ids = []
    for match in re.findall(r"\bCWE-\d+\b", text, re.IGNORECASE):
        cwe_id = match.upper()
        if cwe_id not in seen:
            ids.append(cwe_id)
            seen.add(cwe_id)
    return ids


def retrieval_backend_for_mode(mode: str) -> str:
    return "lexical"


def build_synthesizer(args: argparse.Namespace) -> ExtractiveSynthesizer | GraniteGroundedSynthesizer:
    if args.synthesizer == "extractive":
        return ExtractiveSynthesizer()
    if args.synthesizer == "granite":
        return GraniteGroundedSynthesizer(
            OpenAIChatClient(
                endpoint=args.llm_endpoint,
                model=args.llm_model,
                max_tokens=args.llm_max_tokens,
                timeout_s=args.llm_timeout_s,
            )
        )
    raise ValueError(f"Unknown synthesizer: {args.synthesizer}")


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


def build_agent(corpus: CorpusIndex, mode: str, args: argparse.Namespace) -> AgenticRAG:
    planner: QueryPlanner | None = None
    verifier: EvidenceVerifier | None = None
    if mode == "agentic_lexical_no_graph":
        planner = NoGraphPlanner()
    if mode == "agentic_lexical_no_verifier":
        verifier = NoVerifier()
    search_backend = build_search_backend(corpus, retrieval_backend_for_mode(mode))
    return AgenticRAG(
        corpus,
        AgenticRAGConfig(
            retrieve_k=args.retrieve_k,
            max_steps=args.max_steps,
            evidence_budget=args.evidence_budget,
        ),
        planner=planner,
        verifier=verifier,
        search_backend=search_backend,
        synthesizer=build_synthesizer(args),
    )


def run_lexical_baseline(corpus: CorpusIndex, question: str, args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    evidence = corpus.search(question, k=args.evidence_budget)
    verifier = EvidenceVerifier()
    verification = verifier.verify(question, evidence)
    answer = build_synthesizer(args).synthesize(question, evidence, verification)
    elapsed = time.perf_counter() - start
    trace = [
        {
            "step": 1,
            "action": "search",
            "value": question,
            "rationale": "single-pass lexical baseline",
            "new_evidence": [ev.identifier for ev in evidence],
            "new_evidence_count": len(evidence),
            "supported": verification.supported,
            "confidence": verification.confidence,
            "missing": verification.missing,
        }
    ]
    return {
        "question": question,
        "answer": answer,
        "verification": verification,
        "evidence": evidence,
        "trace": trace,
        "duration_s": elapsed,
        "retrieval_backend": "lexical",
        "synthesizer": args.synthesizer,
        "llm_model": args.llm_model if args.synthesizer == "granite" else "",
    }


def run_mode(
    corpus: CorpusIndex,
    mode: str,
    row: dict[str, str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    if mode == "lexical_baseline":
        return run_lexical_baseline(corpus, row["Prompt"], args)
    agent = build_agent(corpus, mode, args)
    start = time.perf_counter()
    result = agent.answer(row["Prompt"])
    result["duration_s"] = time.perf_counter() - start
    result["retrieval_backend"] = retrieval_backend_for_mode(mode)
    result["synthesizer"] = args.synthesizer
    result["llm_model"] = args.llm_model if args.synthesizer == "granite" else ""
    return result


def run_one(
    corpus: CorpusIndex,
    mode: str,
    row: dict[str, str],
    cve_cwe_index: dict[str, list[str]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    cve_id = extract_cve_id(row["URL"])
    gt_cwe = row["GT"].strip().upper()
    result = run_mode(corpus, mode, row, args)
    answer = result["answer"]
    passed = answer_contains_cwe(answer, gt_cwe)
    trace = result["trace"]
    evidence = result["evidence"]
    evidence_identifiers = [ev.identifier for ev in evidence]
    trace_actions = Counter(str(step.get("action") or "") for step in trace)
    evidence_tools = Counter(ev.tool for ev in evidence)
    verifier_rejections = sum(1 for step in trace if not step.get("supported"))
    gold_in_evidence = any(
        gt_cwe == ev.identifier.upper() or gt_cwe in ev.chunk.text.upper()
        for ev in evidence
    )
    predicted_cwes = extract_cwe_ids(answer)
    failure_reason = classify_failure(
        passed=passed,
        evidence=evidence,
        gold_in_evidence=gold_in_evidence,
        predicted_cwes=predicted_cwes,
        verification=result["verification"],
    )
    return {
        "mode": mode,
        "retrieval_backend": result.get("retrieval_backend", retrieval_backend_for_mode(mode)),
        "synthesizer": result.get("synthesizer", args.synthesizer),
        "llm_model": result.get("llm_model", args.llm_model if args.synthesizer == "granite" else ""),
        "cve_id": cve_id,
        "gt_cwe": gt_cwe,
        "predicted_cwes": predicted_cwes,
        "passed": passed,
        "nvd_mapped": bool(cve_cwe_index.get(cve_id)),
        "answer": answer,
        "duration_s": result["duration_s"],
        "tool_calls": len(trace),
        "trace_actions": dict(trace_actions),
        "evidence_count": len(evidence),
        "evidence_identifiers": evidence_identifiers,
        "evidence_tools": dict(evidence_tools),
        "gold_in_evidence": gold_in_evidence,
        "verifier_rejections": verifier_rejections,
        "graph_expansions": trace_actions.get("expand", 0),
        "failure_reason": failure_reason,
        "verification": asdict(result["verification"]),
        "trace": trace,
        "evidence": serialize_result(result)["evidence"],
    }


def classify_failure(
    passed: bool,
    evidence: list[Evidence],
    gold_in_evidence: bool,
    predicted_cwes: list[str],
    verification: Verification,
) -> str:
    if passed:
        return ""
    if not evidence:
        return "no_evidence"
    if not gold_in_evidence:
        return "gold_not_in_evidence"
    if not verification.supported:
        return "unsupported_evidence"
    if predicted_cwes:
        return "wrong_cwe"
    return "gold_in_evidence_answer_missing"


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
    evidence_counts = [int(row.get("evidence_count") or 0) for row in results]
    verifier_rejections = [int(row.get("verifier_rejections") or 0) for row in results]
    graph_expansions = [int(row.get("graph_expansions") or 0) for row in results]
    gold_evidence_hits = sum(1 for row in results if row.get("gold_in_evidence"))
    failure_reasons = Counter(str(row.get("failure_reason") or "") for row in results if not row["passed"])
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
        "avg_evidence_count": sum(evidence_counts) / len(evidence_counts) if evidence_counts else 0.0,
        "gold_evidence_hits": gold_evidence_hits,
        "gold_evidence_recall": gold_evidence_hits / total if total else 0.0,
        "avg_verifier_rejections": (
            sum(verifier_rejections) / len(verifier_rejections) if verifier_rejections else 0.0
        ),
        "avg_graph_expansions": sum(graph_expansions) / len(graph_expansions) if graph_expansions else 0.0,
        "failure_reasons": dict(failure_reasons),
    }


def summarize_by_mode(results: list[dict[str, Any]]) -> dict[str, Any]:
    modes = sorted({str(row.get("mode") or "unknown") for row in results})
    return {mode: summarize([row for row in results if row.get("mode") == mode]) for mode in modes}


def run_eval(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = load_rows(args.dataset)
    cve_cwe_index_path = resolve_data_path(args.root, args.cve_cwe_index, DEFAULT_CVE_CWE_INDEX)
    cve_cwe_index = load_cve_cwe_index(cve_cwe_index_path)
    rows = filter_rows(rows, cve_cwe_index, args.nvd_mapped, args.nvd_unmapped)
    rows = sample_rows(rows, args.n, args.seed)

    corpus = CorpusIndex.default(args.root)

    results = []
    for row_index, row in enumerate(rows, 1):
        for mode in args.modes:
            result = run_one(corpus, mode, row, cve_cwe_index, args)
            result["index"] = row_index
            results.append(result)
            status = "PASS" if result["passed"] else "FAIL"
            print(
                f"[{row_index}/{len(rows)}] {mode} {result['cve_id']} GT={result['gt_cwe']} {status} "
                f"tools={result['tool_calls']} evidence={result['evidence_count']} "
                f"gold_evidence={result['gold_in_evidence']} time={result['duration_s']:.3f}s"
            )
            if args.debug_failures and not result["passed"]:
                print(result["answer"][:1000].replace("\n", " "))

    summary = summarize_by_mode(results) if len(args.modes) > 1 else summarize(results)
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
    parser.add_argument(
        "--synthesizer",
        choices=["extractive", "granite"],
        default="extractive",
        help="Answer backend. granite calls an OpenAI-compatible local endpoint.",
    )
    parser.add_argument("--llm-endpoint", default="http://localhost:8000/v1/chat/completions")
    parser.add_argument("--llm-model", default=DEFAULT_GRANITE_MODEL)
    parser.add_argument("--llm-max-tokens", type=int, default=256)
    parser.add_argument("--llm-timeout-s", type=float, default=120.0)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=EVAL_MODES,
        default=["agentic_lexical"],
        help="Evaluation modes to run. Multiple modes are written to one trace log with a mode field.",
    )
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
    print_summary(summary)
    print(f"  Trace log: {args.trace_log}")
    return 0


def print_summary(summary: dict[str, Any]) -> None:
    if "accuracy" in summary:
        print_one_summary(summary, indent="  ")
        return
    for mode, mode_summary in summary.items():
        print(f"  {mode}:")
        print_one_summary(mode_summary, indent="    ")


def print_one_summary(summary: dict[str, Any], indent: str = "  ") -> None:
    print(f"{indent}Passed: {summary['passed']}/{summary['total']} ({summary['accuracy'] * 100:.1f}%)")
    if summary["mapped_total"]:
        mapped_acc = summary["mapped_passed"] / summary["mapped_total"]
        print(f"{indent}NVD-mapped: {summary['mapped_passed']}/{summary['mapped_total']} ({mapped_acc * 100:.1f}%)")
    if summary["unmapped_total"]:
        unmapped_acc = summary["unmapped_passed"] / summary["unmapped_total"]
        print(f"{indent}NVD-unmapped: {summary['unmapped_passed']}/{summary['unmapped_total']} ({unmapped_acc * 100:.1f}%)")
    print(f"{indent}Gold evidence recall: {summary['gold_evidence_recall'] * 100:.1f}%")
    print(f"{indent}Avg tool calls: {summary['avg_tool_calls']:.2f}")
    print(f"{indent}Avg evidence: {summary['avg_evidence_count']:.2f}")
    print(f"{indent}Avg verifier rejections: {summary['avg_verifier_rejections']:.2f}")
    print(f"{indent}Avg graph expansions: {summary['avg_graph_expansions']:.2f}")
    print(f"{indent}Avg duration: {summary['avg_duration_s']:.3f}s")
    if summary["failure_reasons"]:
        print(f"{indent}Failure reasons: {summary['failure_reasons']}")


if __name__ == "__main__":
    raise SystemExit(main())
