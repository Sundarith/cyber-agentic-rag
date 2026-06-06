"""
Evaluate the agentic RAG prototype on SecEval multiple-choice questions.

SecEval is broad cybersecurity QA rather than CTI root-cause mapping. This
harness keeps retrieval traces explicit and scores exact answer-letter sets,
including multi-select answers such as "AC" or "BD".

Examples:
    python3 eval_seceval.py 25
    python3 eval_seceval.py 25 --modes lexical_baseline agentic_lexical
    python3 eval_seceval.py 10 --selector granite --llm-model ibm-granite/granite-4.1-8b
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from agentic_rag import (
    DEFAULT_GRANITE_MODEL,
    AgenticRAG,
    AgenticRAGConfig,
    CorpusIndex,
    OpenAIChatClient,
    build_search_backend,
)
from agentic_rag.planner import QueryPlanner
from agentic_rag.schema import Action, ActionType, AgentState, Evidence, Verification
from agentic_rag.synthesizer import _truncate
from agentic_rag.verifier import EvidenceVerifier

DEFAULT_DATASET = Path("data/seceval/questions.json")
DEFAULT_TRACE_LOG = Path("logs/seceval_eval.jsonl")
SEED = 42
EVAL_MODES = (
    "lexical_baseline",
    "agentic_lexical",
    "agentic_lexical_no_verifier",
    "agentic_lexical_no_graph",
    "agentic_hybrid",
    "agentic_hybrid_no_verifier",
    "agentic_hybrid_no_graph",
)
ANSWER_RE = re.compile(r"\b[ABCD]{1,4}\b", re.IGNORECASE)
TOKEN_RE = re.compile(r"[a-zA-Z0-9_.-]+")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "because", "but", "by", "can",
    "could", "for", "from", "in", "is", "it", "of", "on", "or", "should",
    "that", "the", "this", "to", "with", "which", "why", "you",
}


class NoGraphPlanner(QueryPlanner):
    def initial_actions(self, question: str, retrieve_k: int) -> list[Action]:
        return [action for action in super().initial_actions(question, retrieve_k) if action.kind != ActionType.EXPAND]

    def next_actions(self, state: AgentState, verification: Verification, retrieve_k: int) -> list[Action]:
        return [action for action in super().next_actions(state, verification, retrieve_k) if action.kind != ActionType.EXPAND]


class NoVerifier(EvidenceVerifier):
    def verify(self, question: str, evidence: list[Evidence]) -> Verification:
        cited_ids = []
        seen: set[str] = set()
        for ev in evidence:
            if ev.identifier and ev.identifier not in seen:
                cited_ids.append(ev.identifier)
                seen.add(ev.identifier)
        return Verification(False, 0.0, ["verifier_disabled"], cited_ids)


def normalize_answer(value: str) -> str:
    letters = [ch for ch in str(value).upper() if ch in "ABCD"]
    seen: set[str] = set()
    normalized = []
    for label in letters:
        if label not in seen:
            normalized.append(label)
            seen.add(label)
    return "".join(label for label in "ABCD" if label in normalized)


def extract_answer_letters(text: str) -> str:
    for line in reversed(str(text).splitlines()):
        cleaned = re.sub(r"[^A-D]", "", line.upper())
        stripped = re.sub(r"[^A-Z]", "", line.upper())
        if stripped == cleaned and re.fullmatch(r"[ABCD]{1,4}", cleaned or ""):
            return normalize_answer(cleaned)
        label_match = re.search(r"\b(?:FINAL|ANSWER|OUTPUT)\s*[:=-]\s*([ABCD]{1,4})\b", line, re.IGNORECASE)
        if label_match:
            return normalize_answer(label_match.group(1))
    return ""


def choice_label(choice: str, fallback_index: int) -> str:
    match = re.match(r"\s*([A-D])\s*[:.)-]\s*", choice, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return "ABCD"[fallback_index]


def choice_text(choice: str) -> str:
    return re.sub(r"^\s*[A-D]\s*[:.)-]\s*", "", choice, flags=re.IGNORECASE).strip()


def load_rows(dataset: Path) -> list[dict[str, Any]]:
    if not dataset.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset}. Clone SecEval under data/seceval "
            "or pass --dataset /path/to/questions.json."
        )
    raw = json.loads(dataset.read_text(encoding="utf-8"))
    rows = raw if isinstance(raw, list) else list(raw.values())
    valid = []
    for row in rows:
        answer = normalize_answer(str(row.get("answer") or ""))
        choices = row.get("choices") or []
        if answer and len(choices) == 4:
            normalized = dict(row)
            normalized["answer"] = answer
            valid.append(normalized)
    return valid


def sample_rows(rows: list[dict[str, Any]], n: int | None, seed: int) -> list[dict[str, Any]]:
    if n is None or n >= len(rows):
        return list(rows)
    rng = random.Random(seed)
    return rng.sample(rows, n)


def filter_topics(rows: list[dict[str, Any]], topics: list[str]) -> list[dict[str, Any]]:
    if not topics:
        return rows
    wanted = {topic.lower() for topic in topics}
    return [
        row for row in rows
        if any(str(topic).lower() in wanted for topic in row.get("topics") or [])
    ]


def retrieval_backend_for_mode(mode: str) -> str:
    return "legacy_hybrid" if "hybrid" in mode else "lexical"


def build_query(row: dict[str, Any]) -> str:
    choices = "\n".join(str(choice) for choice in row.get("choices") or [])
    keyword = str(row.get("keyword") or "")
    topics = ", ".join(str(topic) for topic in row.get("topics") or [])
    return (
        f"Cybersecurity multiple-choice question:\n{row.get('question', '')}\n\n"
        f"Choices:\n{choices}\n\n"
        f"Keyword: {keyword}\nTopics: {topics}"
    )


def build_agent(corpus: CorpusIndex, mode: str, args: argparse.Namespace) -> AgenticRAG:
    planner: QueryPlanner | None = None
    verifier: EvidenceVerifier | None = None
    if mode in {"agentic_lexical_no_graph", "agentic_hybrid_no_graph"}:
        planner = NoGraphPlanner()
    if mode in {"agentic_lexical_no_verifier", "agentic_hybrid_no_verifier"}:
        verifier = NoVerifier()
    return AgenticRAG(
        corpus,
        AgenticRAGConfig(
            retrieve_k=args.retrieve_k,
            max_steps=args.max_steps,
            evidence_budget=args.evidence_budget,
        ),
        planner=planner,
        verifier=verifier,
        search_backend=build_search_backend(corpus, retrieval_backend_for_mode(mode)),
    )


def run_retrieval(corpus: CorpusIndex, mode: str, query: str, args: argparse.Namespace) -> dict[str, Any]:
    if mode == "lexical_baseline":
        start = time.perf_counter()
        evidence = corpus.search(query, k=args.evidence_budget)
        verification = EvidenceVerifier().verify(query, evidence)
        duration_s = time.perf_counter() - start
        trace = [
            {
                "step": 1,
                "action": "search",
                "value": query,
                "rationale": "single-pass lexical baseline",
                "new_evidence": [ev.identifier for ev in evidence],
                "new_evidence_count": len(evidence),
                "supported": verification.supported,
                "confidence": verification.confidence,
                "missing": verification.missing,
            }
        ]
        return {
            "evidence": evidence,
            "verification": verification,
            "trace": trace,
            "duration_s": duration_s,
            "retrieval_backend": "lexical",
        }

    agent = build_agent(corpus, mode, args)
    start = time.perf_counter()
    result = agent.answer(query)
    result["duration_s"] = time.perf_counter() - start
    result["retrieval_backend"] = retrieval_backend_for_mode(mode)
    return result


def select_answer(row: dict[str, Any], evidence: list[Evidence], args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    start = time.perf_counter()
    if args.selector == "heuristic":
        selected, debug = heuristic_select(row, evidence)
        debug["selection_duration_s"] = time.perf_counter() - start
        return selected, debug
    if args.selector == "granite":
        selected, raw = granite_select(row, evidence, args)
        return selected, {
            "selector": "granite",
            "raw_model_answer": raw,
            "selection_duration_s": time.perf_counter() - start,
        }
    raise ValueError(f"Unknown selector: {args.selector}")


def heuristic_select(row: dict[str, Any], evidence: list[Evidence]) -> tuple[str, dict[str, Any]]:
    evidence_text = " ".join((ev.snippet or ev.chunk.text) for ev in evidence).lower()
    question_tokens = set(content_tokens(str(row.get("question") or "")))
    scores: dict[str, float] = {}
    for index, choice in enumerate(row.get("choices") or []):
        label = choice_label(str(choice), index)
        text = choice_text(str(choice))
        tokens = set(content_tokens(text))
        evidence_hits = sum(1 for token in tokens if token in evidence_text)
        question_overlap = len(tokens & question_tokens)
        scores[label] = float(2.0 * evidence_hits + 0.25 * question_overlap)

    if not scores:
        return "", {"selector": "heuristic", "option_scores": {}}
    best_score = max(scores.values())
    if best_score <= 0:
        selected = max(scores, key=lambda label: (scores[label], label))
    else:
        selected_labels = [
            label for label in "ABCD"
            if scores.get(label, 0.0) >= max(best_score - 1.0, best_score * 0.80)
        ]
        selected = "".join(selected_labels) or max(scores, key=scores.get)
    return normalize_answer(selected), {"selector": "heuristic", "option_scores": scores}


def granite_select(row: dict[str, Any], evidence: list[Evidence], args: argparse.Namespace) -> tuple[str, str]:
    client = OpenAIChatClient(
        endpoint=args.llm_endpoint,
        model=args.llm_model,
        max_tokens=args.llm_max_tokens,
        timeout_s=args.llm_timeout_s,
    )
    evidence_blocks = []
    for idx, ev in enumerate(evidence[: args.granite_evidence_k], start=1):
        text = re.sub(r"\s+", " ", ev.snippet or ev.chunk.text).strip()
        evidence_blocks.append(
            f"[{idx}] {ev.identifier} | {ev.chunk.source} | {ev.chunk.name}\n"
            f"{_truncate(text, args.granite_evidence_chars)}"
        )
    prompt = "\n".join(
        [
            "Return only the correct answer letter set. Do not explain.",
            "Your entire response must be only A, B, C, D, or a concatenated multi-answer such as AC or ABD.",
            "",
            "Question:",
            str(row.get("question") or ""),
            "",
            "Choices:",
            *[str(choice) for choice in row.get("choices") or []],
            "",
            "Retrieved evidence:",
            *evidence_blocks,
            "",
            "Select the complete correct answer letter set. Some questions have multiple correct choices.",
            "Evaluate A, B, C, and D independently.",
            "Return only the letters. No prose. No markdown. No punctuation.",
        ]
    )
    raw = client.complete(
        [
            {
                "role": "system",
                "content": (
                    "You are a cybersecurity multiple-choice evaluator. Use the retrieved evidence when relevant. "
                    "Return the full set of correct option letters, not just the best single option. "
                    "Output only letters A-D with no explanation."
                ),
            },
            {"role": "user", "content": prompt},
        ]
    )
    return extract_answer_letters(raw), raw


def content_tokens(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) >= 3 and token.lower() not in STOPWORDS
    ]


def serialize_evidence(evidence: list[Evidence]) -> list[dict[str, Any]]:
    return [
        {
            "identifier": ev.identifier,
            "source": ev.chunk.source,
            "type": ev.chunk.type,
            "name": ev.chunk.name,
            "score": ev.score,
            "tool": ev.tool,
            "reason": ev.reason,
        }
        for ev in evidence
    ]


def classify_failure(correct: bool, selected: str, evidence: list[Evidence], selection_debug: dict[str, Any]) -> str:
    if correct:
        return ""
    if not selected:
        return "no_parseable_answer"
    if not evidence:
        return "no_evidence"
    scores = selection_debug.get("option_scores") or {}
    if scores and max(scores.values(), default=0.0) <= 0:
        return "low_evidence_overlap"
    return "wrong_choice"


def run_one(corpus: CorpusIndex, mode: str, row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    query = build_query(row)
    retrieval = run_retrieval(corpus, mode, query, args)
    evidence = retrieval["evidence"]
    selected, selection_debug = select_answer(row, evidence, args)
    gold = normalize_answer(str(row.get("answer") or ""))
    correct = selected == gold
    trace = retrieval["trace"]
    retrieval_duration_s = float(retrieval["duration_s"])
    selection_duration_s = float(selection_debug.get("selection_duration_s") or 0.0)
    duration_s = retrieval_duration_s + selection_duration_s
    return {
        "id": row.get("id", ""),
        "mode": mode,
        "selector": args.selector,
        "llm_model": args.llm_model if args.selector == "granite" else "",
        "retrieval_backend": retrieval.get("retrieval_backend", retrieval_backend_for_mode(mode)),
        "question": row.get("question", ""),
        "choices": row.get("choices") or [],
        "gold_answer": gold,
        "selected_answer": selected,
        "correct": correct,
        "topics": row.get("topics") or [],
        "keyword": row.get("keyword", ""),
        "source": row.get("source", ""),
        "duration_s": duration_s,
        "retrieval_duration_s": retrieval_duration_s,
        "selection_duration_s": selection_duration_s,
        "tool_calls": len(trace),
        "trace_actions": dict(Counter(str(step.get("action") or "") for step in trace)),
        "evidence_count": len(evidence),
        "evidence_identifiers": [ev.identifier for ev in evidence],
        "evidence": serialize_evidence(evidence),
        "verification": asdict(retrieval["verification"]),
        "trace": trace,
        "selection": selection_debug,
        "failure_reason": classify_failure(correct, selected, evidence, selection_debug),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    correct = sum(1 for row in results if row["correct"])
    durations = [float(row.get("duration_s") or 0.0) for row in results]
    evidence_counts = [int(row.get("evidence_count") or 0) for row in results]
    tool_calls = [int(row.get("tool_calls") or 0) for row in results]
    failures = Counter(str(row.get("failure_reason") or "") for row in results if not row["correct"])
    topic_totals: dict[str, int] = defaultdict(int)
    topic_correct: dict[str, int] = defaultdict(int)
    for row in results:
        for topic in row.get("topics") or ["unknown"]:
            topic_totals[str(topic)] += 1
            topic_correct[str(topic)] += int(bool(row["correct"]))
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "avg_duration_s": sum(durations) / len(durations) if durations else 0.0,
        "avg_evidence_count": sum(evidence_counts) / len(evidence_counts) if evidence_counts else 0.0,
        "avg_tool_calls": sum(tool_calls) / len(tool_calls) if tool_calls else 0.0,
        "failure_reasons": dict(failures),
        "topic_accuracy": {
            topic: {
                "total": topic_totals[topic],
                "correct": topic_correct[topic],
                "accuracy": topic_correct[topic] / topic_totals[topic],
            }
            for topic in sorted(topic_totals)
        },
    }


def summarize_by_mode(results: list[dict[str, Any]]) -> dict[str, Any]:
    modes = sorted({str(row.get("mode") or "unknown") for row in results})
    return {mode: summarize([row for row in results if row.get("mode") == mode]) for mode in modes}


def run_eval(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = filter_topics(load_rows(args.dataset), args.topics)
    rows = sample_rows(rows, args.n, args.seed)
    corpus = CorpusIndex.default(args.root)

    results = []
    for row_index, row in enumerate(rows, 1):
        for mode in args.modes:
            result = run_one(corpus, mode, row, args)
            result["index"] = row_index
            results.append(result)
            status = "PASS" if result["correct"] else "FAIL"
            print(
                f"[{row_index}/{len(rows)}] {mode} {status} "
                f"gold={result['gold_answer']} selected={result['selected_answer'] or '-'} "
                f"evidence={result['evidence_count']} tools={result['tool_calls']} "
                f"time={result['duration_s']:.3f}s"
            )
            if args.debug_failures and not result["correct"]:
                print((result["question"] or "")[:300].replace("\n", " "))
                print(f"choices={result['choices']}")
                print(f"selection={result['selection']}")

    summary = summarize_by_mode(results) if len(args.modes) > 1 else summarize(results)
    write_jsonl(args.trace_log, results)
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return results, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate agentic RAG on SecEval multiple-choice QA.")
    parser.add_argument("n", nargs="?", type=int, default=None, help="Sample size. Defaults to all valid rows.")
    parser.add_argument("--root", type=Path, default=Path("."), help="Repository/data root.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--trace-log", type=Path, default=DEFAULT_TRACE_LOG)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--topics", nargs="*", default=[], help="Optional SecEval topics to include.")
    parser.add_argument("--selector", choices=["heuristic", "granite"], default="heuristic")
    parser.add_argument("--llm-endpoint", default="http://localhost:8000/v1/chat/completions")
    parser.add_argument("--llm-model", default=DEFAULT_GRANITE_MODEL)
    parser.add_argument("--llm-max-tokens", type=int, default=32)
    parser.add_argument("--llm-timeout-s", type=float, default=120.0)
    parser.add_argument("--granite-evidence-k", type=int, default=8)
    parser.add_argument("--granite-evidence-chars", type=int, default=700)
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
    parser.add_argument("--debug-failures", action="store_true")
    return parser


def print_summary(summary: dict[str, Any]) -> None:
    if "accuracy" in summary:
        print_one_summary(summary, "  ")
        return
    for mode, mode_summary in summary.items():
        print(f"  {mode}:")
        print_one_summary(mode_summary, "    ")


def print_one_summary(summary: dict[str, Any], indent: str) -> None:
    print(f"{indent}Correct: {summary['correct']}/{summary['total']} ({summary['accuracy'] * 100:.1f}%)")
    print(f"{indent}Avg evidence: {summary['avg_evidence_count']:.2f}")
    print(f"{indent}Avg tool calls: {summary['avg_tool_calls']:.2f}")
    print(f"{indent}Avg duration: {summary['avg_duration_s']:.3f}s")
    if summary["failure_reasons"]:
        print(f"{indent}Failure reasons: {summary['failure_reasons']}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _results, summary = run_eval(args)
    print("\nSecEval summary")
    print_summary(summary)
    print(f"  Trace log: {args.trace_log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
