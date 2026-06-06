"""
Interactive chat shell for manually probing the agentic CTI-RAG system.

This is intentionally separate from the experiment/evaluation harnesses. It is
for qualitative testing: type a question, inspect the answer, trace, and
evidence, then ask another question.

Example:
    python3 chat_agentic_rag.py
    python3 chat_agentic_rag.py --synthesizer granite --max-steps 8
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from agentic_rag import (
    DEFAULT_GRANITE_MODEL,
    AgenticRAG,
    AgenticRAGConfig,
    CorpusIndex,
    GraniteStructuredSynthesizer,
    OpenAIChatClient,
    build_search_backend,
)
from agentic_rag.schema import Evidence
from agentic_rag.synthesizer import StructuredSynthesisResult, endpoint_reachable


HELP_TEXT = """Commands:
  /help      Show this help
  /trace     Toggle trace display
  /evidence  Toggle evidence display
  /json      Toggle JSON output
  /quit      Exit
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chat with the local agentic CTI-RAG system.")
    parser.add_argument("--root", type=Path, default=Path("."), help="Repository/data root.")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--retrieve-k", type=int, default=8)
    parser.add_argument("--evidence-budget", type=int, default=14)
    parser.add_argument(
        "--retrieval-backend",
        choices=["lexical", "legacy_hybrid"],
        default="lexical",
        help="Search backend. legacy_hybrid lazy-loads the legacy dense/BM25 stack.",
    )
    parser.add_argument(
        "--synthesizer",
        choices=["extractive", "granite"],
        default="extractive",
        help="Answer synthesis backend. granite calls an OpenAI-compatible local endpoint.",
    )
    parser.add_argument(
        "--controller",
        choices=["auto", "deterministic", "granite"],
        default="auto",
        help="auto = LLM-orchestrated when the endpoint is reachable, else deterministic; "
             "deterministic = regex planner loop; granite = force the LLM-orchestrated ReAct loop.",
    )
    parser.add_argument("--llm-endpoint", default="http://localhost:8000/v1/chat/completions")
    parser.add_argument("--llm-model", default=DEFAULT_GRANITE_MODEL)
    parser.add_argument("--llm-max-tokens", type=int, default=384)
    parser.add_argument("--llm-timeout-s", type=float, default=120.0)
    parser.add_argument("--no-trace", action="store_true", help="Hide trace by default.")
    parser.add_argument("--no-evidence", action="store_true", help="Hide evidence by default.")
    parser.add_argument("--json", action="store_true", help="Print each response as JSON.")
    return parser


def build_agent(args: argparse.Namespace) -> tuple[AgenticRAG, str, GraniteStructuredSynthesizer | None]:
    corpus = CorpusIndex.default(args.root)
    search_backend = build_search_backend(corpus, args.retrieval_backend)
    # Resolve the agentic-by-default controller: use the LLM loop when the
    # endpoint is reachable, otherwise fall back to the deterministic baseline.
    controller = args.controller
    if controller == "auto":
        if endpoint_reachable(args.llm_endpoint, min(args.llm_timeout_s, 3.0)):
            controller = "granite"
            print("[controller] LLM endpoint reachable -> agentic (granite) controller.")
        else:
            controller = "deterministic"
            print("[controller] LLM endpoint unreachable -> deterministic baseline. "
                  "Start the vLLM endpoint for the agentic loop.")
    structured_synthesizer = None
    if args.synthesizer == "granite":
        structured_synthesizer = GraniteStructuredSynthesizer(
            OpenAIChatClient(
                endpoint=args.llm_endpoint,
                model=args.llm_model,
                max_tokens=args.llm_max_tokens,
                timeout_s=args.llm_timeout_s,
            )
        )
    orchestrator = None
    if controller == "granite":
        from agentic_rag import GraniteOrchestrator

        orchestrator = GraniteOrchestrator(
            corpus,
            client=OpenAIChatClient(
                endpoint=args.llm_endpoint,
                model=args.llm_model,
                max_tokens=args.llm_max_tokens,
                timeout_s=args.llm_timeout_s,
            ),
            search_backend=search_backend,
        )
    agent = AgenticRAG(
        corpus,
        AgenticRAGConfig(
            max_steps=args.max_steps,
            retrieve_k=args.retrieve_k,
            evidence_budget=args.evidence_budget,
            controller=controller,
        ),
        search_backend=search_backend,
        orchestrator=orchestrator,
    )
    return agent, search_backend.name, structured_synthesizer


def result_to_jsonable(result: dict[str, Any], retrieval_backend: str) -> dict[str, Any]:
    verification = result["verification"]
    return {
        "question": result["question"],
        "answer": result["answer"],
        "verification": {
            "supported": verification.supported,
            "confidence": verification.confidence,
            "missing": verification.missing,
            "cited_ids": verification.cited_ids,
        },
        "trace": result["trace"],
        "final_answer": result.get("final_answer") or build_final_answer(result),
        "final_answer_source": result.get("final_answer_source", "deterministic"),
        "final_answer_warning": result.get("final_answer_warning", ""),
        "structured_answer": (
            result["structured_answer"].to_dict()
            if isinstance(result.get("structured_answer"), StructuredSynthesisResult)
            else None
        ),
        "evidence": [
            {
                "identifier": ev.identifier,
                "source": ev.chunk.source,
                "type": ev.chunk.type,
                "name": ev.chunk.name,
                "score": ev.score,
                "tool": ev.tool,
                "reason": ev.reason,
                "retrieval_backend": retrieval_backend,
            }
            for ev in result["evidence"]
        ],
    }


def format_result(
    result: dict[str, Any],
    retrieval_backend: str,
    show_trace: bool = True,
    show_evidence: bool = True,
) -> str:
    verification = result["verification"]
    lines = [
        "",
        "Retrieval Draft",
        "---------------",
        str(result.get("answer") or "").strip(),
        "",
        f"Verifier: supported={verification.supported} confidence={verification.confidence:.2f}",
    ]
    if verification.missing:
        lines.append("Missing: " + ", ".join(verification.missing))
    if verification.cited_ids:
        lines.append("Cited IDs: " + ", ".join(verification.cited_ids[:10]))
    if result.get("final_answer_warning"):
        lines.append("Final answer warning: " + str(result["final_answer_warning"]))

    structured_answer = result.get("structured_answer")
    if isinstance(structured_answer, StructuredSynthesisResult):
        lines += ["", "Granite Structured Answer", "-------------------------"]
        lines += format_structured_answer(structured_answer)

    if show_trace:
        lines += ["", "Trace", "-----"]
        if result["trace"]:
            for step in result["trace"]:
                lines.append(format_trace_step(step))
        else:
            lines.append("(no tool calls)")

    if show_evidence:
        lines += ["", "Evidence", "--------"]
        if result["evidence"]:
            for idx, ev in enumerate(result["evidence"], start=1):
                lines.append(format_evidence(idx, ev, retrieval_backend))
        else:
            lines.append("(no evidence)")
    path_text = build_path_answer(result)
    if path_text:
        lines += ["", "ATT&CK -> CAPEC -> CWE Path", "---------------------------", path_text]
    lines += ["", "Final Answer", "------------", str(result.get("final_answer") or build_final_answer(result)).strip()]
    return "\n".join(lines)


def format_structured_answer(structured_answer: StructuredSynthesisResult) -> list[str]:
    validation = structured_answer.validation
    lines = [
        f"answerable={structured_answer.answerable} confidence={structured_answer.confidence}",
        f"selected_cwe={structured_answer.selected_cwe or '-'}",
        "selected_path=" + (" -> ".join(structured_answer.selected_path) if structured_answer.selected_path else "-"),
        "cited_evidence_ids="
        + (", ".join(structured_answer.cited_evidence_ids) if structured_answer.cited_evidence_ids else "-"),
        f"validation_supported={validation.supported}",
    ]
    if structured_answer.reason:
        lines.append("reason=" + structured_answer.reason)
    if validation.warnings:
        lines.append("warnings=" + ", ".join(validation.warnings))
    if validation.unsupported_ids:
        lines.append("unsupported_ids=" + ", ".join(validation.unsupported_ids))
    if validation.candidate_paths:
        rendered = [" -> ".join(path) for path in validation.candidate_paths[:5]]
        if len(validation.candidate_paths) > 5:
            rendered.append(f"... ({len(validation.candidate_paths)} paths total)")
        lines.append("candidate_paths=" + "; ".join(rendered))
    if structured_answer.error:
        lines.append("error=" + structured_answer.error)
    return lines


def build_final_answer(result: dict[str, Any]) -> str:
    question = str(result.get("question") or "")
    if asks_for_cwe(question):
        structured_path = validated_structured_path(result.get("structured_answer"))
        if structured_path:
            return format_polished_path_answer(structured_path, result.get("evidence") or [])

        paths = trace_cwe_path_lists(result.get("trace") or [])
        if paths:
            if len(paths) == 1:
                return format_polished_path_answer(paths[0], result.get("evidence") or [])
            formatted = ", ".join(format_path(path) for path in paths[:5])
            if len(paths) > 5:
                formatted += f", ... ({len(paths)} paths total)"
            return f"Complete candidate path(s): {formatted}"

        if asks_for_relation_path(question):
            return "I do not have enough validated evidence to identify one complete ATT&CK -> CAPEC -> CWE path."

        cwe_candidates = evidence_cwe_candidates(result.get("evidence") or [])
        if len(cwe_candidates) == 1:
            cwe_id, name = cwe_candidates[0]
            return f"{cwe_id} - {name}"
        if len(cwe_candidates) > 1:
            rendered = ", ".join(f"{cwe_id} ({name})" for cwe_id, name in cwe_candidates[:8])
            if len(cwe_candidates) > 8:
                rendered += f", ... ({len(cwe_candidates)} candidates total)"
            return (
                "Multiple CWE candidates were found, but the trace did not identify one unambiguous "
                f"complete ATT&CK -> CAPEC -> CWE path: {rendered}"
            )
        return "I do not have enough validated evidence to identify one complete ATT&CK -> CAPEC -> CWE path."

    answer = str(result.get("answer") or "").strip()
    return answer or "No final answer was produced."


def build_path_answer(result: dict[str, Any]) -> str:
    structured_path = validated_structured_path(result.get("structured_answer"))
    if structured_path:
        return format_path(structured_path)
    paths = trace_cwe_path_lists(result.get("trace") or [])
    if len(paths) == 1:
        return format_path(paths[0])
    if len(paths) > 1:
        formatted = ", ".join(format_path(path) for path in paths[:5])
        if len(paths) > 5:
            formatted += f", ... ({len(paths)} paths total)"
        return formatted
    return ""


def build_user_facing_answer(
    result: dict[str, Any],
    structured_synthesizer: GraniteStructuredSynthesizer | None = None,
) -> tuple[str, str]:
    structured_path = validated_structured_path(result.get("structured_answer"))
    if structured_path and structured_synthesizer is not None:
        try:
            return (
                structured_synthesizer.compose_grounded_answer(
                    str(result.get("question") or ""),
                    result.get("evidence") or [],
                    structured_path,
                ),
                "granite_composed",
            )
        except Exception as exc:
            result["final_answer_warning"] = str(exc)
            return format_polished_path_answer(structured_path, result.get("evidence") or []), "deterministic_fallback"

    deterministic_paths = trace_cwe_path_lists(result.get("trace") or [])
    if deterministic_paths:
        return format_polished_path_answer(deterministic_paths[0], result.get("evidence") or []), "deterministic_path"
    return build_final_answer(result), "deterministic"


def validated_structured_path(value: Any) -> list[str]:
    if not isinstance(value, StructuredSynthesisResult):
        return []
    if not value.validation.supported or not value.answerable:
        return []
    if not value.selected_path:
        return []
    return [str(identifier).upper() for identifier in value.selected_path]


def format_polished_path_answer(path: list[str], evidence: list[Evidence]) -> str:
    if len(path) != 3:
        return "Complete candidate path: " + format_path(path)
    attack_id, capec_id, cwe_id = [identifier.upper() for identifier in path]
    names = evidence_name_map(evidence)
    attack = format_id_name(attack_id, names)
    capec = format_id_name(capec_id, names)
    cwe = format_id_name(cwe_id, names)
    return f"{cwe} is the connected CWE weakness for ATT&CK {attack}, through {capec}."


def format_id_name(identifier: str, names: dict[str, str]) -> str:
    name = names.get(identifier.upper(), "")
    return f"{identifier} ({name})" if name and name != identifier else identifier


def evidence_name_map(evidence: list[Evidence]) -> dict[str, str]:
    names = {}
    for ev in evidence:
        identifier = ev.identifier.upper()
        if identifier and identifier not in names:
            names[identifier] = ev.chunk.name or identifier
    return names


def format_path(path: list[str]) -> str:
    return " -> ".join(path)


def asks_for_cwe(question: str) -> bool:
    return bool(re.search(r"\b(cwe|weakness|root cause|underlies|underlying)\b", question, re.IGNORECASE))


def asks_for_relation_path(question: str) -> bool:
    return bool(
        re.search(
            r"\b(att&ck|attack technique|capec|attack pattern|connected|through|related)\b",
            question,
            re.IGNORECASE,
        )
    )


def trace_cwe_paths(trace: list[dict[str, Any]]) -> list[str]:
    return [format_path(path) for path in trace_cwe_path_lists(trace)]


def trace_cwe_path_lists(trace: list[dict[str, Any]]) -> list[list[str]]:
    attack_to_capecs: dict[str, list[str]] = {}
    capec_to_cwes: dict[str, list[str]] = {}
    for step in trace:
        if str(step.get("action") or "") != "expand":
            continue
        src = str(step.get("expanded_from") or step.get("value") or "").upper()
        new_ids = [str(identifier).upper() for identifier in step.get("new_evidence") or []]
        if is_attack_id(src):
            attack_to_capecs[src] = [identifier for identifier in new_ids if identifier.startswith("CAPEC-")]
        if src.startswith("CAPEC-"):
            cwes = [identifier for identifier in new_ids if identifier.startswith("CWE-")]
            if cwes:
                capec_to_cwes[src] = cwes

    paths = []
    for attack_id, capec_ids in attack_to_capecs.items():
        for capec_id in capec_ids:
            for cwe_id in capec_to_cwes.get(capec_id, []):
                paths.append([attack_id, capec_id, cwe_id])
    return dedupe_paths(paths)


def evidence_cwe_candidates(evidence: list[Evidence]) -> list[tuple[str, str]]:
    candidates = []
    seen: set[str] = set()
    for ev in evidence:
        identifier = ev.identifier.upper()
        if identifier.startswith("CWE-") and identifier not in seen:
            candidates.append((identifier, ev.chunk.name or identifier))
            seen.add(identifier)
    return candidates


def is_attack_id(identifier: str) -> bool:
    return bool(re.match(r"^T\d{4}(?:\.\d{3})?$", identifier or "", re.IGNORECASE))


def dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped = []
    for value in values:
        if value not in seen:
            deduped.append(value)
            seen.add(value)
    return deduped


def dedupe_paths(values: list[list[str]]) -> list[list[str]]:
    seen: set[tuple[str, ...]] = set()
    deduped = []
    for value in values:
        key = tuple(value)
        if key not in seen:
            deduped.append(value)
            seen.add(key)
    return deduped


def format_trace_step(step: dict[str, Any]) -> str:
    action = step.get("action", "")
    value = step.get("value", "")
    new_evidence = step.get("new_evidence") or []
    confidence = float(step.get("confidence") or 0.0)
    relation_hops = step.get("relation_hop_completed") or []
    gold_hops = step.get("gold_hop_completed") or []
    suffix = ""
    if relation_hops:
        suffix += " relation_hop=" + ",".join(relation_hops)
    if gold_hops:
        suffix += " gold_hop=" + ",".join(gold_hops)
    return f"- {step.get('step')}: {action} {value} -> {new_evidence} confidence={confidence:.2f}{suffix}"


def format_evidence(index: int, evidence: Evidence, retrieval_backend: str) -> str:
    title = evidence.chunk.name or evidence.chunk.type or evidence.chunk.source
    return (
        f"{index}. {evidence.identifier} | {evidence.chunk.source} | {title} "
        f"| tool={evidence.tool} | score={evidence.score:.3f} | backend={retrieval_backend}"
    )


def handle_command(command: str, state: dict[str, bool]) -> bool:
    normalized = command.strip().lower()
    if normalized in {"/q", "/quit", "/exit"}:
        return False
    if normalized == "/help":
        print(HELP_TEXT)
        return True
    if normalized == "/trace":
        state["show_trace"] = not state["show_trace"]
        print(f"Trace display: {'on' if state['show_trace'] else 'off'}")
        return True
    if normalized == "/evidence":
        state["show_evidence"] = not state["show_evidence"]
        print(f"Evidence display: {'on' if state['show_evidence'] else 'off'}")
        return True
    if normalized == "/json":
        state["json_output"] = not state["json_output"]
        print(f"JSON output: {'on' if state['json_output'] else 'off'}")
        return True
    print("Unknown command. Type /help for commands.")
    return True


def chat_loop(
    agent: AgenticRAG,
    retrieval_backend: str,
    structured_synthesizer: GraniteStructuredSynthesizer | None,
    args: argparse.Namespace,
) -> None:
    state = {
        "show_trace": not args.no_trace,
        "show_evidence": not args.no_evidence,
        "json_output": bool(args.json),
    }
    print("Agentic CTI-RAG chat")
    print("Type a question, or /help for commands. Use /quit to exit.")
    while True:
        try:
            question = input("\nQuestion> ").strip()
        except EOFError:
            print()
            break
        if not question:
            continue
        if question.startswith("/"):
            if not handle_command(question, state):
                break
            continue
        result = agent.answer(question)
        if structured_synthesizer is not None:
            result["structured_answer"] = structured_synthesizer.synthesize_structured(
                question,
                result["evidence"],
                result["verification"],
                result["trace"],
            )
        result["final_answer"], result["final_answer_source"] = build_user_facing_answer(result, structured_synthesizer)
        if state["json_output"]:
            print(json.dumps(result_to_jsonable(result, retrieval_backend), indent=2))
        else:
            print(
                format_result(
                    result,
                    retrieval_backend,
                    show_trace=state["show_trace"],
                    show_evidence=state["show_evidence"],
                )
            )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print("Loading corpus and agent...")
    agent, retrieval_backend, structured_synthesizer = build_agent(args)
    chat_loop(agent, retrieval_backend, structured_synthesizer, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
