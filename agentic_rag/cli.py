from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agent import AgenticRAG, AgenticRAGConfig
from .corpus import CorpusIndex
from .orchestrator import GraniteOrchestrator
from .retrieval import build_search_backend
from .synthesizer import DEFAULT_GRANITE_MODEL, GraniteGroundedSynthesizer, OpenAIChatClient, endpoint_reachable


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the agentic CTI-RAG prototype.")
    parser.add_argument("question", help="Question to ask over the local CTI corpus.")
    parser.add_argument("--root", default=".", help="Repository/data root.")
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--retrieve-k", type=int, default=8)
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
    parser.add_argument("--llm-max-tokens", type=int, default=256)
    parser.add_argument("--json", action="store_true", help="Print machine-readable result.")
    args = parser.parse_args(argv)

    corpus = CorpusIndex.default(Path(args.root))
    search_backend = build_search_backend(corpus, args.retrieval_backend)
    controller = args.controller
    if controller == "auto":
        controller = "granite" if endpoint_reachable(args.llm_endpoint, 3.0) else "deterministic"
    synthesizer = None
    if args.synthesizer == "granite":
        synthesizer = GraniteGroundedSynthesizer(
            OpenAIChatClient(
                endpoint=args.llm_endpoint,
                model=args.llm_model,
                max_tokens=args.llm_max_tokens,
            )
        )
    orchestrator = None
    if controller == "granite":
        orchestrator = GraniteOrchestrator(
            corpus,
            client=OpenAIChatClient(
                endpoint=args.llm_endpoint, model=args.llm_model, max_tokens=args.llm_max_tokens
            ),
            search_backend=search_backend,
        )
    agent = AgenticRAG(
        corpus,
        AgenticRAGConfig(max_steps=args.max_steps, retrieve_k=args.retrieve_k, controller=controller),
        search_backend=search_backend,
        synthesizer=synthesizer,
        orchestrator=orchestrator,
    )
    result = agent.answer(args.question)
    if args.json:
        print(
            json.dumps(
                {
                    "question": result["question"],
                    "answer": result["answer"],
                    "verification": result["verification"].__dict__,
                    "resolution": result.get("resolution"),
                    "route": result.get("route"),
                    "reverse_answer": result.get("reverse_answer"),
                    "trace": result["trace"],
                    "evidence": [
                        {
                            "identifier": ev.identifier,
                            "source": ev.chunk.source,
                            "type": ev.chunk.type,
                            "name": ev.chunk.name,
                            "score": ev.score,
                            "tool": ev.tool,
                            "retrieval_backend": search_backend.name,
                        }
                        for ev in result["evidence"]
                    ],
                },
                indent=2,
            )
        )
    else:
        print(result["answer"])
        resolution = result.get("resolution")
        if resolution and resolution.get("chosen_id"):
            flag = " (ambiguous)" if resolution.get("ambiguous") else ""
            print(
                f"\nResolved entity: '{resolution['query']}' -> {resolution['chosen_id']} "
                f"[{resolution['type']}] via {resolution['mode']} match{flag}"
            )
        reverse_answer = result.get("reverse_answer")
        if reverse_answer:
            preview = ", ".join(reverse_answer[:20])
            more = f" (+{len(reverse_answer) - 20} more)" if len(reverse_answer) > 20 else ""
            print(f"\nMatching {result['route']['target_type']} set ({len(reverse_answer)}): {preview}{more}")
        print("\nTrace:")
        for step in result["trace"]:
            print(
                f"- {step['step']}: {step['action']} {step['value']} "
                f"-> {step['new_evidence']} confidence={step['confidence']:.2f}"
            )
    return 0
