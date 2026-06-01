from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agent import AgenticRAG, AgenticRAGConfig
from .corpus import CorpusIndex


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the agentic CTI-RAG prototype.")
    parser.add_argument("question", help="Question to ask over the local CTI corpus.")
    parser.add_argument("--root", default=".", help="Repository/data root.")
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--retrieve-k", type=int, default=8)
    parser.add_argument("--json", action="store_true", help="Print machine-readable result.")
    args = parser.parse_args(argv)

    corpus = CorpusIndex.default(Path(args.root))
    agent = AgenticRAG(
        corpus,
        AgenticRAGConfig(max_steps=args.max_steps, retrieve_k=args.retrieve_k),
    )
    result = agent.answer(args.question)
    if args.json:
        print(
            json.dumps(
                {
                    "question": result["question"],
                    "answer": result["answer"],
                    "verification": result["verification"].__dict__,
                    "trace": result["trace"],
                    "evidence": [
                        {
                            "identifier": ev.identifier,
                            "source": ev.chunk.source,
                            "type": ev.chunk.type,
                            "name": ev.chunk.name,
                            "score": ev.score,
                            "tool": ev.tool,
                        }
                        for ev in result["evidence"]
                    ],
                },
                indent=2,
            )
        )
    else:
        print(result["answer"])
        print("\nTrace:")
        for step in result["trace"]:
            print(
                f"- {step['step']}: {step['action']} {step['value']} "
                f"-> {step['new_evidence']} confidence={step['confidence']:.2f}"
            )
    return 0
