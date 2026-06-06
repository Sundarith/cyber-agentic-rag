"""
Evaluate agentic RAG on controlled ATT&CK -> CAPEC -> CWE gold paths.

The diagnostic benchmark gives the agent an ATT&CK technique as the starting
point, hides the CAPEC/CWE path from retrieval, and scores whether the final
evidence trace recovers the expected multihop path.

Examples:
    python3 eval_agentic_multihop.py 25
    python3 eval_agentic_multihop.py 25 --modes lexical_baseline agentic_lexical agentic_lexical_no_graph
    python3 eval_agentic_multihop.py 10 --selector granite --modes no_retrieval agentic_lexical
"""
from __future__ import annotations

import argparse
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
    GraniteOrchestrator,
    GraniteStructuredSynthesizer,
    OpenAIChatClient,
    build_search_backend,
)
from agentic_rag.planner import QueryPlanner
from agentic_rag.schema import Action, ActionType, AgentState, Evidence, Verification
from agentic_rag.synthesizer import ExtractiveSynthesizer, StructuredSynthesisResult
from agentic_rag.verifier import EvidenceVerifier

DEFAULT_TRACE_LOG = Path("logs/agentic_multihop_eval.jsonl")
DEFAULT_ATTACK_CAPEC_RELATIONS = Path("data/processed/capec_attack_relations.json")
DEFAULT_CAPEC_CWE_RELATIONS = Path("data/processed/capec_cwe_relations.json")
SEED = 42
EVAL_MODES = (
    "no_retrieval",
    "lexical_baseline",
    "agentic_lexical",
    "agentic_lexical_no_verifier",
    "agentic_lexical_no_graph",
    "agentic_granite",
)
CHOICE_LABELS = "ABCD"
CWE_RE = re.compile(r"\bCWE-\d+\b", re.IGNORECASE)
LETTER_RE = re.compile(r"\b[ABCD]\b", re.IGNORECASE)


class NoGraphPlanner(QueryPlanner):
    def initial_actions(self, question: str, retrieve_k: int) -> list[Action]:
        return [action for action in super().initial_actions(question, retrieve_k) if action.kind != ActionType.EXPAND]

    def next_actions(self, state: AgentState, verification: Verification, retrieve_k: int) -> list[Action]:
        return [action for action in super().next_actions(state, verification, retrieve_k) if action.kind != ActionType.EXPAND]

    def priority_actions_after_step(
        self,
        state: AgentState,
        action: Action,
        new_evidence: list[Evidence],
        verification: Verification,
        retrieve_k: int,
    ) -> list[Action]:
        return []


class NoVerifier(EvidenceVerifier):
    def verify(self, question: str, evidence: list[Evidence]) -> Verification:
        cited_ids = []
        seen: set[str] = set()
        for ev in evidence:
            if ev.identifier and ev.identifier not in seen:
                cited_ids.append(ev.identifier)
                seen.add(ev.identifier)
        return Verification(False, 0.0, ["verifier_disabled"], cited_ids)


def normalize_id(identifier: str) -> str:
    return str(identifier or "").strip().upper()


def load_relation_map(path: Path, key: str) -> dict[str, list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Relation file not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = raw.get(key, {})
    if not isinstance(rows, dict):
        raise ValueError(f"Relation file {path} missing object key {key!r}")
    return {
        normalize_id(src): [normalize_id(dst) for dst in dsts]
        for src, dsts in rows.items()
        if isinstance(dsts, list)
    }


def sample_rows(rows: list[dict[str, Any]], n: int | None, seed: int) -> list[dict[str, Any]]:
    if n is None or n >= len(rows):
        return list(rows)
    rng = random.Random(seed)
    return rng.sample(rows, n)


def build_diagnostic_items(
    corpus: CorpusIndex,
    attack_capec_relations: Path,
    capec_cwe_relations: Path,
    seed: int = SEED,
    require_single_cwe_capec: bool = True,
    hidden_id: bool = False,
    entry: str = "technique",
) -> list[dict[str, Any]]:
    attack_to_capec = load_relation_map(attack_capec_relations, "tech_to_capec")
    capec_to_cwe = load_relation_map(capec_cwe_relations, "capec_to_cwe")
    known = set(corpus.by_identifier)
    cwe_ids = sorted(identifier for identifier in known if identifier.startswith("CWE-"))
    if len(cwe_ids) < 4:
        raise ValueError("Need at least four CWE chunks to generate multiple-choice items")

    paths: list[tuple[str, str, str]] = []
    for attack_id, capec_ids in attack_to_capec.items():
        if attack_id not in known:
            continue
        for capec_id in capec_ids:
            if capec_id not in known:
                continue
            related_cwes = [cwe_id for cwe_id in capec_to_cwe.get(capec_id, []) if cwe_id in known]
            if require_single_cwe_capec and len(related_cwes) != 1:
                continue
            for cwe_id in related_cwes:
                paths.append((attack_id, capec_id, cwe_id))
    paths = sorted(set(paths))

    if entry == "cwe":
        return _build_reverse_cwe_items(corpus, paths, capec_to_cwe, known, capec_cwe_relations)

    rng = random.Random(seed)
    items = []
    if entry == "capec":
        units = sorted({(capec_id, cwe_id) for _a, capec_id, cwe_id in paths})
        triples = [(capec_id, capec_id, cwe_id) for capec_id, cwe_id in units]
    else:
        triples = [(attack_id, capec_id, cwe_id) for attack_id, capec_id, cwe_id in paths]

    for index, (entry_id, capec_id, cwe_id) in enumerate(triples, start=1):
        distractors = choose_distractors(cwe_ids, cwe_id, rng)
        choice_ids = distractors + [cwe_id]
        rng.shuffle(choice_ids)
        gold_label = CHOICE_LABELS[choice_ids.index(cwe_id)]
        choices = [
            f"{label}: {choice_id} - {display_name(corpus, choice_id)}"
            for label, choice_id in zip(CHOICE_LABELS, choice_ids)
        ]
        entry_name = display_name(corpus, entry_id)
        if entry == "capec":
            entry_type = "attack_pattern"
            gold_path = [capec_id, cwe_id]
            required_hops = ["capec_to_cwe"]
            if hidden_id:
                question = f"For the attack pattern {entry_name}, which listed CWE weakness is its root cause?"
            else:
                question = (
                    f"For CAPEC pattern {entry_id} ({entry_name}), which listed CWE weakness is its root cause?"
                )
        else:
            entry_type = "technique"
            gold_path = [entry_id, capec_id, cwe_id]
            required_hops = ["attack_to_capec", "capec_to_cwe"]
            if hidden_id:
                question = (
                    f"For {entry_name}, which listed CWE weakness is "
                    "connected through its related CAPEC attack pattern?"
                )
            else:
                question = (
                    f"For ATT&CK technique {entry_id} ({entry_name}), which listed CWE weakness is "
                    "connected through its related CAPEC attack pattern?"
                )
        items.append(
            {
                "question_id": f"diagnostic-{entry}-cwe-{index:05d}",
                "question_type": f"diagnostic_{entry}_to_cwe" + ("_hidden" if hidden_id else ""),
                "direction": "forward",
                "entry": entry,
                "entry_id": entry_id,
                "entry_type": entry_type,
                "hidden_id": hidden_id,
                "gold_entity_name": entry_name,
                "question": question,
                "answer": f"{cwe_id} - {display_name(corpus, cwe_id)}",
                "choices": choices,
                "gold_answer": gold_label,
                "gold_cwe": cwe_id,
                "gold_path": gold_path,
                "required_hops": required_hops,
                "distractor_ids": distractors,
                "source_relation_files": [str(attack_capec_relations), str(capec_cwe_relations)],
            }
        )
    return items


def _build_reverse_cwe_items(
    corpus: CorpusIndex,
    paths: list[tuple[str, str, str]],
    capec_to_cwe: dict[str, list[str]],
    known: set[str],
    capec_cwe_relations: Path,
) -> list[dict[str, Any]]:
    """Reverse questions: given a CWE name, recover the set of CAPECs that reference it."""
    # CWE -> all CAPECs (present in corpus) that reference it.
    cwe_to_capecs: dict[str, list[str]] = {}
    for capec_id, cwes in capec_to_cwe.items():
        if capec_id not in known:
            continue
        for cwe_id in cwes:
            if cwe_id in known:
                cwe_to_capecs.setdefault(cwe_id, []).append(capec_id)

    target_cwes = sorted({cwe_id for _a, _c, cwe_id in paths if cwe_to_capecs.get(cwe_id)})
    items = []
    for index, cwe_id in enumerate(target_cwes, start=1):
        gold_set = sorted(set(cwe_to_capecs[cwe_id]))
        cwe_name = display_name(corpus, cwe_id)
        items.append(
            {
                "question_id": f"diagnostic-cwe-capecs-{index:05d}",
                "question_type": "diagnostic_cwe_to_capecs",
                "direction": "reverse",
                "entry": "cwe",
                "entry_id": cwe_id,
                "entry_type": "weakness",
                "hidden_id": True,
                "gold_entity_name": cwe_name,
                "question": f"Which attack patterns reference {cwe_name}?",
                "gold_set": gold_set,
                "gold_target_type": "attack_pattern",
                "gold_path": [cwe_id],
                "required_hops": ["cwe_to_capec"],
                "source_relation_files": [str(capec_cwe_relations)],
            }
        )
    return items


def choose_distractors(cwe_ids: list[str], gold_cwe: str, rng: random.Random) -> list[str]:
    candidates = [cwe_id for cwe_id in cwe_ids if cwe_id != gold_cwe]
    return sorted(rng.sample(candidates, 3))


def display_name(corpus: CorpusIndex, identifier: str) -> str:
    chunk = corpus.by_identifier.get(normalize_id(identifier))
    return chunk.name if chunk and chunk.name else normalize_id(identifier)


def choice_id(choice: str) -> str:
    match = CWE_RE.search(choice)
    return normalize_id(match.group(0)) if match else ""


def choice_map(item: dict[str, Any]) -> dict[str, str]:
    mapping = {}
    for index, choice in enumerate(item.get("choices") or []):
        if index < len(CHOICE_LABELS):
            mapping[CHOICE_LABELS[index]] = choice_id(str(choice))
    return mapping


def extract_answer_letter(text: str) -> str:
    for line in reversed(str(text).splitlines()):
        cleaned = re.sub(r"[^A-D]", "", line.upper())
        stripped = re.sub(r"[^A-Z]", "", line.upper())
        if len(cleaned) == 1 and stripped == cleaned:
            return cleaned
        match = re.search(r"\b(?:FINAL|ANSWER|OUTPUT)\s*[:=-]\s*([ABCD])\b", line, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    match = LETTER_RE.search(str(text))
    return match.group(0).upper() if match else ""


def retrieval_backend_for_mode(mode: str) -> str:
    return "lexical"


def build_agent(corpus: CorpusIndex, mode: str, args: argparse.Namespace) -> AgenticRAG:
    planner: QueryPlanner | None = None
    verifier: EvidenceVerifier | None = None
    search_backend = build_search_backend(corpus, retrieval_backend_for_mode(mode))
    if mode == "agentic_granite":
        orchestrator = GraniteOrchestrator(
            corpus,
            client=OpenAIChatClient(
                endpoint=args.llm_endpoint, model=args.llm_model, max_tokens=args.llm_max_tokens,
                timeout_s=args.llm_timeout_s,
            ),
            search_backend=search_backend,
        )
        return AgenticRAG(
            corpus,
            AgenticRAGConfig(controller="granite"),
            search_backend=search_backend,
            orchestrator=orchestrator,
        )
    if mode == "agentic_lexical_no_graph":
        planner = NoGraphPlanner()
    if mode == "agentic_lexical_no_verifier":
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
        search_backend=search_backend,
    )


def run_retrieval(corpus: CorpusIndex, mode: str, item: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    question = str(item.get("question") or "")
    if mode == "no_retrieval":
        return {
            "answer": "",
            "evidence": [],
            "verification": Verification(False, 0.0, ["no_retrieval"], []),
            "trace": [],
            "duration_s": 0.0,
            "retrieval_backend": "none",
        }
    if mode == "lexical_baseline":
        start = time.perf_counter()
        evidence = corpus.search(question, k=args.evidence_budget)
        verification = EvidenceVerifier().verify(question, evidence)
        duration_s = time.perf_counter() - start
        answer = synthesize_choice_answer(question, evidence, verification)
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
            "answer": answer,
            "evidence": evidence,
            "verification": verification,
            "trace": trace,
            "duration_s": duration_s,
            "retrieval_backend": "lexical",
        }

    agent = build_agent(corpus, mode, args)
    start = time.perf_counter()
    result = agent.answer(question)
    result["duration_s"] = time.perf_counter() - start
    result["retrieval_backend"] = retrieval_backend_for_mode(mode)
    return result


def synthesize_choice_answer(question: str, evidence: list[Evidence], verification: Verification) -> str:
    return ExtractiveSynthesizer().synthesize(question, evidence, verification)


def select_answer(
    item: dict[str, Any],
    answer: str,
    evidence: list[Evidence],
    trace: list[dict[str, Any]],
    verification: Verification,
    args: argparse.Namespace,
) -> tuple[str, dict[str, Any]]:
    start = time.perf_counter()
    if args.selector == "heuristic":
        selected, debug = heuristic_select(item, answer, evidence)
        debug["selection_duration_s"] = time.perf_counter() - start
        return selected, debug
    if args.selector == "granite":
        selected, structured = granite_select(item, evidence, trace, verification, args)
        return selected, {
            "selector": "granite",
            "raw_model_answer": structured.raw,
            "structured": structured.to_dict(),
            "unsupported_id_count": len(structured.validation.unsupported_ids),
            "unsupported_citations_count": len(structured.validation.unsupported_citations),
            "validation_supported": structured.validation.supported,
            "validation_warnings": structured.validation.warnings,
            "selection_duration_s": time.perf_counter() - start,
        }
    raise ValueError(f"Unknown selector: {args.selector}")


def heuristic_select(item: dict[str, Any], answer: str, evidence: list[Evidence]) -> tuple[str, dict[str, Any]]:
    choices = choice_map(item)
    answer_upper = str(answer).upper()
    positions = {
        label: answer_upper.find(cwe_id)
        for label, cwe_id in choices.items()
        if cwe_id and cwe_id in answer_upper
    }
    if positions:
        selected = min(positions, key=positions.get)
        return selected, {"selector": "heuristic", "source": "answer_text", "matched_cwe": choices[selected]}

    evidence_ids = [ev.identifier for ev in evidence]
    for ev in evidence:
        ev_id = normalize_id(ev.identifier)
        for label, cwe_id in choices.items():
            if ev_id == cwe_id:
                return label, {"selector": "heuristic", "source": "evidence_order", "matched_cwe": cwe_id}
    return "", {"selector": "heuristic", "source": "none", "evidence_ids": evidence_ids}


def granite_select(
    item: dict[str, Any],
    evidence: list[Evidence],
    trace: list[dict[str, Any]],
    verification: Verification,
    args: argparse.Namespace,
) -> tuple[str, StructuredSynthesisResult]:
    synthesizer = GraniteStructuredSynthesizer(
        OpenAIChatClient(
            endpoint=args.llm_endpoint,
            model=args.llm_model,
            max_tokens=args.llm_max_tokens,
            timeout_s=args.llm_timeout_s,
        ),
        max_evidence=args.granite_evidence_k,
        max_evidence_chars=args.granite_evidence_chars,
    )
    structured = synthesizer.synthesize_structured(
        str(item.get("question") or ""),
        evidence,
        verification,
        trace,
        choices=[str(choice) for choice in item.get("choices") or []],
    )
    selected = ""
    if structured.validation.supported and structured.selected_cwe:
        for label, cwe_id in choice_map(item).items():
            if structured.selected_cwe == cwe_id:
                selected = label
                break
    return selected, structured


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


def trace_ids_by_step(trace: list[dict[str, Any]]) -> list[tuple[int, list[str]]]:
    rows = []
    for step in trace:
        rows.append(
            (
                int(step.get("step") or 0),
                [normalize_id(identifier) for identifier in step.get("new_evidence") or []],
            )
        )
    return rows


def first_seen_step(trace: list[dict[str, Any]], wanted: set[str]) -> int | None:
    seen: set[str] = set()
    for step, identifiers in trace_ids_by_step(trace):
        seen.update(identifiers)
        if wanted <= seen:
            return step
    return None


def annotate_gold_hops(item: dict[str, Any], trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    gold_path = [normalize_id(identifier) for identifier in item.get("gold_path") or []]
    if len(gold_path) != 3:
        return trace
    attack_id, capec_id, cwe_id = gold_path
    seen: set[str] = set()
    annotated = []
    completed: set[str] = set()
    for step in trace:
        step_copy = dict(step)
        seen.update(normalize_id(identifier) for identifier in step.get("new_evidence") or [])
        gold_completed = []
        if "attack_to_capec" not in completed and {attack_id, capec_id} <= seen:
            gold_completed.append("attack_to_capec")
            completed.add("attack_to_capec")
        if "capec_to_cwe" not in completed and {capec_id, cwe_id} <= seen:
            gold_completed.append("capec_to_cwe")
            completed.add("capec_to_cwe")
        step_copy["gold_hop_completed"] = gold_completed
        annotated.append(step_copy)
    return annotated


def score_path(item: dict[str, Any], evidence: list[Evidence], trace: list[dict[str, Any]]) -> dict[str, Any]:
    gold_path = [normalize_id(identifier) for identifier in item.get("gold_path") or []]
    cwe_id = gold_path[-1]
    # Hops are consecutive node pairs; works for 3-node (technique entry) and
    # 2-node (CAPEC entry) gold paths alike.
    first_hop = set(gold_path[:2])
    second_hop = set(gold_path[-2:])
    final_ids = {normalize_id(ev.identifier) for ev in evidence}
    found = [identifier for identifier in gold_path if identifier in final_ids]
    initial_ids = set(trace_ids_by_step(trace)[0][1]) if trace else set()
    full_path = set(gold_path)
    return {
        "gold_path_found": found,
        "gold_path_recall": len(found) / len(gold_path) if gold_path else 0.0,
        "partial_path_recall": len(found) > 0,
        "full_path_retrieved": full_path <= final_ids,
        "first_hop_success": first_hop <= final_ids,
        "second_hop_success": second_hop <= final_ids,
        "gold_first_seen_step": first_seen_step(trace, {cwe_id}),
        "full_path_first_seen_step": first_seen_step(trace, full_path),
        "initial_full_path_miss": not full_path <= initial_ids,
        "recovery_after_initial_miss": not full_path <= initial_ids and full_path <= final_ids,
    }


def score_resolution(item: dict[str, Any], resolution: dict[str, Any] | None) -> dict[str, Any]:
    """Score entity resolution for hidden-ID items.

    ``evaluated`` is True only when this item required resolution (hidden-ID)
    and the mode actually ran a resolver, so resolution metrics are not diluted
    by explicit-ID items or non-agentic baselines.
    """
    gold_path = [normalize_id(identifier) for identifier in item.get("gold_path") or []]
    # The entity the question names (technique/CAPEC/CWE) is what resolution
    # must recover -- not necessarily gold_path[0] for reverse/CAPEC entries.
    gold_entry = normalize_id(str(item.get("entry_id"))) if item.get("entry_id") else (gold_path[0] if gold_path else "")
    evaluated = bool(item.get("hidden_id")) and resolution is not None
    chosen = normalize_id(str(resolution.get("chosen_id"))) if resolution else ""
    candidate_ids = {normalize_id(str(c.get("identifier"))) for c in (resolution.get("candidates") or [])} if resolution else set()
    return {
        "resolution_evaluated": evaluated,
        "resolution_chosen_id": chosen,
        "resolution_correct": evaluated and chosen == gold_entry,
        # Distinguishes genuine ambiguity (gold was a candidate but not chosen)
        # from a true resolution miss (gold absent entirely).
        "resolution_gold_in_candidates": evaluated and gold_entry in candidate_ids,
        "resolution_ambiguous": bool(resolution.get("ambiguous")) if resolution else False,
        "resolution_mode": str(resolution.get("mode")) if resolution else "",
    }


def classify_failure(
    correct: bool,
    path_scores: dict[str, Any],
    selected: str,
    evidence: list[Evidence],
    resolution_scores: dict[str, Any] | None = None,
) -> str:
    if correct and path_scores["full_path_retrieved"]:
        return ""
    if correct and not path_scores["full_path_retrieved"]:
        return "answer_correct_without_gold_path"
    if path_scores["full_path_retrieved"]:
        return "gold_path_found_answer_wrong"
    # A hidden-ID item whose entry entity never resolved correctly failed at
    # resolution, not traversal -- surface that distinctly.
    if (
        resolution_scores
        and resolution_scores.get("resolution_evaluated")
        and not resolution_scores.get("resolution_correct")
        and not path_scores["full_path_retrieved"]
    ):
        return "resolution_failure"
    if path_scores["second_hop_success"]:
        return "first_hop_or_start_missing"
    if path_scores["first_hop_success"]:
        return "second_hop_missing"
    if not selected:
        return "no_answer_selected"
    if not evidence:
        return "no_evidence"
    return "gold_path_missing"


def run_one(corpus: CorpusIndex, mode: str, item: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if item.get("direction") == "reverse":
        return run_one_reverse(corpus, mode, item, args)
    retrieval = run_retrieval(corpus, mode, item, args)
    evidence = retrieval["evidence"]
    answer = str(retrieval.get("answer") or "")
    trace = annotate_gold_hops(item, retrieval["trace"])
    selected, selection_debug = select_answer(item, answer, evidence, trace, retrieval["verification"], args)
    correct = selected == str(item.get("gold_answer") or "")
    path_scores = score_path(item, evidence, trace)
    resolution = retrieval.get("resolution")
    resolution_scores = score_resolution(item, resolution)
    retrieval_duration_s = float(retrieval["duration_s"])
    selection_duration_s = float(selection_debug.get("selection_duration_s") or 0.0)
    duration_s = retrieval_duration_s + selection_duration_s
    trace_actions = Counter(str(step.get("action") or "") for step in trace)
    return {
        "question_id": item.get("question_id", ""),
        "question_type": item.get("question_type", ""),
        "mode": mode,
        "selector": args.selector,
        "llm_model": args.llm_model if args.selector == "granite" else "",
        "retrieval_backend": retrieval.get("retrieval_backend", retrieval_backend_for_mode(mode)),
        "question": item.get("question", ""),
        "choices": item.get("choices") or [],
        "answer": answer,
        "gold_answer": item.get("gold_answer", ""),
        "selected_answer": selected,
        "correct": correct,
        "gold_cwe": item.get("gold_cwe", ""),
        "gold_path": item.get("gold_path") or [],
        "required_hops": item.get("required_hops") or [],
        "distractor_ids": item.get("distractor_ids") or [],
        "duration_s": duration_s,
        "retrieval_duration_s": retrieval_duration_s,
        "selection_duration_s": selection_duration_s,
        "tool_calls": len(trace),
        "trace_actions": dict(trace_actions),
        "verifier_rejections": sum(1 for step in trace if not step.get("supported")),
        "graph_expansions": trace_actions.get("expand", 0),
        "evidence_count": len(evidence),
        "evidence_identifiers": [ev.identifier for ev in evidence],
        "evidence": serialize_evidence(evidence),
        "verification": asdict(retrieval["verification"]),
        "trace": trace,
        "selection": selection_debug,
        "unsupported_id_count": int(selection_debug.get("unsupported_id_count") or 0),
        "selector_validation_supported": bool(selection_debug.get("validation_supported", True)),
        "hidden_id": bool(item.get("hidden_id")),
        "direction": item.get("direction", "forward"),
        "entry": item.get("entry", "technique"),
        "gold_entity_name": item.get("gold_entity_name", ""),
        "resolution": resolution,
        **path_scores,
        **resolution_scores,
        "failure_reason": classify_failure(correct, path_scores, selected, evidence, resolution_scores),
    }


def run_one_reverse(corpus: CorpusIndex, mode: str, item: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Score a reverse set question: given a CWE, recover the CAPEC set referencing it.

    The agent answers reverse questions deterministically from the graph
    (``reverse_answer``); we score that set against gold. Non-agentic baselines
    have no reverse answer and score zero recall.
    """
    retrieval = run_retrieval(corpus, mode, item, args)
    predicted = {normalize_id(x) for x in (retrieval.get("reverse_answer") or [])}
    gold = {normalize_id(x) for x in (item.get("gold_set") or [])}
    hits = predicted & gold
    recall = len(hits) / len(gold) if gold else 0.0
    precision = len(hits) / len(predicted) if predicted else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    perfect = bool(gold) and predicted == gold

    resolution = retrieval.get("resolution")
    resolution_scores = score_resolution(item, resolution)
    trace = retrieval.get("trace") or []
    trace_actions = Counter(str(step.get("action") or "") for step in trace)
    if not resolution_scores.get("resolution_evaluated") or resolution_scores.get("resolution_correct"):
        failure = "" if perfect else ("partial_reverse_set" if recall > 0 else "reverse_set_missing")
    else:
        failure = "resolution_failure"
    return {
        "question_id": item.get("question_id", ""),
        "question_type": item.get("question_type", ""),
        "mode": mode,
        "direction": "reverse",
        "entry": item.get("entry", "cwe"),
        "selector": args.selector,
        "retrieval_backend": retrieval.get("retrieval_backend", retrieval_backend_for_mode(mode)),
        "question": item.get("question", ""),
        "gold_entity_name": item.get("gold_entity_name", ""),
        "gold_set": sorted(gold),
        "gold_set_size": len(gold),
        "predicted_set": sorted(predicted),
        "predicted_set_size": len(predicted),
        "set_precision": precision,
        "set_recall": recall,
        "set_f1": f1,
        # Forward-compatible keys so summarize() can aggregate uniformly.
        "correct": perfect,
        "gold_answer": "",
        "selected_answer": "",
        "gold_path": item.get("gold_path") or [],
        "gold_path_recall": recall,
        "partial_path_recall": recall > 0,
        "full_path_retrieved": perfect,
        "first_hop_success": recall > 0,
        "second_hop_success": perfect,
        "gold_first_seen_step": None,
        "full_path_first_seen_step": None,
        "initial_full_path_miss": not perfect,
        "recovery_after_initial_miss": False,
        "duration_s": float(retrieval.get("duration_s") or 0.0),
        "retrieval_duration_s": float(retrieval.get("duration_s") or 0.0),
        "selection_duration_s": 0.0,
        "tool_calls": len(trace),
        "trace_actions": dict(trace_actions),
        "verifier_rejections": sum(1 for step in trace if not step.get("supported")),
        "graph_expansions": trace_actions.get("expand", 0),
        "evidence_count": len(predicted),
        "evidence_identifiers": sorted(predicted),
        "verification": asdict(retrieval["verification"]),
        "trace": trace,
        "unsupported_id_count": 0,
        "selector_validation_supported": True,
        "hidden_id": bool(item.get("hidden_id")),
        "resolution": resolution,
        **resolution_scores,
        "failure_reason": failure,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    correct = sum(1 for row in results if row["correct"])
    full_paths = sum(1 for row in results if row["full_path_retrieved"])
    first_hop = sum(1 for row in results if row["first_hop_success"])
    second_hop = sum(1 for row in results if row["second_hop_success"])
    recoveries = sum(1 for row in results if row["recovery_after_initial_miss"])
    recovery_candidates = sum(1 for row in results if row["initial_full_path_miss"])
    durations = [float(row.get("duration_s") or 0.0) for row in results]
    evidence_counts = [int(row.get("evidence_count") or 0) for row in results]
    tool_calls = [int(row.get("tool_calls") or 0) for row in results]
    path_recalls = [float(row.get("gold_path_recall") or 0.0) for row in results]
    verifier_rejections = [int(row.get("verifier_rejections") or 0) for row in results]
    graph_expansions = [int(row.get("graph_expansions") or 0) for row in results]
    unsupported_id_counts = [int(row.get("unsupported_id_count") or 0) for row in results]
    unsupported_rows = sum(1 for count in unsupported_id_counts if count > 0)
    validation_failures = sum(1 for row in results if not row.get("selector_validation_supported", True))
    failures = Counter(str(row.get("failure_reason") or "") for row in results if row.get("failure_reason"))
    resolution_rows = [row for row in results if row.get("resolution_evaluated")]
    resolution_total = len(resolution_rows)
    resolution_correct = sum(1 for row in resolution_rows if row.get("resolution_correct"))
    resolution_gold_in_candidates = sum(1 for row in resolution_rows if row.get("resolution_gold_in_candidates"))
    resolution_ambiguous = sum(1 for row in resolution_rows if row.get("resolution_ambiguous"))
    reverse_rows = [row for row in results if row.get("direction") == "reverse"]
    reverse_recall = [float(row.get("set_recall") or 0.0) for row in reverse_rows]
    reverse_precision = [float(row.get("set_precision") or 0.0) for row in reverse_rows]
    reverse_summary = {}
    if reverse_rows:
        reverse_summary = {
            "reverse_total": len(reverse_rows),
            "reverse_exact_set_match": sum(1 for row in reverse_rows if row.get("correct")),
            "reverse_avg_set_recall": sum(reverse_recall) / len(reverse_recall),
            "reverse_avg_set_precision": sum(reverse_precision) / len(reverse_precision),
        }
    return {
        "total": total,
        "correct": correct,
        "answer_accuracy": correct / total if total else 0.0,
        "full_path_retrieved": full_paths,
        "full_path_recall": full_paths / total if total else 0.0,
        "avg_gold_path_recall": sum(path_recalls) / len(path_recalls) if path_recalls else 0.0,
        "first_hop_success": first_hop,
        "first_hop_rate": first_hop / total if total else 0.0,
        "second_hop_success": second_hop,
        "second_hop_rate": second_hop / total if total else 0.0,
        "recovery_after_initial_miss": recoveries,
        "recovery_candidates": recovery_candidates,
        "recovery_rate_after_initial_miss": recoveries / recovery_candidates if recovery_candidates else 0.0,
        "avg_duration_s": sum(durations) / len(durations) if durations else 0.0,
        "avg_evidence_count": sum(evidence_counts) / len(evidence_counts) if evidence_counts else 0.0,
        "avg_tool_calls": sum(tool_calls) / len(tool_calls) if tool_calls else 0.0,
        "avg_verifier_rejections": sum(verifier_rejections) / len(verifier_rejections) if verifier_rejections else 0.0,
        "avg_graph_expansions": sum(graph_expansions) / len(graph_expansions) if graph_expansions else 0.0,
        "unsupported_id_rows": unsupported_rows,
        "unsupported_id_rate": unsupported_rows / total if total else 0.0,
        "avg_unsupported_ids": sum(unsupported_id_counts) / len(unsupported_id_counts) if unsupported_id_counts else 0.0,
        "selector_validation_failures": validation_failures,
        "selector_validation_failure_rate": validation_failures / total if total else 0.0,
        "resolution_evaluated": resolution_total,
        "entity_resolution_correct": resolution_correct,
        "entity_resolution_accuracy": resolution_correct / resolution_total if resolution_total else 0.0,
        "resolution_gold_in_candidates": resolution_gold_in_candidates,
        "entity_resolution_recall_at_k": resolution_gold_in_candidates / resolution_total if resolution_total else 0.0,
        "resolution_ambiguous": resolution_ambiguous,
        "ambiguity_rate": resolution_ambiguous / resolution_total if resolution_total else 0.0,
        **reverse_summary,
        "failure_reasons": dict(failures),
    }


def summarize_by_mode(results: list[dict[str, Any]]) -> dict[str, Any]:
    modes = sorted({str(row.get("mode") or "unknown") for row in results})
    return {mode: summarize([row for row in results if row.get("mode") == mode]) for mode in modes}


def run_eval(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    corpus = CorpusIndex.default(args.root)
    attack_capec_relations = resolve_data_path(args.root, args.attack_capec_relations, DEFAULT_ATTACK_CAPEC_RELATIONS)
    capec_cwe_relations = resolve_data_path(args.root, args.capec_cwe_relations, DEFAULT_CAPEC_CWE_RELATIONS)
    items = build_diagnostic_items(
        corpus,
        attack_capec_relations=attack_capec_relations,
        capec_cwe_relations=capec_cwe_relations,
        seed=args.seed,
        require_single_cwe_capec=not args.include_multi_cwe_capec,
        hidden_id=args.hidden_id,
        entry=args.entry,
    )
    items = sample_rows(items, args.n, args.seed)
    if args.write_dataset:
        write_jsonl(args.write_dataset, items)

    results = []
    for row_index, item in enumerate(items, 1):
        for mode in args.modes:
            result = run_one(corpus, mode, item, args)
            result["index"] = row_index
            results.append(result)
            status = "PASS" if result["correct"] else "FAIL"
            path_status = "PATH" if result["full_path_retrieved"] else f"PATH={result['gold_path_recall']:.2f}"
            print(
                f"[{row_index}/{len(items)}] {mode} {status} {path_status} "
                f"gold={result['gold_answer']} selected={result['selected_answer'] or '-'} "
                f"tools={result['tool_calls']} evidence={result['evidence_count']} "
                f"time={result['duration_s']:.3f}s"
            )
            if args.debug_failures and result["failure_reason"]:
                print(result["question"])
                print(f"gold_path={result['gold_path']}")
                print(f"evidence={result['evidence_identifiers']}")
                print(f"failure={result['failure_reason']}")

    summary = summarize_by_mode(results) if len(args.modes) > 1 else summarize(results)
    write_jsonl(args.trace_log, results)
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return results, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate diagnostic ATT&CK -> CAPEC -> CWE multihop retrieval.")
    parser.add_argument("n", nargs="?", type=int, default=None, help="Sample size. Defaults to all generated items.")
    parser.add_argument("--root", type=Path, default=Path("."), help="Repository/data root.")
    parser.add_argument("--attack-capec-relations", type=Path, default=DEFAULT_ATTACK_CAPEC_RELATIONS)
    parser.add_argument("--capec-cwe-relations", type=Path, default=DEFAULT_CAPEC_CWE_RELATIONS)
    parser.add_argument("--trace-log", type=Path, default=DEFAULT_TRACE_LOG)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--write-dataset", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--include-multi-cwe-capec", action="store_true")
    parser.add_argument(
        "--hidden-id",
        action="store_true",
        help="Phrase questions with the entity name only (no ID), forcing entity resolution before traversal.",
    )
    parser.add_argument(
        "--entry",
        choices=["technique", "capec", "cwe"],
        default="technique",
        help="Entity the question is addressed by: technique/capec (forward to CWE) or cwe (reverse to CAPEC set).",
    )
    parser.add_argument("--selector", choices=["heuristic", "granite"], default="heuristic")
    parser.add_argument("--llm-endpoint", default="http://localhost:8000/v1/chat/completions")
    parser.add_argument("--llm-model", default=DEFAULT_GRANITE_MODEL)
    parser.add_argument("--llm-max-tokens", type=int, default=384)
    parser.add_argument("--llm-timeout-s", type=float, default=120.0)
    parser.add_argument("--granite-evidence-k", type=int, default=14)
    parser.add_argument("--granite-evidence-chars", type=int, default=700)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=EVAL_MODES,
        default=["agentic_lexical"],
        help="Evaluation modes to run. Multiple modes are written to one trace log with a mode field.",
    )
    parser.add_argument("--retrieve-k", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--evidence-budget", type=int, default=14)
    parser.add_argument("--debug-failures", action="store_true")
    return parser


def resolve_data_path(root: Path, value: Path, default_value: Path) -> Path:
    if value.is_absolute() or value != default_value:
        return value
    return root / value


def print_summary(summary: dict[str, Any]) -> None:
    if "answer_accuracy" in summary:
        print_one_summary(summary, "  ")
        return
    for mode, mode_summary in summary.items():
        print(f"  {mode}:")
        print_one_summary(mode_summary, "    ")


def print_one_summary(summary: dict[str, Any], indent: str) -> None:
    print(f"{indent}Answer accuracy: {summary['correct']}/{summary['total']} ({summary['answer_accuracy'] * 100:.1f}%)")
    print(f"{indent}Full path recall: {summary['full_path_retrieved']}/{summary['total']} ({summary['full_path_recall'] * 100:.1f}%)")
    print(f"{indent}Avg gold path recall: {summary['avg_gold_path_recall'] * 100:.1f}%")
    print(f"{indent}First-hop rate: {summary['first_hop_rate'] * 100:.1f}%")
    print(f"{indent}Second-hop rate: {summary['second_hop_rate'] * 100:.1f}%")
    print(f"{indent}Recovery after initial miss: {summary['recovery_rate_after_initial_miss'] * 100:.1f}%")
    print(f"{indent}Avg evidence: {summary['avg_evidence_count']:.2f}")
    print(f"{indent}Avg tool calls: {summary['avg_tool_calls']:.2f}")
    print(f"{indent}Avg verifier rejections: {summary['avg_verifier_rejections']:.2f}")
    print(f"{indent}Avg graph expansions: {summary['avg_graph_expansions']:.2f}")
    print(f"{indent}Unsupported-ID rate: {summary['unsupported_id_rate'] * 100:.1f}%")
    print(f"{indent}Selector validation failures: {summary['selector_validation_failure_rate'] * 100:.1f}%")
    if summary.get("resolution_evaluated"):
        print(
            f"{indent}Entity resolution accuracy: {summary['entity_resolution_correct']}/"
            f"{summary['resolution_evaluated']} ({summary['entity_resolution_accuracy'] * 100:.1f}%)"
        )
        print(
            f"{indent}Entity resolution recall@k: {summary['resolution_gold_in_candidates']}/"
            f"{summary['resolution_evaluated']} ({summary['entity_resolution_recall_at_k'] * 100:.1f}%)"
        )
        print(f"{indent}Ambiguity rate: {summary['ambiguity_rate'] * 100:.1f}%")
    if summary.get("reverse_total"):
        print(
            f"{indent}Reverse set: exact={summary['reverse_exact_set_match']}/{summary['reverse_total']} "
            f"avg recall={summary['reverse_avg_set_recall'] * 100:.1f}% "
            f"avg precision={summary['reverse_avg_set_precision'] * 100:.1f}%"
        )
    print(f"{indent}Avg duration: {summary['avg_duration_s']:.3f}s")
    if summary["failure_reasons"]:
        print(f"{indent}Failure reasons: {summary['failure_reasons']}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _results, summary = run_eval(args)
    print("\nAgentic multihop diagnostic summary")
    print_summary(summary)
    print(f"  Trace log: {args.trace_log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
