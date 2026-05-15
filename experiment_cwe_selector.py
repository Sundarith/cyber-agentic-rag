"""
Offline experiment for deterministic CWE selection on CTI-RCM failures.

This intentionally does not import better_rag.py, so it can run without loading
embeddings or contacting vLLM. It replays saved failure debug records and tests
small, high-precision selector rules against the CWE candidates that were already
present in the final retrieval context.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


FAILURES_PATH = Path("eval_failures_debug.jsonl")


@dataclass(frozen=True)
class SelectorRule:
    name: str
    pattern: re.Pattern[str]
    cwe_id: str


SELECTOR_RULES: tuple[SelectorRule, ...] = (
    SelectorRule("null_pointer_deref", re.compile(r"\bnull pointer dereference\b|\bnull dereference\b", re.I), "CWE-476"),
    SelectorRule("use_after_free", re.compile(r"\buse[- ]after[- ]free\b|\buaf\b", re.I), "CWE-416"),
    SelectorRule("out_of_bounds_read", re.compile(r"\bout[- ]of[- ]bounds read\b", re.I), "CWE-125"),
    SelectorRule("out_of_bounds_write", re.compile(r"\bout[- ]of[- ]bounds write\b", re.I), "CWE-787"),
    SelectorRule("path_traversal", re.compile(r"\b(path|directory) traversal\b", re.I), "CWE-22"),
    SelectorRule("dangerous_file_upload", re.compile(r"\bunrestricted (file )?upload\b|\barbitrary file upload\b|\bupload[^.]{0,80}dangerous file\b", re.I), "CWE-434"),
    SelectorRule("improper_certificate_validation", re.compile(r"\b(improper|missing|incorrect|fails? to|does not) [^.]{0,60}certificate validation\b|\bvalidate [^.]{0,40}certificate\b", re.I), "CWE-295"),
    SelectorRule("xss", re.compile(r"\bcross[- ]site scripting\b|\bXSS\b", re.I), "CWE-79"),
    SelectorRule("csrf", re.compile(r"\bcross[- ]site request forgery\b|\bCSRF\b", re.I), "CWE-352"),
)


def cwe_ids_from_answer(answer: str) -> list[str]:
    seen: list[str] = []
    for match in re.findall(r"\bCWE-\d+\b", answer or "", re.I):
        cwe_id = match.upper()
        if cwe_id not in seen:
            seen.append(cwe_id)
    return seen


def candidate_cwes(row: dict) -> set[str]:
    debug = row.get("debug") or {}
    cwes = {cwe.upper() for cwe in debug.get("final_context_cwes", [])}
    knn = debug.get("knn_cwe") or {}
    for cwe_id in (knn.get("weights") or {}):
        cwes.add(cwe_id.upper())
    return cwes


def select_cwe(prompt: str, candidates: set[str]) -> tuple[str, str]:
    for rule in SELECTOR_RULES:
        if rule.cwe_id in candidates and rule.pattern.search(prompt):
            return rule.cwe_id, rule.name
    return "", ""


def main() -> int:
    if not FAILURES_PATH.exists():
        print(f"Missing {FAILURES_PATH}; run eval_rcm.py with --debug-failures first.")
        return 1

    rows = [json.loads(line) for line in FAILURES_PATH.open(encoding="utf-8") if line.strip()]
    attempted = []
    fixed = []
    not_fixed = []

    for row in rows:
        predicted = (cwe_ids_from_answer(row.get("answer", "")) or [""])[-1]
        gt = row["gt_cwe"].upper()
        candidates = candidate_cwes(row)
        selected, reason = select_cwe(row.get("prompt", ""), candidates)
        if not selected or selected == predicted:
            continue
        item = {
            "cve_id": row["cve_id"],
            "gt": gt,
            "predicted": predicted,
            "selected": selected,
            "reason": reason,
            "candidates": sorted(candidates),
        }
        attempted.append(item)
        if selected == gt:
            fixed.append(item)
        else:
            not_fixed.append(item)

    print(f"Failures replayed: {len(rows)}")
    print(f"Selector changed an answer: {len(attempted)}")
    print(f"Would fix saved failures: {len(fixed)}")
    print(f"Would remain wrong / risk regression: {len(not_fixed)}")

    if fixed:
        print("\nLikely fixes:")
        for item in fixed:
            print(
                f"  {item['cve_id']}: {item['predicted']} -> {item['selected']} "
                f"(GT {item['gt']}, rule={item['reason']})"
            )

    if not_fixed:
        print("\nRisky changes:")
        for item in not_fixed:
            print(
                f"  {item['cve_id']}: {item['predicted']} -> {item['selected']} "
                f"(GT {item['gt']}, rule={item['reason']})"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
