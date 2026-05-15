"""
Build and replay a system-derived CWE phrase index.

This experiment uses every CWE, not only CTI-RCM failures. Phrases come from
the official CWE XML plus corpus-derived acronym transfer from official names.
It does not import better_rag.py, load embeddings, or call the LLM.
"""
from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path


CWE_XML_PATH = Path("data/raw/cwec_latest.xml")
FAILURES_PATH = Path("eval_failures_debug.jsonl")
DIAGNOSTICS_PATH = Path("analysis/rcm_recall_diagnostics.jsonl")
OUT_INDEX = Path("data/processed/cwe_phrase_index.json")
OUT_REPORT = Path("analysis/cwe_phrase_index_replay.md")

STOP_PHRASES = {
    "weakness",
    "category",
    "view",
    "improper",
    "incorrect",
    "missing",
    "protection",
    "validation",
    "authorization",
    "authentication",
    "input validation",
    "access control",
    "sensitive information",
}

GENERIC_PREFIXES = (
    "improper ",
    "incorrect ",
    "missing ",
    "insufficient ",
    "incomplete ",
    "uncontrolled ",
    "untrusted ",
    "use of ",
    "reliance on ",
    "failure to ",
)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize(text: str) -> str:
    text = text.lower()
    text = text.replace("cross-site", "cross site")
    text = text.replace("use-after-free", "use after free")
    text = re.sub(r"['`\"]", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def phrase_ok(phrase: str, raw: str = "", source: str = "") -> bool:
    phrase = normalize(phrase)
    if not phrase or phrase in STOP_PHRASES:
        return False
    terms = phrase.split()
    raw_is_acronym = bool(re.fullmatch(r"[A-Z][A-Z0-9]{1,8}", raw.strip()))
    source_is_acronym = source == "corpus_acronym_transfer"
    if len(phrase) < 4 and not (raw_is_acronym or source_is_acronym):
        return False
    if len(terms) == 1 and not (raw_is_acronym or source_is_acronym):
        return False
    return True


def add_phrase(index: dict[str, dict], cwe_id: str, phrase: str, source: str) -> None:
    norm = normalize(phrase)
    if not phrase_ok(norm, phrase, source):
        return
    entry = index.setdefault(cwe_id, {"phrases": {}})
    current = entry["phrases"].setdefault(norm, {"phrase": norm, "sources": []})
    if source not in current["sources"]:
        current["sources"].append(source)


def unwrapped_name_phrases(name: str) -> list[str]:
    phrases = [name]
    for raw in re.findall(r"\(([^)]+)\)", name):
        phrases.append(raw.strip(" '\""))
    for raw in re.findall(r"'([^']+)'", name):
        phrases.append(raw.strip())

    base = re.sub(r"\([^)]*\)", " ", name)
    base = re.sub(r"'[^']*'", " ", base)
    base = normalize(base)
    for prefix in GENERIC_PREFIXES:
        if base.startswith(prefix):
            phrases.append(base[len(prefix):])
    return phrases


def load_official_cwes() -> dict[str, dict]:
    cwes: dict[str, dict] = {}
    if not CWE_XML_PATH.exists():
        return cwes

    root = ET.parse(CWE_XML_PATH).getroot()
    for elem in root.iter():
        if not (elem.tag.endswith("Weakness") or elem.tag.endswith("Category") or elem.tag.endswith("View")):
            continue
        cwe_num = elem.attrib.get("ID")
        if not cwe_num:
            continue
        cwe_id = f"CWE-{cwe_num}"
        row = {
            "name": elem.attrib.get("Name", ""),
            "alternate_terms": [],
        }
        for child in elem:
            if not child.tag.endswith("Alternate_Terms"):
                continue
            for alt in child:
                for node in alt:
                    if node.tag.endswith("Term") and node.text:
                        row["alternate_terms"].append(node.text.strip())
        cwes[cwe_id] = row
    return cwes


def acronym_candidates(name: str) -> list[tuple[str, str]]:
    pairs = []
    for match in re.finditer(r"\(([^)]+)\)", name):
        alias = match.group(1).strip()
        if not re.fullmatch(r"[A-Z][A-Z0-9]{1,8}", alias):
            continue
        before = name[: match.start()]
        # Prefer the phrase nearest the acronym. Category names often look like
        # "OWASP ... - Cross-Site Scripting (XSS)".
        fragment = re.split(r"\s+-\s+|\s+:\s+", before)[-1].strip()
        if phrase_ok(fragment, fragment):
            pairs.append((normalize(fragment), alias))
    return pairs


def build_phrase_index() -> dict[str, dict]:
    cwes = load_official_cwes()
    index: dict[str, dict] = {}

    for cwe_id, row in cwes.items():
        for phrase in unwrapped_name_phrases(row.get("name", "")):
            add_phrase(index, cwe_id, phrase, "cwe_name")
        for phrase in row.get("alternate_terms", []):
            add_phrase(index, cwe_id, phrase, "alternate_term")

    # Build a global official phrase -> acronym map, then transfer acronyms to
    # CWEs whose official names contain that phrase. This gives CWE-79 "xss"
    # because official CWE category names define Cross-Site Scripting (XSS).
    acronym_by_phrase: dict[str, set[str]] = defaultdict(set)
    for row in cwes.values():
        for phrase, acronym in acronym_candidates(row.get("name", "")):
            acronym_by_phrase[phrase].add(acronym)
    for cwe_id, row in cwes.items():
        normalized_name = normalize(row.get("name", ""))
        for phrase, acronyms in acronym_by_phrase.items():
            if phrase and re.search(rf"\b{re.escape(phrase)}\b", normalized_name):
                for acronym in acronyms:
                    add_phrase(index, cwe_id, acronym, "corpus_acronym_transfer")

    serializable = {}
    for cwe_id, entry in sorted(index.items()):
        phrases = sorted(entry["phrases"].values(), key=lambda p: (p["phrase"], p["sources"]))
        serializable[cwe_id] = {"phrases": phrases}
    return serializable


def phrase_hits(prompt: str, phrases: list[dict]) -> list[dict]:
    q = normalize(prompt)
    hits = []
    for phrase in phrases:
        p = phrase["phrase"]
        if re.search(rf"\b{re.escape(p)}\b", q):
            hits.append(phrase)
    return hits


def answer_cwe(answer: str) -> str:
    ids = re.findall(r"\bCWE-\d+\b", answer or "", re.IGNORECASE)
    return ids[-1].upper() if ids else ""


def candidate_pool(row: dict, diag: dict, mode: str) -> set[str]:
    debug = row.get("debug") or {}
    pool = {cwe.upper() for cwe in debug.get("final_context_cwes", [])}
    if mode in {"final_knn", "final_knn_top10", "final_knn_top50"}:
        pool.update(cwe.upper() for cwe in (debug.get("knn_cwe") or {}).get("weights", {}))
    if mode in {"final_knn_top10", "final_knn_top50"}:
        limit = 10 if mode == "final_knn_top10" else 50
        for candidate in (diag.get("cwe_only_top50") or [])[:limit]:
            pool.add(candidate.get("identifier", "").upper())
    return {cwe for cwe in pool if cwe.startswith("CWE-")}


def cwe_rank(diag: dict, cwe_id: str) -> int:
    for idx, candidate in enumerate(diag.get("cwe_only_top50") or [], start=1):
        if candidate.get("identifier", "").upper() == cwe_id:
            return idx
    return 999


def select_with_phrases(row: dict, diag: dict, index: dict, mode: str) -> dict | None:
    pool = candidate_pool(row, diag, mode)
    scored = []
    for cwe_id in pool:
        hits = phrase_hits(row.get("prompt", ""), index.get(cwe_id, {}).get("phrases", []))
        hits = [hit for hit in hits if "alternate_term" in hit.get("sources", [])]
        if not hits:
            continue
        best_hit = max(hits, key=lambda h: (len(h["phrase"].split()), len(h["phrase"])))
        rank = cwe_rank(diag, cwe_id)
        in_final = cwe_id in {c.upper() for c in (row.get("debug") or {}).get("final_context_cwes", [])}
        in_knn = cwe_id in ((row.get("debug") or {}).get("knn_cwe") or {}).get("weights", {})
        scored.append({
            "cwe": cwe_id,
            "phrase": best_hit["phrase"],
            "sources": best_hit["sources"],
            "rank": rank,
            "in_final": in_final,
            "in_knn": in_knn,
            "score_tuple": (
                1 if in_final else 0,
                len(best_hit["phrase"].split()),
                len(best_hit["phrase"]),
                1 if in_knn else 0,
                -rank,
            ),
        })
    if not scored:
        return None
    scored.sort(key=lambda item: item["score_tuple"], reverse=True)
    return scored[0]


def replay(index: dict) -> dict:
    failures = load_jsonl(FAILURES_PATH)
    diagnostics = {row["cve_id"]: row for row in load_jsonl(DIAGNOSTICS_PATH)}
    modes = ["final", "final_knn", "final_knn_top10", "final_knn_top50"]
    results = {}
    for mode in modes:
        changed = []
        fixed = []
        wrong = []
        same = []
        for row in failures:
            diag = diagnostics.get(row["cve_id"], {})
            selected = select_with_phrases(row, diag, index, mode)
            predicted = answer_cwe(row.get("answer", ""))
            if not selected or selected["cwe"] == predicted:
                same.append(row["cve_id"])
                continue
            item = {
                "cve_id": row["cve_id"],
                "gt": row["gt_cwe"].upper(),
                "predicted": predicted,
                **{k: v for k, v in selected.items() if k != "score_tuple"},
            }
            changed.append(item)
            if selected["cwe"] == row["gt_cwe"].upper():
                fixed.append(item)
            else:
                wrong.append(item)
        results[mode] = {"changed": changed, "fixed": fixed, "wrong": wrong, "same": same}
    return results


def write_report(index: dict, results: dict) -> None:
    lines = [
        "# CWE Phrase Index Replay",
        "",
        f"CWEs indexed: **{len(index)}**",
        f"Total phrases: **{sum(len(v['phrases']) for v in index.values())}**",
        "",
        "## Replay Summary",
        "",
    ]
    for mode, result in results.items():
        lines.append(
            f"- `{mode}`: changed {len(result['changed'])}, "
            f"fixed {len(result['fixed'])}, wrong/risky {len(result['wrong'])}"
        )

    for mode, result in results.items():
        lines.extend(["", f"## Mode `{mode}`", ""])
        if result["fixed"]:
            lines.append("Likely fixes:")
            for item in result["fixed"]:
                lines.append(
                    f"- {item['cve_id']}: `{item['predicted']}` -> `{item['cwe']}` "
                    f"(GT `{item['gt']}`, phrase `{item['phrase']}`, sources {item['sources']}, rank {item['rank']})"
                )
        if result["wrong"]:
            lines.append("")
            lines.append("Risky changes:")
            for item in result["wrong"]:
                lines.append(
                    f"- {item['cve_id']}: `{item['predicted']}` -> `{item['cwe']}` "
                    f"(GT `{item['gt']}`, phrase `{item['phrase']}`, sources {item['sources']}, rank {item['rank']})"
                )
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    index = build_phrase_index()
    OUT_INDEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_INDEX.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    results = replay(index)
    write_report(index, results)
    print(f"Wrote {OUT_INDEX}")
    print(f"Wrote {OUT_REPORT}")
    for mode, result in results.items():
        print(
            f"{mode}: changed={len(result['changed'])} "
            f"fixed={len(result['fixed'])} wrong={len(result['wrong'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
