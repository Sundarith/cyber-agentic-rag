"""
Build augmented CWE chunks by appending sampled real-CVE descriptions for each CWE.

Source MITRE CWE descriptions are abstract/normative; CVE descriptions are concrete
and specific. The bi-encoder embeds them in different neighborhoods even when they
describe the same weakness. Appending a few labeled real-CVE descriptions to each
CWE chunk shifts that CWE's embedding toward actual CVE-prose semantic space.

Inputs:
  data/processed/cwe_chunks.jsonl       — original MITRE CWE chunks
  data/processed/cve_cwe_index.json     — CVE-ID → [CWE-ID,...] (only mapped CVEs have non-empty lists)
  data/processed/cve_chunks.jsonl       — full CVE chunk records

Output:
  data/processed/cwe_chunks_augmented.jsonl

CTI-Bench unmapped test CVEs are by construction NOT in the mapping index (NVD has
no structured CWE for them), so they cannot leak into the augmentation. This is
training-side mapped corpus only.
"""
import json
import random
import re
from collections import defaultdict
from pathlib import Path

CWE_CHUNKS = Path("data/processed/cwe_chunks.jsonl")
CVE_CHUNKS = Path("data/processed/cve_chunks.jsonl")
CVE_CWE_INDEX = Path("data/processed/cve_cwe_index.json")
OUT_PATH = Path("data/processed/cwe_chunks_augmented.jsonl")

SAMPLES_PER_CWE = 3
DESC_MAX_CHARS = 280
SEED = 42


def cve_description_from_text(text: str) -> str:
    """Extract the CVE description from a chunk's text (skip markdown header)."""
    if not text:
        return ""
    lines = text.split("\n", 2)
    body = lines[2] if len(lines) > 2 else (lines[1] if len(lines) > 1 else "")
    body = body.strip()
    # CVE chunk descriptions often start directly. Truncate at first paragraph break.
    para = body.split("\n\n", 1)[0].strip()
    para = re.sub(r"\s+", " ", para)
    return para[:DESC_MAX_CHARS]


def build_reverse_index() -> dict[str, list[str]]:
    """CWE-ID → [CVE-IDs that NVD labeled with that CWE]."""
    print(f"Loading {CVE_CWE_INDEX}...")
    mapping = json.loads(CVE_CWE_INDEX.read_text())
    rev: dict[str, list[str]] = defaultdict(list)
    for cve_id, cwe_ids in mapping.items():
        for cwe_id in cwe_ids:
            rev[cwe_id.upper()].append(cve_id.upper())
    print(f"  Reverse index: {len(rev)} CWEs have at least one mapped CVE")
    return rev


def sample_cves_per_cwe(rev: dict[str, list[str]]) -> dict[str, list[str]]:
    """Pick SAMPLES_PER_CWE deterministic samples per CWE."""
    rng = random.Random(SEED)
    sampled: dict[str, list[str]] = {}
    for cwe_id, cves in rev.items():
        n = min(SAMPLES_PER_CWE, len(cves))
        # Sort for determinism, then sample
        sampled[cwe_id] = rng.sample(sorted(cves), n) if n > 0 else []
    total_samples = sum(len(v) for v in sampled.values())
    print(f"  Sampled {total_samples} CVE references across {len(sampled)} CWEs "
          f"(cap {SAMPLES_PER_CWE} per CWE)")
    return sampled


def collect_cve_descriptions(needed: set[str]) -> dict[str, str]:
    """Single-pass scan of cve_chunks.jsonl to extract descriptions for `needed` CVE IDs."""
    print(f"Scanning {CVE_CHUNKS} for {len(needed)} CVE descriptions...")
    descs: dict[str, str] = {}
    with CVE_CHUNKS.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            cve_id = (r.get("identifier") or "").upper()
            if cve_id in needed:
                d = cve_description_from_text(r.get("text") or "")
                if d:
                    descs[cve_id] = d
                if len(descs) == len(needed):
                    break
    print(f"  Collected {len(descs)} CVE descriptions")
    return descs


def augment_chunk_text(orig_text: str, cve_examples: list[tuple[str, str]]) -> str:
    """Append a 'Real-world examples' section to an existing CWE chunk text."""
    if not cve_examples:
        return orig_text
    lines = ["", "## Real-world examples"]
    for cve_id, desc in cve_examples:
        lines.append(f"- {cve_id}: {desc}")
    return orig_text.rstrip() + "\n" + "\n".join(lines) + "\n"


def main() -> None:
    rev = build_reverse_index()
    sampled = sample_cves_per_cwe(rev)
    needed: set[str] = set()
    for cves in sampled.values():
        needed.update(cves)
    descs = collect_cve_descriptions(needed)

    print(f"Reading {CWE_CHUNKS} and writing augmented chunks...")
    total = 0
    augmented_count = 0
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CWE_CHUNKS.open() as f, OUT_PATH.open("w") as out:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            total += 1
            cwe_id = (r.get("identifier") or "").upper()
            if cwe_id in sampled:
                examples = [
                    (cve, descs[cve])
                    for cve in sampled[cwe_id]
                    if cve in descs
                ]
                if examples:
                    r["text"] = augment_chunk_text(r.get("text") or "", examples)
                    r["augmented"] = True
                    r["augmented_with"] = [cve for cve, _ in examples]
                    augmented_count += 1
            out.write(json.dumps(r) + "\n")
    print(f"  Wrote {total} CWE chunks ({augmented_count} augmented) → {OUT_PATH}")


if __name__ == "__main__":
    main()
