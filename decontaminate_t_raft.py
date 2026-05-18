"""
Strip CTI-Bench test CVE-IDs from data/processed/t_raft_dataset.jsonl.

CTI-Bench's 2000 test CVEs are derived from NVD, so they overlap the
150k labeled training set by construction. Without this step the
fine-tuned model would memorize test answers.

Pulls test CVE-IDs from the URL column of both rcm TSVs (authoritative)
and writes a decontaminated dataset back to t_raft_dataset.jsonl
(original preserved as t_raft_dataset.jsonl.predecon).
"""
import csv
import json
import re
import shutil
from pathlib import Path

DATASET = Path("data/processed/t_raft_dataset.jsonl")
BACKUP = Path("data/processed/t_raft_dataset.jsonl.predecon")
TSV_FILES = [
    Path("data/cti-bench/data/cti-rcm.tsv"),
    Path("data/cti-bench/data/cti-rcm-2021.tsv"),
]
CVE_RE = re.compile(r"CVE-\d{4}-\d+")


def load_test_cve_ids() -> set[str]:
    ids: set[str] = set()
    for p in TSV_FILES:
        with p.open() as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                m = CVE_RE.search(row.get("URL", ""))
                if m:
                    ids.add(m.group(0))
    return ids


def main() -> None:
    test_ids = load_test_cve_ids()
    print(f"CTI-Bench test CVE-IDs: {len(test_ids)}")

    if not BACKUP.exists():
        print(f"Backing up {DATASET} -> {BACKUP}")
        shutil.copy(DATASET, BACKUP)
    else:
        print(f"Backup already exists at {BACKUP}, filtering from it")

    src = BACKUP
    tmp = DATASET.with_suffix(".jsonl.tmp")
    kept = 0
    dropped = 0
    with src.open() as fin, tmp.open("w") as fout:
        for line in fin:
            row = json.loads(line)
            if row["cve_id"] in test_ids:
                dropped += 1
                continue
            fout.write(line)
            kept += 1

    tmp.replace(DATASET)
    print(f"Kept: {kept}")
    print(f"Dropped (contamination): {dropped}")
    print(f"Wrote: {DATASET}")


if __name__ == "__main__":
    main()
