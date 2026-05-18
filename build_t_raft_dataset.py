"""
Generate T-RAFT dataset from NVD-mapped CVEs.
Uses better_rag._cwe_only_candidates to find hard taxonomic distractors.
"""
import json
import itertools
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

import better_rag

CVE_CHUNKS = Path("data/processed/cve_chunks.jsonl")
OUT_FILE = Path("data/processed/t_raft_dataset.jsonl")
WORKERS = 16

def main():
    print("Loading CVE chunks...")
    mapped_cves = []
    with CVE_CHUNKS.open("r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            if c.get("cwe_ids"):
                mapped_cves.append(c)
    
    print(f"Found {len(mapped_cves)} mapped CVEs.")
    
    _counter = itertools.count()
    _embedders = [
        better_rag.embedder,
        better_rag._embedder_1 if better_rag._embedder_1 else better_rag.embedder
    ]

    def _thread_init():
        idx = next(_counter)
        emb = _embedders[idx % len(_embedders)]
        better_rag._thread_local.embedder = emb

    write_lock = threading.Lock()
    
    def process_cve(cve):
        desc = cve["text"]
        true_cwes = [c.upper() for c in cve["cwe_ids"]]
        
        # Get distractors
        candidates = better_rag._cwe_only_candidates(desc, k=5)
        distractors = []
        for cand in candidates:
            cand_id = cand["identifier"]
            if cand_id not in true_cwes:
                distractors.append(cand_id)
                
        return {
            "cve_id": cve["identifier"],
            "description": desc,
            "oracle_cwes": true_cwes,
            "distractor_cwes": distractors
        }

    print("Generating T-RAFT dataset...")
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUT_FILE.open("w", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=WORKERS, initializer=_thread_init) as executor:
            futures = [executor.submit(process_cve, c) for c in mapped_cves]
            for future in tqdm(as_completed(futures), total=len(mapped_cves)):
                res = future.result()
                if res:
                    with write_lock:
                        f.write(json.dumps(res) + "\n")
                        
    print(f"\nDone. Wrote to {OUT_FILE}")

if __name__ == "__main__":
    main()
