"""
Prepare T-RAFT dataset for Embedding Fine-Tuning (BAAI/bge-small-en-v1.5).
Converts synthetic (Query, Oracle, Distractors) tuples into {"query": "", "pos": [""], "neg": [""]} format.
"""
import json
from pathlib import Path

IN_FILE = Path("data/processed/t_raft_dataset.jsonl")
OUT_FILE = Path("data/processed/t_raft_embed_train.jsonl")
CWE_CHUNKS = Path("data/processed/cwe_chunks.jsonl")

def main():
    print("Loading CWE chunks for context building...")
    cwe_map = {}
    with CWE_CHUNKS.open("r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            cwe_map[c.get("identifier", "").upper()] = c

    print(f"Reading T-RAFT dataset from {IN_FILE}...")
    dataset = []
    with IN_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            dataset.append(json.loads(line))
            
    print(f"Formatting {len(dataset)} examples for Embedding Fine-Tuning...")
    
    processed_count = 0
    with OUT_FILE.open("w", encoding="utf-8") as f:
        for row in dataset:
            desc = row["description"]
            oracle_cwes = row["oracle_cwes"]
            distractor_cwes = row["distractor_cwes"]
            
            if not oracle_cwes:
                continue
                
            oracle_id = oracle_cwes[0]
            oracle_chunk = cwe_map.get(oracle_id)
            if not oracle_chunk:
                continue
                
            pos_passage = f"[WEAKNESS {oracle_chunk['identifier']} — {oracle_chunk['name']}]\n{oracle_chunk['text']}"
            
            neg_passages = []
            for d_id in distractor_cwes:
                d_chunk = cwe_map.get(d_id)
                if d_chunk:
                    neg_passages.append(f"[WEAKNESS {d_chunk['identifier']} — {d_chunk['name']}]\n{d_chunk['text']}")
            
            if not neg_passages:
                continue
                
            out_row = {
                "query": desc,
                "pos": [pos_passage],
                "neg": neg_passages
            }
            f.write(json.dumps(out_row) + "\n")
            processed_count += 1

    print(f"Done. Wrote {processed_count} embedding training examples to {OUT_FILE}")

if __name__ == "__main__":
    main()
