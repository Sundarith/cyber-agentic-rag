"""
Prepare T-RAFT dataset for Qwen2.5-7B Instruct fine-tuning.
Converts synthetic (Query, Oracle, Distractors) tuples into {"messages": [...]} format.
"""
import json
import random
from pathlib import Path

IN_FILE = Path("data/processed/t_raft_dataset.jsonl")
OUT_FILE = Path("data/processed/t_raft_qwen_train.jsonl")
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
            
    print(f"Formatting {len(dataset)} examples for Qwen2.5-7B-Instruct...")
    
    SYSTEM_PROMPT = """You are a Cyber Threat Intelligence expert. Answer the question based ONLY on the provided context.
    
CRITICAL RULES:
1. For CVE weakness questions: if the context lists an explicit CWE ID, state it. If the weakness shows 'n/a', you MUST analyze the vulnerability description and determine the most likely CWE.
2. Always end your answer with the CWE ID on its own line."""

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
                
            context_chunks = [oracle_chunk]
            for d_id in distractor_cwes[:4]:
                if d_id in cwe_map:
                    context_chunks.append(cwe_map[d_id])
                    
            random.shuffle(context_chunks)
            
            context_parts = []
            for c in context_chunks:
                context_parts.append(f"[WEAKNESS {c['identifier']} — {c['name']}]\n{c['text']}")
            context_str = "\n\n---\n\n".join(context_parts)
            
            user_msg = f"Context:\n{context_str}\n\nQUESTION: What CWE weakness is associated with the following description?\n{desc}"
            
            cot = (f"Let's think step by step. We need to identify the root cause weakness from the description. "
                   f"Evaluating the retrieved candidates, the distractors present near-miss behaviors, but {oracle_id} "
                   f"({oracle_chunk.get('name', '')}) perfectly matches the root cause described.\n"
                   f"The vulnerability description matches {oracle_id}: {oracle_chunk.get('name', '')}.\n{oracle_id}")
            
            message_row = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": cot}
                ]
            }
            f.write(json.dumps(message_row) + "\n")
            processed_count += 1

    print(f"Done. Wrote {processed_count} conversational training examples to {OUT_FILE}")

if __name__ == "__main__":
    main()
