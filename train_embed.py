"""
Fine-tune BAAI/bge-small-en-v1.5 using T-RAFT extracted triplets.
Uses MultipleNegativesRankingLoss to push Oracle CWEs closer to descriptions and pull Taxonomic Distractors away.
"""
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

MODEL_ID = "BAAI/bge-small-en-v1.5"
TRAIN_FILE = Path("data/processed/t_raft_embed_train.jsonl")
OUTPUT_DIR = Path("data/models/bge-small-t-raft")

def main():
    print(f"Loading base model: {MODEL_ID}")
    model = SentenceTransformer(MODEL_ID)

    print(f"Loading training data from {TRAIN_FILE}...")
    train_examples = []
    with TRAIN_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            query = data["query"]
            pos = data["pos"][0]
            neg = data["neg"]
            
            # Create an InputExample for each hard negative
            for n in neg:
                train_examples.append(InputExample(texts=[query, pos, n]))
                
    print(f"Created {len(train_examples)} triplet examples for training.")
    
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    
    # MultipleNegativesRankingLoss is extremely effective for triplets with hard negatives
    train_loss = losses.MultipleNegativesRankingLoss(model=model)
    
    OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    
    print("Starting fine-tuning...")
    # Train for 1 epoch. BGE-small fine-tunes quickly.
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=1,
        warmup_steps=100,
        output_path=str(OUTPUT_DIR),
        show_progress_bar=True
    )
    print(f"Fine-tuning complete. Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
