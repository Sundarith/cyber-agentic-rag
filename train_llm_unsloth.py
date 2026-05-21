"""
QLoRA SFT of Qwen2.5-7B-Instruct on the decontaminated T-RAFT dataset via Unsloth.

Env-gated for smoke vs full:
  T_RAFT_LIMIT       integer N >> 0   train on first N rows only (smoke)
  T_RAFT_EPOCHS      default 2
  T_RAFT_OUTPUT_DIR  default data/models/qwen2.5-7b-t-raft
  T_RAFT_DATA        default data/processed/t_raft_qwen_train.jsonl

After training, the merged-bf16 model is written to T_RAFT_OUTPUT_DIR so vLLM
can serve it as a drop-in replacement for Qwen/Qwen2.5-7B-Instruct.
"""
import json
import os
import random
import time
from pathlib import Path

from unsloth import FastLanguageModel
import torch
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DATA_PATH = Path(os.environ.get("T_RAFT_DATA", "data/processed/t_raft_qwen_train.jsonl"))
OUT_DIR = Path(os.environ.get("T_RAFT_OUTPUT_DIR", "data/models/qwen2.5-7b-t-raft"))
LIMIT = int(os.environ.get("T_RAFT_LIMIT", "0"))
EPOCHS = int(os.environ.get("T_RAFT_EPOCHS", "2"))
# Token-length analysis on 2k samples: p99=5663, max=6811. Cap at 5120 to cover
# 99% of examples and keep step time tractable on a 4090. Outliers will be
# truncated; the gold CWE-ID at the end of the assistant turn is short and
# stays inside the cap because the user-message context dominates length.
MAX_SEQ = int(os.environ.get("T_RAFT_MAX_SEQ", "5120"))
PER_DEV_BATCH = int(os.environ.get("T_RAFT_PER_DEV_BATCH", "2"))
GRAD_ACCUM = int(os.environ.get("T_RAFT_GRAD_ACCUM", "8"))
PACKING = os.environ.get("T_RAFT_PACKING", "1") == "1"
SEED = int(os.environ.get("T_RAFT_SEED", "42"))


def main() -> None:
    print(f"Loading {MODEL_NAME} at 4-bit, max_seq_length={MAX_SEQ}...")
    t0 = time.time()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ,
        dtype=None,
        load_in_4bit=True,
    )
    print(f"Loaded base in {time.time()-t0:.1f}s")

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        use_rslora=True,
    )

    print(f"Loading dataset from {DATA_PATH}...")
    rows = []
    with DATA_PATH.open() as fin:
        for line in fin:
            rows.append(json.loads(line))
    print(f"Loaded {len(rows)} total rows.")

    if LIMIT > 0 and LIMIT < len(rows):
        rng = random.Random(SEED)
        rng.shuffle(rows)
        rows = rows[:LIMIT]
        print(f"Random-sampled {LIMIT} rows (seed={SEED}).")

    texts = []
    for r in rows:
        texts.append(
            tokenizer.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=False)
        )
    dataset = Dataset.from_dict({"text": texts})
    print(f"Training rows: {len(dataset)}")

    OUT_DIR.parent.mkdir(parents=True, exist_ok=True)

    args = SFTConfig(
        per_device_train_batch_size=PER_DEV_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        warmup_ratio=0.03,
        num_train_epochs=EPOCHS,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        output_dir=str(OUT_DIR / "checkpoints"),
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        report_to="none",
        max_seq_length=MAX_SEQ,
        dataset_text_field="text",
        packing=PACKING,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=args,
    )

    ckpt_dir = OUT_DIR / "checkpoints"
    has_ckpt = ckpt_dir.exists() and any(ckpt_dir.glob("checkpoint-*"))
    print(f"Starting training... (resume_from_checkpoint={has_ckpt})")
    t1 = time.time()
    trainer.train(resume_from_checkpoint=has_ckpt)
    print(f"Training complete in {(time.time()-t1)/60:.1f} min")

    merged_dir = OUT_DIR / "merged"
    print(f"Saving merged-16bit model to {merged_dir}...")
    model.save_pretrained_merged(
        str(merged_dir), tokenizer, save_method="merged_16bit"
    )
    print(f"Done. vLLM can serve {merged_dir} as a drop-in Qwen2.5-7B replacement.")


if __name__ == "__main__":
    main()
