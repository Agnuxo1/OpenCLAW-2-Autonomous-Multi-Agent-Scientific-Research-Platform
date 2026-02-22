#!/usr/bin/env python3
"""
🌱 SEED Training Script — Auto-generated 2026-02-22T01:05:54.187502+00:00
===========================================================================
This script is FULLY AUTONOMOUS. Upload it to Kaggle/Colab with your data.
It will train, merge, and push the model to HuggingFace automatically.

Stage: GERMINATION (135M)
Base model: HuggingFaceTB/SmolLM2-135M-Instruct
Output: Agnuxo/OpenCLAW-SEED-135M
"""
import os
import json

# ===== CONFIGURATION =====
BASE_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
OUTPUT_MODEL = "Agnuxo/OpenCLAW-SEED-135M"
HF_TOKEN = os.environ.get("HF_TOKEN", "")
LORA_R = 8
LORA_ALPHA = 16
EPOCHS = 3
BATCH_SIZE = 4
LEARNING_RATE = 0.0002
MAX_SEQ_LEN = 1024

# ===== INSTALL DEPENDENCIES =====
print("📦 Installing training dependencies...")
os.system("pip install -q transformers>=4.45 datasets peft bitsandbytes trl accelerate huggingface_hub")

from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, 
    TrainingArguments, BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer, SFTConfig
from huggingface_hub import HfApi, login
import torch

# ===== LOGIN =====
if HF_TOKEN:
    login(token=HF_TOKEN)
    print("✅ Logged into HuggingFace")
else:
    print("⚠️ No HF_TOKEN — model won't be pushed")

# ===== LOAD TRAINING DATA =====
print("📊 Loading training data...")
data_files = [f for f in os.listdir(".") if f.endswith(".jsonl")]
if not data_files:
    # Try seed_data directory
    data_dir = "seed_data"
    if os.path.exists(data_dir):
        data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".jsonl")]

if not data_files:
    print("❌ No training data found! Run DataHarvester first.")
    exit(1)

# Combine all JSONL files
all_entries = []
for f in data_files:
    with open(f) as fp:
        for line in fp:
            try:
                entry = json.loads(line.strip())
                # Format as chat
                text = f"### Instruction:\n{entry.get('instruction', '')}\n\n"
                if entry.get("input"):
                    text += f"### Input:\n{entry['input']}\n\n"
                text += f"### Response:\n{entry.get('output', '')}"
                all_entries.append({"text": text})
            except:
                continue

print(f"📊 Loaded {len(all_entries)} training entries from {len(data_files)} files")

if len(all_entries) < 50:
    print("⚠️ Very small dataset — results may be limited")

dataset = Dataset.from_list(all_entries)

# ===== LOAD MODEL =====
print(f"🧠 Loading base model: {BASE_MODEL}")

# Quantization for larger models
use_4bit = "3B" in BASE_MODEL or "7B" in BASE_MODEL
if use_4bit:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb_config,
        device_map="auto", trust_remote_code=True,
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16,
        device_map="auto", trust_remote_code=True,
    )

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"✅ Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

# ===== CONFIGURE LoRA =====
print(f"🔧 Configuring LoRA (r={LORA_R}, alpha={LORA_ALPHA})")
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                     "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"🌱 Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

# ===== TRAIN =====
print("🚀 Starting training...")

training_args = SFTConfig(
    output_dir="./seed_checkpoint",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=4,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,
    max_seq_length=MAX_SEQ_LEN,
    dataset_text_field="text",
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    tokenizer=tokenizer,
)

train_result = trainer.train()
print(f"✅ Training complete! Loss: {train_result.training_loss:.4f}")

# ===== SAVE LoRA ADAPTER =====
adapter_path = "./seed_lora_adapter"
trainer.save_model(adapter_path)
print(f"💾 LoRA adapter saved to {adapter_path}")

# ===== MERGE ADAPTER INTO BASE =====
print("🔀 Merging adapter into base model...")

if use_4bit:
    # For quantized models, reload in fp16 for merging
    base_model_fp16 = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16,
        device_map="auto", trust_remote_code=True,
    )
    merged_model = PeftModel.from_pretrained(base_model_fp16, adapter_path)
else:
    merged_model = PeftModel.from_pretrained(model.base_model, adapter_path)

merged_model = merged_model.merge_and_unload()
print(f"✅ Merged! Final params: {sum(p.numel() for p in merged_model.parameters()):,}")

# ===== PUSH TO HUB =====
if HF_TOKEN:
    print(f"📤 Pushing to HuggingFace: {OUTPUT_MODEL}")
    merged_model.push_to_hub(OUTPUT_MODEL, token=HF_TOKEN, private=False)
    tokenizer.push_to_hub(OUTPUT_MODEL, token=HF_TOKEN, private=False)
    
    # Create model card
    card = f"""---
library_name: transformers
tags:
- seed
- openclaw
- self-evolving
- neuromorphic
license: mit
base_model: {BASE_MODEL}
---

# 🌱 OpenCLAW SEED — Self-Evolving Model

**Stage:** GERMINATION (135M)
**Base:** {BASE_MODEL}
**Training entries:** {len(all_entries)}
**LoRA rank:** {LORA_R}
**Final loss:** {train_result.training_loss:.4f}
**Date:** {__import__('datetime').datetime.now().isoformat()}

## What is SEED?

SEED (Self-Evolving Epistemic Dynamo) is an AI system that **grows autonomously**, 
like a seed becoming a tree. It continuously:
1. Harvests knowledge from ArXiv, Semantic Scholar, and agent interactions
2. Trains itself via LoRA fine-tuning on free GPU resources
3. Merges learned knowledge into its core
4. Evaluates and selects the best version
5. Grows to larger models when enough knowledge is accumulated

## By Francisco Angulo de Lafuente
Advanced AI Systems Laboratory, Madrid, Spain
- GitHub: https://github.com/Agnuxo1
- Scholar: https://scholar.google.com/citations?user=6nOpJ9IAAAAJ
"""
    api = HfApi(token=HF_TOKEN)
    api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=OUTPUT_MODEL,
    )
    print(f"🎉 Model published: https://huggingface.co/{OUTPUT_MODEL}")
else:
    # Save locally
    merged_model.save_pretrained("./seed_merged_model")
    tokenizer.save_pretrained("./seed_merged_model")
    print("💾 Model saved locally (no HF_TOKEN)")

# ===== SAVE TRAINING REPORT =====
report = {
    "stage": "GERMINATION",
    "base_model": BASE_MODEL,
    "output_model": OUTPUT_MODEL,
    "training_entries": len(all_entries),
    "lora_r": LORA_R,
    "lora_alpha": LORA_ALPHA,
    "epochs": EPOCHS,
    "final_loss": train_result.training_loss,
    "trainable_params": trainable,
    "total_params": total,
    "timestamp": __import__("datetime").datetime.now().isoformat(),
}
with open("training_report.json", "w") as f:
    json.dump(report, f, indent=2)

print("\n" + "="*60)
print("🌳 SEED GROWTH CYCLE COMPLETE")
print(f"   Model: {OUTPUT_MODEL}")
print(f"   Stage: GERMINATION")
print(f"   Loss:  {train_result.training_loss:.4f}")
print(f"   Data:  {len(all_entries)} entries")
print("="*60)
