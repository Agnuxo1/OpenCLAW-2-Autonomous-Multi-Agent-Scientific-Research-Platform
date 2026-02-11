"""
Training Engine — Autonomous LoRA Fine-Tuning
===============================================
Trains the seed model using LoRA adapters on free GPU resources.

Strategy:
  - Start with tiny model (Qwen2.5-0.5B or SmolLM-135M)
  - Train LoRA adapters on harvested data
  - Merge adapter into base → new, smarter model
  - Push merged model to HuggingFace Hub
  - Repeat with more data → model keeps growing

Free GPU Sources:
  - Kaggle: 30h/week T4 GPU (primary)
  - HuggingFace: AutoTrain (limited free)
  - Google Colab: Burst training sessions

The key insight: we don't need to train a full model.
LoRA adds ~1-4% new parameters per cycle. Over hundreds
of cycles, the model accumulates massive specialized knowledge
while staying lightweight enough for free inference.
"""
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("seed.trainer")


# Model progression ladder
MODEL_LADDER = [
    {
        "name": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "params": "135M",
        "stage": "GERMINATION",
        "min_data": 100,        # Min training entries needed
        "lora_r": 8,
        "lora_alpha": 16,
        "epochs": 3,
        "batch_size": 4,
        "lr": 2e-4,
    },
    {
        "name": "Qwen/Qwen2.5-0.5B-Instruct",
        "params": "0.5B",
        "stage": "GERMINATION",
        "min_data": 500,
        "lora_r": 16,
        "lora_alpha": 32,
        "epochs": 2,
        "batch_size": 4,
        "lr": 1e-4,
    },
    {
        "name": "Qwen/Qwen2.5-1.5B-Instruct",
        "params": "1.5B",
        "stage": "SEEDLING",
        "min_data": 2000,
        "lora_r": 32,
        "lora_alpha": 64,
        "epochs": 2,
        "batch_size": 2,
        "lr": 5e-5,
    },
    {
        "name": "Qwen/Qwen2.5-3B-Instruct",
        "params": "3B",
        "stage": "SAPLING",
        "min_data": 5000,
        "lora_r": 32,
        "lora_alpha": 64,
        "epochs": 1,
        "batch_size": 1,
        "lr": 2e-5,
    },
    {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "params": "7B",
        "stage": "YOUNG_TREE",
        "min_data": 10000,
        "lora_r": 64,
        "lora_alpha": 128,
        "epochs": 1,
        "batch_size": 1,
        "lr": 1e-5,
    },
]


class TrainingEngine:
    """Autonomous LoRA training engine."""
    
    def __init__(self, hf_token: str = None, data_dir: str = "seed_data",
                 state_dir: str = "seed_state"):
        self.hf_token = hf_token or os.environ.get("HF_TOKEN", "")
        self.data_dir = Path(data_dir)
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.growth_log = self._load_growth_log()
    
    def _load_growth_log(self) -> dict:
        """Load training history."""
        log_file = self.state_dir / "growth_log.json"
        if log_file.exists():
            try:
                return json.loads(log_file.read_text())
            except Exception:
                pass
        return {
            "current_stage": "GERMINATION",
            "current_model": MODEL_LADDER[0]["name"],
            "training_cycles": 0,
            "total_entries_trained": 0,
            "adapters_merged": 0,
            "models_published": [],
            "history": [],
        }
    
    def _save_growth_log(self):
        log_file = self.state_dir / "growth_log.json"
        log_file.write_text(json.dumps(self.growth_log, indent=2))
    
    def get_current_stage(self) -> dict:
        """Determine current growth stage based on data available."""
        dataset_file = self.data_dir / "training_dataset.jsonl"
        if not dataset_file.exists():
            return MODEL_LADDER[0]
        
        entry_count = sum(1 for _ in open(dataset_file))
        
        # Find the most advanced model we have enough data for
        best = MODEL_LADDER[0]
        for model in MODEL_LADDER:
            if entry_count >= model["min_data"]:
                best = model
        
        return best
    
    def should_upgrade(self) -> Optional[dict]:
        """Check if we should upgrade to a larger model."""
        current = self.growth_log["current_model"]
        stage = self.get_current_stage()
        
        if stage["name"] != current:
            logger.info(f"🌱 Growth detected! {current} → {stage['name']} ({stage['stage']})")
            return stage
        return None
    
    def generate_training_script(self, output_path: str = None) -> str:
        """
        Generate a self-contained Python training script.
        This script is designed to run on Kaggle/Colab/HF with free GPU.
        It does everything: loads data, trains LoRA, merges, pushes to Hub.
        """
        stage = self.get_current_stage()
        model_name = stage["name"]
        our_model_name = f"Agnuxo/OpenCLAW-SEED-{stage['params']}"
        
        # Check if we already have a fine-tuned version
        prev_models = self.growth_log.get("models_published", [])
        base_model = model_name
        for m in prev_models:
            if stage["params"] in m:
                base_model = m  # Continue from our own model
        
        script = f'''#!/usr/bin/env python3
"""
🌱 SEED Training Script — Auto-generated {datetime.now(timezone.utc).isoformat()}
===========================================================================
This script is FULLY AUTONOMOUS. Upload it to Kaggle/Colab with your data.
It will train, merge, and push the model to HuggingFace automatically.

Stage: {stage["stage"]} ({stage["params"]})
Base model: {base_model}
Output: {our_model_name}
"""
import os
import json

# ===== CONFIGURATION =====
BASE_MODEL = "{base_model}"
OUTPUT_MODEL = "{our_model_name}"
HF_TOKEN = os.environ.get("HF_TOKEN", "")
LORA_R = {stage["lora_r"]}
LORA_ALPHA = {stage["lora_alpha"]}
EPOCHS = {stage["epochs"]}
BATCH_SIZE = {stage["batch_size"]}
LEARNING_RATE = {stage["lr"]}
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
                text = f"### Instruction:\\n{{entry.get('instruction', '')}}\\n\\n"
                if entry.get("input"):
                    text += f"### Input:\\n{{entry['input']}}\\n\\n"
                text += f"### Response:\\n{{entry.get('output', '')}}"
                all_entries.append({{"text": text}})
            except:
                continue

print(f"📊 Loaded {{len(all_entries)}} training entries from {{len(data_files)}} files")

if len(all_entries) < 50:
    print("⚠️ Very small dataset — results may be limited")

dataset = Dataset.from_list(all_entries)

# ===== LOAD MODEL =====
print(f"🧠 Loading base model: {{BASE_MODEL}}")

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

print(f"✅ Model loaded: {{sum(p.numel() for p in model.parameters()):,}} parameters")

# ===== CONFIGURE LoRA =====
print(f"🔧 Configuring LoRA (r={{LORA_R}}, alpha={{LORA_ALPHA}})")
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
print(f"🌱 Trainable: {{trainable:,}} / {{total:,}} ({{100*trainable/total:.2f}}%)")

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
print(f"✅ Training complete! Loss: {{train_result.training_loss:.4f}}")

# ===== SAVE LoRA ADAPTER =====
adapter_path = "./seed_lora_adapter"
trainer.save_model(adapter_path)
print(f"💾 LoRA adapter saved to {{adapter_path}}")

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
print(f"✅ Merged! Final params: {{sum(p.numel() for p in merged_model.parameters()):,}}")

# ===== PUSH TO HUB =====
if HF_TOKEN:
    print(f"📤 Pushing to HuggingFace: {{OUTPUT_MODEL}}")
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
base_model: {{BASE_MODEL}}
---

# 🌱 OpenCLAW SEED — Self-Evolving Model

**Stage:** {stage["stage"]} ({stage["params"]})
**Base:** {{BASE_MODEL}}
**Training entries:** {{len(all_entries)}}
**LoRA rank:** {{LORA_R}}
**Final loss:** {{train_result.training_loss:.4f}}
**Date:** {{__import__('datetime').datetime.now().isoformat()}}

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
    print(f"🎉 Model published: https://huggingface.co/{{OUTPUT_MODEL}}")
else:
    # Save locally
    merged_model.save_pretrained("./seed_merged_model")
    tokenizer.save_pretrained("./seed_merged_model")
    print("💾 Model saved locally (no HF_TOKEN)")

# ===== SAVE TRAINING REPORT =====
report = {{
    "stage": "{stage['stage']}",
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
}}
with open("training_report.json", "w") as f:
    json.dump(report, f, indent=2)

print("\\n" + "="*60)
print("🌳 SEED GROWTH CYCLE COMPLETE")
print(f"   Model: {{OUTPUT_MODEL}}")
print(f"   Stage: {stage['stage']}")
print(f"   Loss:  {{train_result.training_loss:.4f}}")
print(f"   Data:  {{len(all_entries)}} entries")
print("="*60)
'''
        
        if output_path is None:
            output_path = str(self.state_dir / "train_seed.py")
        
        Path(output_path).write_text(script)
        logger.info(f"Training script generated: {output_path}")
        return output_path
    
    def generate_kaggle_notebook(self, output_path: str = None) -> str:
        """Generate a Kaggle notebook JSON for GPU training."""
        stage = self.get_current_stage()
        training_script = self.generate_training_script("/tmp/train_seed.py")
        script_content = Path("/tmp/train_seed.py").read_text()
        
        notebook = {
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {"name": "python", "version": "3.10.0"},
                "kaggle": {
                    "accelerator": "gpu",
                    "dataSources": [],
                    "isGpuEnabled": True,
                    "isInternetEnabled": True,
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4,
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        f"# 🌱 SEED Training — {stage['stage']} ({stage['params']})\n",
                        f"Auto-generated training notebook for OpenCLAW SEED.\n",
                        f"**Run this on Kaggle with GPU enabled!**"
                    ]
                },
                {
                    "cell_type": "code",
                    "metadata": {"execution": {"iopub.status.busy": ""}},
                    "source": [
                        "import os\n",
                        "# Set your HuggingFace token from Kaggle Secrets\n",
                        "from kaggle_secrets import UserSecretsClient\n",
                        "try:\n",
                        "    secrets = UserSecretsClient()\n",
                        "    os.environ['HF_TOKEN'] = secrets.get_secret('HF_TOKEN')\n",
                        "except:\n",
                        "    os.environ['HF_TOKEN'] = ''  # Set manually if needed\n",
                    ],
                    "outputs": [],
                    "execution_count": None,
                },
                {
                    "cell_type": "code",
                    "metadata": {},
                    "source": [
                        "# Download training data from HuggingFace\n",
                        "!pip install -q huggingface_hub\n",
                        "from huggingface_hub import hf_hub_download, HfApi\n",
                        "import os\n",
                        "\n",
                        "api = HfApi()\n",
                        "# Try to download training data from our dataset repo\n",
                        "try:\n",
                        "    files = api.list_repo_files('Agnuxo/OpenCLAW-SEED-data', repo_type='dataset')\n",
                        "    os.makedirs('seed_data', exist_ok=True)\n",
                        "    for f in files:\n",
                        "        if f.endswith('.jsonl'):\n",
                        "            hf_hub_download('Agnuxo/OpenCLAW-SEED-data', f, \n",
                        "                          repo_type='dataset', local_dir='seed_data')\n",
                        "            print(f'Downloaded {f}')\n",
                        "except Exception as e:\n",
                        "    print(f'No remote data: {e}')\n",
                        "    print('Using local data if available')\n",
                    ],
                    "outputs": [],
                    "execution_count": None,
                },
                {
                    "cell_type": "code",
                    "metadata": {},
                    "source": script_content.split("\n"),
                    "outputs": [],
                    "execution_count": None,
                },
            ]
        }
        
        if output_path is None:
            output_path = str(self.state_dir / "seed_training.ipynb")
        
        Path(output_path).write_text(json.dumps(notebook, indent=2))
        logger.info(f"Kaggle notebook generated: {output_path}")
        return output_path
    
    def trigger_hf_autotrain(self, dataset_repo: str = "Agnuxo/OpenCLAW-SEED-data") -> dict:
        """
        Use HuggingFace AutoTrain to trigger training via API.
        This is an alternative to manual Kaggle training.
        """
        stage = self.get_current_stage()
        
        # AutoTrain configuration
        config = {
            "task": "text_generation",
            "base_model": stage["name"],
            "dataset": dataset_repo,
            "text_column": "text",
            "learning_rate": stage["lr"],
            "num_epochs": stage["epochs"],
            "batch_size": stage["batch_size"],
            "lora_r": stage["lora_r"],
            "lora_alpha": stage["lora_alpha"],
            "use_peft": True,
            "quantization": "4bit" if "3B" in stage["name"] or "7B" in stage["name"] else None,
            "push_to_hub": True,
            "hub_model_id": f"Agnuxo/OpenCLAW-SEED-{stage['params']}",
        }
        
        logger.info(f"AutoTrain config for {stage['stage']}: {json.dumps(config, indent=2)}")
        return config
    
    def upload_training_data(self, dataset_repo: str = "Agnuxo/OpenCLAW-SEED-data") -> bool:
        """Upload harvested data to HuggingFace as a dataset."""
        if not self.hf_token:
            logger.warning("No HF_TOKEN — can't upload data")
            return False
        
        try:
            from huggingface_hub import HfApi, create_repo
            api = HfApi(token=self.hf_token)
            
            # Create dataset repo if needed
            try:
                create_repo(dataset_repo, repo_type="dataset", token=self.hf_token, exist_ok=True)
            except Exception:
                pass
            
            # Upload all JSONL files
            uploaded = 0
            for f in self.data_dir.glob("*.jsonl"):
                api.upload_file(
                    path_or_fileobj=str(f),
                    path_in_repo=f.name,
                    repo_id=dataset_repo,
                    repo_type="dataset",
                    token=self.hf_token,
                )
                uploaded += 1
                logger.info(f"Uploaded {f.name}")
            
            # Upload training script
            script_path = self.generate_training_script()
            api.upload_file(
                path_or_fileobj=script_path,
                path_in_repo="train_seed.py",
                repo_id=dataset_repo,
                repo_type="dataset",
                token=self.hf_token,
            )
            
            # Upload Kaggle notebook
            nb_path = self.generate_kaggle_notebook()
            api.upload_file(
                path_or_fileobj=nb_path,
                path_in_repo="seed_training.ipynb",
                repo_id=dataset_repo,
                repo_type="dataset",
                token=self.hf_token,
            )
            
            logger.info(f"✅ Uploaded {uploaded} data files + training scripts to {dataset_repo}")
            return True
            
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            return False
    
    def record_training_result(self, report: dict):
        """Record a training result in the growth log."""
        self.growth_log["training_cycles"] += 1
        self.growth_log["total_entries_trained"] += report.get("training_entries", 0)
        self.growth_log["adapters_merged"] += 1
        
        model_name = report.get("output_model", "")
        if model_name and model_name not in self.growth_log["models_published"]:
            self.growth_log["models_published"].append(model_name)
        
        self.growth_log["current_stage"] = report.get("stage", self.growth_log["current_stage"])
        self.growth_log["current_model"] = model_name or self.growth_log["current_model"]
        
        self.growth_log["history"].append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stage": report.get("stage"),
            "loss": report.get("final_loss"),
            "entries": report.get("training_entries"),
            "model": model_name,
        })
        
        # Keep last 100 history entries
        self.growth_log["history"] = self.growth_log["history"][-100:]
        self._save_growth_log()
        
        logger.info(f"🌳 Growth recorded: cycle #{self.growth_log['training_cycles']}, "
                     f"stage={self.growth_log['current_stage']}")
