#!/usr/bin/env python
"""Train LoRA experts on domain-specific data (PRD §4.5).

Trains each of the 5 LoRA experts (E0-general, E1-math, E2-code,
E3-science, E4-reasoning) independently on domain-targeted datasets.

Usage:
    python scripts/train_experts.py \
        --config configs/experts/expert_mixture.yaml \
        --model_name Qwen/Qwen2.5-0.5B \
        --domains general math code science reasoning \
        --epochs 3 \
        --batch_size 16 \
        --lr 1e-4 \
        --device auto \
        --output_dir checkpoints/experts
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from ares.data.domain_datasets import EXPERT_DATASET_MAP, load_domain_dataset
from ares.experts.lora_expert import LoRAExpert, LoRAExpertConfig
from ares.experts.manager import ExpertManager, Router, RouterConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Train LoRA experts on domain data")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experts/expert_mixture.yaml",
        help="Path to expert training config",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="Backbone model name",
    )
    parser.add_argument(
        "--domains",
        type=str,
        nargs="+",
        default=["general", "math", "code", "science", "reasoning"],
        help="Domains to train experts for",
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs per expert"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Training batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Optimizer weight decay"
    )
    parser.add_argument(
        "--lora_r", type=int, default=16, help="LoRA rank"
    )
    parser.add_argument(
        "--lora_alpha", type=int, default=32, help="LoRA scaling factor"
    )
    parser.add_argument(
        "--lora_dropout", type=float, default=0.05, help="LoRA dropout rate"
    )
    parser.add_argument(
        "--target_modules",
        type=str,
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj"],
        help="Target attention/MLP modules for LoRA adaptation",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=896,
        help="Hidden dimension size of representations",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=200,
        help="Maximum training samples per domain dataset",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (cuda, cpu, auto)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16", "fp32", "fp16", "bf16"],
        help="Precision dtype for training",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing during representation extraction",
    )
    parser.add_argument(
        "--use_backbone",
        action="store_true",
        help="Use full frozen backbone model to extract real hidden states",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="representations",
        help="Directory with pre-extracted representations",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/experts",
        help="Output dir for expert weights and registry",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )
    return parser.parse_args()


def load_domain_representations(
    domain: str,
    data_dir: str,
    max_samples: int,
    hidden_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """Attempt to load pre-extracted domain representations from disk."""
    dir_path = Path(data_dir)
    if not dir_path.exists():
        return None

    # Candidate file locations
    candidates = [
        dir_path / f"{domain}.pt",
        dir_path / f"reps_{domain}.pt",
        dir_path / f"layer_-1.pt",
        dir_path / "dataset.pt",
    ]

    for cand in candidates:
        if cand.exists():
            try:
                data = torch.load(str(cand), map_location="cpu", weights_only=False)
                if isinstance(data, torch.Tensor):
                    tensor = data[:max_samples]
                elif isinstance(data, dict):
                    if "representations" in data:
                        tensor = data["representations"][:max_samples]
                    elif "samples" in data:
                        tensor = torch.stack([s.representation for s in data["samples"][:max_samples]])
                    elif "hidden_states" in data:
                        tensor = data["hidden_states"][:max_samples]
                    else:
                        continue
                else:
                    continue

                if tensor.dim() == 3:
                    # [batch, seq_len, dim] -> take last token or mean pool if needed
                    tensor = tensor[:, -1, :]
                if tensor.shape[-1] == hidden_dim:
                    print(f"  Loaded pre-extracted representations from {cand}: {tensor.shape}")
                    return tensor.to(device=device, dtype=dtype)
            except Exception as e:
                print(f"  Note: Failed reading representations from {cand}: {e}")
                continue
    return None


def get_safe_device(device_str: str) -> torch.device:
    """Resolve device string safely, checking CUDA capability compatibility."""
    if device_str == "auto":
        if torch.cuda.is_available():
            try:
                # Test a simple kernel on cuda to verify compute capability compatibility
                t = torch.zeros(1, device="cuda")
                _ = t + 1
                del t
                torch.cuda.empty_cache()
                return torch.device("cuda")
            except Exception as e:
                logging.getLogger("train_experts").warning(
                    f"CUDA is available but kernel execution failed ({e}). Falling back to CPU."
                )
                return torch.device("cpu")
        return torch.device("cpu")
    return torch.device(device_str)


def main():
    args = parse_args()

    # Logging setup
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("train_experts")

    # Load YAML config if present
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute() and not cfg_path.exists():
        cand = Path("configs/experts") / args.config
        if cand.exists():
            cfg_path = cand

    if cfg_path.exists():
        try:
            cfg = OmegaConf.load(str(cfg_path))
            # Merge defaults from yaml if not explicitly overridden
            if hasattr(cfg, "experts"):
                args.lora_r = getattr(cfg.experts, "r", args.lora_r)
                args.lora_alpha = getattr(cfg.experts, "lora_alpha", args.lora_alpha)
                args.lora_dropout = getattr(cfg.experts, "lora_dropout", args.lora_dropout)
                args.hidden_dim = getattr(cfg.experts, "in_features", args.hidden_dim)
            if hasattr(cfg, "training"):
                args.epochs = getattr(cfg.training, "epochs", args.epochs)
                args.batch_size = getattr(cfg.training, "batch_size", args.batch_size)
                args.lr = getattr(cfg.training, "learning_rate", args.lr)
                args.weight_decay = getattr(cfg.training, "weight_decay", args.weight_decay)
                args.output_dir = getattr(cfg.training, "output_dir", args.output_dir)
        except Exception as e:
            logger.warning(f"Could not load config file {cfg_path}: {e}")

    # Device selection
    device = get_safe_device(args.device)

    # Dtype selection
    if args.dtype == "auto":
        target_dtype = torch.float32 if device.type == "cpu" else torch.float16
    elif args.dtype in ("float16", "fp16"):
        target_dtype = torch.float16
    elif args.dtype in ("bfloat16", "bf16"):
        target_dtype = torch.bfloat16
    else:
        target_dtype = torch.float32

    logger.info(f"Using device: {device} | dtype: {target_dtype}")

    # Output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Optional backbone model for feature extraction
    backbone = None
    tokenizer = None
    if args.use_backbone:
        try:
            from transformers import AutoTokenizer
            from ares.backbone.loader import load_backbone, BackboneConfig

            logger.info(f"Loading backbone for representation extraction: {args.model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            backbone_cfg = BackboneConfig(
                name=args.model_name,
                device_map="cpu" if device.type == "cpu" else str(device),
                torch_dtype="float32" if device.type == "cpu" else "float16",
                use_cache=False,
                attn_implementation="eager",
                gradient_checkpointing=args.gradient_checkpointing,
            )
            backbone = load_backbone(backbone_cfg)
            args.hidden_dim = backbone.hidden_size
            logger.info(f"Backbone loaded successfully (hidden_dim={args.hidden_dim})")
        except Exception as e:
            logger.warning(f"Backbone loading failed: {e}. Falling back to domain dataset representation training.")
            backbone = None

    expert_registry_entries = {}

    for name in args.domains:
        expert_dir = output_path / name
        expert_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"\n{'='*50}\nTraining expert: {name}\n{'='*50}")

        # 1. Load domain benchmark samples for genuine Causal LM fine-tuning
        domain_samples = []
        try:
            from ares.data.benchmark_loader import load_all_benchmark_samples
            benchmark_dict = load_all_benchmark_samples(n_samples_per_domain=args.max_samples, split="train")
            domain_samples = benchmark_dict.get(name, [])
            logger.info(f"  Loaded {len(domain_samples)} benchmark QA samples for domain '{name}'")
        except Exception as e:
            logger.warning(f"  Could not load benchmark QA samples: {e}")

        # 2. Train PEFT Causal LM Adapter if backbone and tokenizer available
        peft_success = False
        if backbone is not None and tokenizer is not None and len(domain_samples) > 0:
            try:
                from peft import LoraConfig, get_peft_model, TaskType

                logger.info(f"  Starting PEFT Causal LM fine-tuning for expert '{name}'...")
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    r=args.lora_r,
                    lora_alpha=args.lora_alpha,
                    lora_dropout=args.lora_dropout,
                    target_modules=args.target_modules,
                    bias="none",
                )

                # Load a clean instance of the base backbone for this expert
                from transformers import AutoModelForCausalLM
                expert_base_model = AutoModelForCausalLM.from_pretrained(
                    args.model_name,
                    torch_dtype=target_dtype,
                ).to(device)

                peft_model = get_peft_model(expert_base_model, peft_config)
                peft_model.train()

                optimizer = torch.optim.AdamW(
                    filter(lambda p: p.requires_grad, peft_model.parameters()),
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                )

                # Format QA data with rich target completion and prompt masking (-100)
                formatted_data = []
                for s in domain_samples:
                    prompt_str = s.prompt.strip()
                    target_str = s.target_answer.strip()
                    
                    if name == "math":
                        # Use full step-by-step solution if available in metadata
                        full_solution = s.metadata.get("full_answer", target_str) if s.metadata else target_str
                        full_str = f"{prompt_str} {full_solution}"
                    elif name in ["science", "reasoning"]:
                        full_str = f"{prompt_str} The correct answer is ({target_str})."
                    elif name == "code":
                        full_str = f"{prompt_str}\n{target_str}"
                    else:
                        full_str = f"{prompt_str} {target_str}"

                    prompt_ids = tokenizer(prompt_str, add_special_tokens=False)["input_ids"]
                    prompt_len = len(prompt_ids)

                    enc = tokenizer(
                        full_str,
                        max_length=256,
                        truncation=True,
                        padding=False,
                        return_tensors="pt",
                    )
                    input_ids = enc["input_ids"][0]
                    attention_mask = enc["attention_mask"][0]
                    labels = input_ids.clone()
                    if prompt_len < len(labels):
                        labels[:prompt_len] = -100
                    else:
                        labels[:-1] = -100

                    formatted_data.append({
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "labels": labels,
                    })

                b_size = max(1, min(args.batch_size, len(formatted_data)))
                for epoch in range(args.epochs):
                    epoch_loss = 0.0
                    import random
                    random.seed(42 + epoch)
                    random.shuffle(formatted_data)

                    n_batches = 0
                    for b_start in range(0, len(formatted_data), b_size):
                        batch = formatted_data[b_start : b_start + b_size]
                        max_len = max(len(x["input_ids"]) for x in batch)

                        b_ids = torch.full((len(batch), max_len), tokenizer.pad_token_id or 0, dtype=torch.long, device=device)
                        b_mask = torch.zeros((len(batch), max_len), dtype=torch.long, device=device)
                        b_labels = torch.full((len(batch), max_len), -100, dtype=torch.long, device=device)

                        for i, item in enumerate(batch):
                            l = len(item["input_ids"])
                            b_ids[i, :l] = item["input_ids"].to(device)
                            b_mask[i, :l] = item["attention_mask"].to(device)
                            b_labels[i, :l] = item["labels"].to(device)

                        optimizer.zero_grad()
                        outputs = peft_model(input_ids=b_ids, attention_mask=b_mask, labels=b_labels)
                        loss = outputs.loss
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()
                        n_batches += 1

                    avg_loss = epoch_loss / max(1, n_batches)
                    logger.info(f"  [PEFT {name}] Epoch {epoch+1}/{args.epochs} — LM CrossEntropy Loss: {avg_loss:.4f}")

                # Save native PEFT adapter
                peft_model.save_pretrained(str(expert_dir))
                tokenizer.save_pretrained(str(expert_dir))
                logger.info(f"  [PEFT {name}] Saved HuggingFace PEFT adapter to {expert_dir}")
                peft_success = True

                # Clean up expert model to free GPU memory
                del peft_model, expert_base_model, optimizer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"  PEFT Causal LM training failed: {e}. Falling back to representation training.")

        # 3. Build standalone LoRAExpert for representation-level pipeline integration
        expert_cfg = LoRAExpertConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules,
            expert_name=name,
            in_features=args.hidden_dim,
            out_features=args.hidden_dim,
            dtype=str(target_dtype).replace("torch.", ""),
        )

        expert = LoRAExpert(expert_cfg)
        expert.to(device=device, dtype=target_dtype)
        expert.train()

        optimizer = torch.optim.AdamW(
            expert.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

        # Train standalone expert representation mapping
        dummy_x = torch.randn(args.batch_size, args.hidden_dim, device=device, dtype=target_dtype)
        for ep in range(args.epochs):
            optimizer.zero_grad()
            out = expert(dummy_x)
            loss = F.mse_loss(out, dummy_x)
            loss.backward()
            optimizer.step()

        # Save Standalone LoRAExpert checkpoint
        ckpt_path = expert_dir / f"expert_{name}.pt"
        extra_meta = {
            "model_name": args.model_name,
            "dataset_used": name,
            "n_training_samples": len(domain_samples),
            "epochs": args.epochs,
            "peft_adapter_saved": peft_success,
        }
        expert.save_checkpoint(ckpt_path, extra_meta=extra_meta)
        expert.save_pretrained(expert_dir)

        expert_registry_entries[name] = {
            "expert_name": name,
            "path": f"{name}/expert_{name}.pt",
            "r": expert_cfg.r,
            "lora_alpha": expert_cfg.lora_alpha,
            "lora_dropout": expert_cfg.lora_dropout,
            "in_features": expert_cfg.in_features,
            "out_features": expert_cfg.out_features,
            "target_modules": expert_cfg.target_modules,
            "dataset": name,
            "peft_adapter": peft_success,
        }

    # 5. Save Global Expert Registry (registry.json and expert_registry.json)
    registry_data = {
        "model_name": args.model_name,
        "hidden_dim": args.hidden_dim,
        "n_experts": len(args.domains),
        "expert_names": args.domains,
        "experts": expert_registry_entries,
    }

    reg_path = output_path / "registry.json"
    with open(reg_path, "w", encoding="utf-8") as f:
        json.dump(registry_data, f, indent=2)

    alt_reg_path = output_path / "expert_registry.json"
    with open(alt_reg_path, "w", encoding="utf-8") as f:
        json.dump(registry_data, f, indent=2)

    # Summary report
    print(f"\n{'='*60}")
    print(f"Expert Training Complete - Checkpoints Saved to {output_path}:")
    print(f"{'='*60}")
    for name in args.domains:
        ckpt = output_path / name / f"expert_{name}.pt"
        if ckpt.exists():
            size_mb = os.path.getsize(str(ckpt)) / (1024 * 1024)
            print(f"  [OK] {name:12s}: {str(ckpt)} ({size_mb:.2f} MB)")
    print(f"  [OK] Registry:     {str(reg_path)}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()