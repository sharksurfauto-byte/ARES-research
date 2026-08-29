#!/usr/bin/env python
"""Harvest Real Multi-Domain Representations & Correctness Labels (PRD §4.1).

Harvests representations from real benchmark datasets (GSM8K, MBPP, AI2-ARC, WikiText, CommonsenseQA):
1. Evaluates Qwen base model on benchmark prompts using batched GPU generation
2. Evaluates model answers against ground-truth targets to get genuine correctness labels
3. Collects hidden representations from target layers (-1, -6, -12, -24)
4. Saves structured RepresentationDataset partitions ready for GRM, LRM, and Router training.

Usage:
    python scripts/harvest_real_data.py \
        --model_name Qwen/Qwen2.5-0.5B \
        --samples_per_domain 500 \
        --batch_size 16 \
        --output_dir representations/multi_domain
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from ares import RepresentationCollector, RepresentationDataset, RepresentationSample, load_backbone
from ares.data import (
    BenchmarkSample,
    evaluate_prediction,
    load_all_benchmark_samples,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Harvest real multi-domain representations for ARES")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="Backbone model name (Qwen/Qwen2.5-0.5B, 1.5B, 7B)",
    )
    parser.add_argument(
        "--samples_per_domain",
        type=int,
        default=500,
        help="Number of samples to collect per domain (e.g. 500 = 2500 total samples)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for parallel GPU inference",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="representations/multi_domain",
        help="Directory to save harvested representations",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device (cuda, cpu, auto)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Max sequence length for tokenization",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=48,
        help="Max new tokens to generate for evaluation",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Load Backbone Model & Tokenizer
    logger.info(f"Loading backbone model: {args.model_name}")
    backbone = load_backbone(args.model_name, device=device)
    raw_model = getattr(backbone, "model", getattr(backbone, "_model", backbone))
    if hasattr(raw_model, "eval"):
        raw_model.eval()
    # Disable gradient checkpointing for pure inference (saves overhead)
    if hasattr(raw_model, "gradient_checkpointing_disable"):
        raw_model.gradient_checkpointing_disable()
    logger.info(f"Model device: {next(raw_model.parameters()).device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Initialize Representation Collector
    collector = RepresentationCollector(
        backbone=backbone,
        layers=(-1, -6, -12, -24),
        pooling_method="mean",
        device=device,
    )

    # 3. Load Real Benchmark Samples across all 5 domains
    n_train = int(args.samples_per_domain * 0.8)
    n_val = max(10, int(args.samples_per_domain * 0.2))
    logger.info(f"Loading benchmark datasets (Train: {n_train}/domain, Val: {n_val}/domain)...")

    train_domain_samples = load_all_benchmark_samples(
        n_samples_per_domain=n_train,
        split="train",
    )
    val_domain_samples = load_all_benchmark_samples(
        n_samples_per_domain=n_val,
        split="val",
    )

    def process_samples(
        domain_dict: Dict[str, List[BenchmarkSample]],
        split_name: str,
    ) -> RepresentationDataset:
        collected_samples: List[RepresentationSample] = []
        domain_stats = {d: {"total": 0, "correct": 0} for d in domain_dict.keys()}

        logger.info(f"Harvesting representations for split: {split_name.upper()}...")
        for domain, samples in domain_dict.items():
            pbar = tqdm(
                range(0, len(samples), args.batch_size),
                desc=f"[{split_name.upper()}] {domain:10s}",
                unit="batch",
            )
            for batch_idx in pbar:
                batch_samples = samples[batch_idx : batch_idx + args.batch_size]
                prompts = [s.prompt for s in batch_samples]

                # Batched tokenization
                encoded = tokenizer(
                    prompts,
                    padding=True,
                    truncation=True,
                    max_length=args.max_length,
                    return_tensors="pt",
                ).to(device)

                input_ids = encoded["input_ids"]
                attention_mask = encoded["attention_mask"]

                # Batched generation
                with torch.no_grad():
                    gen_ids = raw_model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                    
                    # Evaluate correctness per sample in batch
                    prompt_len = input_ids.shape[1]
                    batch_labels = []
                    for idx, s in enumerate(batch_samples):
                        gen_text = tokenizer.decode(gen_ids[idx][prompt_len:], skip_special_tokens=True)
                        is_correct = evaluate_prediction(
                            prediction=gen_text,
                            target=s.target_answer,
                            eval_type=s.eval_type,
                        )
                        domain_stats[domain]["total"] += 1
                        if is_correct:
                            domain_stats[domain]["correct"] += 1
                        batch_labels.append(1 if is_correct else 0)

                    # Collect multi-layer representations
                    labels_tensor = torch.tensor(batch_labels, device=device)
                    pooled, logits, meta_samples = collector.collect(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels_tensor,
                        metadata={"domain": domain},
                    )

                if meta_samples:
                    collected_samples.extend(meta_samples)

                # Update live progress bar with rolling accuracy
                tot = domain_stats[domain]["total"]
                cor = domain_stats[domain]["correct"]
                acc = (cor / tot * 100.0) if tot > 0 else 0.0
                pbar.set_postfix({"samples": tot, "base_acc": f"{acc:.1f}%"})

        # Print domain accuracy summary
        print(f"\n{'='*55}")
        print(f"--- {split_name.upper()} BASE MODEL ACCURACY SUMMARY ---")
        print(f"{'='*55}")
        for d, stats in domain_stats.items():
            tot = stats["total"]
            acc = (stats["correct"] / tot * 100.0) if tot > 0 else 0.0
            print(f"  {d:12s}: {acc:5.1f}% Accuracy ({stats['correct']}/{tot})")
        print(f"{'='*55}\n")

        return RepresentationDataset(collected_samples)

    # 4. Process Train & Validation Datasets
    train_dataset = process_samples(train_domain_samples, "train")
    val_dataset = process_samples(val_domain_samples, "val")

    # 5. Save Datasets to Disk
    train_path = output_path / "train.pt"
    val_path = output_path / "val.pt"

    train_dataset.save(str(train_path))
    val_dataset.save(str(val_path))

    logger.info(f"Successfully harvested and saved representations:")
    logger.info(f"  Train: {train_path} ({len(train_dataset)} representation vectors)")
    logger.info(f"  Val:   {val_path} ({len(val_dataset)} representation vectors)")
    print(f"\n[DONE] Real multi-domain dataset saved to {output_path}")


if __name__ == "__main__":
    main()
