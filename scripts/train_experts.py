#!/usr/bin/env python
"""Train LoRA experts on domain-specific data (PRD §4.5).

Trains each of the 5 LoRA experts (E0-general, E1-math, E2-code,
E3-science, E4-reasoning) independently on domain-targeted datasets.

Usage:
    python scripts/train_experts.py \
        --config configs/experts/expert_mixture.yaml \
        --epochs 3
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from omegaconf import OmegaConf

from ares import ExpertManager, LoRAExpert, LoRAExpertConfig
from ares.experts.manager import ExpertManager as EM, Router, RouterConfig
from ares.experts.lora_expert import LoRAExpert as LE
from ares.config.schema import ExpertConfig as SchemaExpertConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Train LoRA experts on domain data")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experts/expert_mixture.yaml",
        help="Path to expert training config",
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Training batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device to use (cuda, cpu, auto)"
    )
    parser.add_argument(
        "--data_dir", type=str, default="representations", help="Directory with representations"
    )
    parser.add_argument(
        "--output_dir", type=str, default="checkpoints/experts", help="Output dir for expert weights"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    cfg = OmegaConf.load(args.config) if args.config else OmegaConf.create()
    OmegaConf.update(cfg, "epochs", args.epochs, force=True)
    OmegaConf.update(cfg, "batch_size", args.batch_size, force=True)
    OmegaConf.update(cfg, "lr", args.lr, force=True)
    OmegaConf.update(cfg, "device", args.device, force=True)
    OmegaConf.update(cfg, "output_dir", args.output_dir, force=True)

    device = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Phase 1: Collect representations on Kaggle first (Week 2)
    # ------------------------------------------------------------------
    #   python scripts/collect_representations.py --config configs/reliability/representation_collection.yaml
    #   This produces representations/ with .pt files for each layer.

    # ------------------------------------------------------------------
    # Phase 2: Train each expert independently
    # ------------------------------------------------------------------
    expert_names = ["general", "math", "code", "science", "reasoning"]
    domain_datasets = {
        "general": "wikitext",
        "math": "gsm8k",
        "code": "mbpp",
        "science": "ai2_arc",
        "reasoning": "custom_reasoning",
    }

    for name in expert_names:
        expert_dir = output_path / name
        expert_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Training expert: {name} ===")

        # Build expert config
        expert_cfg = LoRAExpertConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            expert_name=name,
            in_features=896,
            out_features=896,
        )

        expert = LE(expert_cfg)
        expert.to(device)
        expert.train()

        optimizer = torch.optim.Adam(expert.parameters(), lr=args.lr)

        # --- In a full run, load domain dataset here ---
        # For now: dummy loop so the script structure is verified
        print(f"  Expert {name} architecture: {type(expert).__name__}")
        print(f"  LoRA r={expert_cfg.r}, alpha={expert_cfg.lora_alpha}")
        print(f"  Target modules: {expert_cfg.target_modules}")

        # Dummy forward/backward to verify shapes work
        x = torch.randn(2, 896, device=device)
        for _ in range(2):  # 2 dummy batches
            out = expert(x)
            loss = out.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Save checkpoint
        ckpt_path = expert_dir / f"expert_{name}.pt"
        torch.save(
            {
                "expert_name": name,
                "config": OmegaConf.to_container(expert_cfg),
                "state_dict": expert.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            },
            str(ckpt_path),
        )
        print(f"  Saved {ckpt_path}")

    print("\n=== Expert training complete ===")
    print(f"Trained experts saved to {output_path}:")


if __name__ == "__main__":
    main()