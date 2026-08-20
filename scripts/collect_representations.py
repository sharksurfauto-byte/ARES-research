#!/usr/bin/env python
"""Collect representations from frozen backbone (PRD §11 #4).

Extracts multi-layer hidden states from Qwen2.5 and saves them as a dataset
for training GRM/LRM reliability probes.

Usage:
    python scripts/collect_representations.py \
        --config configs/reliability/representation_collection.yaml \
        --model_name Qwen/Qwen2.5-0.5B \
        --max_samples 100 \
        --analyze
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from omegaconf import DictConfig, OmegaConf

from ares import load_backbone, RepresentationCollector, CollectorConfig, BackboneConfig
from ares.utils.checkpoint import verify_checkpoint
from ares.utils.wandb_utils import init_wandb, log_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Collect representations for ARES")
    parser.add_argument(
        "--config",
        type=str,
        default="representation_collection.yaml",
        help="Config filename (under configs/reliability/)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="Model name from HuggingFace"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=100,
        help="Maximum number of samples to collect"
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Run analysis on collected representations"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="representations",
        help="Output directory for saved representations"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for collection"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (cuda, cpu, auto)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve config path
    config_path = Path("configs/reliability") / args.config

    # Load config
    config_dict = OmegaConf.load(str(config_path))
    cfg = OmegaConf.structured(CollectorConfig) if hasattr(CollectorConfig, '__dataclass_fields__') else config_dict

    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    try:
        # Load backbone
        logger.info(f"Loading backbone: {args.model_name} on {device}")
        backbone_cfg = {
            "name": args.model_name,
            "revision": "main",
            "torch_dtype": "bfloat16" if not args.model_name.__contains__("4bit") else "float16",
            "device_map": "auto",
            "use_cache": False,
            "attn_implementation": "eager",
            "load_in_4bit": "7B" in args.model_name and "4bit" in args.model_name,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": "bfloat16",
            "bnb_4bit_use_double_quant": True,
            "use_peft": False,
            "gradient_checkpointing": True,
            "hidden_state_layers": (-1, -6, -12, -24),
        }
        backbone = load_backbone(BackboneConfig.from_dict(backbone_cfg))

        # Create collector
        collector = RepresentationCollector(
            backbone=backbone,
            layers=cfg.default_layers if hasattr(cfg, 'default_layers') else (-1, -6, -12, -24),
            pooling_method=cfg.default_pooling if hasattr(cfg, 'default_pooling') else "mean",
            device=str(device),
        )

        # Create sample dataset (simple synthetic data for verification)
        # In practice, this would use real datasets (wikitext, gsm8k, etc.)
        from torch.utils.data import DataLoader, Dataset
        import random

        class SyntheticDataset(Dataset):
            def __init__(self, n_samples, max_len=32, vocab_size=151936):
                self.n_samples = n_samples
                self.max_len = max_len
                self.vocab_size = vocab_size

            def __len__(self):
                return self.n_samples

            def __getitem__(self, idx):
                input_ids = torch.randint(0, self.vocab_size, (1, torch.randint(1, self.max_len, (1,)).item()))
                attention_mask = torch.ones(1, input_ids.shape[1])
                return {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                }

            def custom_getitem(self, idx, domain="general", task="classification", metadata=None):
                input_ids = torch.randint(0, self.vocab_size, (1, torch.randint(1, self.max_len, (1,)).item()))
                attention_mask = torch.ones(1, input_ids.shape[1])
                return {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "domain": domain,
                    "task": task,
                    "metadata": metadata or {},
                }

        dataset = SyntheticDataset(max_samples=args.max_samples)

        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
        )

        # Collect representations
        logger.info(f"Collecting representations from {len(dataset)} samples...")
        collector.collect_to_dataset(
            dataloader=dataloader,
            output_dir=args.output_dir,
            save_samples=True,
        )

        # Run analysis if requested
        if args.analyze:
            logger.info("Running representation analysis...")
            # Load saved representations and analyze
            import os
            save_dir = Path(args.output_dir)
            if save_dir.exists():
                pt_file = list(save_dir.glob("*.pt"))
                if pt_file:
                    data = torch.load(pt_file[0])
                    samples = data.get("samples", [])
                    logger.info(f"Analyzed {len(samples)} samples")
                    # Print sample info
                    if samples:
                        s = samples[0]
                        logger.info(f"Sample domain: {s.domain}, task: {s.task}")
                        logger.info(f"Representation shape: {s.representation.shape}")
                        logger.info(f"Prediction: {s.prediction}, Correctness: {s.correctness}")

        logger.info("Representation collection complete!")
        print(f"Collected representations saved to {args.output_dir}/")

    except Exception as e:
        logger.error(f"Representation collection failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()