#!/usr/bin/env python
"""Self-supervised pretraining for GRM (PRD §4.2).

Pretrains the Global Reliability Model on unlabeled data using:
1. Contrastive loss between representations from adjacent layers
2. Reconstruction loss (autoencoder-style)

Usage:
    python scripts/pretrain_grm.py \
        --config configs/reliability/self_supervised.yaml \
        --model_name Qwen/Qwen2.5-0.5B \
        --max_samples 5000 \
        --epochs 5
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from omegaconf import OmegaConf

from ares import GRM, RepresentationCollector, load_backbone
from ares.backbone.config import BackboneConfig
from ares.grm.pretraining import (
    GRMPretrainer,
    PretrainingConfig,
    create_pretraining_dataloader,
)
from ares.utils.wandb_utils import init_wandb


def parse_args():
    parser = argparse.ArgumentParser(description="Self-supervised pretraining for GRM")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/reliability/self_supervised.yaml",
        help="Path to pretraining config",
    )
    parser.add_argument(
        "--model_name", type=str, default="Qwen/Qwen2.5-0.5B", help="Backbone model name"
    )
    parser.add_argument(
        "--max_samples", type=int, default=5000, help="Number of samples to collect for pretraining"
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of pretraining epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument(
        "--device", type=str, default="auto", help="Device to use (cuda, cpu, auto)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/grm_pretrained",
        help="Output directory for pretrained checkpoint",
    )
    parser.add_argument(
        "--dataset", type=str, default="wikitext", help="Unlabeled dataset (wikitext, synthetic)"
    )
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")
    return parser.parse_args()


def collect_unlabeled_representations(
    model_name: str,
    max_samples: int,
    device: torch.device,
    dataset: str = "wikitext",
) -> list[torch.Tensor]:
    """
    Collect multi-layer representations from unlabeled data.

    Args:
        model_name: Backbone model name
        max_samples: Number of samples to collect
        device: Computation device
        dataset: Dataset name

    Returns:
        List of [N, hidden_dim] tensors, one per layer
    """
    # Load backbone
    backbone = load_backbone(model_name, device=device)

    # Create collector
    collector = RepresentationCollector(
        backbone=backbone,
        layers=(-1, -6, -12, -24),
        pooling_method="mean",
        device=device,
    )

    # Create simple dataloader with unlabeled data

    if dataset == "wikitext":
        from datasets import load_dataset

        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train[:1%]")
        texts = ds["text"][:max_samples]
    else:
        # Synthetic data for testing
        texts = [
            "This is a sample text for representation collection. " * 10 for _ in range(max_samples)
        ]

    # Tokenize
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Collect representations
    all_layer_reps = {layer: [] for layer in (-1, -6, -12, -24)}

    for i in range(0, len(texts), 8):  # Process in small batches
        batch_texts = texts[i : i + 8]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            pooled, _, _ = collector.collect(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )

        # pooled is list of [batch, hidden_dim] per layer
        for layer_idx, layer in enumerate((-1, -6, -12, -24)):
            all_layer_reps[layer].append(pooled[layer_idx].cpu())

    # Concatenate per layer
    layer_tensors = []
    for layer in (-1, -6, -12, -24):
        tensor = torch.cat(all_layer_reps[layer], dim=0)[:max_samples]
        layer_tensors.append(tensor)
        print(f"Layer {layer}: {tensor.shape}")

    return layer_tensors


def main():
    args = parse_args()

    # Resolve config path
    config_path = Path(args.config)
    if not config_path.is_absolute():
        if args.config.startswith("configs/"):
            config_path = Path(__file__).parent.parent / args.config
        else:
            config_path = Path(__file__).parent.parent / "configs" / args.config

    config_dict = OmegaConf.load(str(config_path))

    # Override with CLI args
    if args.epochs:
        config_dict.pretraining.training.epochs = args.epochs
    if args.batch_size:
        config_dict.pretraining.training.batch_size = args.batch_size
    if args.learning_rate:
        config_dict.pretraining.training.learning_rate = args.learning_rate

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    try:
        # Initialize W&B
        wandb_logger = None
        if not args.no_wandb:
            wandb_logger = init_wandb(
                config=dict(config_dict),
                project="ares-research",
                run_name=f"grm_pretrain_{args.model_name.split('/')[-1]}",
            )

        logger.info(f"Loading backbone: {args.model_name}")
        # Build BackboneConfig (same pattern as collect_representations.py)
        backbone_cfg = {
            "name": args.model_name,
            "revision": "main",
            "torch_dtype": (
                "float32"
                if device.type == "cpu"
                else ("bfloat16" if "4bit" not in args.model_name else "float16")
            ),
            "device_map": "cpu" if device.type == "cpu" else "auto",
            "use_cache": False,
            "attn_implementation": "eager",
            "load_in_4bit": (
                False
                if device.type == "cpu"
                else ("7B" in args.model_name and "4bit" in args.model_name)
            ),
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": "bfloat16",
            "bnb_4bit_use_double_quant": True,
            "use_peft": False,
            "gradient_checkpointing": True,
            "hidden_state_layers": (-1, -6, -12, -24),
        }
        backbone = load_backbone(BackboneConfig.from_dict(backbone_cfg))
        input_dim = backbone.hidden_size  # 896 for Qwen2.5-0.5B

        logger.info(f"Collecting {args.max_samples} unlabeled samples from {args.dataset}")
        layer_reps = collect_unlabeled_representations(
            model_name=args.model_name,
            max_samples=args.max_samples,
            device=device,
            dataset=args.dataset,
        )

        # Create pretraining config
        pretrain_config = PretrainingConfig(
            contrastive=config_dict.pretraining.contrastive,
            reconstruction=config_dict.pretraining.reconstruction,
            learning_rate=config_dict.pretraining.training.learning_rate,
            batch_size=config_dict.pretraining.training.batch_size,
            epochs=config_dict.pretraining.training.epochs,
            warmup_steps=config_dict.pretraining.training.warmup_steps,
            weight_decay=config_dict.pretraining.training.weight_decay,
            lr_step_size=config_dict.pretraining.training.lr_step_size,
            lr_gamma=config_dict.pretraining.training.lr_gamma,
        )

        # Initialize GRM
        grm = GRM(
            input_dim=input_dim,
            hidden_dim=512,
            num_layers=2,
            num_heads=4,
            dropout=0.1,
            domain_classes=5,
        ).to(device)

        # Create pretrainer
        pretrainer = GRMPretrainer(
            model=grm,
            device=device,
            config=pretrain_config,
            wandb_logger=wandb_logger,
        )

        # Create dataloaders
        train_loader = create_pretraining_dataloader(
            layer_reps,
            pretrain_config,
            shuffle=True,
        )

        # Simple 90/10 split for validation
        n = len(layer_reps[0])
        n_val = max(1, n // 10)
        val_reps = [r[-n_val:] for r in layer_reps]
        train_reps = [r[:-n_val] for r in layer_reps]

        val_loader = create_pretraining_dataloader(
            val_reps,
            pretrain_config,
            shuffle=False,
        )

        logger.info(f"Train samples: {len(train_reps[0])}, Val samples: {len(val_reps[0])}")

        # Run pretraining
        logger.info("Starting self-supervised pretraining...")
        history = pretrainer.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            epochs=pretrain_config.epochs,
        )

        # Save pretrained checkpoint
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        pretrained_path = output_path / "grm_pretrained.pt"

        pretrainer.save_pretrained(
            str(pretrained_path),
            config=dict(config_dict),
        )

        logger.info(f"Pretrained GRM saved to {pretrained_path}")
        print("\nPretraining complete!")
        print(f"Pretrained checkpoint: {pretrained_path}")
        print(f"Final train loss: {history['train'][-1]['loss']:.4f}")
        if history["val"]:
            print(f"Final val loss: {history['val'][-1]['val_loss']:.4f}")

    except Exception as e:
        logger.error(f"Pretraining failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
