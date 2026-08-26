#!/usr/bin/env python
"""Train Router MLP with Supervised Oracle Pretraining (PRD §4.4, Option A).

Generates oracle routing targets: route to expert if base would be wrong,
else route to base. Trains the Router MLP with cross-entropy + Switch
Transformer load-balancing loss.

Usage:
    python scripts/train_router.py \
        --config configs/experts/router_config.yaml \
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

from ares import ExpertManager, Router, RouterConfig
from ares.experts.manager import ExpertManager as EM, Router as RouterModule
from ares.experts.lora_expert import LoRAExpert, LoRAExpertConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Train Router with oracle pretraining")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experts/router_config.yaml",
        help="Path to router training config",
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of training epochs"
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
        "--expert_checkpoints_dir",
        type=str,
        default="checkpoints/experts",
        help="Directory with trained expert .pt files",
    )
    return parser.parse_args()


def generate_oracle_targets(expert_checkpoints_dir: str, data_loader, device):
    """Generate oracle routing labels.

    For each sample:
      - Run base (frozen Qwen2.5) forward to get prediction
      - If base would be wrong → label = expert_id (1..5)
      - If base would be correct → label = 0 (base route)

    Returns routing_labels: tensor [batch], where 0=base, 1..5=expert
    """
    from ares.backbone.loader import load_backbone
    from ares.representations import RepresentationCollector

    # Load frozen backbone
    backbone = load_backbone("Qwen/Qwen2.5-0.5B", device=device)
    backbone.eval()

    # Load representation collector
    collector = RepresentationCollector(
        backbone=backbone,
        layers=(-1, -6, -12, -24),
        pooling_method="mean",
        device=device,
    )

    router = RouterModule(RouterConfig(input_dim=896, hidden_dim=256, n_experts=5))
    router.to(device)
    router.train()

    # Collect representations + build oracle labels
    routing_labels = []
    representations = []

    for batch in data_loader:
        batch = batch.to(device)
        with torch.no_grad():
            # Collect representations (pooled hidden state from layer -1)
            reps = collector.collect(batch)

        # Run each expert and check if base would be wrong
        # ... (oracle logic: compare base prediction vs expert predictions)
        # For now, placeholder labels (all routed to base = 0)
        B = reps.shape[0]
        labels = torch.zeros(B, dtype=torch.long, device=device)
        routing_labels.append(labels)
        representations.append(reps)

    labels_tensor = torch.cat(routing_labels, dim=0)
    reps_tensor = torch.cat(representations, dim=0)
    return reps_tensor, labels_tensor


def main():
    args = parse_args()

    # Load config
    cfg = OmegaConf.load(args.config) if args.config else OmegaConf.create()
    OmegaConf.update(cfg, "epochs", args.epochs, force=True)
    OmegaConf.update(cfg, "batch_size", args.batch_size, force=True)
    OmegaConf.update(cfg, "lr", args.lr, force=True)
    OmegaConf.update(cfg, "device", args.device, force=True)

    device = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    # Create output directory for router checkpoints
    output_path = Path("checkpoints/router")
    output_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # TODO: Load data (representations from Week 2 collect step)
    # ------------------------------------------------------------------
    # python scripts/collect_representations.py first
    # Then: python scripts/train_router.py

    # For now, quick verification that the router trains without errors
    router = RouterModule(RouterConfig(input_dim=896, hidden_dim=256, n_experts=5))
    router.to(device)
    router.train()

    optimizer = torch.optim.Adam(router.parameters(), lr=args.lr)

    # Dummy forward/backward loop
    for epoch in range(args.epochs):
        print(f"\n=== Epoch {epoch + 1}/{args.epochs} ===")

        # Generate dummy batch
        x = torch.randn(args.batch_size, 896, device=device)

        # Forward through router
        routing_probs = router(x)  # [B, 6] softmax over {base, e0..e4}
        loss_ce = -(routing_probs[:, 0] * 0.5).sum()  # simplified CE: favor base initially

        # Load-balancing loss (Switch Transformer aux loss)
        lb_loss = router.module.load_balancing_loss(routing_probs) if hasattr(router, 'module') else router.load_balancing_loss(routing_probs)

        # Total loss: CE + λ * LB (λ = 0.01 initially)
        total_loss = loss_ce + 0.01 * lb_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Log
        with torch.no_grad():
            ent = -(routing_probs * (routing_probs + 1e-8).log()).sum(dim=-1).mean()
        print(f"  CE loss: {loss_ce.item():.4f} | LB loss: {lb_loss.item():.4f} | Entropy: {ent.item():.4f}")

    # Save router checkpoint
    ckpt_path = output_path / "router.pt"
    torch.save(
        {
            "router_state_dict": router.state_dict(),
            "config": OmegaConf.to_container(cfg),
            "epochs": args.epochs,
        },
        str(ckpt_path),
    )
    print(f"\nSaved router checkpoint to {ckpt_path}")

    print("\n=== Router training complete ===")


if __name__ == "__main__":
    main()