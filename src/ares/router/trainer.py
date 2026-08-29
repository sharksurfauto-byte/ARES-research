"""Router Training Loop and Trainer (PRD §4.4).

Implements supervised oracle pretraining for the Router MLP with
Cross-Entropy loss and Switch Transformer load-balancing loss.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .architecture import Router, RouterConfig
from .loss import RouterLoss, SwitchLoadBalancingLoss, generate_oracle_targets
from ..utils.checkpoint import compute_state_dict_sha256


class RouterTrainer:
    """Trainer for the Router MLP Network (PRD §4.4, Option A).

    Trains the Router to predict oracle routing decisions (route to expert if
    base would be wrong, else route to base) using cross-entropy + Switch
    Transformer load-balancing loss.
    """

    def __init__(
        self,
        router: Router,
        device: torch.device,
        config: dict[str, Any] | None = None,
        wandb_logger: Any | None = None,
    ):
        """Initialize RouterTrainer.

        Args:
            router: Router model instance
            device: Computation device
            config: Training configuration dictionary
            wandb_logger: Optional W&B logger
        """
        self.router = router.to(device)
        self.device = device
        self.config = config or {}
        self.wandb_logger = wandb_logger

        lr = self.config.get("learning_rate", 1e-4)
        weight_decay = self.config.get("weight_decay", 0.01)
        lambda_lb = self.config.get("lambda_lb", 0.01)

        self.loss_fn = RouterLoss(
            n_classes=self.router.n_classes,
            lambda_lb=lambda_lb,
        )

        self.optimizer = torch.optim.AdamW(
            self.router.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.get("epochs", 5),
            eta_min=lr * 0.01,
        )

        self.epoch = 0
        self.global_step = 0

    def train_epoch(
        self,
        dataloader: DataLoader,
    ) -> dict[str, float]:
        """Train router for one epoch.

        Args:
            dataloader: DataLoader yielding (representations, targets) batches

        Returns:
            Dictionary of average training metrics for the epoch
        """
        self.router.train()

        total_loss_sum = 0.0
        ce_loss_sum = 0.0
        lb_loss_sum = 0.0
        correct_sum = 0
        total_samples = 0
        abstain_count = 0

        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                x, targets = batch[0].to(self.device), batch[1].to(self.device)
            elif isinstance(batch, dict):
                x = batch["representation"].to(self.device)
                targets = batch["target"].to(self.device)
            else:
                raise ValueError(f"Unsupported batch format: {type(batch)}")

            batch_size = x.size(0)
            logits = self.router.get_logits(x)
            probs = torch.softmax(logits / self.router.temperature, dim=-1)

            loss, loss_dict = self.loss_fn(logits, probs, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Tracking
            total_loss_sum += loss.item() * batch_size
            ce_loss_sum += loss_dict["ce_loss"].item() * batch_size
            lb_loss_sum += loss_dict["lb_loss"].item() * batch_size

            preds = probs.argmax(dim=-1)
            correct_sum += (preds == targets).sum().item()
            abstain_count += (preds == 0).sum().item()
            total_samples += batch_size
            self.global_step += 1

        if total_samples == 0:
            return {}

        metrics = {
            "train/total_loss": total_loss_sum / total_samples,
            "train/ce_loss": ce_loss_sum / total_samples,
            "train/lb_loss": lb_loss_sum / total_samples,
            "train/accuracy": correct_sum / total_samples,
            "train/abstain_rate": abstain_count / total_samples,
        }

        return metrics

    def evaluate(
        self,
        dataloader: DataLoader,
    ) -> dict[str, float]:
        """Evaluate router on validation dataloader.

        Args:
            dataloader: DataLoader yielding (representations, targets) batches

        Returns:
            Dictionary of evaluation metrics
        """
        self.router.eval()

        total_loss_sum = 0.0
        ce_loss_sum = 0.0
        lb_loss_sum = 0.0
        correct_sum = 0
        total_samples = 0
        abstain_count = 0

        entropy_sum = 0.0
        route_counts = torch.zeros(self.router.n_classes, device=self.device)

        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    x, targets = batch[0].to(self.device), batch[1].to(self.device)
                elif isinstance(batch, dict):
                    x = batch["representation"].to(self.device)
                    targets = batch["target"].to(self.device)
                else:
                    raise ValueError(f"Unsupported batch format: {type(batch)}")

                batch_size = x.size(0)
                logits = self.router.get_logits(x)
                probs = torch.softmax(logits / self.router.temperature, dim=-1)

                loss, loss_dict = self.loss_fn(logits, probs, targets)

                total_loss_sum += loss.item() * batch_size
                ce_loss_sum += loss_dict["ce_loss"].item() * batch_size
                lb_loss_sum += loss_dict["lb_loss"].item() * batch_size

                preds = probs.argmax(dim=-1)
                correct_sum += (preds == targets).sum().item()
                abstain_count += (preds == 0).sum().item()
                total_samples += batch_size

                # Routing entropy: -sum(p * log(p))
                batch_entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1).mean().item()
                entropy_sum += batch_entropy * batch_size

                for idx in range(self.router.n_classes):
                    route_counts[idx] += (preds == idx).sum()

        if total_samples == 0:
            return {}

        metrics = {
            "val/total_loss": total_loss_sum / total_samples,
            "val/ce_loss": ce_loss_sum / total_samples,
            "val/lb_loss": lb_loss_sum / total_samples,
            "val/accuracy": correct_sum / total_samples,
            "val/abstain_rate": abstain_count / total_samples,
            "val/entropy": entropy_sum / total_samples,
        }

        # Per route percentages
        for idx in range(self.router.n_classes):
            route_name = "base" if idx == 0 else f"expert_{idx-1}"
            metrics[f"val/route_pct_{route_name}"] = (route_counts[idx] / total_samples).item()

        return metrics

    def train(
        self,
        train_representations: torch.Tensor,
        train_targets: torch.Tensor,
        val_representations: torch.Tensor | None = None,
        val_targets: torch.Tensor | None = None,
        epochs: int = 5,
        batch_size: int = 16,
    ) -> list[dict[str, float]]:
        """Run full training loop over representation tensors.

        Args:
            train_representations: [N, input_dim] training representation tensor
            train_targets: [N] target route indices (0=base, 1..5=experts)
            val_representations: Optional [M, input_dim] validation representation tensor
            val_targets: Optional [M] validation target route indices
            epochs: Number of training epochs
            batch_size: Batch size

        Returns:
            List of metrics dictionaries for each epoch
        """
        train_dataset = TensorDataset(train_representations, train_targets)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_loader = None
        if val_representations is not None and val_targets is not None:
            val_dataset = TensorDataset(val_representations, val_targets)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        history: list[dict[str, float]] = []

        for epoch in range(epochs):
            self.epoch = epoch + 1
            train_metrics = self.train_epoch(train_loader)
            self.scheduler.step()

            epoch_metrics = {**train_metrics, "epoch": self.epoch}

            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                epoch_metrics.update(val_metrics)

            history.append(epoch_metrics)

        return history

    def save_checkpoint(
        self,
        path: str | Path,
        epoch: int | None = None,
        metrics: dict[str, float] | None = None,
    ) -> str:
        """Save router checkpoint compatible with run_ares_pipeline.py and ARES utils.

        Args:
            path: Destination file path
            epoch: Current epoch
            metrics: Metrics dict

        Returns:
            Saved file path string
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state_dict = self.router.state_dict()
        sha256 = compute_state_dict_sha256(state_dict)

        checkpoint_data = {
            # Key expected by run_ares_pipeline.py
            "router_state_dict": state_dict,
            # Key standard in ARES checkpointing
            "model_state_dict": state_dict,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": {
                "input_dim": self.router.input_dim,
                "hidden_dim": self.router.hidden_dim,
                "num_layers": self.router.num_layers,
                "n_experts": self.router.n_experts,
                "dropout": self.router.dropout_rate,
                "temperature": self.router.temperature,
                "top_k": self.router.top_k,
                "routing_mode": self.router.routing_mode,
            },
            "epoch": epoch if epoch is not None else self.epoch,
            "global_step": self.global_step,
            "metrics": metrics or {},
            "sha256": sha256,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        torch.save(checkpoint_data, str(path))
        return str(path)

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        """Load router checkpoint.

        Args:
            path: Checkpoint file path

        Returns:
            Loaded checkpoint dictionary
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Router checkpoint not found at {path}")

        ckpt = torch.load(str(path), map_location=self.device, weights_only=False)

        state_dict = ckpt.get("router_state_dict", ckpt.get("model_state_dict", ckpt))
        self.router.load_state_dict(state_dict)

        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "epoch" in ckpt:
            self.epoch = ckpt["epoch"]
        if "global_step" in ckpt:
            self.global_step = ckpt["global_step"]

        return ckpt
