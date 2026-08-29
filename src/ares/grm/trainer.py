"""GRM (Global Reliability Model) training loop (PRD §4.2).

Implements supervised training on (representation, correctness_label) pairs
plus optional self-supervised pretraining.
"""

from typing import Any

import torch
import torch.nn as nn

from ..utils.checkpoint import load_checkpoint, save_checkpoint
from ..utils.ddp import is_main_process
from ..utils.wandb_utils import log_metrics
from .architecture import GRM


class GRMTrainer:
    """Trainer for the Global Reliability Model.

    PRD §4.2: GRM Training
    1. Supervised phase: Input pooled representation + correctness label
       - Loss: BCE for feasibility + CE for domain classification
    2. Self-supervised phase (optional): Contrastive loss + reconstruction
    """

    def __init__(
        self,
        model: GRM,
        device: torch.device,
        config: dict[str, Any] | None = None,
        wandb_logger: Any | None = None,
    ):
        """Initialize GRM trainer.

        Args:
            model: GRM instance
            device: Computation device
            config: Training configuration
            wandb_logger: W&B logger instance
        """
        self.model = model
        self.device = device
        self.config = config or {}
        self.wandb_logger = wandb_logger

        # Loss functions
        self.domain_criterion = nn.CrossEntropyLoss()
        self.feasibility_criterion = nn.BCELoss()

        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.get("learning_rate", 1e-4),
            weight_decay=self.config.get("weight_decay", 1e-4),
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.config.get("lr_step_size", 10),
            gamma=self.config.get("lr_gamma", 0.5),
        )

        # Training stats
        self.epoch = 0
        self.global_step = 0

    def _move_to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Move batch tensors to device."""
        moved = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                moved[k] = v.to(self.device)
            else:
                moved[k] = v
        return moved

    def train_epoch(
        self,
        representations: torch.Tensor,
        domain_labels: torch.Tensor,
        feasibility_labels: torch.Tensor,
    ) -> dict[str, float]:
        """Train one epoch.

        Args:
            representations: [batch, input_dim] pooled hidden states
            domain_labels: [batch] domain class indices (0=general, 1=math, 2=code, 3=science, 4=reasoning)
            feasibility_labels: [batch] binary labels (0=unreliable, 1=reliable)

        Returns:
            Dictionary of average losses and metrics
        """
        self.model.train()
        total_loss = 0.0
        total_domain_loss = 0.0
        total_feasibility_loss = 0.0
        correct_domain = 0
        correct_feasibility = 0
        total_samples = 0

        # Create dataset indices for shuffling
        n = representations.shape[0]
        permutation = torch.randperm(n)

        # Process in batches
        batch_size = self.config.get("batch_size", 32)
        for start in range(0, n, batch_size):
            indices = permutation[start : start + batch_size]
            batch_repr = representations[indices].to(self.device)
            batch_domain = domain_labels[indices].to(self.device)
            batch_feasibility = feasibility_labels[indices].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            domain_logits, feasibility, global_rel = self.model(batch_repr)

            # Ensure valid label ranges
            batch_domain_clamped = torch.clamp(batch_domain.long(), min=0, max=self.model.domain_classes - 1)
            batch_feas_clamped = torch.clamp(batch_feasibility.float(), min=0.0, max=1.0)

            # Compute domain classification loss
            domain_loss = self.domain_criterion(domain_logits, batch_domain_clamped)

            # Squeeze and clamp feasibility to (1e-7, 1-1e-7) to prevent log(0) CUDA device-side assertions
            feasibility_squeezed = torch.clamp(feasibility.squeeze(-1), min=1e-7, max=1.0 - 1e-7)
            feasibility_loss = self.feasibility_criterion(
                feasibility_squeezed, batch_feas_clamped
            )

            # Combine losses
            loss = domain_loss + feasibility_loss

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Stats
            total_loss += loss.item()
            total_domain_loss += domain_loss.item()
            total_feasibility_loss += feasibility_loss.item()

            # Accuracy
            pred_domain = torch.argmax(domain_logits, dim=-1)
            correct_domain += (pred_domain == batch_domain_clamped).sum().item()
            correct_feasibility += (
                ((feasibility_squeezed > 0.5).float() == batch_feas_clamped).sum().item()
            )
            total_samples += batch_repr.size(0)

        # Step scheduler
        self.scheduler.step()

        # Average metrics
        n_batches = max(1, (n + batch_size - 1) // batch_size)
        metrics = {
            "loss": total_loss / n_batches,
            "domain_loss": total_domain_loss / n_batches,
            "feasibility_loss": total_feasibility_loss / n_batches,
            "domain_accuracy": correct_domain / total_samples if total_samples > 0 else 0.0,
            "feasibility_accuracy": (
                correct_feasibility / total_samples if total_samples > 0 else 0.0
            ),
            "epoch": self.epoch,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }

        return metrics

    def train(
        self,
        representations: torch.Tensor,
        domain_labels: torch.Tensor,
        feasibility_labels: torch.Tensor,
        epochs: int = 10,
        val_representations: torch.Tensor | None = None,
        val_domain_labels: torch.Tensor | None = None,
        val_feasibility_labels: torch.Tensor | None = None,
    ) -> dict[str, list[float]]:
        """Full training loop.

        Args:
            representations: [N, input_dim] tensor of pooled representations
            domain_labels: [N] tensor of domain class indices
            feasibility_labels: [N] tensor of binary feasibility labels
            epochs: Number of training epochs
            val_representations: Optional validation representations
            val_domain_labels: Optional validation domain labels
            val_feasibility_labels: Optional validation feasibility labels

        Returns:
            Dictionary of training history
        """
        history = {"train": [], "val": []}
        representations = representations.detach()
        domain_labels = domain_labels.detach()
        feasibility_labels = feasibility_labels.detach()
        if val_representations is not None:
            val_representations = val_representations.detach()
        if val_domain_labels is not None:
            val_domain_labels = val_domain_labels.detach()
        if val_feasibility_labels is not None:
            val_feasibility_labels = val_feasibility_labels.detach()

        for self.epoch in range(1, epochs + 1):
            # Train one epoch
            train_metrics = self.train_epoch(
                representations,
                domain_labels,
                feasibility_labels,
            )
            history["train"].append(train_metrics)

            # Validation
            if val_representations is not None:
                val_metrics = self._validate(
                    val_representations,
                    val_domain_labels,
                    val_feasibility_labels,
                )
                history["val"].append(val_metrics)
            else:
                history["val"].append({"epoch": self.epoch})

            # Log to W&B
            if self.wandb_logger is not None:
                log_metrics(
                    self.wandb_logger,
                    {f"grm/{k}": v for k, v in train_metrics.items()},
                    step=self.epoch,
                )

            if is_main_process():
                print(
                    f"GRM Epoch {self.epoch}/{epochs} | "
                    f"Loss: {train_metrics['loss']:.4f} | "
                    f"Domain Acc: {train_metrics['domain_accuracy']:.4f} | "
                    f"Feasibility Acc: {train_metrics['feasibility_accuracy']:.4f}"
                )

        return history

    def _validate(
        self,
        representations: torch.Tensor,
        domain_labels: torch.Tensor,
        feasibility_labels: torch.Tensor,
    ) -> dict[str, float]:
        """Validation pass."""
        self.model.eval()
        total_loss = 0.0
        correct_domain = 0
        correct_feasibility = 0
        total_samples = representations.shape[0]

        batch_size = 32
        with torch.no_grad():
            for start in range(0, total_samples, batch_size):
                end = min(start + batch_size, total_samples)
                batch_repr = representations[start:end].to(self.device)
                batch_domain = domain_labels[start:end].to(self.device)
                batch_feasibility = feasibility_labels[start:end].to(self.device)

                batch_domain_clamped = torch.clamp(batch_domain.long(), min=0, max=self.model.domain_classes - 1)
                batch_feas_clamped = torch.clamp(batch_feasibility.float(), min=0.0, max=1.0)

                domain_logits, feasibility, global_rel = self.model(batch_repr)
                domain_loss = nn.CrossEntropyLoss()(domain_logits, batch_domain_clamped)
                feasibility_squeezed = torch.clamp(feasibility.squeeze(-1), min=1e-7, max=1.0 - 1e-7)
                feasibility_loss = nn.BCELoss()(feasibility_squeezed, batch_feas_clamped)
                loss = domain_loss + feasibility_loss

                total_loss += loss.item()
                pred_domain = torch.argmax(domain_logits, dim=-1)
                correct_domain += (pred_domain == batch_domain_clamped).sum().item()
                correct_feasibility += (
                    ((feasibility_squeezed > 0.5).float() == batch_feas_clamped)
                    .sum()
                    .item()
                )

        n_batches = max(1, (total_samples + batch_size - 1) // batch_size)
        return {
            "val_loss": total_loss / n_batches,
            "val_domain_accuracy": correct_domain / total_samples if total_samples > 0 else 0.0,
            "val_feasibility_accuracy": (
                correct_feasibility / total_samples if total_samples > 0 else 0.0
            ),
        }

    def save(self, path: str, config: dict[str, Any] | None = None):
        """Save model checkpoint.

        Args:
            path: Output path
            config: Optional config dictionary
        """
        save_checkpoint(
            model=self.model,
            path=path,
            config=config,
            verify_sha256=True,
        )
        if is_main_process():
            print(f"GRM checkpoint saved to {path}")

    @classmethod
    def load(
        cls,
        model: GRM,
        path: str,
        device: torch.device,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Load model checkpoint.

        Args:
            model: GRM instance to load into
            path: Checkpoint path
            device: Device to load to
            config: Optional config dictionary

        Returns:
            Checkpoint metadata
        """
        metadata = load_checkpoint(
            path=path,
            model=model,
            device=device,
            verify_sha256=True,
        )
        if is_main_process():
            print(f"GRM checkpoint loaded from {path}")
        return metadata
