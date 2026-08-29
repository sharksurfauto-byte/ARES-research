"""LRM (Local Reliability Model) training loop (PRD §4.3).

Implements token-wise correctness prediction training.
"""

from typing import Any

import torch

from ..utils.checkpoint import load_checkpoint, save_checkpoint
from ..utils.wandb_utils import log_metrics
from .architecture import LRM


class LRMTrainer:
    """Trainer for the Local Reliability Model.

    PRD §4.3: LRM Training
    1. Per-token correctness prediction
    2. Binary classification: correct/incorrect given token hidden state
    3. Loss: Weighted BCE to handle class imbalance
    """

    def __init__(
        self,
        model: LRM,
        device: torch.device,
        config: dict[str, Any] | None = None,
        wandb_logger: Any | None = None,
    ):
        """Initialize LRM trainer.

        Args:
            model: LRM instance
            device: Computation device
            config: Training configuration
            wandb_logger: W&B logger instance
        """
        self.model = model.to(device)
        self.device = device
        self.config = config or {}
        self.wandb_logger = wandb_logger

        # Class weighting for imbalance
        self.pos_weight = float(self.config.get("pos_weight", 1.0))

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
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

    def train_epoch(
        self,
        token_hidden_states: torch.Tensor,
        correctness_labels: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Train one epoch.

        Args:
            token_hidden_states: [batch, seq_len, hidden_dim] or [N, hidden_dim] hidden states
            correctness_labels: [batch, seq_len] or [N] binary labels (1=correct, 0=incorrect)
            attention_mask: Optional [batch, seq_len] or [N] mask (1 for valid, 0 for padding)

        Returns:
            Dictionary of average losses and metrics
        """
        self.model.train()
        accumulated_loss = 0.0
        total_correct = 0
        total_tokens = 0

        # Handle batch & sequence dimensions
        if token_hidden_states.dim() == 3:
            n, seq_len, hidden_dim = token_hidden_states.shape
            flat_size = n * seq_len
            flat_hidden = token_hidden_states.reshape(flat_size, hidden_dim)
            flat_labels = correctness_labels.reshape(-1).float()
            if attention_mask is not None:
                flat_mask = attention_mask.reshape(-1).float()
            else:
                flat_mask = torch.ones_like(flat_labels)
        else:
            flat_size, hidden_dim = token_hidden_states.shape
            flat_hidden = token_hidden_states
            flat_labels = correctness_labels.reshape(-1).float()
            if attention_mask is not None:
                flat_mask = attention_mask.reshape(-1).float()
            else:
                flat_mask = torch.ones_like(flat_labels)

        if flat_labels.size(0) != flat_size:
            raise ValueError(
                f"correctness_labels size ({flat_labels.size(0)}) does not match token_hidden_states size ({flat_size})"
            )
        if flat_mask.size(0) != flat_size:
            raise ValueError(
                f"attention_mask size ({flat_mask.size(0)}) does not match token_hidden_states size ({flat_size})"
            )

        permutation = torch.randperm(flat_size)
        batch_size = self.config.get("batch_size", 32)

        for start in range(0, flat_size, batch_size):
            end = min(start + batch_size, flat_size)
            batch_indices = permutation[start:end]

            batch_hidden = torch.nan_to_num(flat_hidden[batch_indices].to(self.device).float(), nan=0.0)
            batch_labels = torch.clamp(
                torch.nan_to_num(flat_labels[batch_indices].to(self.device).float(), nan=0.0),
                min=0.0,
                max=1.0,
            )
            batch_mask = torch.clamp(
                torch.nan_to_num(flat_mask[batch_indices].to(self.device).float(), nan=0.0),
                min=0.0,
                max=1.0,
            )

            self.optimizer.zero_grad()

            correctness_prob, failure_risk = self.model(batch_hidden)
            prob_flat = correctness_prob.reshape(-1)

            eps = 1e-7
            prob_clamped = prob_flat.clamp(min=eps, max=1.0 - eps)
            weight = batch_labels * self.pos_weight + (1.0 - batch_labels)
            element_loss = -weight * (
                batch_labels * torch.log(prob_clamped)
                + (1.0 - batch_labels) * torch.log(1.0 - prob_clamped)
            )
            masked_loss = element_loss * batch_mask
            valid_count = batch_mask.sum()

            if valid_count.item() > 0:
                loss = (masked_loss.sum() / valid_count).clone()
                loss.backward()
                self.optimizer.step()

                accumulated_loss += loss.item() * valid_count.item()
                pred_correct = (prob_flat > 0.5).float()
                correct_tokens = (((pred_correct == (batch_labels > 0.5).float()) * batch_mask).sum().item())
                total_correct += correct_tokens
                total_tokens += valid_count.item()

        avg_loss = accumulated_loss / total_tokens if total_tokens > 0 else 0.0
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0

        self.scheduler.step()

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "epoch": self.epoch,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }

    def train(
        self,
        token_hidden_states: torch.Tensor,
        correctness_labels: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        epochs: int = 10,
        val_hidden_states: torch.Tensor | None = None,
        val_labels: torch.Tensor | None = None,
        val_mask: torch.Tensor | None = None,
    ) -> dict[str, list[float]]:
        """Full training loop.

        Args:
            token_hidden_states: Hidden states tensor
            correctness_labels: Binary correctness labels
            attention_mask: Optional attention mask
            epochs: Number of training epochs
            val_hidden_states: Optional validation hidden states
            val_labels: Optional validation labels
            val_mask: Optional validation mask

        Returns:
            Dictionary of training history
        """
        history = {"train": [], "val": []}

        for self.epoch in range(1, epochs + 1):
            # Train one epoch
            train_metrics = self.train_epoch(
                token_hidden_states=token_hidden_states,
                correctness_labels=correctness_labels,
                attention_mask=attention_mask,
            )
            history["train"].append(train_metrics)

            # Validation
            if val_hidden_states is not None and val_labels is not None:
                val_metrics = self._validate(
                    val_hidden_states,
                    val_labels,
                    val_mask,
                )
                history["val"].append(val_metrics)
            else:
                history["val"].append({"epoch": self.epoch})

            # Log to W&B
            if self.wandb_logger is not None:
                log_metrics(
                    self.wandb_logger,
                    {f"lrm/{k}": v for k, v in train_metrics.items()},
                    step=self.epoch,
                )

        return history

    def _validate(
        self,
        val_hidden_states: torch.Tensor,
        val_labels: torch.Tensor,
        val_mask: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Validation pass."""
        self.model.eval()
        accumulated_loss = 0.0
        total_correct = 0
        total_tokens = 0

        if val_hidden_states.dim() == 3:
            n, seq_len, hidden_dim = val_hidden_states.shape
            flat_size = n * seq_len
            flat_hidden = val_hidden_states.reshape(flat_size, hidden_dim)
            flat_labels = val_labels.reshape(-1).float()
            if val_mask is not None:
                flat_mask = val_mask.reshape(-1).float()
            else:
                flat_mask = torch.ones_like(flat_labels)
        else:
            flat_size, hidden_dim = val_hidden_states.shape
            flat_hidden = val_hidden_states
            flat_labels = val_labels.reshape(-1).float()
            if val_mask is not None:
                flat_mask = val_mask.reshape(-1).float()
            else:
                flat_mask = torch.ones_like(flat_labels)

        batch_size = 32
        with torch.no_grad():
            for start in range(0, flat_size, batch_size):
                end = min(start + batch_size, flat_size)
                batch_indices = torch.arange(start, end)
                batch_hidden = torch.nan_to_num(flat_hidden[batch_indices].to(self.device).float(), nan=0.0)
                batch_labels = torch.clamp(
                    torch.nan_to_num(flat_labels[batch_indices].to(self.device).float(), nan=0.0),
                    min=0.0,
                    max=1.0,
                )
                batch_mask = torch.clamp(
                    torch.nan_to_num(flat_mask[batch_indices].to(self.device).float(), nan=0.0),
                    min=0.0,
                    max=1.0,
                )

                correctness_prob, failure_risk = self.model(batch_hidden)
                prob_flat = correctness_prob.reshape(-1)

                eps = 1e-7
                prob_clamped = prob_flat.clamp(min=eps, max=1.0 - eps)
                weight = batch_labels * self.pos_weight + (1.0 - batch_labels)
                element_loss = -weight * (
                    batch_labels * torch.log(prob_clamped)
                    + (1.0 - batch_labels) * torch.log(1.0 - prob_clamped)
                )
                masked_loss = element_loss * batch_mask
                valid_count = batch_mask.sum()

                if valid_count.item() > 0:
                    accumulated_loss += masked_loss.sum().item()
                    pred_correct = (prob_flat > 0.5).float()
                    correct_tokens = (((pred_correct == (batch_labels > 0.5).float()) * batch_mask).sum().item())
                    total_correct += correct_tokens
                    total_tokens += valid_count.item()

        return {
            "val_loss": accumulated_loss / total_tokens if total_tokens > 0 else 0.0,
            "val_accuracy": total_correct / total_tokens if total_tokens > 0 else 0.0,
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

    @classmethod
    def load(
        cls,
        model: LRM,
        path: str,
        device: torch.device,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Load model checkpoint.

        Args:
            model: LRM instance to load into
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
        return metadata
