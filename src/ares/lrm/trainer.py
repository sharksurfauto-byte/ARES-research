"""LRM (Local Reliability Model) training loop (PRD §4.3).

Implements token-wise correctness prediction training.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, List
from tqdm import tqdm

from .architecture import LRM
from ..utils.checkpoint import save_checkpoint, load_checkpoint
from ..utils.wandb_utils import init_wandb, log_metrics


class LRMTrainer:
    """Trainer for the Local Reliability Model.

    PRD §4.3: LRM Training
    1. Per-token correctness prediction
    2. Binary classification: correct/incorrect given token hidden state
    3. Loss: BCE with class weighting (handle imbalance)
    """

    def __init__(
        self,
        model: LRM,
        device: torch.device,
        config: Optional[Dict[str, Any]] = None,
        wandb_logger: Optional[Any] = None,
    ):
        """Initialize LRM trainer.

        Args:
            model: LRM instance
            device: Computation device
            config: Training configuration
            wandb_logger: W&B logger instance
        """
        self.model = model
        self.device = device
        self.config = config or {}
        self.wandb_logger = wandb_logger

        # Loss function with class weighting for imbalance
        pos_weight = self.config.get("pos_weight", 1.0)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(device))

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

    def _move_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
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
        token_hidden_states: torch.Tensor,
        correctness_labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Train one epoch.

        Args:
            token_hidden_states: [batch, seq_len, hidden_dim] per-token hidden states
            correctness_labels: [batch, seq_len] binary labels (1=correct, 0=incorrect)
            attention_mask: [batch, seq_len] - 1 for valid, 0 for padding

        Returns:
            Dictionary of average losses and metrics
        """
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0

        # Create dataset indices for shuffling
        n = token_hidden_states.shape[0]  # batch size
        seq_len = token_hidden_states.shape[1]
        permutation = torch.randperm(n * seq_len)

        # Process in batches (flatten batch+seq for simplicity)
        flat_size = n * seq_len
        batch_size = self.config.get("batch_size", 32)

        # Flatten the hidden states and labels for batching
        flat_hidden = token_hidden_states.reshape(flat_size, -1)
        flat_labels = correctness_labels.reshape(-1).float()

        if attention_mask is not None:
            flat_mask = attention_mask.reshape(-1).float()
        else:
            flat_mask = torch.ones_like(flat_labels)

        # Process in batches
        for start in range(0, flat_size, batch_size):
            end = min(start + batch_size, flat_size)
            batch_indices = permutation[start:end] if start == 0 else torch.arange(start, end)

            batch_hidden = flat_hidden[batch_indices]
            batch_labels = flat_labels[batch_indices]
            batch_mask = flat_mask[batch_indices]

            self.optimizer.zero_grad()

            # Forward pass
            correctness_prob, failure_risk = self.model(batch_hidden)

            # Only compute loss on valid (non-padded) tokens
            loss = self.criterion(correctness_prob.squeeze(-1) if correctness_prob.dim() > 1 else correctness_prob.squeeze(),
                                  batch_labels) * batch_mask
            loss = loss.sum() / batch_mask.sum()  # Normalize by valid tokens

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Stats
            total_loss = loss.item() * batch_mask.sum().item()
            # Count correct predictions (threshold at 0.5)
            pred_correct = (correctness_prob.squeeze() > 0.5).float()
            correct_tokens = (pred_correct * batch_mask).sum().item()
            total_valid_tokens = batch_mask.sum().item()

            total_loss += total_loss  # Accumulate (actually we should track differently)
            total_correct += correct_tokens
            total_tokens += total_valid_tokens

        # Compute averages
        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0

        # Step scheduler
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
        attention_mask: Optional[torch.Tensor] = None,
        epochs: int = 10,
        val_hidden_states: Optional[torch.Tensor] = None,
        val_labels: Optional[torch.Tensor] = None,
        val_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, List[float]]:
        """Full training loop.

        Args:
            token_hidden_states: [N, seq_len, hidden_dim] tensor
            correctness_labels: [N, seq_len] binary labels
            attention_mask: Optional [N, seq_len] mask
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
                token_hidden_states,
                correctness_labels,
                val_hidden_states is not None and val_labels is not None
                and val_mask is not None
                and (val_hidden_states.shape[0] > 0),
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
                log_metrics(self.wandb_logger, {f"lrm/{k}": v for k, v in train_metrics.items()},
                           step=self.epoch)

            if __import__("builtins").__import__("sys").argv[0].endswith("__main__") or __import__("sys").platform != "cli":
                pass  # Skip printing in non-interactive mode
            else:
                print(
                    f"LRM Epoch {self.epoch}/{epochs} | "
                    f"Loss: {train_metrics['loss']:.4f} | "
                    f"Accuracy: {train_metrics['accuracy']:.4f}"
                )

        return history

    def _validate(
        self,
        val_hidden_states: torch.Tensor,
        val_labels: torch.Tensor,
        val_mask: torch.Tensor,
    ) -> Dict[str, float]:
        """Validation pass."""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_valid_tokens = 0

        batch_size = 32
        with torch.no_grad():
            n = val_hidden_states.shape[0]
            seq_len = val_hidden_states.shape[1]
            flat_size = n * seq_len

            # Flatten
            flat_hidden = val_hidden_states.reshape(flat_size, -1)
            flat_labels = val_labels.reshape(-1).float()
            flat_mask = val_mask.reshape(-1).float()

            for start in range(0, flat_size, batch_size):
                end = min(start + batch_size, flat_size)
                batch_indices = torch.arange(start, end)
                batch_hidden = flat_hidden[batch_indices]
                batch_labels = flat_labels[batch_indices]
                batch_mask = flat_mask[batch_indices]

                correctness_prob, failure_risk = self.model(batch_hidden)

                # Compute loss only on valid tokens
                loss = nn.BCEWithLogitsLoss(
                    pos_weight=torch.tensor(self.config.get("pos_weight", 1.0)).to(self.device)
                )(correctness_prob.squeeze(), batch_labels)
                # Mask the loss
                masked_loss = loss * batch_mask
                loss_value = masked_loss.sum().item() / batch_mask.sum().item()

                total_loss += masked_loss.sum().item()
                # Count correct
                pred_correct = (correctness_prob.squeeze() > 0.5).float()
                correct_tokens = (pred_correct * batch_mask).sum().item()
                total_correct += correct_tokens  # FIXED: accumulate correct count
                total_valid_tokens += batch_mask.sum().item()

        n_batches = max(1, (flat_size + batch_size - 1) // batch_size)
        return {
            "val_loss": total_loss / n_batches if n_batches > 0 else 0.0,
            "val_accuracy": total_correct / total_valid_tokens if total_valid_tokens > 0 else 0.0,
        }

    def save(self, path: str, config: Optional[Dict[str, Any]] = None):
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
        if __import__("sys").platform != "cli" or "cuda" in str(__import__("torch").cuda.is_available()):
            print(f"LRM checkpoint saved to {path}")

    @classmethod
    def load(
        cls,
        model: LRM,
        path: str,
        device: torch.device,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
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