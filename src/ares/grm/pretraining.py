"""GRM Self-Supervised Pretraining (PRD §4.2).

Implements contrastive learning and reconstruction loss on unlabeled representations
before supervised fine-tuning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..utils.checkpoint import save_checkpoint
from ..utils.wandb_utils import log_metrics
from .architecture import GRM

logger = logging.getLogger(__name__)


@dataclass
class ContrastiveConfig:
    """Configuration for contrastive loss."""

    temperature: float = 0.07
    loss_weight: float = 1.0


@dataclass
class ReconstructionConfig:
    """Configuration for reconstruction loss."""

    loss_weight: float = 0.5
    hidden_dim: int = 512


@dataclass
class PretrainingConfig:
    """Configuration for GRM self-supervised pretraining."""

    contrastive: ContrastiveConfig = field(default_factory=ContrastiveConfig)
    reconstruction: ReconstructionConfig = field(default_factory=ReconstructionConfig)
    learning_rate: float = 5e-5
    batch_size: int = 16
    epochs: int = 5
    warmup_steps: int = 100
    weight_decay: float = 1e-4
    lr_step_size: int = 10
    lr_gamma: float = 0.5


class ReconstructionHead(nn.Module):
    """Autoencoder-style reconstruction head.

    Reconstructs input_dim from GRM's hidden_dim (CLS token).
    """

    def __init__(self, grm_hidden_dim: int, input_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(grm_hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: CLS token [batch, grm_hidden_dim] -> reconstructed input [batch, input_dim]."""
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return reconstructed


class GRMPretrainer:
    """Self-supervised pretraining for GRM using contrastive + reconstruction losses."""

    def __init__(
        self,
        model: GRM,
        device: torch.device,
        config: PretrainingConfig,
        wandb_logger: Any | None = None,
    ):
        """
        Args:
            model: GRM instance to pretrain
            device: Computation device
            config: PretrainingConfig with hyperparameters
            wandb_logger: Optional W&B logger
        """
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.wandb_logger = wandb_logger

        # Contrastive loss temperature (learnable)
        self.temperature = nn.Parameter(torch.tensor(config.contrastive.temperature))

        # Reconstruction head: reconstructs input_dim from GRM's hidden_dim (CLS token)
        self.recon_head = ReconstructionHead(
            grm_hidden_dim=model.hidden_dim,
            input_dim=model.input_dim,
            hidden_dim=config.reconstruction.hidden_dim,
        ).to(device)

        # Combined optimizer for GRM + reconstruction head + temperature
        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.recon_head.parameters()) + [self.temperature],
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config.lr_step_size,
            gamma=config.lr_gamma,
        )

        self.global_step = 0
        self.epoch = 0

    def contrastive_loss(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
    ) -> torch.Tensor:
        """
        InfoNCE contrastive loss between two views of the same sample.

        Args:
            z1: [batch, hidden_dim] - representations from view 1 (e.g., layer -1)
            z2: [batch, hidden_dim] - representations from view 2 (e.g., layer -6)

        Returns:
            Scalar contrastive loss
        """
        batch_size = z1.shape[0]

        # Normalize embeddings
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        # Similarity matrix: [batch, batch]
        logits = torch.matmul(z1, z2.T) / self.temperature.clamp(min=0.01)

        # Positive pairs are on diagonal
        labels = torch.arange(batch_size, device=self.device)

        # Symmetric loss: both directions
        loss_i2j = F.cross_entropy(logits, labels)
        loss_j2i = F.cross_entropy(logits.T, labels)

        return (loss_i2j + loss_j2i) / 2

    def reconstruction_loss(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
    ) -> torch.Tensor:
        """
        MSE reconstruction loss.

        Args:
            original: [batch, input_dim] - original pooled representations
            reconstructed: [batch, input_dim] - decoder output

        Returns:
            Scalar MSE loss
        """
        return F.mse_loss(reconstructed, original)

    def forward_grm_views(
        self,
        representations: list[torch.Tensor],
        layer_indices: tuple[int, int] = (0, 1),
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get GRM outputs for two different layer views.

        Args:
            representations: List of pooled representations per layer [batch, input_dim]
            layer_indices: Which two layers to use as positive pair

        Returns:
            Tuple of (z1, z2, recon_input) where z1,z2 are CLS tokens from GRM
        """
        layer_a = representations[layer_indices[0]].to(self.device)
        layer_b = representations[layer_indices[1]].to(self.device)

        # GRM expects [batch, input_dim] and returns cls_token from transformer
        _, _, _ = self.model(layer_a)  # Run forward to get cls_token
        _, _, _ = self.model(layer_b)

        # We need to access the internal cls_token - modify GRM forward or add method
        # For now, we'll extract from the transformer output directly
        z1 = self._get_cls_token(layer_a)
        z2 = self._get_cls_token(layer_b)

        return z1, z2, layer_a  # Use layer_a as reconstruction target

    def _get_cls_token(self, x: torch.Tensor) -> torch.Tensor:
        """Extract CLS token from GRM transformer (internal method)."""
        if x.dim() == 2:
            x_seq = x.unsqueeze(1)
        else:
            x_seq = x

        if self.model.input_projection is not None:
            x_proj = self.model.input_projection(x_seq)
        else:
            x_proj = x_seq

        x_trans = self.model.transformer(x_proj)
        cls_token = x_trans[:, 0, :]  # [batch, hidden_dim]
        return cls_token

    def train_epoch(
        self,
        dataloader: DataLoader,
    ) -> dict[str, float]:
        """Train one epoch of self-supervised pretraining."""
        self.model.train()
        self.recon_head.train()

        total_loss = 0.0
        total_contrastive = 0.0
        total_reconstruction = 0.0
        num_batches = 0

        for batch in dataloader:
            # Batch contains list of representations per layer
            # Shape: [num_layers, batch, input_dim]
            reps_list = batch["representations"].to(self.device)  # [L, B, D]

            self.optimizer.zero_grad()

            batch_loss = 0.0
            batch_contrastive = 0.0
            batch_reconstruction = 0.0

            # For each sample in batch, use different layer pairs as positives
            num_layers = reps_list.shape[0]
            batch_size = reps_list.shape[1]

            for i in range(batch_size):
                # Get representations for this sample across all layers
                sample_reps = reps_list[:, i, :]  # [L, D]

                # Use adjacent layers as positive pairs
                for layer_idx in range(num_layers - 1):
                    z1 = sample_reps[layer_idx : layer_idx + 1]  # [1, D]
                    z2 = sample_reps[layer_idx + 1 : layer_idx + 2]  # [1, D]

                    # Get CLS tokens from GRM
                    cls1 = self._get_cls_token(z1)
                    cls2 = self._get_cls_token(z2)

                    # Contrastive loss
                    c_loss = self.contrastive_loss(cls1, cls2)
                    batch_contrastive += c_loss.item()
                    batch_loss += self.config.contrastive.loss_weight * c_loss

                    # Reconstruction loss (reconstruct layer representation from CLS)
                    recon1 = self.recon_head(cls1)
                    r_loss = self.reconstruction_loss(z1, recon1)
                    batch_reconstruction += r_loss.item()
                    batch_loss += self.config.reconstruction.loss_weight * r_loss

            if batch_loss > 0:
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.parameters()) + list(self.recon_head.parameters()),
                    max_norm=1.0,
                )
                self.optimizer.step()

            total_loss += batch_loss if isinstance(batch_loss, float) else batch_loss.item()
            total_contrastive += batch_contrastive
            total_reconstruction += batch_reconstruction
            num_batches += 1

        self.scheduler.step()
        self.epoch += 1

        return {
            "loss": total_loss / max(1, num_batches),
            "contrastive_loss": total_contrastive / max(1, num_batches),
            "reconstruction_loss": total_reconstruction / max(1, num_batches),
            "temperature": self.temperature.item(),
            "epoch": self.epoch,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader | None = None,
        epochs: int | None = None,
    ) -> dict[str, list[dict[str, float]]]:
        """Full pretraining loop."""
        epochs = epochs or self.config.epochs
        history = {"train": [], "val": []}

        for self.epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(train_dataloader)
            history["train"].append(train_metrics)

            if self.wandb_logger is not None:
                log_metrics(
                    self.wandb_logger,
                    {f"grm_pretrain/{k}": v for k, v in train_metrics.items()},
                    step=self.epoch,
                )

            logger.info(
                f"GRM Pretrain Epoch {self.epoch}/{epochs} | "
                f"Loss: {train_metrics['loss']:.4f} | "
                f"Contrastive: {train_metrics['contrastive_loss']:.4f} | "
                f"Reconstruction: {train_metrics['reconstruction_loss']:.4f} | "
                f"Temp: {train_metrics['temperature']:.4f}"
            )

            # Validation
            if val_dataloader is not None:
                val_metrics = self._validate(val_dataloader)
                history["val"].append(val_metrics)
                if self.wandb_logger is not None:
                    log_metrics(
                        self.wandb_logger,
                        {f"grm_pretrain/val_{k}": v for k, v in val_metrics.items()},
                        step=self.epoch,
                    )

        return history

    def _validate(
        self,
        dataloader: DataLoader,
    ) -> dict[str, float]:
        """Validation pass."""
        self.model.eval()
        self.recon_head.eval()

        total_loss = 0.0
        total_contrastive = 0.0
        total_reconstruction = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                reps_list = batch["representations"].to(self.device)
                num_layers = reps_list.shape[0]
                batch_size = reps_list.shape[1]

                batch_loss = 0.0
                batch_contrastive = 0.0
                batch_reconstruction = 0.0

                for i in range(batch_size):
                    sample_reps = reps_list[:, i, :]
                    for layer_idx in range(num_layers - 1):
                        z1 = sample_reps[layer_idx : layer_idx + 1]
                        z2 = sample_reps[layer_idx + 1 : layer_idx + 2]

                        cls1 = self._get_cls_token(z1)
                        cls2 = self._get_cls_token(z2)

                        c_loss = self.contrastive_loss(cls1, cls2)
                        batch_contrastive += c_loss.item()
                        batch_loss += self.config.contrastive.loss_weight * c_loss

                        recon1 = self.recon_head(cls1)
                        r_loss = self.reconstruction_loss(z1, recon1)
                        batch_reconstruction += r_loss.item()
                        batch_loss += self.config.reconstruction.loss_weight * r_loss

                total_loss += batch_loss
                total_contrastive += batch_contrastive
                total_reconstruction += batch_reconstruction
                num_batches += 1

        return {
            "val_loss": total_loss / max(1, num_batches),
            "val_contrastive_loss": total_contrastive / max(1, num_batches),
            "val_reconstruction_loss": total_reconstruction / max(1, num_batches),
        }

    def save_pretrained(self, path: str, config: dict[str, Any] | None = None):
        """Save pretrained GRM checkpoint."""
        save_checkpoint(
            model=self.model,
            path=path,
            config=config
            or {
                "pretraining_config": self.config.__dict__,
                "epoch": self.epoch,
            },
            verify_sha256=True,
        )
        logger.info(f"Pretrained GRM saved to {path}")


def create_pretraining_dataloader(
    representations: list[torch.Tensor],
    config: PretrainingConfig,
    shuffle: bool = True,
) -> DataLoader:
    """
    Create DataLoader for pretraining from list of representation tensors.

    Args:
        representations: List of [batch, input_dim] tensors, one per layer
        config: PretrainingConfig
        shuffle: Whether to shuffle

    Returns:
        DataLoader yielding dict with "representations": [L, B, D]
    """
    # Stack representations: [L, B, D]
    stacked = (
        torch.stack(representations, dim=0)
        if isinstance(representations[0], torch.Tensor)
        else representations
    )

    # Create dataset
    from torch.utils.data import TensorDataset

    dataset = TensorDataset(stacked.permute(1, 0, 2))  # [B, L, D]

    def collate_fn(batch):
        # batch is list of tuples (one element per sample since TensorDataset has 1 tensor)
        # Each element is [L, D], stack to [B, L, D] then permute to [L, B, D]
        tensors = [item[0] for item in batch]  # Extract tensor from tuple
        return {"representations": torch.stack(tensors, dim=0).permute(1, 0, 2)}  # [L, B, D]

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )
