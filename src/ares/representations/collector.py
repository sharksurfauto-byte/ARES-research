"""Representation Collector (PRD §3.2.2).

Extracts hidden states from multiple layers of the frozen backbone,
with support for multiple pooling methods and dataset storage.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple, TYPE_CHECKING

# TYPE_CHECKING imports avoid circular dependencies at runtime
if TYPE_CHECKING:
    from ares.backbone.base import Backbone
    from ares.representations.dataset import RepresentationDataset

import torch.nn.functional as F
from dataclasses import dataclass, field

from .pooling import last_token_pool, mean_pool, max_pool

# Type aliases
LayerIndices = Tuple[int, ...]
HiddenStates = List[torch.Tensor]  # List of [batch, seq_len, hidden_dim]


@dataclass
class RepresentationSample:
    """Single sample from the representation collector (PRD §3.2.2)."""
    sample_id: str
    domain: str  # general, math, code, science, reasoning
    task: str
    layer: int
    representation: torch.Tensor  # [hidden_dim] after pooling
    logits: torch.Tensor
    prediction: str
    correctness: bool
    confidence: float
    entropy: float
    margin: float
    attention_mask: Optional[torch.Tensor] = None


@dataclass
class CollectorConfig:
    """Configuration for RepresentationCollector."""
    # Which layers to extract from (negative = from end)
    # PRD §3.2.2: {-1, -6, -12, -24}
    default_layers: LayerIndices = (-1, -6, -12, -24)

    # Pooling method for representation vectors
    # PRD §3.2.2: last-token, mean-pooled, max-pooled
    default_pooling: str = "mean"

    # Whether to extract per-token representations for LRM
    extract_per_token: bool = True

    # Device for computation
    device: str = "auto"


class RepresentationCollector:
    """Extract multi-layer hidden states from frozen backbone.

    Based on PRD §3.2.2: Representation Collector (layers {-1, -6, -12, -24},
    pooled hidden states). Collects representations with metadata for
    GRM/LRM training and evaluation.
    """

    def __init__(
        self,
        backbone,
        layers: LayerIndices = None,
        pooling_method: str = "mean",
        device: str = "auto",
    ):
        """Initialize the collector.

        Args:
            backbone: Frozen Qwen2.5 backbone
            layers: Layer indices to extract (default: {-1, -6, -12, -24})
            pooling_method: Pooling strategy ("last_token", "mean", "max")
            device: Computation device
        """
        self.backbone = backbone
        # Use provided layers or fall back to standard layers
        if layers is not None:
            self.layers = layers
        elif hasattr(backbone, 'hidden_state_layers'):
            self.layers = backbone.hidden_state_layers
        else:
            self.layers = (-1, -6, -12, -24)
        self.pooling_method = pooling_method
        self.device = torch.device(device) if device != "auto" else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        # Ensure backbone settings (safe check - works for QwenBackbone)
        try:
            if hasattr(self.backbone, '_model') and hasattr(self.backbone._model, 'config'):
                self.backbone._model.config.use_cache = False
        except Exception:
            pass
        try:
            self.backbone._model.eval()
        except Exception:
            pass

    @classmethod
    def from_config(
        cls,
        backbone,
        config: CollectorConfig,
    ) -> "RepresentationCollector":
        """Create collector from config."""
        return cls(
            backbone=backbone,
            layers=config.default_layers,
            pooling_method=config.default_pooling,
            device=config.device,
        )

    def _run_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool = True,
        output_attentions: bool = False,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Run forward pass and extract hidden states.

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            output_hidden_states: Whether to return hidden states from all layers
            output_attentions: Whether to return attention weights

        Returns:
            Tuple of (hidden_states from target layers, logits)
        """
        with torch.no_grad():
            outputs = self.backbone.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
                # use_cache is handled by the backbone wrapper
            )

        # Extract hidden states - handle different output formats
        hidden_states = []
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            # Handle tuple/list of tensors
            if isinstance(outputs.hidden_states, (list, tuple)):
                hidden_states = list(outputs.hidden_states)
            else:
                hidden_states = [outputs.hidden_states]

        logits = outputs.logits  # [batch, seq_len, vocab_size]

        return hidden_states, logits

    def _pool_representations(
        self,
        hidden_states: List[torch.Tensor],
        attention_mask: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Apply pooling to extract representation vectors from each layer.

        Args:
            hidden_states: List of [batch, seq_len, hidden_dim] from target layers
            attention_mask: [batch, seq_len]

        Returns:
            List of [batch, hidden_dim] pooled representations
        """
        pooled = []
        for hs in hidden_states:
            if self.pooling_method == "last_token":
                vec = last_token_pool(hs, attention_mask)
            elif self.pooling_method == "mean":
                vec = mean_pool(hs, attention_mask)
            elif self.pooling_method == "max":
                vec = max_pool(hs, attention_mask)
            else:
                vec = mean_pool(hs, attention_mask)  # default
            pooled.append(vec)
        return pooled

    def collect(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, Optional[List["RepresentationSample"]]]:
        """Collect representations from the backbone.

        Main entry point (PRD §3.2.2). Extracts hidden states from multiple layers,
        pools them, and optionally returns per-sample metadata.

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            labels: Optional labels for correctness computation
            metadata: Optional dict with 'domain', 'task', etc.

        Returns:
            Tuple of (pooled_representations, logits, optional_samples)
        """
        # Run forward pass
        hidden_states, logits = self._run_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Pool representations
        pooled = self._pool_representations(hidden_states, attention_mask)

        # Create sample metadata (optional)
        samples = None
        if metadata is not None:
            samples = []
            batch_size = input_ids.shape[0]

            # Get domain and task from metadata (use defaults if not provided)
            domain = metadata.get("domain", "general")
            task = metadata.get("task", "classification")

            # Compute prediction from logits (take last token)
            preds = torch.argmax(logits, dim=-1)  # [batch, seq_len]

            for i in range(batch_size):
                # Take last token's prediction
                last_pred = preds[i, -1] if preds.dim() > 1 else preds[i]
                last_logits = logits[i, -1] if logits.dim() > 2 else logits[i]
                probs = torch.softmax(last_logits, dim=-1)
                top2_vals = torch.topk(probs, k=min(2, probs.size(-1))).values
                margin_val = (top2_vals[0] - top2_vals[1]).item() if top2_vals.size(0) >= 2 else top2_vals[0].item()

                sample_repr = pooled[-1][i] if len(pooled) > 0 else torch.zeros(1)

                sample = RepresentationSample(
                    sample_id=f"{metadata.get('prefix', 'sample')}_{i}",
                    domain=domain,
                    task=task,
                    layer=int(self.layers[-1]) if len(self.layers) > 0 else 0,
                    representation=sample_repr,
                    logits=last_logits,
                    prediction=str(last_pred.item()),
                    correctness=bool(labels is not None and i < labels.shape[0] and labels[i].item() == last_pred.item()),
                    confidence=probs.max().item(),
                    entropy=self._compute_entropy(probs),
                    margin=margin_val,
                    attention_mask=attention_mask[i] if attention_mask is not None else None,
                )
                samples.append(sample)

        return pooled, logits, samples

    @staticmethod
    def _compute_entropy(probs: torch.Tensor) -> float:
        """Compute Shannon entropy of probability distribution."""
        # Avoid log(0)
        eps = 1e-7
        probs = probs.clamp(min=eps)
        return -(probs * probs.log()).sum().item()

    def collect_to_dataset(
        self,
        dataloader: torch.utils.data.DataLoader,
        output_dir: Optional[str] = None,
        save_samples: bool = True,
    ) -> Tuple[List["RepresentationSample"], List[torch.Tensor]]:
        """Collect representations from a dataloader and optionally save to dataset.

        Args:
            dataloader: DataLoader yielding batches
            output_dir: Optional directory to save representations
            save_samples: Whether to create RepresentationSample objects

        Returns:
            Tuple of (samples, all_representations)
        """
        all_samples = []
        all_representations = []

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)

            labels = batch.get("labels", None)
            metadata = batch.get("metadata", {})

            # Collect representations
            pooled, logits, samples = self.collect(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                metadata=metadata,
            )

            all_representations.extend(pooled)

            if save_samples and samples is not None:
                all_samples.extend(samples)

        # Save to dataset if output_dir provided
        if output_dir is not None and save_samples:
            import os
            os.makedirs(output_dir, exist_ok=True)
            output_path = f"{output_dir}/representations.pt"
            torch.save({
                "samples": all_samples,
                "representations": all_representations,
            }, output_path)
            print(f"Saved {len(all_samples)} representations to {output_path}")

        return all_samples, all_representations

    def get_representation_dim(self) -> int:
        """Get the dimension of pooled representations.

        Returns:
            hidden_dim (e.g., 896 for Qwen2.5-0.5B)
        """
        # Run a quick forward pass to determine dim
        test_input = torch.randint(0, 1000, (1, 32), device=self.device)
        test_mask = torch.ones(1, 32, device=self.device)

        _, _ = self._run_forward(test_input, test_mask)

        # After pooling with mean pooling, dim = backbone hidden size
        # Use safe access
        try:
            return self.backbone.hidden_size
        except (AttributeError, TypeError):
            return 896  # Default for Qwen2.5-0.5B

    def __call__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Allow collector to be called directly."""
        return self.collect(input_ids=input_ids, attention_mask=attention_mask, **kwargs)