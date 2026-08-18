"""Abstract backbone interface for ARES.

This defines the contract that all backbone implementations must follow,
enabling model-agnostic design (PRD §7.4 #6, §8.2 #1).
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import torch
from torch import nn


class Backbone(ABC):
    """Abstract base class for frozen pretrained language model backbones.

    All adaptation in ARES happens via LoRA experts and router —
    the backbone weights are never updated (PRD §7.4 #1).
    """

    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = True,
        output_attentions: bool = True,
        **kwargs
    ) -> Any:
        """Forward pass through the backbone.

        Args:
            input_ids: Token IDs of shape [batch_size, seq_len]
            attention_mask: Attention mask of shape [batch_size, seq_len]
            output_hidden_states: Whether to return hidden states from all layers
            output_attentions: Whether to return attention weights

        Returns:
            Model outputs containing logits, hidden_states, and optionally attentions
        """
        pass

    @abstractmethod
    def get_hidden_states(
        self,
        layer_indices: List[int]
    ) -> List[torch.Tensor]:
        """Extract hidden states from specified layers.

        Args:
            layer_indices: Layer indices (negative = from end, e.g., -1 = last layer)

        Returns:
            List of hidden state tensors, each of shape [batch_size, seq_len, hidden_dim]
        """
        pass

    @abstractmethod
    def get_model_config(self) -> Dict[str, Any]:
        """Get the model's configuration as a dictionary.

        Returns:
            Dictionary with model configuration (hidden_size, num_layers, vocab_size, etc.)
        """
        pass

    @abstractmethod
    def get_device(self) -> torch.device:
        """Get the device the model is on."""
        pass

    @abstractmethod
    def get_dtype(self) -> torch.dtype:
        """Get the model's dtype."""
        pass

    @property
    @abstractmethod
    def hidden_size(self) -> int:
        """Hidden dimension size."""
        pass

    @property
    @abstractmethod
    def num_layers(self) -> int:
        """Number of transformer layers."""
        pass

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Vocabulary size."""
        pass


class QwenBackbone(Backbone):
    """Concrete implementation for Qwen2.5 models.

    Wraps a Hugging Face AutoModelForCausalLM with the required
    configuration for ARES (use_cache=False, eager attention, etc.).
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        hidden_state_layers: Tuple[int, ...] = (-1, -6, -12, -24)
    ):
        """Initialize the Qwen backbone wrapper.

        Args:
            model: Loaded Hugging Face model
            config: Model configuration dictionary
            hidden_state_layers: Which layers to extract hidden states from
        """
        self._model = model
        self._config = config
        self._hidden_state_layers = hidden_state_layers

        # Ensure critical settings
        self._model.config.use_cache = False
        if hasattr(self._model.config, "attn_implementation"):
            self._model.config.attn_implementation = "eager"

        # Freeze all parameters (PRD §7.4 #1)
        for param in self._model.parameters():
            param.requires_grad = False

        self._model.eval()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = True,
        output_attentions: bool = True,
        **kwargs
    ) -> Any:
        """Forward pass with hidden states and attentions."""
        return self._model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            use_cache=False,  # Critical for dynamic expert switching
            **kwargs
        )

    def get_hidden_states(self, layer_indices: List[int]) -> List[torch.Tensor]:
        """Extract hidden states from specified layers.

        Note: This requires a forward pass with output_hidden_states=True first.
        The hidden states are cached from the last forward call.
        """
        # Get the last forward output (should have hidden_states)
        # In practice, this is called after forward() with output_hidden_states=True
        raise NotImplementedError(
            "get_hidden_states should be called via RepresentationCollector "
            "which handles the forward pass and extraction"
        )

    def get_model_config(self) -> Dict[str, Any]:
        """Return model configuration."""
        return {
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "vocab_size": self.vocab_size,
            "model_type": self._config.get("model_type", "qwen2"),
            "architectures": self._config.get("architectures", []),
        }

    def get_device(self) -> torch.device:
        """Get model device."""
        return next(self._model.parameters()).device

    def get_dtype(self) -> torch.dtype:
        """Get model dtype."""
        return next(self._model.parameters()).dtype

    @property
    def hidden_size(self) -> int:
        return self._model.config.hidden_size

    @property
    def num_layers(self) -> int:
        return self._model.config.num_hidden_layers

    @property
    def vocab_size(self) -> int:
        return self._model.config.vocab_size

    def __call__(self, *args, **kwargs):
        """Allow calling the backbone directly."""
        return self.forward(*args, **kwargs)