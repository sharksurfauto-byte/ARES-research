"""Pooling strategies for representation extraction (PRD §3.2.2).

Multiple pooling methods to reduce hidden states from [batch, seq_len, hidden_dim]
to single vectors of dimension hidden_dim.
"""

import torch
import torch.nn.functional as F
from enum import Enum
from typing import Literal, Optional


class PoolMethod(Enum):
    """Pooling methods for representation extraction."""
    LAST_TOKEN = "last_token"
    MEAN = "mean"
    MAX = "max"
    CLS = "cls"  # First token (for models with CLS token)


PoolMethodType = Literal["last_token", "mean", "max", "cls"]


def pool_hidden_state(
    hidden_state: torch.Tensor,
    method: PoolMethodType = "mean",
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pool a hidden state tensor to a single vector.

    Args:
        hidden_state: Tensor of shape [batch, seq_len, hidden_dim]
        method: Pooling method ("last_token", "mean", "max", "cls")
        attention_mask: Optional attention mask for masking

    Returns:
        Pooled tensor of shape [batch, hidden_dim]
    """
    if method == "last_token":
        return last_token_pool(hidden_state, attention_mask)
    elif method == "mean":
        return mean_pool(hidden_state, attention_mask)
    elif method == "max":
        return max_pool(hidden_state, attention_mask)
    elif method == "cls":
        return cls_pool(hidden_state)
    else:
        raise ValueError(f"Unknown pooling method: {method}")


def last_token_pool(
    hidden_state: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pool by taking the last valid token.

    Args:
        hidden_state: [batch, seq_len, hidden_dim]
        attention_mask: [batch, seq_len] - 1 for valid, 0 for padding

    Returns:
        [batch, hidden_dim] - last token from each sequence
    """
    if attention_mask is None:
        # Use last position
        return hidden_state[:, -1, :]

    # Ensure attention_mask is on same device as hidden_state
    attention_mask = attention_mask.to(hidden_state.device)
    # Find last valid position for each batch
    lengths = attention_mask.sum(dim=1).long()  # [batch]
    batch_indices = torch.arange(hidden_state.size(0), device=hidden_state.device)

    return hidden_state[batch_indices, lengths - 1, :]


def mean_pool(
    hidden_state: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mean pool over sequence dimension.

    Args:
        hidden_state: [batch, seq_len, hidden_dim]
        attention_mask: [batch, seq_len] - 1 for valid, 0 for padding

    Returns:
        [batch, hidden_dim] - mean of all tokens
    """
    if attention_mask is not None:
        # Ensure attention_mask is on same device as hidden_state
        attention_mask = attention_mask.to(hidden_state.device)
        # Mask out padding tokens
        mask_expanded = attention_mask.unsqueeze(-1).float()
        hidden_state = hidden_state * mask_expanded
        sum_hidden = hidden_state.sum(dim=1)
        lengths = attention_mask.sum(dim=1, keepdim=True).float()
        return sum_hidden / lengths.clamp(min=1.0)
    else:
        return hidden_state.mean(dim=1)


def max_pool(
    hidden_state: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Max pool over sequence dimension.

    Args:
        hidden_state: [batch, seq_len, hidden_dim]
        attention_mask: [batch, seq_len] - 1 for valid, 0 for padding

    Returns:
        [batch, hidden_dim] - max of all tokens
    """
    if attention_mask is not None:
        # Ensure attention_mask is on same device as hidden_state
        attention_mask = attention_mask.to(hidden_state.device)
        # Mask out padding tokens with very negative value
        mask_expanded = attention_mask.unsqueeze(-1).float()
        hidden_state = hidden_state.masked_fill(mask_expanded == 0, -1e9)

    return hidden_state.max(dim=1).values


def cls_pool(hidden_state: torch.Tensor) -> torch.Tensor:
    """Take the first token (CLS) as representation.

    Args:
        hidden_state: [batch, seq_len, hidden_dim]

    Returns:
        [batch, hidden_dim] - first token from each sequence
    """
    return hidden_state[:, 0, :]


def concatenate_layers(
    layer_outputs: list,
    pooling_method: PoolMethodType = "mean",
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Concatenate representations from multiple layers.

    Args:
        layer_outputs: List of [batch, seq_len, hidden_dim] tensors
        pooling_method: Pooling method to apply
        attention_mask: Optional attention mask

    Returns:
        [batch, num_layers * hidden_dim] - concatenated pooled representations
    """
    pooled = []
    for hidden_state in layer_outputs:
        pooled.append(pool_hidden_state(hidden_state, pooling_method, attention_mask))

    return torch.cat(pooled, dim=-1)


def aggregate_layers(
    layer_outputs: list,
    pooling_method: PoolMethodType = "mean",
    attention_mask: Optional[torch.Tensor] = None,
    aggregation: Literal["concat", "mean", "max", "weighted"] = "concat",
) -> torch.Tensor:
    """Aggregate representations from multiple layers.

    Args:
        layer_outputs: List of [batch, seq_len, hidden_dim] tensors
        pooling_method: Pooling method to apply
        attention_mask: Optional attention mask
        aggregation: How to combine layers ("concat", "mean", "max", "weighted")

    Returns:
        Aggregated representation tensor
    """
    pooled = []
    for hidden_state in layer_outputs:
        pooled.append(pool_hidden_state(hidden_state, pooling_method, attention_mask))

    stacked = torch.stack(pooled, dim=0)  # [num_layers, batch, hidden_dim]

    if aggregation == "concat":
        return torch.cat(pooled, dim=-1)
    elif aggregation == "mean":
        return stacked.mean(dim=0)
    elif aggregation == "max":
        return stacked.max(dim=0).values
    elif aggregation == "weighted":
        # Simple weighted average by layer depth (later layers more important)
        weights = torch.linspace(0.5, 1.5, stacked.size(0), device=stacked.device)
        weights = weights / weights.sum()
        return (stacked * weights.view(-1, 1, 1)).sum(dim=0)
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")