"""Local Reliability Model architecture (PRD §3.2.4).

LRM takes per-token hidden states and outputs:
- correctness_prob: per-token P(correct|H_token) in [0,1]
- failure_risk: per-token 1 - correctness_prob
- Token-level reliability scores
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any


class LRM(nn.Module):
    """Local Reliability Model — 2-layer transformer over token-wise representations.

    Based on PRD §3.2.4 architecture:
    - Input: per-token hidden states [batch, seq_len, hidden_dim]
    - Output: correctness_prob, failure_risk (per-token)

    Architecture:
    - 2-layer transformer (applied per-token or with pooling)
    - hidden_dim = 2 × input_dim (per PRD §7.4 design)
    - 4 attention heads
    - Dropout = 0.1
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        """Initialize LRM.

        Args:
            input_dim: Dimension of input per-token hidden states
            hidden_dim: Transformer hidden dimension (default: 2 × input_dim per PRD)
            num_layers: Number of transformer layers (default: 2)
            num_heads: Number of attention heads (default: 4)
            dropout: Dropout probability
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        # Build transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim * 4,  # 4× expansion
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        # Input projection if needed
        self.input_projection = None
        if self.input_dim != self.hidden_dim:
            self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)

        # Output head for binary classification
        self.output_head = nn.Linear(self.hidden_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through LRM.

        Args:
            x: Per-token hidden states [batch, seq_len, input_dim]

        Returns:
            Tuple of (correctness_prob, failure_risk)
            - correctness_prob: [batch, seq_len] — P(correct|H_token)
            - failure_risk: [batch, seq_len] — 1 - correctness_prob
        """
        is_2d = (x.dim() == 2)
        if is_2d:
            x_seq = x.unsqueeze(1)
        else:
            x_seq = x

        # Project to hidden_dim if needed
        if self.input_projection is not None:
            x_seq = self.input_projection(x_seq)

        # Apply transformer
        x_trans = self.transformer(x_seq)  # [batch, seq_len, hidden_dim]

        # Output projection for binary classification
        logits = self.output_head(x_trans).squeeze(-1)  # [batch, seq_len]

        if is_2d:
            logits = logits.squeeze(-1)

        # Sigmoid for probability
        correctness_prob = torch.sigmoid(logits)
        failure_risk = 1.0 - correctness_prob

        return correctness_prob, failure_risk