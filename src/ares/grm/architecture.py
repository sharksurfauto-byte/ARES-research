"""Global Reliability Model architecture (PRD §3.2.3).

GRM takes pooled hidden representations and outputs:
- domain_logits: 5-class (general, math, code, science, reasoning)
- feasibility: scalar [0,1] — "is this representation reliable?"
- global_reliability: scalar [0,1]
"""


import torch
import torch.nn as nn


class GRM(nn.Module):
    """Global Reliability Model — 2-layer transformer encoder.

    Based on PRD §3.2.3 architecture:
    - Input: pooled hidden representation [batch, hidden_dim]
    - Output: domain_logits, feasibility, global_reliability

    Architecture:
    - 2-layer transformer encoder
    - hidden_dim = 2 × input_dim
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
        domain_classes: int = 5,
    ):
        """Initialize GRM.

        Args:
            input_dim: Dimension of input pooled representation
            hidden_dim: Transformer hidden dimension (default: 2 × input_dim per PRD)
            num_layers: Number of transformer layers (default: 2)
            num_heads: Number of attention heads (default: 4)
            dropout: Dropout probability
            domain_classes: Number of domain classes (default: 5)
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.domain_classes = domain_classes

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

        # Output heads
        self.domain_head = nn.Linear(self.hidden_dim, self.domain_classes)
        self.feasibility_head = nn.Linear(self.hidden_dim, 1)
        self.global_head = nn.Linear(self.hidden_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through GRM.

        Args:
            x: Pooled hidden representation [batch, input_dim]

        Returns:
            Tuple of (domain_logits, feasibility, global_reliability)
            - domain_logits: [batch, domain_classes]
            - feasibility: [batch, 1] — "is this representation reliable?"
            - global_reliability: [batch, 1]
        """
        # Ensure 3D input [batch, seq_len, input_dim] for transformer
        if x.dim() == 2:
            x_seq = x.unsqueeze(1)
        else:
            x_seq = x

        # Project to hidden_dim if needed
        if self.input_projection is not None:
            x_proj = self.input_projection(x_seq)
        else:
            x_proj = x_seq

        # Apply transformer
        x_trans = self.transformer(x_proj)  # [batch, seq_len, hidden_dim]

        # Take the [CLS] token (first token) as the aggregated representation
        cls_token = x_trans[:, 0, :]  # [batch, hidden_dim]

        # Output heads
        domain_logits = self.domain_head(cls_token)  # [batch, domain_classes]
        feasibility = torch.sigmoid(self.feasibility_head(cls_token))  # [batch, 1]
        global_reliability = torch.sigmoid(self.global_head(cls_token))  # [batch, 1]

        return domain_logits, feasibility, global_reliability
