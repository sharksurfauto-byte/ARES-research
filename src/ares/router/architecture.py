"""Router Network architecture (PRD §3.2.5).

Routes input representations to base path or specialized LoRA experts
using a learned MLP router. Supports soft routing, top-k routing, and
Gumbel-Softmax exploration.

Classes: {0: base, 1: expert_0, 2: expert_1, 3: expert_2, 4: expert_3, 5: expert_4}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

EXPERT_NAMES = ["general", "math", "code", "science", "reasoning"]
ROUTE_NAMES = ["base", "expert_general", "expert_math", "expert_code", "expert_science", "expert_reasoning"]


@dataclass
class RouterConfig:
    """Router MLP configuration (PRD §3.2.5)."""

    input_dim: int = 896
    hidden_dim: int = 256
    num_layers: int = 2
    n_experts: int = 5
    dropout: float = 0.1
    temperature: float = 1.0
    top_k: int = 1
    routing_mode: str = "soft"  # "soft", "top_k", "gumbel_softmax"
    expert_names: list[str] = field(
        default_factory=lambda: ["general", "math", "code", "science", "reasoning"]
    )

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RouterConfig":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)


@dataclass
class RoutingOutput:
    """Structured routing output container."""

    routing_probs: torch.Tensor
    logits: torch.Tensor
    selected_experts: torch.Tensor
    confidence: torch.Tensor
    base_prob: torch.Tensor
    expert_probs: torch.Tensor


class Router(nn.Module):
    """MLP router that produces routing probabilities over base and experts.

    Routes to n_experts + 1 classes (0: base, 1..n: experts).
    Outputs probability distribution over all routes, confidence scores,
    and supports soft, top-k, and Gumbel-Softmax routing modes.
    """

    def __init__(self, config: RouterConfig | None = None, **kwargs: Any):
        super().__init__()
        if config is None:
            config = RouterConfig(**kwargs)
        elif kwargs:
            for k, v in kwargs.items():
                if hasattr(config, k):
                    setattr(config, k, v)

        self.config = config
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.n_experts = config.n_experts
        self.n_classes = config.n_experts + 1
        self.dropout_rate = config.dropout
        self.temperature = config.temperature
        self.top_k = config.top_k
        self.routing_mode = config.routing_mode
        self.expert_names = config.expert_names[: config.n_experts]

        # Build MLP layers (PRD §3.2.5: 2-layer MLP hidden=256, GELU)
        layers: list[nn.Module] = []
        if self.num_layers <= 1:
            layers.append(nn.Linear(self.input_dim, self.n_classes))
        else:
            layers.append(nn.Linear(self.input_dim, self.hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(self.dropout_rate))
            for _ in range(self.num_layers - 2):
                layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(self.dropout_rate))
            layers.append(nn.Linear(self.hidden_dim, self.n_classes))

        self.mlp = nn.Sequential(*layers)

    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Get raw router logits (before softmax).

        Args:
            x: Input representation [..., input_dim]

        Returns:
            Raw logits [..., n_classes]
        """
        orig_dtype = x.dtype
        w_dtype = next(self.mlp.parameters()).dtype
        if orig_dtype != w_dtype:
            x = x.to(w_dtype)
        logits = self.mlp(x)
        return logits.to(orig_dtype)

    def forward(
        self,
        x: torch.Tensor,
        mode: str | None = None,
        temperature: float | None = None,
        hard: bool = False,
        return_dict: bool = False,
    ) -> torch.Tensor | RoutingOutput:
        """Compute routing probabilities and decisions.

        Args:
            x: Input representation [..., input_dim]
            mode: Routing mode ('soft', 'top_k', 'gumbel_softmax'). If None, uses config.routing_mode.
            temperature: Softmax/Gumbel temperature. If None, uses config.temperature.
            hard: For Gumbel-Softmax: if True, returns one-hot discretization with straight-through gradient.
            return_dict: If True, returns a RoutingOutput dataclass.

        Returns:
            Routing probabilities [..., n_classes] or RoutingOutput dataclass.
        """
        temp = temperature if temperature is not None else self.temperature
        mode = mode if mode is not None else self.routing_mode
        logits = self.get_logits(x)

        if mode == "gumbel_softmax":
            routing_probs = F.gumbel_softmax(logits, tau=temp, hard=hard, dim=-1)
        elif mode == "top_k":
            routing_probs = self._top_k_routing(logits, temp=temp)
        else:  # "soft" or default
            routing_probs = F.softmax(logits / temp, dim=-1)

        if not return_dict:
            return routing_probs

        selected = routing_probs.argmax(dim=-1)
        confidence = routing_probs.max(dim=-1).values
        base_prob = routing_probs[..., 0:1]
        expert_probs = routing_probs[..., 1:]

        return RoutingOutput(
            routing_probs=routing_probs,
            logits=logits,
            selected_experts=selected,
            confidence=confidence,
            base_prob=base_prob,
            expert_probs=expert_probs,
        )

    def _top_k_routing(self, logits: torch.Tensor, temp: float = 1.0) -> torch.Tensor:
        """Top-k sparse routing with probability renormalization."""
        k = min(self.top_k, self.n_classes)
        top_k_logits, top_k_indices = torch.topk(logits, k=k, dim=-1)
        top_k_probs = F.softmax(top_k_logits / temp, dim=-1)

        zeros = torch.zeros_like(logits)
        routing_probs = zeros.scatter(-1, top_k_indices, top_k_probs)
        return routing_probs

    def route(
        self,
        x: torch.Tensor,
        top_k: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform discrete routing decision.

        Args:
            x: Input representation [batch, input_dim]
            top_k: Optional override for top-k selection

        Returns:
            Tuple of:
            - selected_indices: [batch] or [batch, k] index of selected route(s)
            - confidence: [batch] confidence scores in [0, 1]
            - routing_probs: [batch, n_classes] full routing probability distribution
        """
        k = top_k if top_k is not None else self.top_k
        logits = self.get_logits(x)
        probs = F.softmax(logits / self.temperature, dim=-1)

        if k == 1:
            selected = probs.argmax(dim=-1)
            confidence = probs.max(dim=-1).values
        else:
            confidence, selected = torch.topk(probs, k=min(k, self.n_classes), dim=-1)

        return selected, confidence, probs

    def save_checkpoint(self, filepath: str | Path) -> Path:
        """Save router weights and configuration.

        Args:
            filepath: Destination file path

        Returns:
            Path of saved file
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "router_state_dict": self.state_dict(),
                "model_state_dict": self.state_dict(),
                "config": self.config.to_dict(),
            },
            str(path),
        )
        return path

    @classmethod
    def load_checkpoint(
        cls,
        filepath: str | Path,
        device: torch.device | str = "cpu",
    ) -> "Router":
        """Load router from checkpoint.

        Args:
            filepath: Checkpoint file path
            device: Target device

        Returns:
            Instantiated Router
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Router checkpoint not found at: {path}")

        checkpoint = torch.load(str(path), map_location=device, weights_only=False)
        if "config" in checkpoint:
            cfg = RouterConfig.from_dict(checkpoint["config"])
        else:
            cfg = RouterConfig()

        router = cls(cfg)
        state_dict = checkpoint.get(
            "router_state_dict", checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
        )
        router.load_state_dict(state_dict)
        router.to(device)
        return router
