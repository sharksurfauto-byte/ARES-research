"""Expert Manager with Router (PRD §3.2.5).

Routes input representations to base path or specialized LoRA experts
using a learned MLP router. Combines expert outputs with routing weights.

Router architecture: Linear(input_dim → 256) → GELU → Dropout → Linear(256 → n_classes) → Softmax
Classes: {base, expert_0, expert_1, expert_2, expert_3, expert_4}
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lora_expert import LoRAExpert, LoRAExpertConfig


EXPERT_NAMES = ["general", "math", "code", "science", "reasoning"]


@dataclass
class RouterConfig:
    """Router MLP configuration."""

    input_dim: int = 896
    hidden_dim: int = 256
    n_experts: int = 5
    dropout: float = 0.1
    temperature: float = 1.0
    top_k: int = 1


class Router(nn.Module):
    """MLP router that produces routing probabilities over experts.

    Routes to n_experts + 1 classes (base + n_experts).
    Output is a probability distribution over all routes.
    """

    def __init__(self, config: RouterConfig):
        super().__init__()
        self.config = config

        n_classes = config.n_experts + 1

        self.mlp = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, n_classes),
        )
        self.temperature = config.temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute routing probabilities.

        Args:
            x: Input representation [batch, input_dim]

        Returns:
            Routing probabilities [batch, n_experts + 1]
            Index 0 = base, indices 1..n = experts
        """
        logits = self.mlp(x)
        return F.softmax(logits / self.temperature, dim=-1)

    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Get raw router logits (before softmax).

        Args:
            x: Input representation [batch, input_dim]

        Returns:
            Raw logits [batch, n_experts + 1]
        """
        return self.mlp(x)


class ExpertManager(nn.Module):
    """Manages routing and expert selection.

    Coordinates the Router with a set of LoRA experts.
    Routes input to base or a specialized expert, then combines
    outputs using the routing weights.
    """

    def __init__(
        self,
        input_dim: int = 896,
        hidden_dim: int = 256,
        n_experts: int = 5,
        expert_names: list[str] | None = None,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        router_dropout: float = 0.1,
        router_temperature: float = 1.0,
        top_k: int = 1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.n_experts = n_experts
        self.top_k = top_k
        expert_names = expert_names or EXPERT_NAMES[:n_experts]

        router_cfg = RouterConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_experts=n_experts,
            dropout=router_dropout,
            temperature=router_temperature,
            top_k=top_k,
        )
        self.router = Router(router_cfg)

        self.experts = nn.ModuleList()
        for name in expert_names:
            expert_cfg = LoRAExpertConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                expert_name=name,
                in_features=input_dim,
                out_features=input_dim,
            )
            self.experts.append(LoRAExpert(expert_cfg))

        self._usage_counts: torch.Tensor | None = None

    def forward(
        self,
        x: torch.Tensor,
        return_routing_info: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """Route input and compute expert-weighted output.

        Uses soft routing: output = sum(prob_i * expert_i(x)) for all routes.
        The base route (index 0) returns x unchanged.

        Args:
            x: Input representation [batch, input_dim]
            return_routing_info: If True, also return routing metadata

        Returns:
            output: Expert-adapted representation [batch, input_dim]
            info (optional): Dict with routing_probs, selected_experts, etc.
        """
        routing_probs = self.router(x)

        base_prob = routing_probs[:, 0:1]
        expert_probs = routing_probs[:, 1:]

        output = base_prob * x

        for i, expert in enumerate(self.experts):
            expert_out = expert(x)
            output = output + expert_probs[:, i:i+1] * expert_out

        if self._usage_counts is None or self._usage_counts.shape[0] != self.n_experts + 1:
            self._usage_counts = torch.zeros(
                self.n_experts + 1, device=x.device, dtype=torch.float32
            )

        with torch.no_grad():
            selected = routing_probs.argmax(dim=-1)
            for idx in range(self.n_experts + 1):
                self._usage_counts[idx] += (selected == idx).sum().float()

        if return_routing_info:
            info = {
                "routing_probs": routing_probs,
                "selected_experts": selected,
                "expert_probs": expert_probs,
                "base_prob": base_prob,
            }
            return output, info

        return output

    def load_balancing_loss(self, routing_probs: torch.Tensor) -> torch.Tensor:
        """Compute load balancing loss to encourage balanced expert usage.

        Uses the auxiliary loss from Switch Transformer:
        L_balance = n_classes * sum(f_i * P_i)
        where f_i = fraction of tokens routed to expert i
              P_i = mean routing probability for expert i

        Args:
            routing_probs: Router output [batch, n_experts + 1]

        Returns:
            Scalar load balancing loss
        """
        n_classes = self.n_experts + 1

        selected = routing_probs.argmax(dim=-1)
        f = torch.zeros(n_classes, device=routing_probs.device)
        for i in range(n_classes):
            f[i] = (selected == i).float().mean()

        p = routing_probs.mean(dim=0)

        return n_classes * (f * p).sum()

    def get_usage_stats(self) -> dict[str, float]:
        """Get expert usage statistics.

        Returns:
            Dict mapping expert name to usage fraction
        """
        if self._usage_counts is None:
            return {}

        total = self._usage_counts.sum().item()
        if total == 0:
            return {}

        stats = {"base": self._usage_counts[0].item() / total}
        for i, expert in enumerate(self.experts):
            stats[expert.expert_name] = self._usage_counts[i + 1].item() / total
        return stats

    def reset_usage_stats(self):
        """Reset usage counters."""
        self._usage_counts = None

    def get_num_trainable_params(self) -> dict[str, int]:
        """Get parameter counts per component."""
        router_params = sum(p.numel() for p in self.router.parameters() if p.requires_grad)
        expert_params = {}
        for expert in self.experts:
            expert_params[expert.expert_name] = expert.get_num_trainable_params()
        return {
            "router": router_params,
            "experts": expert_params,
            "total": router_params + sum(expert_params.values()),
        }