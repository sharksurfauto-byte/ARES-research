"""LoRA Expert module (PRD §3.2.6).

Provides LoRA-based domain-specialized experts that add low-rank
adaptations on top of frozen backbone linear layers.

Experts: E0-general, E1-math, E2-code, E3-science, E4-reasoning
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class LoRAExpertConfig:
    """Configuration for a single LoRA expert adapter."""

    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    expert_name: str = "general"
    in_features: int = 896
    out_features: int = 896

    @property
    def scaling(self) -> float:
        return self.lora_alpha / self.r


class LoRALayer(nn.Module):
    """Single LoRA adapter: x → x + scaling * (B @ A @ x).

    A is initialized with Kaiming uniform, B is initialized to zero
    so the adapter starts as identity.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
    ):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute LoRA delta: scaling * B(A(dropout(x))).

        Args:
            x: Input tensor [..., in_features]

        Returns:
            LoRA delta [..., out_features]
        """
        return self.scaling * self.lora_B(self.lora_A(self.dropout(x)))


class LoRAExpert(nn.Module):
    """Domain-specialized LoRA expert.

    Wraps multiple LoRA adapters (one per target module) to specialize
    the frozen backbone for a specific domain.

    Forward pass takes a representation tensor and returns the adapted
    representation (input + LoRA delta).
    """

    def __init__(self, config: LoRAExpertConfig):
        super().__init__()
        self.config = config
        self.expert_name = config.expert_name

        self.lora_layers = nn.ModuleDict()
        for module_name in config.target_modules:
            self.lora_layers[module_name] = LoRALayer(
                in_features=config.in_features,
                out_features=config.out_features,
                r=config.r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
            )

        self.gate = nn.Sequential(
            nn.Linear(config.in_features, config.r),
            nn.GELU(),
            nn.Linear(config.r, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        module_name: str | None = None,
    ) -> torch.Tensor:
        """Apply LoRA adaptation to input representation.

        When module_name is specified, applies only that adapter.
        When None, applies all adapters and averages.

        Args:
            x: Input representation [..., in_features]
            module_name: Specific target module name, or None for all

        Returns:
            Adapted representation [..., out_features]
        """
        if module_name is not None:
            if module_name not in self.lora_layers:
                raise ValueError(
                    f"Unknown module '{module_name}'. "
                    f"Available: {list(self.lora_layers.keys())}"
                )
            delta = self.lora_layers[module_name](x)
            return x + delta

        delta_sum = torch.zeros_like(x)
        for lora_layer in self.lora_layers.values():
            delta_sum = delta_sum + lora_layer(x)
        delta_avg = delta_sum / len(self.lora_layers)

        gate_weight = self.gate(x)
        return x + gate_weight * delta_avg

    def get_num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def specialization_score(self, x: torch.Tensor) -> torch.Tensor:
        """Compute how much this expert modifies the input.

        Args:
            x: Input representation [batch, in_features]

        Returns:
            Score [batch, 1] indicating adaptation magnitude
        """
        with torch.no_grad():
            adapted = self.forward(x)
            diff = (adapted - x).norm(dim=-1, keepdim=True)
            input_norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            return diff / input_norm
