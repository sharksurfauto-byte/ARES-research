"""LoRA Expert module (PRD §3.2.6).

Provides LoRA-based domain-specialized experts that add low-rank
adaptations on top of frozen backbone linear layers.

Experts: E0-general, E1-math, E2-code, E3-science, E4-reasoning
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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
    dtype: str = "float32"

    @property
    def scaling(self) -> float:
        return self.lora_alpha / self.r if self.r > 0 else 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LoRAExpertConfig:
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)


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
        self.scaling = lora_alpha / r if r > 0 else 1.0

        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        if self.r > 0:
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute LoRA delta: scaling * B(A(dropout(x))).

        Args:
            x: Input tensor [..., in_features]

        Returns:
            LoRA delta [..., out_features]
        """
        # Match layer weight dtype
        orig_dtype = x.dtype
        w_dtype = self.lora_A.weight.dtype
        if orig_dtype != w_dtype:
            x = x.to(w_dtype)
        delta = self.scaling * self.lora_B(self.lora_A(self.dropout(x)))
        return delta.to(orig_dtype)


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
        orig_dtype = x.dtype
        gate_dtype = next(self.gate.parameters()).dtype
        if x.dtype != gate_dtype:
            x_for_gate = x.to(gate_dtype)
        else:
            x_for_gate = x

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

        gate_weight = self.gate(x_for_gate).to(orig_dtype)
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

    def save_checkpoint(
        self,
        filepath: Union[str, Path],
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save expert weights and configuration to a checkpoint file.

        Args:
            filepath: Target file path
            extra_meta: Optional extra metadata to store

        Returns:
            Path of saved file
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "expert_name": self.expert_name,
            "config": self.config.to_dict(),
            "state_dict": self.state_dict(),
        }
        if extra_meta:
            payload["metadata"] = extra_meta

        torch.save(payload, str(path))
        return path

    @classmethod
    def load_checkpoint(
        cls,
        filepath: Union[str, Path],
        device: Union[torch.device, str] = "cpu",
    ) -> LoRAExpert:
        """Load expert from a checkpoint file.

        Args:
            filepath: Checkpoint path (.pt)
            device: Target device

        Returns:
            Instantiated and loaded LoRAExpert
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Expert checkpoint not found at: {path}")

        checkpoint = torch.load(str(path), map_location=device, weights_only=False)

        if "config" in checkpoint:
            cfg = LoRAExpertConfig.from_dict(checkpoint["config"])
        else:
            cfg = LoRAExpertConfig()

        expert = cls(cfg)
        state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
        expert.load_state_dict(state_dict)
        expert.to(device)
        return expert

    def save_pretrained(self, save_directory: Union[str, Path]) -> Path:
        """Save adapter config and weights in standard HuggingFace/PEFT directory structure.

        Args:
            save_directory: Target directory

        Returns:
            Path of directory
        """
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save config json
        config_path = save_dir / "adapter_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        # Save checkpoint .pt
        self.save_checkpoint(save_dir / f"expert_{self.expert_name}.pt")
        return save_dir

    @classmethod
    def from_pretrained(
        cls,
        load_directory: Union[str, Path],
        device: Union[torch.device, str] = "cpu",
    ) -> LoRAExpert:
        """Load expert from a directory.

        Args:
            load_directory: Directory containing adapter_config.json or expert_*.pt
            device: Target device

        Returns:
            Instantiated LoRAExpert
        """
        load_dir = Path(load_directory)
        pt_files = list(load_dir.glob("*.pt"))
        if not pt_files:
            raise FileNotFoundError(f"No .pt checkpoint found in {load_dir}")
        return cls.load_checkpoint(pt_files[0], device=device)

    def to_peft_state_dict(self) -> dict[str, torch.Tensor]:
        """Convert state dict to PEFT-compatible naming."""
        peft_dict = {}
        for key, val in self.state_dict().items():
            # Example: lora_layers.q_proj.lora_A.weight -> lora_layers.q_proj.lora_A.default.weight
            if "lora_A.weight" in key:
                peft_key = key.replace("lora_A.weight", "lora_A.default.weight")
            elif "lora_B.weight" in key:
                peft_key = key.replace("lora_B.weight", "lora_B.default.weight")
            else:
                peft_key = key
            peft_dict[peft_key] = val
        return peft_dict
