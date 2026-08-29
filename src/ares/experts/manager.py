import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lora_expert import LoRAExpert, LoRAExpertConfig


from ..router import EXPERT_NAMES, Router, RouterConfig



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
        self.expert_names = expert_names or EXPERT_NAMES[:n_experts]

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
        for name in self.expert_names:
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
            x: Input representation [batch, input_dim] or [batch, seq_len, input_dim]
            return_routing_info: If True, also return routing metadata

        Returns:
            output: Expert-adapted representation with same shape as x
            info (optional): Dict with routing_probs, selected_experts, etc.
        """
        routing_probs = self.router(x)

        base_prob = routing_probs[..., 0:1]
        expert_probs = routing_probs[..., 1:]

        output = base_prob * x

        for i, expert in enumerate(self.experts):
            expert_out = expert(x)
            output = output + expert_probs[..., i : i + 1] * expert_out

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
            routing_probs: Router output [..., n_experts + 1]

        Returns:
            Scalar load balancing loss
        """
        n_classes = self.n_experts + 1
        flat_probs = routing_probs.view(-1, n_classes)

        selected = flat_probs.argmax(dim=-1)
        f = torch.zeros(n_classes, device=routing_probs.device)
        for i in range(n_classes):
            f[i] = (selected == i).float().mean()

        p = flat_probs.mean(dim=0)

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

    def load_expert(
        self,
        expert_name_or_idx: Union[str, int],
        checkpoint_path: Union[str, Path],
        strict: bool = True,
    ) -> bool:
        """Load weights for a single expert from checkpoint.

        Args:
            expert_name_or_idx: Expert name (e.g. 'math') or integer index (0..n_experts-1)
            checkpoint_path: Path to .pt checkpoint file
            strict: Whether to enforce strict key matching

        Returns:
            True if loaded successfully
        """
        path = Path(checkpoint_path)
        if not path.exists():
            if strict:
                raise FileNotFoundError(f"Checkpoint not found: {path}")
            return False

        if isinstance(expert_name_or_idx, int):
            idx = expert_name_or_idx
        else:
            names = [e.expert_name for e in self.experts]
            if expert_name_or_idx not in names:
                if strict:
                    raise ValueError(f"Unknown expert name '{expert_name_or_idx}'. Available: {names}")
                return False
            idx = names.index(expert_name_or_idx)

        checkpoint = torch.load(str(path), map_location=next(self.parameters()).device, weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
        self.experts[idx].load_state_dict(state_dict)
        return True

    def load_experts(
        self,
        experts_dir: Union[str, Path],
        strict: bool = False,
    ) -> dict[str, bool]:
        """Load all available experts from a directory.

        Looks for subdirectories with expert names containing `expert_<name>.pt`.

        Args:
            experts_dir: Root directory for experts (e.g. checkpoints/experts)
            strict: Whether to raise error if an expert is missing

        Returns:
            Dict mapping expert name to load success status
        """
        root = Path(experts_dir)
        results = {}
        for i, expert in enumerate(self.experts):
            name = expert.expert_name
            # Check root / name / expert_name.pt, or root / expert_name.pt
            cand1 = root / name / f"expert_{name}.pt"
            cand2 = root / f"expert_{name}.pt"
            cand3 = root / name / "adapter_model.pt"

            target_path = cand1 if cand1.exists() else (cand2 if cand2.exists() else (cand3 if cand3.exists() else None))
            if target_path:
                loaded = self.load_expert(i, target_path, strict=strict)
                results[name] = loaded
            else:
                if strict:
                    raise FileNotFoundError(f"No checkpoint found for expert '{name}' in {root}")
                results[name] = False
        return results

    def save_experts(self, output_dir: Union[str, Path]) -> dict[str, Path]:
        """Save all experts into output directory.

        Args:
            output_dir: Output base directory

        Returns:
            Dict mapping expert name to saved path
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        saved = {}
        for expert in self.experts:
            exp_dir = out_path / expert.expert_name
            exp_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = exp_dir / f"expert_{expert.expert_name}.pt"
            expert.save_checkpoint(ckpt_path)
            saved[expert.expert_name] = ckpt_path
        return saved

    def save_registry(
        self,
        output_dir: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save registry.json describing all experts.

        Args:
            output_dir: Output directory
            metadata: Optional additional metadata

        Returns:
            Path to registry.json
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        registry_data = {
            "input_dim": self.input_dim,
            "n_experts": self.n_experts,
            "expert_names": [e.expert_name for e in self.experts],
            "experts": {
                e.expert_name: {
                    "r": e.config.r,
                    "lora_alpha": e.config.lora_alpha,
                    "lora_dropout": e.config.lora_dropout,
                    "in_features": e.config.in_features,
                    "out_features": e.config.out_features,
                    "target_modules": e.config.target_modules,
                    "path": f"{e.expert_name}/expert_{e.expert_name}.pt",
                }
                for e in self.experts
            },
        }
        if metadata:
            registry_data["metadata"] = metadata

        reg_file = out_path / "registry.json"
        with open(reg_file, "w", encoding="utf-8") as f:
            json.dump(registry_data, f, indent=2)
        return reg_file

    def save_router(self, save_path: Union[str, Path]) -> Path:
        """Save the router weights."""
        return self.router.save_checkpoint(save_path)

    def load_router(self, load_path: Union[str, Path]) -> bool:
        """Load router weights."""
        path = Path(load_path)
        if not path.exists():
            return False
        checkpoint = torch.load(str(path), map_location=next(self.parameters()).device, weights_only=False)
        state_dict = checkpoint.get("router_state_dict", checkpoint.get("state_dict", checkpoint))
        self.router.load_state_dict(state_dict)
        return True