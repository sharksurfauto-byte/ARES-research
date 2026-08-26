"""Expert System module (PRD §3.2.5, §3.2.6).

Provides LoRA-based domain-specialized experts and a learned MLP router
for selective expert routing.
"""

from .lora_expert import LoRAExpert, LoRAExpertConfig, LoRALayer
from .manager import ExpertManager, Router, RouterConfig

__all__ = [
    "LoRAExpert",
    "LoRAExpertConfig",
    "LoRALayer",
    "ExpertManager",
    "Router",
    "RouterConfig",
]