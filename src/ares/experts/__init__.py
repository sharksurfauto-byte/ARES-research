"""Expert System module (PRD §3.2.6, §4.5).

Provides LoRA-based domain-specialized experts for selective expert routing.
"""

from .lora_expert import LoRAExpert, LoRAExpertConfig
from .manager import ExpertManager

__all__ = [
    "LoRAExpert",
    "LoRAExpertConfig",
    "ExpertManager",
]