"""Global Reliability Model module (PRD §3.2.3).

Provides GRM architecture, trainer, and self-supervised pretraining
for global reliability estimation from pooled hidden representations.
"""

from .architecture import GRM
from .pretraining import (
    ContrastiveConfig,
    GRMPretrainer,
    PretrainingConfig,
    ReconstructionConfig,
    create_pretraining_dataloader,
)
from .trainer import GRMTrainer

__all__ = [
    "GRM",
    "GRMTrainer",
    "GRMPretrainer",
    "PretrainingConfig",
    "ContrastiveConfig",
    "ReconstructionConfig",
    "create_pretraining_dataloader",
]
