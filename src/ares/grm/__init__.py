"""Global Reliability Model module (PRD §3.2.3).

Provides GRM architecture and trainer for global reliability estimation
from pooled hidden representations.
"""

from .architecture import GRM
from .trainer import GRMTrainer

__all__ = [
    "GRM",
    "GRMTrainer",
]