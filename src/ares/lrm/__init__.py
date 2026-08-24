"""Local Reliability Model module (PRD §3.2.4).

Provides LRM architecture and trainer for local (token-wise) reliability estimation.
"""

from .architecture import LRM
from .trainer import LRMTrainer

__all__ = [
    "LRM",
    "LRMTrainer",
]
