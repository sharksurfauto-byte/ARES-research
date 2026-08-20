"""Representation Collector module (PRD §3.2.2).

Extracts multi-layer hidden states from frozen backbone for reliability estimation.
"""

from .collector import RepresentationCollector, CollectorConfig
from .pooling import (
    PoolMethod,
    pool_hidden_state,
    last_token_pool,
    mean_pool,
    max_pool,
)
from .collector import RepresentationSample

__all__ = [
    "RepresentationCollector",
    "PoolMethod",
    "pool_hidden_state",
    "last_token_pool",
    "mean_pool",
    "max_pool",
    "RepresentationSample",
    "CollectorConfig",
]