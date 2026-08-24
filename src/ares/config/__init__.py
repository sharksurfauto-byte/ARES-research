"""ARES Configuration module.

Structured configs using Hydra/OmegaConf for type-safe configuration.
"""

from .schema import (
    ARESConfig,
    BackboneConfig,
    CalibrationConfig,
    CheckpointConfig,
    DataConfig,
    DDPConfig,
    ExperimentConfig,
    ExpertConfig,
    GRMConfig,
    LRMConfig,
    RouterConfig,
    TrainingConfig,
    WandbConfig,
)

__all__ = [
    "BackboneConfig",
    "DDPConfig",
    "WandbConfig",
    "CheckpointConfig",
    "ExperimentConfig",
    "DataConfig",
    "TrainingConfig",
    "GRMConfig",
    "LRMConfig",
    "RouterConfig",
    "ExpertConfig",
    "CalibrationConfig",
    "ARESConfig",
]
