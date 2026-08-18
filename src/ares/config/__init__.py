"""ARES Configuration module.

Structured configs using Hydra/OmegaConf for type-safe configuration.
"""

from .schema import (
    BackboneConfig,
    DDPConfig,
    WandbConfig,
    CheckpointConfig,
    ExperimentConfig,
    DataConfig,
    TrainingConfig,
    GRMConfig,
    LRMConfig,
    RouterConfig,
    ExpertConfig,
    CalibrationConfig,
    ARESConfig,
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