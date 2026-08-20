"""ARES: Adaptive Reliability with Expert Specialization.

PRD: ARES_RESEARCH_PRD.md

Updated with Week 2 modules: Representation Collector, GRM, LRM, Calibration.
"""

__version__ = "0.2.0"

from .backbone import (
    BackboneConfig,
    Backbone,
    QwenBackbone,
    load_backbone,
    verify_backbone,
)
from .config import (
    BackboneConfig as ConfigBackboneConfig,
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
from .utils import (
    compute_sha256,
    compute_state_dict_sha256,
    save_checkpoint,
    load_checkpoint,
    verify_checkpoint,
    find_latest_checkpoint,
    CheckpointManager,
    is_distributed,
    get_rank,
    get_world_size,
    is_main_process,
    init_ddp,
    cleanup_ddp,
    wrap_model_ddp,
    reduce_dict,
    all_gather_object,
    broadcast_object,
    synchronize,
    get_device,
    DDPContext,
    WandbLogger,
    init_wandb,
    log_metrics,
    log_model_artifact,
    finish_wandb,
)
from .representations import (
    RepresentationCollector,
    PoolMethod,
    pool_hidden_state,
    last_token_pool,
    mean_pool,
    max_pool,
    RepresentationCollector,
    CollectorConfig,
)
from .grm import GRM, GRMTrainer
from .lrm import LRM, LRMTrainer
from .calibration import (
    TemperatureScaling,
    fit_temperature_scaling,
    fit_isotonic_regression,
    apply_isotonic_regression,
    compute_ece,
    compute_brier_score,
    before_after_calibration,
)

__all__ = [
    # Backbone
    "BackboneConfig",
    "Backbone",
    "QwenBackbone",
    "load_backbone",
    "verify_backbone",
    # Config
    "ConfigBackboneConfig",
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
    # Utils - Checkpoint
    "compute_sha256",
    "compute_state_dict_sha256",
    "save_checkpoint",
    "load_checkpoint",
    "verify_checkpoint",
    "find_latest_checkpoint",
    "CheckpointManager",
    # Utils - DDP
    "is_distributed",
    "get_rank",
    "get_world_size",
    "is_main_process",
    "init_ddp",
    "cleanup_ddp",
    "wrap_model_ddp",
    "reduce_dict",
    "all_gather_object",
    "broadcast_object",
    "synchronize",
    "get_device",
    "DDPContext",
    # Utils - W&B
    "WandbLogger",
    "init_wandb",
    "log_metrics",
    "log_model_artifact",
    "finish_wandb",
    # Representations
    "RepresentationCollector",
    "PoolMethod",
    "pool_hidden_state",
    "last_token_pool",
    "mean_pool",
    "max_pool",
    "RepresentationSample",
    "CollectorConfig",
    "RepresentationDataset",
    # GRM
    "GRM",
    "GRMTrainer",
    # LRM
    "LRM",
    "LRMTrainer",
    # Calibration
    "TemperatureScaling",
    "fit_temperature_scaling",
    "fit_isotonic_regression",
    "apply_isotonic_regression",
    "before_after_calibration",
    "compute_ece",
    "compute_brier_score",
]