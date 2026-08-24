"""ARES: Adaptive Reliability with Expert Specialization.

PRD: ARES_RESEARCH_PRD.md

Updated with Week 2 modules: Representation Collector, GRM, LRM, Calibration.
"""

__version__ = "0.2.0"

from .backbone import (
    Backbone,
    BackboneConfig,
    QwenBackbone,
    load_backbone,
    verify_backbone,
)
from .calibration import (
    TemperatureScaling,
    apply_isotonic_regression,
    before_after_calibration,
    compute_brier_score,
    compute_ece,
    fit_isotonic_regression,
    fit_temperature_scaling,
)
from .config import (
    ARESConfig,
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
from .config import (
    BackboneConfig as ConfigBackboneConfig,
)
from .grm import GRM, GRMTrainer
from .lrm import LRM, LRMTrainer
from .representations import (
    CollectorConfig,
    PoolMethod,
    RepresentationCollector,
    RepresentationDataset,
    RepresentationSample,
    last_token_pool,
    max_pool,
    mean_pool,
    pool_hidden_state,
)
from .utils import (
    CheckpointManager,
    DDPContext,
    WandbLogger,
    all_gather_object,
    broadcast_object,
    cleanup_ddp,
    compute_sha256,
    compute_state_dict_sha256,
    find_latest_checkpoint,
    finish_wandb,
    get_device,
    get_rank,
    get_world_size,
    init_ddp,
    init_wandb,
    is_distributed,
    is_main_process,
    load_checkpoint,
    log_metrics,
    log_model_artifact,
    reduce_dict,
    save_checkpoint,
    synchronize,
    verify_checkpoint,
    wrap_model_ddp,
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
