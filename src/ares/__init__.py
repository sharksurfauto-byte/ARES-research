"""ARES: Adaptive Reliability with Expert Specialization.

A framework for learned adaptive routing on frozen pretrained language models
with GRM+LRM reliability signals and specialized LoRA experts.

PRD: ARES_RESEARCH_PRD.md
"""

__version__ = "0.1.0"

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
]