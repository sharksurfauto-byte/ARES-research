"""ARES Utilities module."""

from .checkpoint import (
    compute_sha256,
    compute_state_dict_sha256,
    save_checkpoint,
    load_checkpoint,
    verify_checkpoint,
    find_latest_checkpoint,
    CheckpointManager,
)
from .ddp import (
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
)
from .wandb_utils import (
    WandbLogger,
    init_wandb,
    log_metrics,
    log_model_artifact,
    finish_wandb,
)

__all__ = [
    # Checkpoint
    "compute_sha256",
    "compute_state_dict_sha256",
    "save_checkpoint",
    "load_checkpoint",
    "verify_checkpoint",
    "find_latest_checkpoint",
    "CheckpointManager",
    # DDP
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
    # W&B
    "WandbLogger",
    "init_wandb",
    "log_metrics",
    "log_model_artifact",
    "finish_wandb",
]