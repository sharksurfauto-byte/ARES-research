"""Distributed Data Parallel utilities for ARES.

Supports 2x T4 Kaggle setup (PRD §7.1, §7.4 #5).
"""

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Optional, Dict, Any, List
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)


def is_distributed() -> bool:
    """Check if distributed training is initialized."""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Get current process rank."""
    if is_distributed():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Get total number of processes."""
    if is_distributed():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    """Check if current process is rank 0."""
    return get_rank() == 0


def init_ddp(
    backend: str = "nccl",
    timeout_minutes: int = 30,
    init_method: str = "env://",
) -> bool:
    """Initialize distributed training.

    Args:
        backend: Communication backend (nccl for GPU, gloo for CPU)
        timeout_minutes: Timeout for operations
        init_method: Initialization method

    Returns:
        True if DDP initialized, False if already initialized or single GPU
    """
    if is_distributed():
        logger.info("DDP already initialized")
        return True

    # Check if we're in a distributed environment
    if "RANK" not in os.environ and "LOCAL_RANK" not in os.environ:
        logger.info("Not in distributed environment (no RANK/LOCAL_RANK)")
        return False

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    logger.info(f"Initializing DDP: rank={rank}, world_size={world_size}, local_rank={local_rank}")

    # Set device for this process
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    # Initialize process group
    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        world_size=world_size,
        rank=rank,
        timeout=timedelta(minutes=timeout_minutes),
    )

    logger.info(f"DDP initialized on {device} (backend={backend})")
    return True


def cleanup_ddp():
    """Clean up distributed training."""
    if is_distributed():
        dist.destroy_process_group()
        logger.info("DDP cleaned up")


def wrap_model_ddp(
    model: torch.nn.Module,
    device_ids: Optional[List[int]] = None,
    output_device: Optional[int] = None,
    find_unused_parameters: bool = False,
    broadcast_buffers: bool = False,
) -> torch.nn.Module:
    """Wrap model with DistributedDataParallel.

    Args:
        model: Model to wrap
        device_ids: GPU device IDs (default: current device)
        output_device: Output device (default: device_ids[0])
        find_unused_parameters: Find unused parameters (needed for dynamic routing)
        broadcast_buffers: Broadcast buffers

    Returns:
        DDP-wrapped model
    """
    if not is_distributed():
        logger.warning("DDP not initialized, returning model unwrapped")
        return model

    if device_ids is None:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device_ids = [local_rank]

    if output_device is None:
        output_device = device_ids[0]

    model = DDP(
        model,
        device_ids=device_ids,
        output_device=output_device,
        find_unused_parameters=find_unused_parameters,
        broadcast_buffers=broadcast_buffers,
    )

    logger.info(f"Model wrapped with DDP (find_unused_parameters={find_unused_parameters})")
    return model


def reduce_dict(
    dictionary: Dict[str, torch.Tensor],
    average: bool = True,
) -> Dict[str, torch.Tensor]:
    """Reduce dictionary of tensors across all processes.

    Args:
        dictionary: Dictionary of tensors to reduce
        average: Whether to average (True) or sum (False)

    Returns:
        Reduced dictionary (same keys, reduced values)
    """
    if not is_distributed():
        return dictionary

    world_size = get_world_size()
    if world_size < 2:
        return dictionary

    with torch.no_grad():
        # Sort keys for consistent ordering
        keys = sorted(dictionary.keys())
        values = [dictionary[k] for k in keys]

        # Stack and reduce
        stacked = torch.stack(values, dim=0)
        dist.all_reduce(stacked, op=dist.ReduceOp.SUM)

        if average:
            stacked /= world_size

        # Reconstruct dictionary
        reduced = {k: v for k, v in zip(keys, stacked.unbind(dim=0))}

    return reduced


def all_gather_object(obj: Any) -> List[Any]:
    """Gather arbitrary Python objects from all processes.

    Args:
        obj: Object to gather

    Returns:
        List of objects from all processes
    """
    if not is_distributed():
        return [obj]

    world_size = get_world_size()
    gathered = [None] * world_size
    dist.all_gather_object(gathered, obj)
    return gathered


def broadcast_object(obj: Any, src: int = 0) -> Any:
    """Broadcast object from src to all processes.

    Args:
        obj: Object to broadcast (only used on src)
        src: Source rank

    Returns:
        Broadcasted object
    """
    if not is_distributed():
        return obj

    obj_list = [obj] if get_rank() == src else [None]
    dist.broadcast_object_list(obj_list, src=src)
    return obj_list[0]


def synchronize():
    """Synchronize all processes."""
    if is_distributed():
        dist.barrier()


def get_device() -> torch.device:
    """Get the device for the current process."""
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


class DDPContext:
    """Context manager for DDP initialization and cleanup."""

    def __init__(
        self,
        backend: str = "nccl",
        timeout_minutes: int = 30,
    ):
        self.backend = backend
        self.timeout_minutes = timeout_minutes
        self.initialized = False

    def __enter__(self):
        self.initialized = init_ddp(backend=self.backend, timeout_minutes=self.timeout_minutes)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.initialized:
            cleanup_ddp()

    def wrap(self, model: torch.nn.Module, **kwargs) -> torch.nn.Module:
        """Wrap model with DDP."""
        return wrap_model_ddp(model, **kwargs)

    def reduce(self, dictionary: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        """Reduce dictionary across processes."""
        return reduce_dict(dictionary, **kwargs)