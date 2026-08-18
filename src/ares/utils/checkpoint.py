"""Checkpoint system with SHA256 verification for ARES.

Implements PRD §6 Week 1: Checkpoint system with SHA256 metadata.
"""

import os
import json
import hashlib
import torch
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union
from datetime import datetime

logger = logging.getLogger(__name__)


def compute_sha256(filepath: Union[str, Path]) -> str:
    """Compute SHA256 hash of a file.

    Args:
        filepath: Path to file

    Returns:
        Hexadecimal SHA256 hash string
    """
    filepath = Path(filepath)
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def compute_state_dict_sha256(state_dict: Dict[str, torch.Tensor]) -> str:
    """Compute SHA256 hash of a model state dict.

    Args:
        state_dict: Model state dictionary

    Returns:
        Hexadecimal SHA256 hash string
    """
    # Serialize tensors to bytes in a deterministic way
    sha256_hash = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        tensor = state_dict[key]
        # Include key in hash
        sha256_hash.update(key.encode())
        # Include tensor data — convert to float32 for deterministic serialization
        # (handles bfloat16, float16, and other dtypes that numpy().tobytes() doesn't support natively)
        if tensor.dtype in (torch.bfloat16, torch.float16):
            tensor = tensor.float()
        sha256_hash.update(tensor.cpu().numpy().tobytes())
    return sha256_hash.hexdigest()


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
    step: int = 0,
    metrics: Optional[Dict[str, float]] = None,
    path: Union[str, Path] = "checkpoints/checkpoint.pt",
    config: Optional[Dict[str, Any]] = None,
    verify_sha256: bool = True,
) -> str:
    """Save model checkpoint with SHA256 verification.

    Args:
        model: Model to save
        optimizer: Optional optimizer state
        scheduler: Optional scheduler state
        epoch: Current epoch
        step: Current step
        metrics: Optional metrics dictionary
        path: Output path
        config: Optional configuration dictionary
        verify_sha256: Whether to compute and store SHA256

    Returns:
        Path to saved checkpoint
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare checkpoint dictionary
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "step": step,
        "metrics": metrics or {},
        "config": config or {},
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    # Compute SHA256 of model weights
    if verify_sha256:
        model_sha256 = compute_state_dict_sha256(model.state_dict())
        checkpoint["model_sha256"] = model_sha256
        logger.info(f"Model SHA256: {model_sha256}")

    # Save checkpoint (single write)
    torch.save(checkpoint, path)

    # Compute and store file SHA256 in sidecar file (avoids double save)
    if verify_sha256:
        file_sha256 = compute_sha256(path)
        sha256_path = path.with_suffix(path.suffix + ".sha256")
        sha256_path.write_text(file_sha256)
        logger.info(f"File SHA256: {file_sha256} (stored in {sha256_path.name})")

    logger.info(f"Checkpoint saved to {path} (epoch={epoch}, step={step})")
    return str(path)


def load_checkpoint(
    path: Union[str, Path],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Union[str, torch.device] = "cpu",
    verify_sha256: bool = True,
    strict: bool = True,
) -> Dict[str, Any]:
    """Load model checkpoint with SHA256 verification.

    Args:
        path: Checkpoint file path
        model: Model to load weights into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        device: Device to load tensors to
        verify_sha256: Whether to verify SHA256
        strict: Whether to enforce strict loading

    Returns:
        Checkpoint metadata dictionary

    Raises:
        ValueError: If SHA256 verification fails
        FileNotFoundError: If checkpoint file doesn't exist
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    logger.info(f"Loading checkpoint from {path}")

    # Load checkpoint
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    # Verify model SHA256 if present
    if verify_sha256 and "model_sha256" in checkpoint:
        # Verify the checkpoint's model_state_dict matches saved SHA256
        checkpoint_sha256 = compute_state_dict_sha256(checkpoint["model_state_dict"])
        saved_sha256 = checkpoint["model_sha256"]
        if checkpoint_sha256 != saved_sha256:
            raise ValueError(
                f"SHA256 mismatch! Expected {saved_sha256}, got {checkpoint_sha256}. "
                f"Checkpoint file may be corrupted."
            )
        logger.info(f"Model SHA256 verified: {saved_sha256}")

    # Verify file SHA256 from sidecar if present
    if verify_sha256:
        sha256_path = path.with_suffix(path.suffix + ".sha256")
        if sha256_path.exists():
            saved_file_sha256 = sha256_path.read_text().strip()
            computed_file_sha256 = compute_sha256(path)
            if computed_file_sha256 != saved_file_sha256:
                raise ValueError(
                    f"File SHA256 mismatch! Expected {saved_file_sha256}, got {computed_file_sha256}. "
                    f"Checkpoint file may be corrupted."
                )
            logger.info(f"File SHA256 verified: {saved_file_sha256}")

    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    # Load optimizer state
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load scheduler state
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # Read file_sha256 from sidecar for metadata
    file_sha256 = ""
    sha256_path = path.with_suffix(path.suffix + ".sha256")
    if sha256_path.exists():
        file_sha256 = sha256_path.read_text().strip()

    metadata = {
        "epoch": checkpoint.get("epoch", 0),
        "step": checkpoint.get("step", 0),
        "metrics": checkpoint.get("metrics", {}),
        "config": checkpoint.get("config", {}),
        "timestamp": checkpoint.get("timestamp", ""),
        "model_sha256": checkpoint.get("model_sha256", ""),
        "file_sha256": file_sha256,
    }

    logger.info(f"Checkpoint loaded: epoch={metadata['epoch']}, step={metadata['step']}")
    return metadata


def verify_checkpoint(path: Union[str, Path]) -> Dict[str, Any]:
    """Verify checkpoint integrity without loading into model.

    Args:
        path: Checkpoint file path

    Returns:
        Dictionary with verification results
    """
    path = Path(path)
    results = {
        "exists": path.exists(),
        "file_sha256": None,
        "model_sha256": None,
        "model_sha256_valid": False,
        "file_sha256_valid": False,
        "keys": [],
        "epoch": None,
        "step": None,
    }

    if not results["exists"]:
        return results

    # Compute file SHA256
    results["file_sha256"] = compute_sha256(path)

    # Check for sidecar SHA256 file
    sha256_path = path.with_suffix(path.suffix + ".sha256")
    saved_file_sha256 = None
    if sha256_path.exists():
        saved_file_sha256 = sha256_path.read_text().strip()
        results["file_sha256_valid"] = (results["file_sha256"] == saved_file_sha256)

    # Load checkpoint metadata
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        results["keys"] = list(checkpoint.keys())
        results["epoch"] = checkpoint.get("epoch")
        results["step"] = checkpoint.get("step")
        results["model_sha256"] = checkpoint.get("model_sha256")

        # Verify model SHA256 if present
        if "model_state_dict" in checkpoint and results["model_sha256"]:
            computed = compute_state_dict_sha256(checkpoint["model_state_dict"])
            results["model_sha256_valid"] = (computed == results["model_sha256"])

        # Verify file SHA256 from sidecar (fallback to checkpoint for backwards compat)
        if not sha256_path.exists() and "file_sha256" in checkpoint:
            results["file_sha256_valid"] = (results["file_sha256"] == checkpoint["file_sha256"])

    except Exception as e:
        results["error"] = str(e)

    return results


def find_latest_checkpoint(
    checkpoint_dir: Union[str, Path],
    pattern: str = "checkpoint_*.pt"
) -> Optional[Path]:
    """Find the latest checkpoint in a directory.

    Args:
        checkpoint_dir: Directory to search
        pattern: Glob pattern for checkpoint files

    Returns:
        Path to latest checkpoint, or None if not found
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = list(checkpoint_dir.glob(pattern))
    if not checkpoints:
        return None
    # Sort by modification time
    checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return checkpoints[0]


class CheckpointManager:
    """Manages checkpoint saving with rotation."""

    def __init__(
        self,
        save_dir: Union[str, Path],
        keep_last_n: int = 3,
        verify_sha256: bool = True,
    ):
        """Initialize checkpoint manager.

        Args:
            save_dir: Directory to save checkpoints
            keep_last_n: Number of recent checkpoints to keep
            verify_sha256: Whether to verify SHA256
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self.verify_sha256 = verify_sha256

    def save(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: int = 0,
        step: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
    ) -> Path:
        """Save checkpoint with automatic rotation.

        Args:
            model: Model to save
            optimizer: Optional optimizer
            scheduler: Optional scheduler
            epoch: Current epoch
            step: Current step
            metrics: Optional metrics
            config: Optional config
            name: Optional custom name

        Returns:
            Path to saved checkpoint
        """
        if name is None:
            name = f"checkpoint_epoch{epoch}_step{step}.pt"
        path = self.save_dir / name

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            step=step,
            metrics=metrics,
            path=path,
            config=config,
            verify_sha256=self.verify_sha256,
        )

        self._rotate()
        return path

    def _rotate(self):
        """Remove old checkpoints beyond keep_last_n."""
        checkpoints = list(self.save_dir.glob("checkpoint_*.pt"))
        checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        for old_checkpoint in checkpoints[self.keep_last_n:]:
            try:
                old_checkpoint.unlink()
                logger.info(f"Removed old checkpoint: {old_checkpoint}")
            except Exception as e:
                logger.warning(f"Failed to remove old checkpoint {old_checkpoint}: {e}")

    def load_latest(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: Union[str, torch.device] = "cpu",
    ) -> Optional[Dict[str, Any]]:
        """Load the latest checkpoint.

        Args:
            model: Model to load into
            optimizer: Optional optimizer
            scheduler: Optional scheduler
            device: Device to load to

        Returns:
            Checkpoint metadata or None if no checkpoint found
        """
        latest = find_latest_checkpoint(self.save_dir)
        if latest is None:
            logger.info("No checkpoint found to load")
            return None
        return load_checkpoint(
            path=latest,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            verify_sha256=self.verify_sha256,
        )