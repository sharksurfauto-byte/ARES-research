"""Weights & Biases integration for ARES.

PRD §6 Week 1: W&B experiment tracking.
"""

import os
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("wandb not available, W&B logging disabled")


class WandbLogger:
    """Wrapper for Weights & Biases logging."""

    def __init__(
        self,
        project: str = "ares-research",
        entity: Optional[str] = None,
        tags: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        mode: str = "online",
        dir: Optional[str] = None,
        name: Optional[str] = None,
        resume: Optional[str] = None,
        id: Optional[str] = None,
    ):
        """Initialize W&B logger.

        Args:
            project: W&B project name
            entity: W&B entity (username or team)
            tags: List of tags for the run
            config: Configuration dictionary to log
            mode: "online", "offline", or "disabled"
            dir: Directory for W&B files
            name: Run name
            resume: Resume mode ("allow", "must", "never", "auto")
            id: Run ID for resuming
        """
        self.project = project
        self.entity = entity
        self.tags = tags or []
        self.config = config or {}
        self.mode = mode
        self.dir = dir
        self.name = name
        self.resume = resume
        self.id = id
        self._run = None
        self._enabled = False

    def init(self) -> bool:
        """Initialize W&B run.

        Returns:
            True if W&B initialized successfully, False otherwise
        """
        if not WANDB_AVAILABLE:
            logger.warning("W&B not available, running in offline mode")
            self._enabled = False
            return False

        if self.mode == "disabled":
            logger.info("W&B disabled by configuration")
            self._enabled = False
            return False

        try:
            self._run = wandb.init(
                project=self.project,
                entity=self.entity,
                tags=self.tags,
                config=self.config,
                mode=self.mode,
                dir=self.dir,
                name=self.name,
                resume=self.resume,
                id=self.id,
                reinit=True,
            )
            self._enabled = True
            logger.info(f"W&B initialized: {self._run.name} ({self._run.id})")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize W&B: {e}")
            self._enabled = False
            return False

    def log(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        commit: bool = True,
    ):
        """Log metrics to W&B.

        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number
            commit: Whether to commit the step
        """
        if not self._enabled or not WANDB_AVAILABLE:
            return

        try:
            wandb.log(metrics, step=step, commit=commit)
        except Exception as e:
            logger.warning(f"W&B log failed: {e}")

    def log_model(
        self,
        path: str,
        name: str = "model",
        type: str = "model",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Log model artifact to W&B.

        Args:
            path: Path to model file or directory
            name: Artifact name
            type: Artifact type
            metadata: Optional metadata
        """
        if not self._enabled or not WANDB_AVAILABLE:
            return

        try:
            artifact = wandb.Artifact(name, type=type, metadata=metadata or {})
            artifact.add_file(path)
            wandb.log_artifact(artifact)
            logger.info(f"Model artifact logged: {name}")
        except Exception as e:
            logger.warning(f"W&B artifact log failed: {e}")

    def log_checkpoint(
        self,
        path: str,
        epoch: int,
        step: int,
        metrics: Optional[Dict[str, float]] = None,
    ):
        """Log checkpoint as artifact.

        Args:
            path: Checkpoint file path
            epoch: Current epoch
            step: Current step
            metrics: Optional metrics
        """
        name = f"checkpoint-epoch{epoch}-step{step}"
        metadata = {
            "epoch": epoch,
            "step": step,
            **(metrics or {}),
        }
        self.log_model(path, name=name, type="model", metadata=metadata)

    def watch(
        self,
        model: "torch.nn.Module",
        log: str = "gradients",
        log_freq: int = 100,
    ):
        """Watch model gradients/parameters.

        Args:
            model: PyTorch model
            log: What to log ("gradients", "parameters", "all", None)
            log_freq: Logging frequency
        """
        if not self._enabled or not WANDB_AVAILABLE:
            return

        try:
            wandb.watch(model, log=log, log_freq=log_freq)
        except Exception as e:
            logger.warning(f"W&B watch failed: {e}")

    def finish(self, exit_code: int = 0):
        """Finish W&B run.

        Args:
            exit_code: Exit code (0 for success)
        """
        if not self._enabled or not WANDB_AVAILABLE:
            return

        try:
            wandb.finish(exit_code=exit_code)
            logger.info("W&B run finished")
        except Exception as e:
            logger.warning(f"W&B finish failed: {e}")

    @property
    def run(self):
        """Get the W&B run object."""
        return self._run

    @property
    def enabled(self) -> bool:
        """Check if W&B is enabled."""
        return self._enabled


def init_wandb(
    config: Dict[str, Any],
    project: str = "ares-research",
    entity: Optional[str] = None,
    tags: Optional[List[str]] = None,
    mode: str = "online",
    dir: Optional[str] = None,
    name: Optional[str] = None,
) -> WandbLogger:
    """Initialize W&B with configuration.

    Args:
        config: Configuration dictionary
        project: W&B project name
        entity: W&B entity
        tags: List of tags
        mode: W&B mode
        dir: W&B directory
        name: Run name

    Returns:
        Initialized WandbLogger
    """
    # Extract W&B config if nested
    wandb_config = config.get("wandb", {})
    experiment_config = config.get("experiment", {})

    logger = WandbLogger(
        project=wandb_config.get("project", project),
        entity=wandb_config.get("entity", entity),
        tags=wandb_config.get("tags", tags or []),
        config=config,
        mode=wandb_config.get("mode", mode),
        dir=wandb_config.get("dir", dir),
        name=name or experiment_config.get("experiment_name"),
    )
    logger.init()
    return logger


def log_metrics(logger: WandbLogger, metrics: Dict[str, Any], step: Optional[int] = None):
    """Log metrics using WandbLogger.

    Args:
        logger: WandbLogger instance
        metrics: Metrics dictionary
        step: Optional step
    """
    logger.log(metrics, step=step)


def log_model_artifact(
    logger: WandbLogger,
    path: str,
    name: str = "model",
    type: str = "model",
):
    """Log model artifact using WandbLogger.

    Args:
        logger: WandbLogger instance
        path: Model path
        name: Artifact name
        type: Artifact type
    """
    logger.log_model(path, name=name, type=type)


def finish_wandb(logger: WandbLogger, exit_code: int = 0):
    """Finish W&B run.

    Args:
        logger: WandbLogger instance
        exit_code: Exit code
    """
    logger.finish(exit_code=exit_code)