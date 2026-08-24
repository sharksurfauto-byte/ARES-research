"""Hydra/OmegaConf structured configs for ARES.

Provides type-safe configuration schemas for all components.
"""

from dataclasses import dataclass, field

from omegaconf import MISSING

from ares.backbone.config import BackboneConfig


@dataclass
class DDPConfig:
    """Distributed Data Parallel configuration."""

    backend: str = "nccl"
    find_unused_parameters: bool = False
    timeout_minutes: int = 30


@dataclass
class WandbConfig:
    """Weights & Biases configuration."""

    project: str = "ares-research"
    entity: str | None = None
    tags: list[str] = field(default_factory=list)
    mode: str = "online"  # online, offline, disabled
    dir: str | None = None


@dataclass
class CheckpointConfig:
    """Checkpoint system configuration."""

    save_dir: str = "checkpoints"
    save_every_n_steps: int = 1000
    save_every_n_epochs: int = 1
    keep_last_n: int = 3
    verify_sha256: bool = True


@dataclass
class ExperimentConfig:
    """Base experiment configuration."""

    seed: int = 42
    output_dir: str = "outputs"
    experiment_name: str = "default"
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    ddp: DDPConfig = field(default_factory=DDPConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)


@dataclass
class DataConfig:
    """Data loading configuration."""

    dataset_name: str = MISSING
    dataset_split: str = "train"
    max_samples: int | None = None
    batch_size: int = 4
    num_workers: int = 4
    max_length: int = 2048


@dataclass
class TrainingConfig:
    """Training configuration."""

    num_epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "bf16"  # bf16, fp16, fp32
    data: DataConfig = field(default_factory=DataConfig)


@dataclass
class GRMConfig:
    """Global Reliability Model configuration (PRD §3.2.3)."""

    hidden_dim: int = 512
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    domain_classes: int = 5  # code, general, math, science, reasoning


@dataclass
class LRMConfig:
    """Local Reliability Model configuration (PRD §3.2.4)."""

    hidden_dim: int = 512
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1


@dataclass
class RouterConfig:
    """Router Network configuration (PRD §3.2.5)."""

    hidden_dim: int = 256
    num_layers: int = 2
    num_experts: int = 5
    dropout: float = 0.1


@dataclass
class ExpertConfig:
    """LoRA Expert configuration (PRD §3.2.6)."""

    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    expert_names: list[str] = field(
        default_factory=lambda: ["general", "math", "code", "science", "reasoning"]
    )


@dataclass
class CalibrationConfig:
    """Calibration configuration (PRD §4.6)."""

    temperature_scaling: bool = True
    isotonic_regression: bool = True
    num_bins: int = 10


# Full config composition
@dataclass
class ARESConfig:
    """Complete ARES configuration."""

    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    grm: GRMConfig = field(default_factory=GRMConfig)
    lrm: LRMConfig = field(default_factory=LRMConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    expert: ExpertConfig = field(default_factory=ExpertConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
