"""Backbone configuration dataclass for ARES."""

from dataclasses import dataclass
from typing import Any


@dataclass
class BackboneConfig:
    """Configuration for loading a frozen pretrained backbone model.

    Based on PRD §7.3 model configuration.
    """

    # Model identification
    name: str = "Qwen/Qwen2.5-0.5B"
    revision: str = "main"

    # Precision and device
    torch_dtype: str = "float16"
    device_map: str | None = None  # None = load to CPU, then .to(device) explicitly

    # Critical flags (PRD §7.4)
    use_cache: bool = False  # Critical for dynamic expert switching
    attn_implementation: str = "eager"  # Required for output_attentions=True

    # 4-bit quantization (bitsandbytes NF4)
    load_in_4bit: bool = False
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_use_double_quant: bool = True

    # LoRA/PEFT (for expert adapters in later weeks)
    use_peft: bool = False
    peft_config: dict[str, Any] | None = None

    # Gradient checkpointing for memory efficiency (enable only for training)
    gradient_checkpointing: bool = False

    # Hidden state extraction layers (negative = from end)
    hidden_state_layers: tuple = (-1, -6, -12, -24)

    def __post_init__(self):
        """Validate configuration."""
        valid_dtypes = {"float16", "bfloat16", "float32"}
        if self.torch_dtype not in valid_dtypes:
            raise ValueError(f"torch_dtype must be one of {valid_dtypes}, got {self.torch_dtype}")

        valid_attn = {"eager", "sdpa", "flash_attention_2"}
        if self.attn_implementation not in valid_attn:
            raise ValueError(
                f"attn_implementation must be one of {valid_attn}, got {self.attn_implementation}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "revision": self.revision,
            "torch_dtype": self.torch_dtype,
            "device_map": self.device_map,
            "use_cache": self.use_cache,
            "attn_implementation": self.attn_implementation,
            "load_in_4bit": self.load_in_4bit,
            "bnb_4bit_quant_type": self.bnb_4bit_quant_type,
            "bnb_4bit_compute_dtype": self.bnb_4bit_compute_dtype,
            "bnb_4bit_use_double_quant": self.bnb_4bit_use_double_quant,
            "use_peft": self.use_peft,
            "peft_config": self.peft_config,
            "gradient_checkpointing": self.gradient_checkpointing,
            "hidden_state_layers": list(self.hidden_state_layers),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BackboneConfig":
        """Create from dictionary."""
        # Handle hidden_state_layers conversion from list to tuple
        if "hidden_state_layers" in data and isinstance(data["hidden_state_layers"], list):
            data["hidden_state_layers"] = tuple(data["hidden_state_layers"])
        return cls(**data)
