"""Model loading utilities for ARES backbone."""

import logging
from typing import Any

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from .base import Backbone, QwenBackbone
from .config import BackboneConfig

logger = logging.getLogger(__name__)


def load_backbone(config_or_name: Any, **kwargs) -> Backbone:
    """Load a pretrained backbone model with ARES-required configuration.

    Args:
        config_or_name: BackboneConfig instance or model_name string
        **kwargs: Overrides when config_or_name is a string (e.g. device)

    Returns:
        Backbone wrapper (QwenBackbone) with frozen weights and correct settings

    Raises:
        ValueError: If model loading fails or config is invalid
    """
    if isinstance(config_or_name, str):
        device = kwargs.pop("device", "cuda" if torch.cuda.is_available() else "cpu")
        device_str = device.type if isinstance(device, torch.device) else str(device)
        cfg_dict = {
            "name": config_or_name,
            "revision": kwargs.pop("revision", "main"),
            # Use float16 (not bfloat16) — T4/V100 don't support bf16 natively
            # and silently upcast to float32, doubling VRAM usage.
            "torch_dtype": "float32" if device_str == "cpu" else "float16",
            # device_map=None avoids accelerate dispatch entirely.
            # We'll .to(device) explicitly after loading.
            "device_map": None,
            "use_cache": False,
            "attn_implementation": "eager",
            "load_in_4bit": (
                False
                if device_str == "cpu"
                else ("7B" in config_or_name and "4bit" in config_or_name)
            ),
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": "float16",
            "use_peft": False,
            # Gradient checkpointing is a TRAINING optimization.
            # Disable by default; scripts that train can enable it explicitly.
            "gradient_checkpointing": False,
            "hidden_state_layers": (-1, -6, -12, -24),
        }
        # Allow callers to override (e.g. gradient_checkpointing=True for training)
        cfg_dict.update(kwargs)
        config = config_or_name_to_config = BackboneConfig.from_dict(cfg_dict)
        config._device_str = device_str  # stash for .to() later
    else:
        config = config_or_name
        config._device_str = getattr(config, "_device_str", None)

    logger.info(f"Loading backbone: {config.name}")

    # Prepare model loading kwargs
    model_kwargs = _build_model_kwargs(config)

    # Load model config first to verify
    model_config = AutoConfig.from_pretrained(
        config.name,
        revision=config.revision,
        trust_remote_code=True,
    )

    # Load model with appropriate quantization
    if config.load_in_4bit:
        logger.info("Loading with 4-bit NF4 quantization (bitsandbytes)")
        model = AutoModelForCausalLM.from_pretrained(
            config.name,
            revision=config.revision,
            config=model_config,
            quantization_config=model_kwargs["quantization_config"],
            device_map=config.device_map,
            torch_dtype=_get_torch_dtype(config.torch_dtype),
            attn_implementation=config.attn_implementation,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
    else:
        logger.info(f"Loading with {config.torch_dtype} precision")
        model = AutoModelForCausalLM.from_pretrained(
            config.name,
            revision=config.revision,
            config=model_config,
            device_map=config.device_map,
            torch_dtype=_get_torch_dtype(config.torch_dtype),
            attn_implementation=config.attn_implementation,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

    # Apply critical ARES settings (PRD §7.4)
    model.config.use_cache = False
    model.config.attn_implementation = "eager"

    # Move model to target device (needed when device_map=None)
    target_device = getattr(config, "_device_str", None)
    if target_device and target_device != "cpu" and config.device_map is None:
        logger.info(f"Moving model to {target_device}")
        model = model.to(target_device)

    # Enable gradient checkpointing only if explicitly requested (training)
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    # Wrap in QwenBackbone
    backbone = QwenBackbone(
        model=model,
        config=model_config.to_dict(),
        hidden_state_layers=config.hidden_state_layers,
    )

    logger.info(
        f"Backbone loaded: {config.name} | "
        f"hidden_size={backbone.hidden_size} | "
        f"num_layers={backbone.num_layers} | "
        f"vocab_size={backbone.vocab_size} | "
        f"device={backbone.get_device()} | "
        f"dtype={backbone.get_dtype()}"
    )

    return backbone


def _build_model_kwargs(config: BackboneConfig) -> dict[str, Any]:
    """Build kwargs for model loading based on config."""
    kwargs = {}

    if config.load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=_get_torch_dtype(config.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
        )

    return kwargs


def _get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert string to torch dtype."""
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    dtype_lower = dtype_str.lower()
    if dtype_lower not in dtype_map:
        raise ValueError(f"Unknown dtype: {dtype_str}. Valid: {list(dtype_map.keys())}")
    return dtype_map[dtype_lower]


def verify_backbone(
    backbone: Backbone, test_input: torch.Tensor | None = None
) -> dict[str, Any]:
    """Verify backbone loads correctly and can run forward pass.

    Args:
        backbone: Loaded backbone instance
        test_input: Optional test input tensor (default: dummy)

    Returns:
        Dictionary with verification results
    """
    device = backbone.get_device()
    dtype = backbone.get_dtype()

    if test_input is None:
        test_input = torch.randint(0, 1000, (1, 32), device=device)

    results = {
        "model_loaded": True,
        "forward_pass": False,
        "hidden_states_extracted": False,
        "logits_shape": None,
        "hidden_states_shapes": None,
        "errors": [],
    }

    try:
        # Forward pass with hidden states
        with torch.no_grad():
            outputs = backbone.forward(
                input_ids=test_input,
                output_hidden_states=True,
                output_attentions=True,
            )

        results["forward_pass"] = True
        results["logits_shape"] = list(outputs.logits.shape)

        # Check hidden states
        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            results["hidden_states_extracted"] = True
            results["hidden_states_shapes"] = [list(h.shape) for h in outputs.hidden_states]
            logger.info(f"Hidden states extracted: {len(outputs.hidden_states)} layers")
            for i, h in enumerate(outputs.hidden_states):
                logger.info(f"  Layer {i}: {h.shape}")

    except Exception as e:
        results["errors"].append(str(e))
        logger.error(f"Verification failed: {e}")

    return results
