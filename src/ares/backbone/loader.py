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
        is_cpu = device_str == "cpu"
        is_7b = "7b" in config_or_name.lower() or "8b" in config_or_name.lower()

        # Determine if 4-bit quantization should be active:
        # Activated when requested explicitly, or when 7B/4bit is indicated and not running on CPU
        requested_4bit = kwargs.pop(
            "load_in_4bit",
            "4bit" in config_or_name.lower() or (is_7b and not is_cpu),
        )
        load_in_4bit = bool(requested_4bit and not is_cpu)

        # Determine precision dtype on GPU
        if is_cpu:
            default_dtype = "float32"
        elif torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            default_dtype = "bfloat16"
        else:
            default_dtype = "float16"

        torch_dtype = kwargs.pop("torch_dtype", default_dtype)
        if torch_dtype == "bfloat16" and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
            logger.warning("bfloat16 requested but not supported on this GPU. Falling back to float16.")
            torch_dtype = "float16"

        # Bitsandbytes requires device_map when load_in_4bit=True
        if load_in_4bit:
            device_map = kwargs.pop("device_map", "auto")
            if device_map is None:
                device_map = "auto"
        else:
            device_map = kwargs.pop("device_map", None)

        bnb_compute = kwargs.pop(
            "bnb_4bit_compute_dtype",
            torch_dtype if torch_dtype != "float32" else "float16",
        )
        if bnb_compute == "bfloat16" and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
            bnb_compute = "float16"

        cfg_dict = {
            "name": config_or_name,
            "revision": kwargs.pop("revision", "main"),
            "torch_dtype": torch_dtype,
            "device_map": device_map,
            "use_cache": True,
            "attn_implementation": "eager",
            "load_in_4bit": load_in_4bit,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": bnb_compute,
            "use_peft": False,
            "gradient_checkpointing": False,
            "hidden_state_layers": (-1, -6, -12, -24),
        }
        # Allow callers to override remaining keys
        cfg_dict.update(kwargs)
        config = config_or_name_to_config = BackboneConfig.from_dict(cfg_dict)
        config._device_str = device_str  # stash for .to() later
    else:
        config = config_or_name
        config._device_str = getattr(config, "_device_str", None)
        is_cpu = getattr(config, "_device_str", None) == "cpu"
        is_7b = "7b" in config.name.lower() or "8b" in config.name.lower()

        # If 7B on GPU without explicit load_in_4bit, default to 4-bit
        if is_7b and not is_cpu and torch.cuda.is_available() and not getattr(config, "load_in_4bit", False):
            config.load_in_4bit = True

        # Ensure device_map is set when 4-bit is enabled (bitsandbytes requirement)
        if config.load_in_4bit and getattr(config, "device_map", None) is None:
            config.device_map = "auto"

        # Check bfloat16 hardware compatibility
        if getattr(config, "torch_dtype", None) == "bfloat16" and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
            config.torch_dtype = "float16"
        if getattr(config, "bnb_4bit_compute_dtype", None) == "bfloat16" and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
            config.bnb_4bit_compute_dtype = "float16"

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

    # Move model to target device (needed when device_map=None and not 4-bit)
    target_device = getattr(config, "_device_str", None)
    if not config.load_in_4bit and target_device and target_device != "cpu" and config.device_map is None:
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
