"""Shared test fixtures for ARES."""


import pytest
import torch
from omegaconf import OmegaConf


@pytest.fixture
def device():
    """Get test device (CPU for CI/incompatible GPUs, CUDA if functional)."""
    if torch.cuda.is_available():
        try:
            t = torch.zeros(1, device="cuda")
            _ = t + 1
            return torch.device("cuda")
        except Exception:
            pass
    return torch.device("cpu")


@pytest.fixture
def tmp_path_fixture(tmp_path):
    """Provide temporary directory path."""
    return tmp_path


@pytest.fixture
def mock_backbone_config():
    """Create a mock backbone config for testing."""
    return {
        "name": "Qwen/Qwen2.5-0.5B",
        "revision": "main",
        "torch_dtype": "float32",  # Use float32 for CPU testing
        "device_map": "cpu",
        "use_cache": False,
        "attn_implementation": "eager",
        "load_in_4bit": False,
        "gradient_checkpointing": False,
        "hidden_state_layers": [-1, -2],
    }


@pytest.fixture
def sample_batch():
    """Create a sample batch for testing."""
    return {
        "input_ids": torch.randint(0, 1000, (2, 16)),
        "attention_mask": torch.ones(2, 16),
    }


@pytest.fixture
def mock_config():
    """Create a mock Hydra config."""
    return OmegaConf.create(
        {
            "backbone": {
                "name": "Qwen/Qwen2.5-0.5B",
                "torch_dtype": "float32",
                "device_map": "cpu",
                "use_cache": False,
                "attn_implementation": "eager",
                "load_in_4bit": False,
                "hidden_state_layers": [-1, -2],
            },
            "experiment": {
                "seed": 42,
                "output_dir": "outputs/test",
            },
        }
    )
