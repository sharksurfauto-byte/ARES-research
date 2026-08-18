"""Tests for backbone loader."""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
from omegaconf import DictConfig, OmegaConf

# Add src to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ares.backbone.config import BackboneConfig
from ares.backbone.base import Backbone, QwenBackbone
from ares.backbone.loader import load_backbone, verify_backbone, _get_torch_dtype


class TestBackboneConfig:
    """Tests for BackboneConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = BackboneConfig()
        assert config.name == "Qwen/Qwen2.5-0.5B"
        assert config.use_cache is False
        assert config.attn_implementation == "eager"
        assert config.load_in_4bit is False
        assert config.torch_dtype == "bfloat16"
        assert config.hidden_state_layers == (-1, -6, -12, -24)

    def test_config_validation_dtype(self):
        """Test dtype validation."""
        with pytest.raises(ValueError):
            BackboneConfig(torch_dtype="invalid")

    def test_config_validation_attn(self):
        """Test attention implementation validation."""
        with pytest.raises(ValueError):
            BackboneConfig(attn_implementation="invalid")

    def test_to_dict(self):
        """Test serialization to dict."""
        config = BackboneConfig()
        d = config.to_dict()
        assert d["name"] == "Qwen/Qwen2.5-0.5B"
        assert d["use_cache"] is False
        assert isinstance(d["hidden_state_layers"], list)

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "name": "test-model",
            "hidden_state_layers": [-1, -2],
        }
        config = BackboneConfig.from_dict(data)
        assert config.name == "test-model"
        assert config.hidden_state_layers == (-1, -2)


class TestTorchDtype:
    """Tests for torch dtype conversion."""

    def test_float16(self):
        assert _get_torch_dtype("float16") == torch.float16

    def test_bfloat16(self):
        assert _get_torch_dtype("bfloat16") == torch.bfloat16

    def test_float32(self):
        assert _get_torch_dtype("float32") == torch.float32

    def test_aliases(self):
        assert _get_torch_dtype("fp16") == torch.float16
        assert _get_torch_dtype("bf16") == torch.bfloat16
        assert _get_torch_dtype("fp32") == torch.float32

    def test_invalid(self):
        with pytest.raises(ValueError):
            _get_torch_dtype("invalid")


class TestBackboneInterface:
    """Tests for Backbone abstract interface."""

    def test_abstract_methods(self):
        """Test that Backbone is abstract."""
        with pytest.raises(TypeError):
            Backbone()

    def test_qwen_backbone_abstract_methods(self):
        """Test QwenBackbone implements required methods."""
        # Check all abstract methods are implemented
        assert hasattr(QwenBackbone, 'forward')
        assert hasattr(QwenBackbone, 'get_hidden_states')
        assert hasattr(QwenBackbone, 'get_model_config')
        assert hasattr(QwenBackbone, 'get_device')
        assert hasattr(QwenBackbone, 'get_dtype')
        assert hasattr(QwenBackbone, 'hidden_size')
        assert hasattr(QwenBackbone, 'num_layers')
        assert hasattr(QwenBackbone, 'vocab_size')


class TestVerifyBackbone:
    """Tests for verify_backbone function."""

    def test_verify_backbone_mock(self):
        """Test verify_backbone with mocked model."""
        # Create mock model
        mock_model = Mock()
        mock_model.config = Mock()
        mock_model.config.hidden_size = 896
        mock_model.config.num_hidden_layers = 24
        mock_model.config.vocab_size = 151936
        mock_model.config.model_type = "qwen2"
        mock_model.config.architectures = ["Qwen2ForCausalLM"]

        # Mock forward output
        mock_output = Mock()
        mock_output.logits = torch.randn(1, 32, 151936)
        mock_output.hidden_states = tuple(
            torch.randn(1, 32, 896) for _ in range(25)
        )
        mock_output.attentions = tuple(
            torch.randn(1, 14, 32, 32) for _ in range(24)
        )
        mock_model.return_value = mock_output
        # parameters() should return an iterator
        mock_model.parameters.side_effect = lambda: iter([torch.randn(10, 10)])

        # Create backbone wrapper
        backbone = QwenBackbone(
            model=mock_model,
            config=mock_model.config.to_dict(),
            hidden_state_layers=(-1, -2),
        )

        # Test verification
        test_input = torch.randint(0, 1000, (1, 32))
        results = verify_backbone(backbone, test_input)

        assert results["model_loaded"] is True
        assert results["forward_pass"] is True
        assert results["logits_shape"] == [1, 32, 151936]
        assert results["hidden_states_extracted"] is True
        assert len(results["hidden_states_shapes"]) == 25


@patch('ares.backbone.loader.AutoModelForCausalLM.from_pretrained')
@patch('ares.backbone.loader.AutoConfig.from_pretrained')
class TestLoadBackbone:
    """Tests for load_backbone function."""

    def test_load_backbone_cpu(self, mock_config_class, mock_model_class):
        """Test loading backbone on CPU."""
        # Setup mocks
        mock_config = Mock()
        mock_config.hidden_size = 896
        mock_config.num_hidden_layers = 24
        mock_config.vocab_size = 151936
        mock_config.model_type = "qwen2"
        mock_config.architectures = ["Qwen2ForCausalLM"]
        mock_config.to_dict.return_value = {
            "hidden_size": 896,
            "num_hidden_layers": 24,
            "vocab_size": 151936,
        }
        mock_config_class.return_value = mock_config

        mock_model = Mock()
        mock_model.config = mock_config
        mock_model.parameters.side_effect = lambda: iter([torch.randn(10, 10)])
        mock_model_class.return_value = mock_model

        # Load backbone
        config = BackboneConfig(
            name="test-model",
            device_map="cpu",
            torch_dtype="float32",
        )
        backbone = load_backbone(config)

        # Verify
        assert isinstance(backbone, QwenBackbone)
        assert backbone.hidden_size == 896
        assert backbone.num_layers == 24
        assert backbone.vocab_size == 151936
        mock_model_class.assert_called_once()

    def test_load_backbone_4bit(self, mock_config_class, mock_model_class):
        """Test loading backbone with 4-bit quantization."""
        # Setup mocks
        mock_config = Mock()
        mock_config.hidden_size = 896
        mock_config.num_hidden_layers = 24
        mock_config.vocab_size = 151936
        mock_config.model_type = "qwen2"
        mock_config.architectures = ["Qwen2ForCausalLM"]
        mock_config.to_dict.return_value = {}
        mock_config_class.return_value = mock_config

        mock_model = Mock()
        mock_model.config = mock_config
        mock_model.parameters.side_effect = lambda: iter([torch.randn(10, 10)])
        mock_model_class.return_value = mock_model

        # Load backbone with 4-bit
        config = BackboneConfig(
            name="test-model",
            device_map="cpu",
            torch_dtype="float32",
            load_in_4bit=True,
        )
        backbone = load_backbone(config)

        # Verify quantization config was passed
        call_kwargs = mock_model_class.call_args[1]
        assert "quantization_config" in call_kwargs


class TestBackboneProperties:
    """Tests for backbone properties."""

    @patch('ares.backbone.loader.AutoModelForCausalLM.from_pretrained')
    @patch('ares.backbone.loader.AutoConfig.from_pretrained')
    def test_frozen_parameters(self, mock_config_class, mock_model_class):
        """Test that backbone parameters are frozen."""
        mock_config = Mock()
        mock_config.hidden_size = 896
        mock_config.num_hidden_layers = 24
        mock_config.vocab_size = 151936
        mock_config.model_type = "qwen2"
        mock_config.architectures = ["Qwen2ForCausalLM"]
        mock_config.to_dict.return_value = {}
        mock_config_class.return_value = mock_config

        # Create mock parameters with requires_grad
        param1 = torch.nn.Parameter(torch.randn(10, 10), requires_grad=True)
        param2 = torch.nn.Parameter(torch.randn(10, 10), requires_grad=True)
        mock_model = Mock()
        mock_model.config = mock_config
        mock_model.parameters.side_effect = lambda: iter([param1, param2])
        mock_model_class.return_value = mock_model

        config = BackboneConfig(name="test-model", device_map="cpu")
        backbone = load_backbone(config)

        # Verify all parameters are frozen
        for param in backbone._model.parameters():
            assert param.requires_grad is False

    @patch('ares.backbone.loader.AutoModelForCausalLM.from_pretrained')
    @patch('ares.backbone.loader.AutoConfig.from_pretrained')
    def test_model_in_eval_mode(self, mock_config_class, mock_model_class):
        """Test that backbone is in eval mode."""
        mock_config = Mock()
        mock_config.hidden_size = 896
        mock_config.num_hidden_layers = 24
        mock_config.vocab_size = 151936
        mock_config.to_dict.return_value = {}
        mock_config_class.return_value = mock_config

        mock_model = Mock()
        mock_model.config = mock_config
        mock_model.parameters.side_effect = lambda: iter([torch.randn(10, 10)])
        mock_model.training = False
        mock_model_class.return_value = mock_model

        config = BackboneConfig(name="test-model", device_map="cpu")
        backbone = load_backbone(config)

        assert backbone._model.training is False