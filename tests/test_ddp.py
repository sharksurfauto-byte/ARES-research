"""Tests for DDP utilities."""

import os

# Add src to path
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ares.utils.ddp import (
    DDPContext,
    all_gather_object,
    broadcast_object,
    get_device,
    get_rank,
    get_world_size,
    init_ddp,
    is_distributed,
    is_main_process,
    reduce_dict,
    synchronize,
    wrap_model_ddp,
)


class TestDistributedHelpers:
    """Tests for distributed helper functions."""

    def test_is_distributed_false_by_default(self):
        """Test is_distributed returns False when not initialized."""
        # Ensure dist is not initialized
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        assert is_distributed() is False

    def test_get_rank_zero_by_default(self):
        """Test get_rank returns 0 when not distributed."""
        assert get_rank() == 0

    def test_get_world_size_one_by_default(self):
        """Test get_world_size returns 1 when not distributed."""
        assert get_world_size() == 1

    def test_is_main_process_true_by_default(self):
        """Test is_main_process returns True when not distributed."""
        assert is_main_process() is True


class TestInitDDp:
    """Tests for DDP initialization."""

    @patch.dict(os.environ, {"RANK": "0", "WORLD_SIZE": "1", "LOCAL_RANK": "0"})
    @patch("torch.distributed.init_process_group")
    def test_init_ddp_single_process(self, mock_init_pg):
        """Test DDP init with single process."""
        result = init_ddp(backend="gloo")
        assert result is True
        mock_init_pg.assert_called_once()

    @patch.dict(os.environ, {"RANK": "0", "WORLD_SIZE": "2", "LOCAL_RANK": "0"})
    @patch("torch.distributed.init_process_group")
    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.set_device")
    def test_init_ddp_multi_gpu(self, mock_set_device, mock_cuda_available, mock_init_pg):
        """Test DDP init with multi-GPU."""
        result = init_ddp(backend="nccl")
        assert result is True
        mock_set_device.assert_called_with(0)
        mock_init_pg.assert_called_once()

    def test_init_ddp_no_env(self):
        """Test DDP init without environment variables."""
        # Remove any existing env vars
        with patch.dict(os.environ, {}, clear=True):
            result = init_ddp()
            assert result is False


class TestWrapModelDDP:
    """Tests for wrap_model_ddp."""

    def test_wrap_model_ddp_not_distributed(self):
        """Test wrap_model_ddp when not distributed."""
        model = torch.nn.Linear(10, 5)
        wrapped = wrap_model_ddp(model)
        assert wrapped is model  # Returns unwrapped

    @patch("ares.utils.ddp.is_distributed", return_value=True)
    @patch("ares.utils.ddp.get_rank", return_value=0)
    @patch("ares.utils.ddp.get_world_size", return_value=2)
    @patch("ares.utils.ddp.DDP")
    def test_wrap_model_ddp_distributed(
        self, mock_ddp_class, mock_world_size, mock_rank, mock_is_dist
    ):
        """Test wrap_model_ddp when distributed."""
        # Mock the DDP constructor to avoid process group requirement
        mock_ddp_instance = Mock()
        mock_ddp_class.return_value = mock_ddp_instance

        model = torch.nn.Linear(10, 5)
        wrapped = wrap_model_ddp(model, find_unused_parameters=True)

        assert wrapped is mock_ddp_instance
        mock_ddp_class.assert_called_once()
        call_kwargs = mock_ddp_class.call_args[1]
        assert call_kwargs["find_unused_parameters"] is True


class TestReduceDict:
    """Tests for reduce_dict."""

    def test_reduce_dict_not_distributed(self):
        """Test reduce_dict when not distributed."""
        d = {"loss": torch.tensor(1.0), "acc": torch.tensor(0.9)}
        result = reduce_dict(d)
        assert result == d

    @patch("ares.utils.ddp.is_distributed", return_value=True)
    @patch("ares.utils.ddp.get_world_size", return_value=2)
    @patch("torch.distributed.all_reduce")
    def test_reduce_dict_distributed(self, mock_all_reduce, mock_world_size, mock_is_dist):
        """Test reduce_dict when distributed."""
        d = {"loss": torch.tensor(1.0), "acc": torch.tensor(0.9)}
        result = reduce_dict(d, average=True)

        mock_all_reduce.assert_called()
        assert "loss" in result
        assert "acc" in result


class TestAllGatherObject:
    """Tests for all_gather_object."""

    def test_all_gather_object_not_distributed(self):
        """Test all_gather_object when not distributed."""
        result = all_gather_object({"key": "value"})
        assert result == [{"key": "value"}]


class TestBroadcastObject:
    """Tests for broadcast_object."""

    def test_broadcast_object_not_distributed(self):
        """Test broadcast_object when not distributed."""
        result = broadcast_object("test", src=0)
        assert result == "test"


class TestSynchronize:
    """Tests for synchronize."""

    def test_synchronize_not_distributed(self):
        """Test synchronize when not distributed."""
        # Should not raise
        synchronize()


class TestGetDevice:
    """Tests for get_device."""

    @patch("torch.cuda.is_available", return_value=False)
    def test_get_device_cpu(self, mock_cuda):
        """Test get_device on CPU."""
        device = get_device()
        assert device.type == "cpu"

    @patch("torch.cuda.is_available", return_value=True)
    @patch.dict(os.environ, {"LOCAL_RANK": "1"})
    def test_get_device_cuda(self, mock_cuda):
        """Test get_device on CUDA."""
        device = get_device()
        assert device.type == "cuda"
        assert device.index == 1


class TestDDPContext:
    """Tests for DDPContext."""

    @patch("ares.utils.ddp.init_ddp", return_value=True)
    @patch("ares.utils.ddp.cleanup_ddp")
    def test_ddp_context_manager(self, mock_cleanup, mock_init):
        """Test DDPContext as context manager."""
        with DDPContext(backend="gloo") as ctx:
            assert ctx.initialized is True

        mock_init.assert_called_once_with(backend="gloo", timeout_minutes=30)
        mock_cleanup.assert_called_once()

    @patch("ares.utils.ddp.init_ddp", return_value=False)
    @patch("ares.utils.ddp.cleanup_ddp")
    def test_ddp_context_not_initialized(self, mock_cleanup, mock_init):
        """Test DDPContext when init fails."""
        with DDPContext() as ctx:
            assert ctx.initialized is False

        mock_cleanup.assert_not_called()

    @patch("ares.utils.ddp.wrap_model_ddp")
    @patch("ares.utils.ddp.init_ddp", return_value=True)
    @patch("ares.utils.ddp.cleanup_ddp")
    def test_ddp_context_wrap(self, mock_cleanup, mock_init, mock_wrap):
        """Test DDPContext wrap method."""
        mock_wrapped = Mock()
        mock_wrap.return_value = mock_wrapped

        with DDPContext() as ctx:
            model = torch.nn.Linear(10, 5)
            wrapped = ctx.wrap(model)

        assert wrapped is mock_wrapped
        mock_wrap.assert_called_once()

    @patch("ares.utils.ddp.reduce_dict")
    @patch("ares.utils.ddp.init_ddp", return_value=True)
    @patch("ares.utils.ddp.cleanup_ddp")
    def test_ddp_context_reduce(self, mock_cleanup, mock_init, mock_reduce):
        """Test DDPContext reduce method."""
        mock_reduce.return_value = {"loss": torch.tensor(0.5)}

        with DDPContext() as ctx:
            result = ctx.reduce({"loss": torch.tensor(1.0)})

        assert result == {"loss": torch.tensor(0.5)}
        mock_reduce.assert_called_once()
