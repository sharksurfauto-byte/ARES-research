"""Tests for checkpoint system."""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import tempfile
import hashlib

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ares.utils.checkpoint import (
    compute_sha256,
    compute_state_dict_sha256,
    save_checkpoint,
    load_checkpoint,
    verify_checkpoint,
    find_latest_checkpoint,
    CheckpointManager,
)


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)


class TestComputeSHA256:
    """Tests for SHA256 computation."""

    def test_compute_sha256_file(self, tmp_path):
        """Test computing SHA256 of a file."""
        test_file = tmp_path / "test.txt"
        test_content = b"Hello, World!"
        test_file.write_bytes(test_content)

        sha256 = compute_sha256(test_file)
        expected = hashlib.sha256(test_content).hexdigest()
        assert sha256 == expected

    def test_compute_sha256_large_file(self, tmp_path):
        """Test computing SHA256 of a larger file."""
        test_file = tmp_path / "large.bin"
        test_content = b"x" * 100000
        test_file.write_bytes(test_content)

        sha256 = compute_sha256(test_file)
        expected = hashlib.sha256(test_content).hexdigest()
        assert sha256 == expected

    def test_compute_state_dict_sha256(self):
        """Test computing SHA256 of state dict."""
        model = SimpleModel()
        state_dict = model.state_dict()

        sha256 = compute_state_dict_sha256(state_dict)
        assert isinstance(sha256, str)
        assert len(sha256) == 64  # SHA256 hex length

    def test_compute_state_dict_sha256_deterministic(self):
        """Test that SHA256 is deterministic for same state dict."""
        model1 = SimpleModel()
        model2 = SimpleModel()
        model2.load_state_dict(model1.state_dict())

        sha1 = compute_state_dict_sha256(model1.state_dict())
        sha2 = compute_state_dict_sha256(model2.state_dict())
        assert sha1 == sha2

    def test_compute_state_dict_sha256_different_models(self):
        """Test that different models have different SHA256."""
        model1 = SimpleModel()
        model2 = SimpleModel()
        # Different initialization
        with torch.no_grad():
            model2.linear.weight.fill_(0.5)

        sha1 = compute_state_dict_sha256(model1.state_dict())
        sha2 = compute_state_dict_sha256(model2.state_dict())
        assert sha1 != sha2


class TestSaveLoadCheckpoint:
    """Tests for save/load checkpoint."""

    def test_save_checkpoint_basic(self, tmp_path):
        """Test basic checkpoint save."""
        model = SimpleModel()
        checkpoint_path = tmp_path / "checkpoint.pt"

        save_checkpoint(
            model=model,
            epoch=5,
            step=100,
            metrics={"loss": 0.5},
            path=checkpoint_path,
            config={"model": "test"},
            verify_sha256=True,
        )

        assert checkpoint_path.exists()

        # Verify checkpoint can be loaded
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        assert checkpoint["epoch"] == 5
        assert checkpoint["step"] == 100
        assert checkpoint["metrics"]["loss"] == 0.5
        assert checkpoint["config"]["model"] == "test"
        assert "model_sha256" in checkpoint
        # file_sha256 is now stored in sidecar file
        sha256_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".sha256")
        assert sha256_path.exists()

    def test_save_checkpoint_with_optimizer(self, tmp_path):
        """Test checkpoint save with optimizer."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        checkpoint_path = tmp_path / "checkpoint.pt"

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=1,
            step=10,
            path=checkpoint_path,
        )

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        assert "optimizer_state_dict" in checkpoint

    def test_load_checkpoint_basic(self, tmp_path):
        """Test basic checkpoint load."""
        model = SimpleModel()
        checkpoint_path = tmp_path / "checkpoint.pt"

        # Save
        save_checkpoint(model=model, epoch=3, step=50, path=checkpoint_path)

        # Create new model and load
        model2 = SimpleModel()
        metadata = load_checkpoint(
            path=checkpoint_path,
            model=model2,
            device="cpu",
        )

        assert metadata["epoch"] == 3
        assert metadata["step"] == 50
        assert metadata["model_sha256"] != ""

        # Verify weights match
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)

    def test_load_checkpoint_with_optimizer(self, tmp_path):
        """Test checkpoint load with optimizer."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        checkpoint_path = tmp_path / "checkpoint.pt"

        save_checkpoint(model=model, optimizer=optimizer, path=checkpoint_path)

        model2 = SimpleModel()
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
        metadata = load_checkpoint(
            path=checkpoint_path,
            model=model2,
            optimizer=optimizer2,
        )

        # Verify optimizer state loaded
        for pg1, pg2 in zip(optimizer.param_groups, optimizer2.param_groups):
            assert pg1["lr"] == pg2["lr"]

    def test_load_checkpoint_sha256_verification(self, tmp_path):
        """Test SHA256 verification on load."""
        model = SimpleModel()
        checkpoint_path = tmp_path / "checkpoint.pt"

        save_checkpoint(model=model, path=checkpoint_path, verify_sha256=True)

        model2 = SimpleModel()
        metadata = load_checkpoint(
            path=checkpoint_path,
            model=model2,
            verify_sha256=True,
        )

        # SHA256 verification passes (no exception raised)
        assert metadata["model_sha256"] != ""

    def test_load_checkpoint_sha256_mismatch(self, tmp_path):
        """Test SHA256 mismatch detection for corrupted checkpoint file."""
        model = SimpleModel()
        checkpoint_path = tmp_path / "checkpoint.pt"

        save_checkpoint(model=model, path=checkpoint_path, verify_sha256=True)

        # Corrupt the checkpoint by modifying the model_state_dict and re-saving
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        # Modify a tensor in the state dict
        for key in checkpoint["model_state_dict"]:
            checkpoint["model_state_dict"][key] = checkpoint["model_state_dict"][key] + 1.0
        torch.save(checkpoint, checkpoint_path)

        model2 = SimpleModel()
        with pytest.raises(ValueError, match="SHA256 mismatch"):
            load_checkpoint(
                path=checkpoint_path,
                model=model2,
                verify_sha256=True,
            )

    def test_load_checkpoint_nonexistent(self, tmp_path):
        """Test loading non-existent checkpoint."""
        model = SimpleModel()
        with pytest.raises(FileNotFoundError):
            load_checkpoint(
                path=tmp_path / "nonexistent.pt",
                model=model,
            )


class TestVerifyCheckpoint:
    """Tests for verify_checkpoint function."""

    def test_verify_checkpoint_valid(self, tmp_path):
        """Test verifying a valid checkpoint."""
        model = SimpleModel()
        checkpoint_path = tmp_path / "checkpoint.pt"

        save_checkpoint(model=model, path=checkpoint_path, verify_sha256=True)

        results = verify_checkpoint(checkpoint_path)

        assert results["exists"] is True
        assert results["file_sha256"] is not None
        assert results["model_sha256"] is not None
        assert results["model_sha256_valid"] is True
        assert results["file_sha256_valid"] is True

    def test_verify_checkpoint_nonexistent(self, tmp_path):
        """Test verifying non-existent checkpoint."""
        results = verify_checkpoint(tmp_path / "nonexistent.pt")
        assert results["exists"] is False


class TestFindLatestCheckpoint:
    """Tests for find_latest_checkpoint."""

    def test_find_latest_checkpoint(self, tmp_path):
        """Test finding latest checkpoint."""
        # Create multiple checkpoints
        for i in range(3):
            checkpoint_path = tmp_path / f"checkpoint_epoch{i}.pt"
            model = SimpleModel()
            save_checkpoint(model=model, epoch=i, path=checkpoint_path)

        latest = find_latest_checkpoint(tmp_path, "checkpoint_*.pt")
        assert latest is not None
        assert latest.name == "checkpoint_epoch2.pt"

    def test_find_latest_checkpoint_empty(self, tmp_path):
        """Test finding latest in empty directory."""
        latest = find_latest_checkpoint(tmp_path, "checkpoint_*.pt")
        assert latest is None


class TestCheckpointManager:
    """Tests for CheckpointManager."""

    def test_manager_save_and_rotate(self, tmp_path):
        """Test checkpoint manager rotation."""
        manager = CheckpointManager(
            save_dir=tmp_path,
            keep_last_n=2,
            verify_sha256=True,
        )

        model = SimpleModel()

        # Save 3 checkpoints
        for i in range(3):
            manager.save(model=model, epoch=i, step=i*10)

        # Only 2 should remain
        checkpoints = list(tmp_path.glob("checkpoint_*.pt"))
        assert len(checkpoints) == 2
        # Should be the latest 2
        epochs = [int(c.stem.split("epoch")[1].split("_")[0]) for c in checkpoints]
        assert set(epochs) == {1, 2}

    def test_manager_load_latest(self, tmp_path):
        """Test loading latest checkpoint via manager."""
        manager = CheckpointManager(save_dir=tmp_path, keep_last_n=3)
        model = SimpleModel()

        # Save checkpoint
        manager.save(model=model, epoch=5, step=100)

        # Load into new model
        model2 = SimpleModel()
        metadata = manager.load_latest(model=model2)

        assert metadata is not None
        assert metadata["epoch"] == 5
        assert metadata["step"] == 100

    def test_manager_load_latest_empty(self, tmp_path):
        """Test loading latest when no checkpoints exist."""
        manager = CheckpointManager(save_dir=tmp_path)
        model = SimpleModel()

        metadata = manager.load_latest(model=model)
        assert metadata is None