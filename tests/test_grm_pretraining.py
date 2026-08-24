"""Tests for GRM Self-Supervised Pretraining (PRD §4.2)."""

import pytest
import torch

from ares.grm import GRM, GRMPretrainer, PretrainingConfig, create_pretraining_dataloader
from ares.grm.pretraining import ContrastiveConfig, ReconstructionConfig, ReconstructionHead


class TestContrastiveLoss:
    """Test contrastive loss computation."""

    @pytest.fixture
    def pretrainer(self):
        """Create a minimal pretrainer for testing loss functions."""
        grm = GRM(input_dim=64, hidden_dim=128, num_layers=2, num_heads=4)
        config = PretrainingConfig(
            contrastive=ContrastiveConfig(temperature=0.07, loss_weight=1.0),
            reconstruction=ReconstructionConfig(loss_weight=0.5, hidden_dim=128),
            learning_rate=1e-3,
            batch_size=4,
        )
        return GRMPretrainer(model=grm, device=torch.device("cpu"), config=config)

    def test_contrastive_loss_basic(self, pretrainer):
        """Test contrastive loss with matching positive pairs."""
        # Create two views of same data (should be similar)
        batch_size = 8
        dim = 128  # hidden_dim
        z1 = torch.randn(batch_size, dim)
        z2 = z1 + 0.01 * torch.randn_like(z1)  # Small noise

        loss = pretrainer.contrastive_loss(z1, z2)
        assert loss.item() >= 0
        assert loss.item() < 10  # Should be relatively small for positive pairs

    def test_contrastive_loss_negative_pairs(self, pretrainer):
        """Test contrastive loss with different samples (should be larger)."""
        batch_size = 8
        dim = 128  # hidden_dim
        z1 = torch.randn(batch_size, dim)
        z2 = torch.randn(batch_size, dim)  # Independent samples

        loss = pretrainer.contrastive_loss(z1, z2)
        assert loss.item() >= 0


class TestReconstructionLoss:
    """Test reconstruction loss."""

    def test_reconstruction_loss(self):
        """Test MSE reconstruction loss."""
        pretrainer = GRMPretrainer.__new__(GRMPretrainer)
        pretrainer.config = PretrainingConfig(
            contrastive=ContrastiveConfig(temperature=0.07, loss_weight=1.0),
            reconstruction=ReconstructionConfig(loss_weight=0.5, hidden_dim=256),
        )

        batch_size = 4
        input_dim = 896
        original = torch.randn(batch_size, input_dim)
        reconstructed = original + 0.1 * torch.randn_like(original)

        loss = pretrainer.reconstruction_loss(original, reconstructed)
        assert loss.item() >= 0
        assert loss.item() < 1.0  # MSE should be small for similar tensors

    def test_reconstruction_head_forward(self):
        """Test ReconstructionHead forward pass."""
        head = ReconstructionHead(grm_hidden_dim=512, input_dim=896, hidden_dim=512)
        x = torch.randn(4, 512)  # Input is GRM hidden_dim
        out = head(x)
        assert out.shape == (4, 896)  # Output is input_dim


class TestPretrainingConfig:
    """Test PretrainingConfig dataclass."""

    def test_default_config(self):
        """Test default config values."""
        config = PretrainingConfig()
        assert config.contrastive.temperature == 0.07
        assert config.contrastive.loss_weight == 1.0
        assert config.reconstruction.loss_weight == 0.5
        assert config.reconstruction.hidden_dim == 512
        assert config.learning_rate == 5e-5
        assert config.batch_size == 16
        assert config.epochs == 5

    def test_custom_config(self):
        """Test custom config values."""
        config = PretrainingConfig(
            contrastive=ContrastiveConfig(temperature=0.1, loss_weight=0.5),
            reconstruction=ReconstructionConfig(loss_weight=1.0, hidden_dim=256),
            learning_rate=1e-4,
            batch_size=32,
            epochs=10,
        )
        assert config.contrastive.temperature == 0.1
        assert config.reconstruction.hidden_dim == 256
        assert config.learning_rate == 1e-4


class TestCreatePretrainingDataloader:
    """Test dataloader creation for pretraining."""

    def test_create_dataloader(self):
        """Test create_pretraining_dataloader produces correct batch format."""
        # Simulate 4 layers, 100 samples, 896 dim
        layer_reps = [torch.randn(100, 896) for _ in range(4)]

        config = PretrainingConfig(batch_size=16)
        loader = create_pretraining_dataloader(layer_reps, config, shuffle=True)

        batch = next(iter(loader))
        assert "representations" in batch
        # Should be [L, B, D] = [4, 16, 896]
        assert batch["representations"].shape == (4, 16, 896)


class TestGRMPretrainer:
    """Test GRMPretrainer initialization and forward."""

    @pytest.fixture
    def grm_model(self):
        """Create a small GRM for testing."""
        return GRM(input_dim=64, hidden_dim=128, num_layers=2, num_heads=4)

    @pytest.fixture
    def pretrain_config(self):
        """Create pretraining config for testing."""
        return PretrainingConfig(
            contrastive=ContrastiveConfig(temperature=0.07, loss_weight=1.0),
            reconstruction=ReconstructionConfig(loss_weight=0.5, hidden_dim=128),
            learning_rate=1e-3,
            batch_size=4,
            epochs=1,
            weight_decay=1e-4,
        )

    def test_pretrainer_init(self, grm_model, pretrain_config):
        """Test GRMPretrainer initialization."""
        device = torch.device("cpu")
        pretrainer = GRMPretrainer(
            model=grm_model,
            device=device,
            config=pretrain_config,
        )
        assert pretrainer.model is grm_model
        assert pretrainer.device == device
        assert hasattr(pretrainer, "recon_head")
        assert hasattr(pretrainer, "optimizer")
        assert hasattr(pretrainer, "scheduler")

    def test_get_cls_token(self, grm_model, pretrain_config):
        """Test _get_cls_token extracts correct shape."""
        device = torch.device("cpu")
        pretrainer = GRMPretrainer(
            model=grm_model,
            device=device,
            config=pretrain_config,
        )

        x = torch.randn(4, 64)  # [batch, input_dim]
        cls_token = pretrainer._get_cls_token(x)
        assert cls_token.shape == (4, 128)  # [batch, hidden_dim]

    def test_train_epoch_runs(self, grm_model, pretrain_config):
        """Test train_epoch runs without error on dummy data."""
        device = torch.device("cpu")
        pretrainer = GRMPretrainer(
            model=grm_model,
            device=device,
            config=pretrain_config,
        )

        # Create dummy dataloader with 4 layers
        layer_reps = [torch.randn(20, 64) for _ in range(4)]
        loader = create_pretraining_dataloader(layer_reps, pretrain_config, shuffle=True)

        metrics = pretrainer.train_epoch(loader)

        assert "loss" in metrics
        assert "contrastive_loss" in metrics
        assert "reconstruction_loss" in metrics
        assert "temperature" in metrics
        assert "epoch" in metrics
        assert metrics["epoch"] == 1
        assert metrics["loss"] >= 0

    def test_validate_runs(self, grm_model, pretrain_config):
        """Test validation runs without error."""
        device = torch.device("cpu")
        pretrainer = GRMPretrainer(
            model=grm_model,
            device=device,
            config=pretrain_config,
        )

        layer_reps = [torch.randn(10, 64) for _ in range(4)]
        loader = create_pretraining_dataloader(layer_reps, pretrain_config, shuffle=False)

        metrics = pretrainer._validate(loader)

        assert "val_loss" in metrics
        assert "val_contrastive_loss" in metrics
        assert "val_reconstruction_loss" in metrics
        assert metrics["val_loss"] >= 0


class TestPretrainingIntegration:
    """Integration test for full pretraining loop."""

    def test_full_pretraining_loop(self):
        """Test full pretraining loop runs for a few steps."""
        device = torch.device("cpu")

        grm = GRM(input_dim=64, hidden_dim=128, num_layers=2, num_heads=4)
        config = PretrainingConfig(
            contrastive=ContrastiveConfig(temperature=0.07, loss_weight=1.0),
            reconstruction=ReconstructionConfig(loss_weight=0.5, hidden_dim=128),
            learning_rate=1e-3,
            batch_size=4,
            epochs=2,
            weight_decay=1e-4,
        )

        pretrainer = GRMPretrainer(
            model=grm,
            device=device,
            config=config,
        )

        # Train data
        train_reps = [torch.randn(40, 64) for _ in range(4)]
        train_loader = create_pretraining_dataloader(train_reps, config, shuffle=True)

        # Val data
        val_reps = [torch.randn(10, 64) for _ in range(4)]
        val_loader = create_pretraining_dataloader(val_reps, config, shuffle=False)

        history = pretrainer.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            epochs=2,
        )

        assert len(history["train"]) == 2
        assert len(history["val"]) == 2
        assert all("loss" in m for m in history["train"])
        assert all("val_loss" in m for m in history["val"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
