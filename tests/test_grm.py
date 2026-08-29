"""Tests for GRM architecture and trainer."""

import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ares.grm import GRM, GRMTrainer


class TestGRMArchitecture:
    def test_grm_forward_2d(self, device):
        model = GRM(input_dim=32, hidden_dim=64, num_layers=2, num_heads=4, domain_classes=5).to(
            device
        )
        x = torch.randn(4, 32, device=device)  # 2D input [batch=4, input_dim=32]
        domain_logits, feasibility, global_rel = model(x)

        assert domain_logits.shape == (4, 5)
        assert feasibility.shape == (4, 1)
        assert global_rel.shape == (4, 1)

    def test_grm_forward_3d(self, device):
        model = GRM(input_dim=32, hidden_dim=64, num_layers=2, num_heads=4, domain_classes=5).to(
            device
        )
        x = torch.randn(4, 3, 32, device=device)  # 3D input [batch=4, seq=3, input_dim=32]
        domain_logits, feasibility, global_rel = model(x)

        assert domain_logits.shape == (4, 5)
        assert feasibility.shape == (4, 1)
        assert global_rel.shape == (4, 1)

    def test_grm_forward_1d(self, device):
        model = GRM(input_dim=16, hidden_dim=32).to(device)
        x = torch.randn(16, device=device)
        domain_logits, feasibility, global_rel = model(x)
        assert domain_logits.shape == (5,)
        assert feasibility.shape == (1,)
        assert global_rel.shape == (1,)

    def test_grm_forward_logits(self, device):
        model = GRM(input_dim=16, hidden_dim=32).to(device)
        x = torch.randn(3, 16, device=device)
        d_logits, f_logits, g_logits = model.forward_logits(x)
        assert d_logits.shape == (3, 5)
        assert f_logits.shape == (3, 1)
        assert g_logits.shape == (3, 1)


class TestGRMTrainer:
    def test_grm_trainer_loop(self, device):
        model = GRM(input_dim=16, hidden_dim=32).to(device)
        trainer = GRMTrainer(
            model=model,
            device=device,
            config={"learning_rate": 1e-3, "batch_size": 4},
        )

        reps = torch.randn(10, 16)
        domain_labels = torch.randint(0, 5, (10,))
        feasibility_labels = torch.randint(0, 2, (10,)).float()

        history = trainer.train(
            representations=reps,
            domain_labels=domain_labels,
            feasibility_labels=feasibility_labels,
            epochs=2,
            val_representations=reps,
            val_domain_labels=domain_labels,
            val_feasibility_labels=feasibility_labels,
        )

        assert len(history["train"]) == 2
        assert "loss" in history["train"][0]
        assert "domain_accuracy" in history["train"][0]

        # Test save & load
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = str(Path(tmp_dir) / "grm.pt")
            trainer.save(save_path)

            model2 = GRM(input_dim=16, hidden_dim=32).to(device)
            meta = GRMTrainer.load(model2, save_path, device=device)
            assert meta is not None
