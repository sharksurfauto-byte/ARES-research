"""Tests for LRM architecture and trainer."""

import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ares.lrm import LRM, LRMTrainer


class TestLRMArchitecture:
    def test_lrm_forward_2d(self, device):
        model = LRM(input_dim=16, hidden_dim=32, num_layers=2, num_heads=4).to(device)
        x = torch.randn(4, 16, device=device)  # 2D input [batch=4, input_dim=16]
        prob, risk = model(x)

        assert prob.shape == (4,)
        assert risk.shape == (4,)

    def test_lrm_forward_3d(self, device):
        model = LRM(input_dim=16, hidden_dim=32, num_layers=2, num_heads=4).to(device)
        x = torch.randn(4, 8, 16, device=device)  # 3D input [batch=4, seq=8, input_dim=16]
        prob, risk = model(x)

    def test_lrm_forward_1d(self, device):
        model = LRM(input_dim=16, hidden_dim=32, num_layers=2, num_heads=4).to(device)
        x = torch.randn(16, device=device)
        prob, risk = model(x)
        assert prob.dim() == 0 or prob.shape == ()
        assert risk.dim() == 0 or risk.shape == ()

    def test_lrm_forward_logits(self, device):
        model = LRM(input_dim=16, hidden_dim=32, num_layers=2, num_heads=4).to(device)
        x = torch.randn(4, 16, device=device)
        logits, risk = model.forward_logits(x)
        assert logits.shape == (4,)
        assert risk.shape == (4,)


class TestLRMTrainer:
    def test_lrm_trainer_loop(self, device):
        model = LRM(input_dim=16, hidden_dim=32).to(device)
        trainer = LRMTrainer(
            model=model,
            device=device,
            config={"learning_rate": 1e-3, "batch_size": 4, "pos_weight": 1.5},
        )

        hidden = torch.randn(8, 5, 16)
        labels = torch.randint(0, 2, (8, 5)).float()
        mask = torch.ones(8, 5)

        history = trainer.train(
            token_hidden_states=hidden,
            correctness_labels=labels,
            attention_mask=mask,
            epochs=2,
            val_hidden_states=hidden,
            val_labels=labels,
            val_mask=mask,
        )

        assert len(history["train"]) == 2
        assert "loss" in history["train"][0]
        assert "accuracy" in history["train"][0]

        # Test save & load
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = str(Path(tmp_dir) / "lrm.pt")
            trainer.save(save_path)

            model2 = LRM(input_dim=16, hidden_dim=32).to(device)
            meta = LRMTrainer.load(model2, save_path, device=device)
            assert meta is not None
