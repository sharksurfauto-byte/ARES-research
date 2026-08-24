"""Tests for representation collection and dataset."""

import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ares.representations import (
    RepresentationCollector,
    RepresentationDataset,
    RepresentationSample,
    last_token_pool,
    max_pool,
    mean_pool,
    pool_hidden_state,
)


class TestPooling:
    def test_pool_methods(self):
        # shape [batch=2, seq_len=4, hidden=8]
        hs = torch.randn(2, 4, 8)
        mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])

        # Last token pool
        p_last = last_token_pool(hs, mask)
        assert p_last.shape == (2, 8)
        assert torch.allclose(p_last[0], hs[0, 2])
        assert torch.allclose(p_last[1], hs[1, 1])

        # Mean pool
        p_mean = mean_pool(hs, mask)
        assert p_mean.shape == (2, 8)

        # Max pool
        p_max = max_pool(hs, mask)
        assert p_max.shape == (2, 8)

        # General pool function
        p_gen = pool_hidden_state(hs, method="mean", attention_mask=mask)
        assert p_gen.shape == (2, 8)


class TestRepresentationCollector:
    def test_collector_mock_backbone(self, device):
        mock_backbone = Mock()
        mock_backbone.hidden_size = 64
        mock_backbone.hidden_state_layers = (-1, -2)

        # Mock forward pass
        mock_outputs = Mock()
        mock_outputs.logits = torch.randn(2, 10, 100)
        mock_outputs.hidden_states = (
            torch.randn(2, 10, 64),
            torch.randn(2, 10, 64),
        )
        mock_backbone.forward.return_value = mock_outputs

        collector = RepresentationCollector(
            backbone=mock_backbone,
            layers=(-1, -2),
            pooling_method="mean",
            device=str(device.type),
        )

        input_ids = torch.randint(0, 100, (2, 10))
        mask = torch.ones(2, 10)
        labels = torch.tensor([5, 12])
        metadata = {"domain": "math", "task": "qa"}

        pooled, logits, samples = collector.collect(
            input_ids=input_ids,
            attention_mask=mask,
            labels=labels,
            metadata=metadata,
        )

        assert len(pooled) == 2  # 2 target layers
        assert pooled[0].shape == (2, 64)
        assert logits.shape == (2, 10, 100)
        assert samples is not None
        assert len(samples) == 2
        assert samples[0].domain == "math"
        assert samples[0].representation.shape == (64,)


class TestRepresentationDataset:
    def test_dataset_save_load(self):
        samples = [
            RepresentationSample(
                sample_id="test_0",
                domain="math",
                task="calc",
                layer=-1,
                representation=torch.randn(16),
                logits=torch.randn(50),
                prediction="12",
                correctness=True,
                confidence=0.9,
                entropy=0.1,
                margin=0.8,
            )
        ]
        reps = [torch.randn(16)]

        ds = RepresentationDataset(samples=samples, representations=reps)
        assert len(ds) == 1

        tensors = ds.get_tensors()
        assert tensors["representations"].shape == (1, 16)
        assert tensors["domain_labels"].shape == (1,)
        assert tensors["feasibility_labels"].shape == (1,)

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "reps.pt"
            ds.save(save_path)
            loaded_ds = RepresentationDataset.load(save_path)
            assert len(loaded_ds) == 1
            assert loaded_ds.samples[0].domain == "math"

            # Test train_test_split
            train_ds, val_ds = ds.train_test_split(test_fraction=0.5)
            assert len(train_ds) + len(val_ds) == 1
