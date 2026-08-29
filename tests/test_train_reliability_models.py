"""Integration tests for scripts/train_reliability_models.py."""

import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import pytest

from ares import GRM, LRM, GRMTrainer, LRMTrainer


def test_train_reliability_models_cli_e2e(tmp_path):
    """Test full CLI invocation with synthetic dataset, calibration, and output directory verification."""
    # 1. Create dummy dataset directory
    data_dir = tmp_path / "representations"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Save synthetic train.pt and val.pt
    train_reps = torch.randn(30, 32)
    train_domain = torch.randint(0, 5, (30,))
    train_feas = torch.randint(0, 2, (30,)).float()

    val_reps = torch.randn(10, 32)
    val_domain = torch.randint(0, 5, (10,))
    val_feas = torch.randint(0, 2, (10,)).float()

    # Save in format load_representations can handle
    torch.save(
        {
            "representations": train_reps,
            "samples": [],
        },
        data_dir / "representations_train.pt",
    )

    output_dir = tmp_path / "checkpoints_reliability"

    script_path = Path(__file__).parent.parent / "scripts" / "train_reliability_models.py"

    cmd = [
        sys.executable,
        str(script_path),
        "--data_dir",
        str(data_dir),
        "--output_dir",
        str(output_dir),
        "--epochs",
        "2",
        "--batch_size",
        "8",
        "--lr",
        "1e-3",
        "--device",
        "cpu",
        "--calibrate",
        "--no_wandb",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Script failed with output:\n{result.stdout}\n{result.stderr}"

    # Verify output files
    grm_pt = output_dir / "grm.pt"
    lrm_pt = output_dir / "lrm.pt"
    grm_calib = output_dir / "grm_calibration.pt"
    lrm_calib = output_dir / "lrm_calibration.pt"

    assert grm_pt.exists(), "grm.pt not found"
    assert lrm_pt.exists(), "lrm.pt not found"
    assert grm_calib.exists(), "grm_calibration.pt not found"
    assert lrm_calib.exists(), "lrm_calibration.pt not found"

    # Verify checkpoints can be loaded back
    grm = GRM(input_dim=32, hidden_dim=512)
    meta_grm = GRMTrainer.load(grm, str(grm_pt), device=torch.device("cpu"))
    assert meta_grm is not None

    lrm = LRM(input_dim=32, hidden_dim=512)
    meta_lrm = LRMTrainer.load(lrm, str(lrm_pt), device=torch.device("cpu"))
    assert meta_lrm is not None

    # Verify calibration artifacts contents
    calib_data = torch.load(grm_calib, weights_only=False)
    assert "temperature" in calib_data
    assert "isotonic_model" in calib_data
    assert "ece_raw" in calib_data
