"""Tests for calibration module."""

import pytest
import torch
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ares.calibration import (
    TemperatureScaling,
    fit_temperature_scaling,
    fit_isotonic_regression,
    apply_isotonic_regression,
    compute_ece,
    compute_brier_score,
    before_after_calibration,
)


class TestTemperatureScaling:
    def test_temperature_scaling_fit(self, device):
        scaler = TemperatureScaling(device=device)
        logits = torch.randn(20, device=device)
        labels = torch.randint(0, 2, (20,), device=device).float()

        results = scaler.fit(logits, labels, epochs=10)
        assert "temperature" in results
        assert results["temperature"] > 0.0

    def test_fit_temperature_scaling_convenience(self, device):
        logits = torch.randn(20, device=device)
        labels = torch.randint(0, 2, (20,), device=device).float()

        results = fit_temperature_scaling(logits, labels, epochs=10)
        assert "temperature" in results


class TestIsotonicAndMetrics:
    def test_isotonic_fit_apply(self):
        scores = np.random.uniform(0, 1, 30)
        labels = np.random.randint(0, 2, 30)

        ir_model = fit_isotonic_regression(scores, labels)
        calibrated = apply_isotonic_regression(scores, ir_model)

        assert calibrated.shape == (30,)
        assert np.all(calibrated >= 0.0) and np.all(calibrated <= 1.0)

    def test_compute_ece(self):
        probs = np.array([0.9, 0.8, 0.7, 0.6, 0.2, 0.1])
        labels = np.array([1, 1, 1, 0, 0, 0])

        ece = compute_ece(probs, labels, n_bins=5)
        assert ece >= 0.0

    def test_compute_brier_score(self):
        probs = np.array([0.9, 0.1])
        labels = np.array([1, 0])
        brier = compute_brier_score(probs, labels)
        assert abs(brier - 0.01) < 1e-4

    def test_before_after_calibration(self):
        raw = np.array([0.9, 0.8, 0.7, 0.6, 0.2, 0.1])
        calibrated = np.array([0.85, 0.8, 0.75, 0.5, 0.15, 0.05])
        labels = np.array([1, 1, 1, 0, 0, 0])

        res = before_after_calibration(raw, calibrated, labels)
        assert "raw_ece" in res
        assert "calibrated_ece" in res
        assert "ece_improvement" in res
