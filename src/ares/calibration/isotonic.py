"""Isotonic regression for reliability calibration (PRD §4.6).

Non-parametric calibration method that fits a monotonic function
to map predicted probabilities to observed frequencies.
"""

from typing import Union
import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression


def _to_numpy(x: Union[np.ndarray, torch.Tensor, list]) -> np.ndarray:
    """Convert input to a flat float numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float64).reshape(-1)
    return np.asarray(x, dtype=np.float64).reshape(-1)


def fit_isotonic_regression(
    scores: Union[np.ndarray, torch.Tensor, list],
    labels: Union[np.ndarray, torch.Tensor, list],
) -> IsotonicRegression:
    """Fit isotonic regression model to calibrate reliability scores.

    Based on PRD §4.6: Fit isotonic regression on (R(x), correctness)
    for non-parametric calibration.

    Args:
        scores: [N] reliability scores R(x) in [0,1]
        labels: [N] binary correctness labels (1=correct, 0=incorrect)

    Returns:
        Fitted IsotonicRegression model
    """
    scores_np = _to_numpy(scores)
    labels_np = _to_numpy(labels)
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(scores_np, labels_np)
    return ir


def apply_isotonic_regression(
    scores: Union[np.ndarray, torch.Tensor, list],
    ir_model: IsotonicRegression,
) -> np.ndarray:
    """Apply fitted isotonic regression to calibrate scores.

    Args:
        scores: [N] raw reliability scores R(x) in [0,1]
        ir_model: Fitted IsotonicRegression model

    Returns:
        [N] calibrated scores in [0,1] as numpy array
    """
    scores_np = _to_numpy(scores)
    return np.clip(ir_model.predict(scores_np), 0.0, 1.0)


def compute_ece(
    probabilities: Union[np.ndarray, torch.Tensor, list],
    labels: Union[np.ndarray, torch.Tensor, list],
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE).

    Args:
        probabilities: [N] predicted probabilities/confidence scores
        labels: [N] binary correctness labels
        n_bins: Number of bins for computation

    Returns:
        ECE value (lower is better, 0 = perfectly calibrated)
    """
    probs_np = _to_numpy(probabilities)
    labels_np = _to_numpy(labels)

    if len(probs_np) == 0:
        return 0.0

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for i in range(n_bins):
        if i == 0:
            in_bin = (probs_np >= bin_lowers[i]) & (probs_np <= bin_uppers[i])
        else:
            in_bin = (probs_np > bin_lowers[i]) & (probs_np <= bin_uppers[i])
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            avg_confidence_in_bin = np.mean(probs_np[in_bin])
            accuracy_in_bin = np.mean(labels_np[in_bin])
            ece += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)

    return float(ece)


def compute_brier_score(
    probabilities: Union[np.ndarray, torch.Tensor, list],
    labels: Union[np.ndarray, torch.Tensor, list],
) -> float:
    """Compute Brier score (proper scoring rule).

    Args:
        probabilities: [N] predicted probabilities
        labels: [N] binary correctness labels (0/1)

    Returns:
        Brier score (lower is better)
    """
    probs_np = _to_numpy(probabilities)
    labels_np = _to_numpy(labels)
    if len(probs_np) == 0:
        return 0.0
    return float(np.mean((probs_np - labels_np) ** 2))


def before_after_calibration(
    raw_scores: Union[np.ndarray, torch.Tensor, list],
    calibrated_scores: Union[np.ndarray, torch.Tensor, list],
    labels: Union[np.ndarray, torch.Tensor, list],
    n_bins: int = 10,
) -> dict[str, float]:
    """Compute ECE before and after calibration.

    Args:
        raw_scores: [N] raw reliability scores before calibration
        calibrated_scores: [N] calibrated scores after calibration
        labels: [N] binary correctness labels
        n_bins: Number of bins

    Returns:
        Dictionary with 'raw_ece', 'calibrated_ece', and 'ece_improvement'
    """
    raw_ece = compute_ece(raw_scores, labels, n_bins)
    calibrated_ece = compute_ece(calibrated_scores, labels, n_bins)

    return {
        "raw_ece": raw_ece,
        "calibrated_ece": calibrated_ece,
        "ece_improvement": raw_ece - calibrated_ece,
    }
