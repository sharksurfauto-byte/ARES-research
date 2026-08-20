"""Isotonic regression for reliability calibration (PRD §4.6).

Non-parametric calibration method that fits a monotonic function
to map predicted probabilities to observed frequencies.
"""

import numpy as np
from typing import Dict, Any, Optional
from sklearn.isotonic import IsotonicRegression


def fit_isotonic_regression(
    scores: np.ndarray,
    labels: np.ndarray,
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
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(scores, labels)
    return ir


def apply_isotonic_regression(
    scores: np.ndarray,
    ir_model: IsotonicRegression,
) -> np.ndarray:
    """Apply fitted isotonic regression to calibrate scores.

    Args:
        scores: [N] raw reliability scores R(x) in [0,1]
        ir_model: Fitted IsotonicRegression model

    Returns:
        [N] calibrated scores in [0,1]
    """
    return ir_model.predict(scores)


def compute_ece(
    probabilities: np.ndarray,
    labels: np.ndarray,
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
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for i in range(n_bins):
        in_bin = (probabilities > bin_lowers[i]) & (probabilities <= bin_uppers[i])
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            avg_confidence_in_bin = np.mean(probabilities[in_bin])
            accuracy_in_bin = np.mean(labels[in_bin])
            ece += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)

    return ece


def compute_brier_score(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Compute Brier score (proper scoring rule).

    Args:
        probabilities: [N] predicted probabilities
        labels: [N] binary correctness labels (0/1)

    Returns:
        Brier score (lower is better)
    """
    return np.mean((probabilities - labels) ** 2)


def before_after_calibration(
    raw_scores: np.ndarray,
    calibrated_scores: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, float]:
    """Compute ECE before and after calibration.

    Args:
        raw_scores: [N] raw reliability scores before calibration
        calibrated_scores: [N] calibrated scores after calibration
        labels: [N] binary correctness labels
        n_bins: Number of bins

    Returns:
        Dictionary with 'raw_ece' and 'calibrated_ece'
    """
    raw_ece = compute_ece(raw_scores, labels, n_bins)
    calibrated_ece = compute_ece(calibrated_scores, labels, n_bins)

    return {
        "raw_ece": raw_ece,
        "calibrated_ece": calibrated_ece,
        "ece_improvement": raw_ece - calibrated_ece,
    }