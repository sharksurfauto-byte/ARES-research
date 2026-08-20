"""Calibration module (PRD §4.6).

Provides temperature scaling and isotonic regression for reliability calibration.
"""

from .temperature import TemperatureScaling, fit_temperature_scaling
from .isotonic import fit_isotonic_regression, apply_isotonic_regression, compute_ece, compute_brier_score, before_after_calibration

__all__ = [
    "TemperatureScaling",
    "fit_temperature_scaling",
    "fit_isotonic_regression",
    "apply_isotonic_regression",
    "compute_ece",
    "compute_brier_score",
    "before_after_calibration",
]