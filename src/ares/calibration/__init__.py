"""Calibration module (PRD §4.6).

Provides temperature scaling and isotonic regression for reliability calibration.
"""

from .isotonic import (
    apply_isotonic_regression,
    before_after_calibration,
    compute_brier_score,
    compute_ece,
    fit_isotonic_regression,
)
from .temperature import TemperatureScaling, fit_temperature_scaling

__all__ = [
    "TemperatureScaling",
    "fit_temperature_scaling",
    "fit_isotonic_regression",
    "apply_isotonic_regression",
    "compute_ece",
    "compute_brier_score",
    "before_after_calibration",
]
