"""ARES Interactive Web Visualizer Dashboard Package."""

from .live_runner import VisualizerRunner, VisualizerExecutionResult
from .presets import PRESET_PROMPTS, PresetPrompt, get_presets_by_domain
from .plot_utils import (
    create_reliability_gauge,
    create_uncertainty_gauge,
    create_router_distribution_chart,
    create_domain_radar_chart,
    create_pareto_frontier_chart,
    create_calibration_diagram,
    render_token_risk_heatmap,
)

__all__ = [
    "VisualizerRunner",
    "VisualizerExecutionResult",
    "PRESET_PROMPTS",
    "PresetPrompt",
    "get_presets_by_domain",
    "create_reliability_gauge",
    "create_uncertainty_gauge",
    "create_router_distribution_chart",
    "create_domain_radar_chart",
    "create_pareto_frontier_chart",
    "create_calibration_diagram",
    "render_token_risk_heatmap",
]
