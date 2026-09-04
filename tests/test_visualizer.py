"""Unit Tests for ARES Web Visualizer Module (PRD §5, §7.4)."""

import pytest
import numpy as np

from ares.visualizer.presets import PRESET_PROMPTS, get_presets_by_domain
from ares.visualizer.plot_utils import (
    HAS_PLOTLY,
    create_reliability_gauge,
    create_uncertainty_gauge,
    create_router_distribution_chart,
    create_domain_radar_chart,
    create_pareto_frontier_chart,
    create_calibration_diagram,
    render_token_risk_heatmap,
)
from ares.visualizer.live_runner import VisualizerRunner, VisualizerExecutionResult


def test_preset_prompts():
    """Verify that preset prompts cover all 5 ARES domains."""
    assert len(PRESET_PROMPTS) >= 5
    domains = {p.domain for p in PRESET_PROMPTS}
    expected_domains = {"math", "code", "science", "reasoning", "general"}
    assert expected_domains.issubset(domains)

    # Test domain filtering
    math_presets = get_presets_by_domain("math")
    assert len(math_presets) >= 1
    assert all(p.domain == "math" for p in math_presets)

    all_presets = get_presets_by_domain("all")
    assert len(all_presets) == len(PRESET_PROMPTS)


def test_plot_gauges():
    """Verify radial gauge generator functions."""
    fig_rel = create_reliability_gauge(0.75, threshold=0.5)
    fig_unc = create_uncertainty_gauge(0.35, 0.20)
    if HAS_PLOTLY:
        assert fig_rel is not None
        assert fig_unc is not None
    else:
        assert fig_rel is None
        assert fig_unc is None


def test_router_distribution_chart():
    """Verify router horizontal probability distribution plot."""
    probs = {"BASE": 0.1, "math": 0.6, "code": 0.1, "science": 0.1, "reasoning": 0.05, "general": 0.05}
    fig = create_router_distribution_chart(probs, selected_route="math")
    if HAS_PLOTLY:
        assert fig is not None
    else:
        assert fig is None


def test_radar_and_pareto_charts():
    """Verify domain radar and Pareto efficiency curves."""
    strategies_acc = {
        "BASE": {"code": 100, "general": 100, "math": 4, "reasoning": 54, "science": 48},
        "DYNAMIC_ARES": {"code": 100, "general": 100, "math": 4, "reasoning": 54, "science": 48},
    }
    radar_fig = create_domain_radar_chart(strategies_acc)
    pareto_data = [
        {"name": "BASE", "invocation_rate": 0.0, "accuracy": 61.2, "latency_ms": 1200},
        {"name": "DYNAMIC_ARES", "invocation_rate": 80.8, "accuracy": 61.2, "latency_ms": 1200},
    ]
    pareto_fig = create_pareto_frontier_chart(pareto_data)
    if HAS_PLOTLY:
        assert radar_fig is not None
        assert pareto_fig is not None
    else:
        assert radar_fig is None
        assert pareto_fig is None


def test_calibration_diagram_and_token_heatmap():
    """Verify calibration reliability curve and HTML token heatmap."""
    confs = np.linspace(0.1, 0.9, 9)
    accs = confs * 0.9
    calib_fig = create_calibration_diagram(confs, accs, confs, ece=0.1911)
    if HAS_PLOTLY:
        assert calib_fig is not None
    else:
        assert calib_fig is None

    tokens = ["def", "solve", "(", "x", ")", ":"]
    risks = [0.1, 0.2, 0.05, 0.8, 0.1, 0.05]
    html = render_token_risk_heatmap(tokens, risks)
    assert "<span" in html
    assert "solve" in html


def test_visualizer_runner_mock_execution():
    """Verify VisualizerRunner simulation mode."""
    runner = VisualizerRunner(force_mock=True)
    assert not runner.is_live

    res = runner.run(
        prompt="Solve the following math problem:\nWhat is 5 + 7?\nAnswer:",
        strategy="dynamic",
    )
    assert isinstance(res, VisualizerExecutionResult)
    assert res.domain_prediction == "math"
    assert res.selected_route == "math"
    assert len(res.tokens) > 0
    assert len(res.token_risks) == len(res.tokens)
    assert 0.0 <= res.global_reliability <= 1.0
