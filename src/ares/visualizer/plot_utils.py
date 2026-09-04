"""Plotly Chart and Diagram Utilities for ARES Visualizer."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


# ─── 1. Dual Reliability & Uncertainty Gauges ────────────────────────────────

def create_reliability_gauge(
    reliability_score: float,
    title: str = "Global Reliability R(x)",
    threshold: float = 0.5,
) -> Any:
    """Create a sleek radial gauge for Global Reliability R(x)."""
    if not HAS_PLOTLY:
        return None

    pct = max(0.0, min(100.0, reliability_score * 100.0))
    bar_color = "#10B981" if reliability_score >= threshold else "#EF4444"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=pct,
            number={"suffix": "%", "font": {"size": 26, "color": "#F8FAFC"}},
            delta={"reference": threshold * 100, "increasing": {"color": "#10B981"}, "decreasing": {"color": "#EF4444"}},
            title={"text": f"<b>{title}</b><br><span style='font-size:0.8em;color:#94A3B8'>Threshold: {threshold:.0%}</span>", "font": {"size": 14, "color": "#E2E8F0"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#64748B"},
                "bar": {"color": bar_color, "thickness": 0.3},
                "bgcolor": "rgba(30, 41, 59, 0.5)",
                "borderwidth": 1,
                "bordercolor": "#334155",
                "steps": [
                    {"range": [0, threshold * 100], "color": "rgba(239, 68, 68, 0.2)"},
                    {"range": [threshold * 100, 100], "color": "rgba(16, 185, 129, 0.2)"},
                ],
                "threshold": {
                    "line": {"color": "#F59E0B", "width": 3},
                    "thickness": 0.8,
                    "value": threshold * 100,
                },
            },
        )
    )

    fig.update_layout(
        height=220,
        margin={"l": 20, "r": 20, "t": 40, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#E2E8F0"},
    )
    return fig


def create_uncertainty_gauge(
    uncertainty_score: float,
    failure_risk: float,
) -> Any:
    """Create a composite Uncertainty & Failure Risk Gauge."""
    if not HAS_PLOTLY:
        return None

    pct = max(0.0, min(100.0, uncertainty_score * 100.0))
    if uncertainty_score < 0.4:
        status = "🟢 Low Uncertainty (Safe)"
        bar_color = "#10B981"
    elif uncertainty_score < 0.7:
        status = "🟡 Moderate Uncertainty"
        bar_color = "#F59E0B"
    else:
        status = "🔴 High Uncertainty (Expert Required)"
        bar_color = "#EF4444"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct,
            number={"suffix": "%", "font": {"size": 26, "color": "#F8FAFC"}},
            title={"text": f"<b>Dual Uncertainty Score</b><br><span style='font-size:0.8em;color:#94A3B8'>{status}</span>", "font": {"size": 14, "color": "#E2E8F0"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#64748B"},
                "bar": {"color": bar_color, "thickness": 0.3},
                "bgcolor": "rgba(30, 41, 59, 0.5)",
                "steps": [
                    {"range": [0, 40], "color": "rgba(16, 185, 129, 0.2)"},
                    {"range": [40, 70], "color": "rgba(245, 158, 11, 0.2)"},
                    {"range": [70, 100], "color": "rgba(239, 68, 68, 0.2)"},
                ],
            },
        )
    )

    fig.update_layout(
        height=220,
        margin={"l": 20, "r": 20, "t": 40, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#E2E8F0"},
    )
    return fig


# ─── 2. Router Probability Distribution Bar ─────────────────────────────────

def create_router_distribution_chart(
    routing_probs: Dict[str, float],
    selected_route: str,
) -> Any:
    """Horizontal bar chart showing softmax router dispatch probabilities."""
    if not HAS_PLOTLY:
        return None

    routes = list(routing_probs.keys())
    probs = [routing_probs[r] * 100.0 for r in routes]

    # Color highlighting for selected route
    colors = [
        "#3B82F6" if r == selected_route else "#475569"
        for r in routes
    ]

    fig = go.Figure(
        go.Bar(
            x=probs,
            y=routes,
            orientation="h",
            marker={"color": colors, "line": {"color": "#60A5FA", "width": [2 if r == selected_route else 0 for r in routes]}},
            text=[f"{p:.1f}%" for p in probs],
            textposition="auto",
            textfont={"color": "#F8FAFC", "size": 12},
        )
    )

    fig.update_layout(
        title={"text": "<b>Router Dispatch Probabilities P(Expert | x)</b>", "font": {"size": 15, "color": "#F1F5F9"}},
        xaxis={"title": "Dispatch Probability (%)", "range": [0, 100], "gridcolor": "#334155", "zeroline": False},
        yaxis={"title": "Routing Target", "autorange": "reversed"},
        height=260,
        margin={"l": 20, "r": 20, "t": 40, "b": 30},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.4)",
        font={"color": "#CBD5E1"},
    )
    return fig


# ─── 3. Domain Radar Chart ───────────────────────────────────────────────────

def create_domain_radar_chart(
    strategies_acc: Dict[str, Dict[str, float]],
) -> Any:
    """Radar chart comparing accuracy profiles across all 5 benchmark domains."""
    if not HAS_PLOTLY:
        return None

    domains = ["Code", "General", "Math", "Reasoning", "Science"]
    fig = go.Figure()

    palette = {
        "BASE": "#94A3B8",
        "DYNAMIC_ARES": "#3B82F6",
        "THRESHOLD_ROUTER": "#10B981",
        "FIXED_EXPERT": "#F59E0B",
        "ORACLE_ROUTER": "#8B5CF6",
    }

    for strat, dom_dict in strategies_acc.items():
        vals = [dom_dict.get(d.lower(), dom_dict.get(d, 0.0)) for d in domains]
        vals.append(vals[0])  # Close the radar polygon
        closed_domains = domains + [domains[0]]

        color = palette.get(strat, "#E2E8F0")
        fig.add_trace(
            go.Scatterpolar(
                r=vals,
                theta=closed_domains,
                fill="toself" if strat in ["DYNAMIC_ARES", "BASE"] else "none",
                name=strat,
                line={"color": color, "width": 2.5 if strat == "DYNAMIC_ARES" else 1.5},
                opacity=0.3 if strat in ["DYNAMIC_ARES", "BASE"] else 1.0,
            )
        )

    fig.update_layout(
        polar={
            "radialaxis": {"visible": True, "range": [0, 100], "gridcolor": "#334155", "tickfont": {"size": 10}},
            "angularaxis": {"gridcolor": "#334155", "tickfont": {"size": 12, "color": "#F8FAFC"}},
            "bgcolor": "rgba(15, 23, 42, 0.5)",
        },
        showlegend=True,
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.2, "xanchor": "center", "x": 0.5},
        title={"text": "<b>Multi-Domain Accuracy Profile (Radar)</b>", "font": {"size": 16, "color": "#F8FAFC"}},
        height=380,
        margin={"l": 40, "r": 40, "t": 50, "b": 40},
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#CBD5E1"},
    )
    return fig


# ─── 4. Pareto Efficiency Curve ──────────────────────────────────────────────

def create_pareto_frontier_chart(
    strategy_metrics: List[Dict[str, Any]],
) -> Any:
    """Pareto Frontier: Compute Invocation Rate (%) vs. Overall Accuracy (%)."""
    if not HAS_PLOTLY:
        return None

    names = [m["name"] for m in strategy_metrics]
    invocations = [m["invocation_rate"] for m in strategy_metrics]
    accuracies = [m["accuracy"] for m in strategy_metrics]
    latencies = [m.get("latency_ms", 1200) for m in strategy_metrics]

    fig = go.Figure()

    # Scatter points
    fig.add_trace(
        go.Scatter(
            x=invocations,
            y=accuracies,
            mode="markers+text",
            text=names,
            textposition="top center",
            marker={
                "size": [18 if "ARES" in n else 14 for n in names],
                "color": ["#3B82F6" if "ARES" in n else "#10B981" if "THRESHOLD" in n else "#64748B" for n in names],
                "line": {"width": 2, "color": "#F8FAFC"},
            },
            hovertext=[f"{n}<br>Invocations: {inv:.1f}%<br>Accuracy: {acc:.1f}%<br>Latency: {lat:.1f}ms" for n, inv, acc, lat in zip(names, invocations, accuracies, latencies)],
            hoverinfo="text",
        )
    )

    # Reference efficiency zone
    fig.add_vrect(
        x0=0, x1=50,
        fillcolor="rgba(16, 185, 129, 0.1)",
        layer="below",
        line_width=0,
        annotation_text="High Compute Savings Zone",
        annotation_position="top left",
        annotation_font={"size": 10, "color": "#10B981"},
    )

    fig.update_layout(
        title={"text": "<b>Compute Efficiency Pareto Frontier (Invocations vs. Accuracy)</b>", "font": {"size": 15, "color": "#F8FAFC"}},
        xaxis={"title": "Expert Invocation Rate (%) — [Lower is More Efficient]", "range": [-5, 110], "gridcolor": "#334155"},
        yaxis={"title": "Overall Accuracy (%) — [Higher is Better]", "range": [0, 100], "gridcolor": "#334155"},
        height=350,
        margin={"l": 40, "r": 30, "t": 50, "b": 40},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.4)",
        font={"color": "#CBD5E1"},
    )
    return fig


# ─── 5. Calibration Reliability Diagram (ECE) ────────────────────────────────

def create_calibration_diagram(
    bins: np.ndarray,
    accs: np.ndarray,
    confs: np.ndarray,
    ece: float = 0.1627,
) -> Any:
    """Reliability curve plotting expected vs empirical confidence bins."""
    if not HAS_PLOTLY:
        return None

    fig = go.Figure()

    # Ideal calibration line (y = x)
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line={"dash": "dash", "color": "#94A3B8", "width": 2},
            name="Perfect Calibration (y = x)",
        )
    )

    # Empirical binned confidence vs accuracy
    fig.add_trace(
        go.Bar(
            x=confs,
            y=accs,
            name="Empirical Accuracy",
            marker={"color": "rgba(59, 130, 246, 0.7)", "line": {"color": "#60A5FA", "width": 1.5}},
            opacity=0.8,
        )
    )

    fig.update_layout(
        title={"text": f"<b>Reliability Diagram (ECE = {ece:.4f})</b>", "font": {"size": 15, "color": "#F8FAFC"}},
        xaxis={"title": "Confidence Bin", "range": [0, 1], "gridcolor": "#334155"},
        yaxis={"title": "Accuracy in Bin", "range": [0, 1], "gridcolor": "#334155"},
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.25, "xanchor": "center", "x": 0.5},
        height=320,
        margin={"l": 40, "r": 20, "t": 40, "b": 40},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.4)",
        font={"color": "#CBD5E1"},
    )
    return fig


# ─── 6. Token Risk Heatmap HTML Generator ────────────────────────────────────

def render_token_risk_heatmap(
    tokens: List[str],
    risk_scores: List[float],
) -> str:
    """Generate HTML text with per-token failure risk color highlighting."""
    html_spans = []
    for tok, r in zip(tokens, risk_scores):
        r_clamped = max(0.0, min(1.0, float(r)))
        # Color interpolation from green (0) to yellow (0.5) to red (1.0)
        if r_clamped < 0.5:
            alpha = r_clamped * 2.0
            bg = f"rgba(16, 185, 129, {0.15 + 0.35 * (1 - alpha)})"
        else:
            alpha = (r_clamped - 0.5) * 2.0
            bg = f"rgba(239, 68, 68, {0.2 + 0.5 * alpha})"

        safe_tok = tok.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
        html_spans.append(
            f"<span style='background-color:{bg}; padding: 2px 4px; margin: 1px; border-radius: 4px; font-family: monospace; font-size: 13px;' title='Failure Risk: {r_clamped:.1%}'>{safe_tok}</span>"
        )

    return f"<div style='background-color: #0F172A; padding: 14px; border-radius: 8px; border: 1px solid #334155; line-height: 1.8;'>{''.join(html_spans)}</div>"
