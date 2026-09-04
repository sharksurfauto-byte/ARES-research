"""Reusable Styled UI Components and Header Cards for ARES Visualizer."""

from __future__ import annotations

from typing import Dict, List, Optional
import streamlit as st


def render_header():
    """Render the top banner and title for the ARES Visualizer."""
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%); padding: 24px; border-radius: 12px; border: 1px solid #334155; margin-bottom: 24px;">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <div>
                    <h1 style="color: #F8FAFC; margin: 0; font-size: 2.2rem; font-weight: 800; letter-spacing: -0.5px;">
                        ⚡ ARES <span style="color: #38BDF8; font-weight: 400; font-size: 1.3rem;">| Autonomous Reliable Expert System</span>
                    </h1>
                    <p style="color: #94A3B8; margin: 8px 0 0 0; font-size: 1.05rem;">
                        Dynamic Routing, Dual Reliability Estimation (GRM + LRM), and Multi-Domain PEFT Expert Specialization
                    </p>
                </div>
                <div style="text-align: right;">
                    <span style="background: rgba(56, 189, 248, 0.15); color: #38BDF8; padding: 6px 14px; border-radius: 20px; font-weight: 600; font-size: 0.85rem; border: 1px solid rgba(56, 189, 248, 0.3);">
                        Qwen2.5-0.5B Backbone
                    </span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_ribbon(
    global_reliability: float,
    local_risk: float,
    uncertainty: float,
    domain: str,
    selected_route: str,
    latency_ms: float,
):
    """Render top-level KPI metrics card ribbon."""
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.metric(
            label="Global Reliability R(x)",
            value=f"{global_reliability:.1%}",
            delta="Feasible" if global_reliability >= 0.5 else "Low Feasibility",
            delta_color="normal" if global_reliability >= 0.5 else "inverse",
        )
    with c2:
        st.metric(
            label="Local Failure Risk",
            value=f"{local_risk:.1%}",
            delta="High Risk" if local_risk >= 0.5 else "Low Risk",
            delta_color="inverse" if local_risk >= 0.5 else "normal",
        )
    with c3:
        st.metric(
            label="Dual Uncertainty",
            value=f"{uncertainty:.1%}",
            delta="Gated to Expert" if uncertainty >= 0.5 else "Base Confident",
            delta_color="off",
        )
    with c4:
        st.metric(
            label="Selected Route",
            value=selected_route.upper(),
            delta=f"Domain: {domain}",
            delta_color="off",
        )
    with c5:
        st.metric(
            label="Total Latency",
            value=f"{latency_ms:.0f} ms",
            delta="Sub-1.3s generation",
            delta_color="off",
        )


def render_route_badge(route_name: str, confidence: float) -> str:
    """Return HTML formatted pill badge for a selected route."""
    colors = {
        "BASE": ("#64748B", "rgba(100, 116, 139, 0.2)"),
        "math": ("#EC4899", "rgba(236, 72, 153, 0.2)"),
        "code": ("#10B981", "rgba(16, 185, 129, 0.2)"),
        "science": ("#3B82F6", "rgba(59, 130, 246, 0.2)"),
        "reasoning": ("#8B5CF6", "rgba(139, 92, 246, 0.2)"),
        "general": ("#F59E0B", "rgba(245, 158, 11, 0.2)"),
    }
    border, bg = colors.get(route_name.lower(), ("#38BDF8", "rgba(56, 189, 248, 0.2)"))
    return f"""
    <span style="background-color: {bg}; color: #F8FAFC; border: 1px solid {border}; padding: 4px 12px; border-radius: 16px; font-weight: 700; font-size: 0.9rem;">
        🎯 Route: {route_name.upper()} ({confidence:.1%})
    </span>
    """
