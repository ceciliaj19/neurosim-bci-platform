"""Reusable cursor trajectory chart component."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from components.ui import COLORS, PLOTLY_LAYOUT


def render_cursor_chart(
    trajectory: list[tuple[float, float]],
    title: str = "Virtual Cursor Position",
    axis_range: float = 15.0,
) -> None:
    """Render the cursor trajectory as a Plotly scatter chart.

    Parameters
    ----------
    trajectory : list[tuple[float, float]]
        Ordered list of ``(x, y)`` positions including the starting point.
    title : str, optional
        Chart title. Default ``"Virtual Cursor Position"``.
    axis_range : float, optional
        Symmetric axis limit (``[-axis_range, axis_range]``). Default ``15.0``.
    """
    x_vals = [p[0] for p in trajectory]
    y_vals = [p[1] for p in trajectory]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode="lines+markers",
        name="Trajectory",
        line=dict(color=COLORS["left_class"], width=2),
        marker=dict(size=6, color=COLORS["left_class"]),
    ))

    fig.add_trace(go.Scatter(
        x=[x_vals[-1]],
        y=[y_vals[-1]],
        mode="markers",
        name="Current",
        marker=dict(size=14, color=COLORS["spike"], symbol="circle"),
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=title,
        xaxis_title="X position",
        yaxis_title="Y position",
        height=380,
        xaxis=dict(range=[-axis_range, axis_range]),
        yaxis=dict(range=[-axis_range, axis_range]),
        legend=dict(x=1, y=1, xanchor="right"),
    )

    st.plotly_chart(fig, use_container_width=True)
