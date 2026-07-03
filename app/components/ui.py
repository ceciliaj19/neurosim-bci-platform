"""NeuroLab — shared UI helpers and design tokens.

Import from any page or component:

    from components.ui import COLORS, PLOTLY_LAYOUT, section_header
"""

from __future__ import annotations

import streamlit as st

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------

COLORS: dict[str, str] = {
    # Brand
    "primary": "#6366f1",        # indigo — main accent, F-I curves, Izhikevich traces
    # EEG class colours (also used for BCI cues)
    "left_class": "#1967d2",     # left-hand imagery / blue traces
    "right_class": "#c5221f",    # right-hand imagery / red traces
    # Semantic
    "success": "#10b981",        # correct predictions, positive outcomes
    "warning": "#f59e0b",        # threshold lines, amber annotations
    "danger": "#ef4444",         # incorrect predictions
    "spike": "#d62728",          # spike markers (high-contrast red)
    # Chart
    "plot_bg": "#f9f9f9",        # Plotly plot area background
}

# System font stack — matches Streamlit's default UI font on most platforms
_FONT_FAMILY = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, sans-serif"

# Base Plotly layout applied to every chart via **PLOTLY_LAYOUT.
# Charts should spread this dict first, then add their own overrides.
#
#   fig.update_layout(**PLOTLY_LAYOUT, title="...", xaxis_title="...")
#
PLOTLY_LAYOUT: dict = {
    "plot_bgcolor": COLORS["plot_bg"],
    "paper_bgcolor": "rgba(0,0,0,0)",   # transparent — blends with Streamlit card bg
    "font": {"family": _FONT_FAMILY, "size": 12},
    "margin": {"l": 56, "r": 32, "t": 48, "b": 48},
}


# ---------------------------------------------------------------------------
# Section header
# ---------------------------------------------------------------------------

def section_header(title: str, icon: str = "") -> None:
    """Render a consistent section heading inside any NeuroLab page.

    Produces a single styled ``<div>`` — 15 px, semibold — that gives every
    section across all pages the same visual weight regardless of Streamlit's
    theme H4 rendering.

    Parameters
    ----------
    title : str
        Section title text.
    icon : str, optional
        Emoji or character prepended to the title with a small gap.

    Usage
    -----
    ::

        st.divider()
        section_header("F-I Curve Experiment")
        st.markdown("Sweep input current and measure steady-state firing rate.")
    """
    prefix = f'<span style="margin-right:7px;">{icon}</span>' if icon else ""
    st.markdown(
        f"<div style='"
        f"font-size:15px;"
        f"font-weight:600;"
        f"letter-spacing:-.01em;"
        f"line-height:1.35;"
        f"margin:4px 0 10px;"
        f"'>{prefix}{title}</div>",
        unsafe_allow_html=True,
    )
