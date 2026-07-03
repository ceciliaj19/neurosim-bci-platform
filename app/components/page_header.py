"""Reusable page header component for NeuroLab."""

import streamlit as st


def render_page_header(icon: str, title: str, subtitle: str) -> None:
    """Render a consistent branded page header followed by a divider.

    Parameters
    ----------
    icon : str
        Emoji or single character shown prominently left of the title.
    title : str
        Page title rendered at heading weight.
    subtitle : str
        One-line description shown in a muted style below the title.
    """
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:16px; padding:8px 0 4px;">
            <span style="font-size:40px; line-height:1;">{icon}</span>
            <div>
                <div style="font-size:28px; font-weight:700; line-height:1.2;">{title}</div>
                <div style="font-size:15px; opacity:0.55; margin-top:4px;">{subtitle}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
