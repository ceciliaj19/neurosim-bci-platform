"""NeuroLab — Settings page.

Preferences are stored in st.session_state under cfg_* keys.
Pages read these keys as default values for their controls.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from components.page_header import render_page_header
from components.ui import section_header

# ---------------------------------------------------------------------------
# Defaults — single source of truth for fallback values used across all pages
# ---------------------------------------------------------------------------

DEFAULTS: dict[str, object] = {
    "cfg_neuron_model":   "LIF",
    "cfg_sim_duration":   500.0,
    "cfg_eeg_trial":      0,
    "cfg_eeg_channels":   5,
    "cfg_plot_theme":     "Default",
}

_NEURON_MODELS = ["LIF", "Izhikevich"]
_PLOT_THEMES   = ["Default", "Dark"]


def _init_defaults() -> None:
    """Write any missing cfg_* keys with their default values."""
    for key, val in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = val


def render() -> None:
    _init_defaults()

    render_page_header(
        "⚙️", "Settings",
        "Global defaults applied across NeuroLab pages",
    )

    st.info(
        "Settings are stored for the current browser session. "
        "They take effect the next time you visit each page or reload a sidebar control.",
        icon="ℹ️",
    )

    # ── NeuroSim ─────────────────────────────────────────────────────────────
    section_header("NeuroSim", icon="🔬")

    ns_model = st.selectbox(
        "Default neuron model",
        options=_NEURON_MODELS,
        index=_NEURON_MODELS.index(st.session_state["cfg_neuron_model"]),
        help="Pre-selects which model section is shown first on the NeuroSim page.",
        key="_cfg_neuron_model_sel",
    )
    st.caption("LIF = Leaky Integrate-and-Fire · Izhikevich = two-variable spiking model")

    ns_duration = st.slider(
        "Default simulation duration (ms)",
        min_value=100.0,
        max_value=1000.0,
        value=float(st.session_state["cfg_sim_duration"]),
        step=50.0,
        help="Sets the initial value of the Simulation Time slider on NeuroSim.",
        key="_cfg_sim_duration_sl",
    )

    # ── EEG Studio & Decoder Lab ─────────────────────────────────────────────
    st.divider()
    section_header("EEG Studio & Decoder Lab", icon="📡")

    eeg_trial = st.number_input(
        "Default EEG trial index",
        min_value=0,
        max_value=9999,
        value=int(st.session_state["cfg_eeg_trial"]),
        step=1,
        help="Trial index pre-selected in EEG Studio and Decoder Lab. "
             "Clamped to the dataset size at runtime.",
        key="_cfg_eeg_trial_ni",
    )

    eeg_channels = st.slider(
        "Default EEG channels displayed",
        min_value=1,
        max_value=20,
        value=int(st.session_state["cfg_eeg_channels"]),
        step=1,
        help="Number of EEG traces shown by default in EEG Studio and Decoder Lab.",
        key="_cfg_eeg_channels_sl",
    )

    # ── Appearance ───────────────────────────────────────────────────────────
    st.divider()
    section_header("Appearance", icon="🎨")

    plot_theme = st.selectbox(
        "Plot theme",
        options=_PLOT_THEMES,
        index=_PLOT_THEMES.index(st.session_state["cfg_plot_theme"]),
        help="Applies to Plotly charts across NeuroSim and EEG Studio.",
        key="_cfg_plot_theme_sel",
    )

    if plot_theme == "Dark":
        st.caption(
            "Dark theme uses a dark plot area and light axis text. "
            "Best with Streamlit's built-in dark mode (Settings → Theme → Dark)."
        )

    # ── Action buttons ───────────────────────────────────────────────────────
    st.divider()
    col_apply, col_reset, _ = st.columns([1, 1, 3])

    with col_apply:
        if st.button("✓ Apply", type="primary", use_container_width=True, key="cfg_apply"):
            st.session_state["cfg_neuron_model"] = ns_model
            st.session_state["cfg_sim_duration"] = float(ns_duration)
            st.session_state["cfg_eeg_trial"]    = int(eeg_trial)
            st.session_state["cfg_eeg_channels"] = int(eeg_channels)
            st.session_state["cfg_plot_theme"]   = plot_theme
            st.success("Settings applied. Navigate to a page to see the updated defaults.", icon="✅")

    with col_reset:
        if st.button("↺ Reset", use_container_width=True, key="cfg_reset"):
            for key, val in DEFAULTS.items():
                st.session_state[key] = val
            st.rerun()

    # ── Current values preview ───────────────────────────────────────────────
    st.divider()
    section_header("Current stored values")

    preview = {
        "Neuron model":          st.session_state["cfg_neuron_model"],
        "Simulation duration":   f"{st.session_state['cfg_sim_duration']:.0f} ms",
        "EEG trial index":       st.session_state["cfg_eeg_trial"],
        "EEG channels":          st.session_state["cfg_eeg_channels"],
        "Plot theme":            st.session_state["cfg_plot_theme"],
    }
    for label, val in preview.items():
        c1, c2 = st.columns([2, 3])
        c1.markdown(f"<span style='opacity:0.55;font-size:13px;'>{label}</span>",
                    unsafe_allow_html=True)
        c2.markdown(f"<span style='font-size:13px;font-weight:600;'>{val}</span>",
                    unsafe_allow_html=True)


render()
