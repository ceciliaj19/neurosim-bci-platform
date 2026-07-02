"""NeuroLab — EEG Studio page: raw EEG trial visualisation."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.signal import welch

from components.eeg_chart import render_eeg_chart
from components.page_header import render_page_header
from neurosim.datasets import DatasetLoader

_LABEL_NAMES = {0: "Left-hand imagery", 1: "Right-hand imagery"}
_LABEL_COLORS = {0: "rgba(25,103,210,0.12)", 1: "rgba(197,34,31,0.12)"}
_LABEL_TEXT_COLORS = {0: "#1967d2", 1: "#c5221f"}


@st.cache_data
def _load_data() -> tuple[np.ndarray, np.ndarray]:
    return DatasetLoader().load_motor_imagery()


def render() -> None:
    render_page_header(
        "📡", "EEG Studio",
        "Explore raw 64-channel motor imagery EEG trials",
    )

    X, y = _load_data()

    with st.sidebar:
        st.markdown("**Dataset**")
        c1, c2 = st.columns(2)
        c1.metric("Trials", X.shape[0])
        c2.metric("Channels", X.shape[1])

        st.divider()
        st.markdown("**Trial Selection**")
        trial_idx = st.slider(
            "Trial index",
            min_value=0,
            max_value=X.shape[0] - 1,
            value=0,
            step=1,
        )
        num_channels = st.slider(
            "Channels to display",
            min_value=1,
            max_value=min(20, X.shape[1]),
            value=5,
            step=1,
        )

    trial = X[trial_idx]
    true_label = int(y[trial_idx])
    label_name = _LABEL_NAMES.get(true_label, str(true_label))
    bg = _LABEL_COLORS.get(true_label, "rgba(100,100,100,0.1)")
    fg = _LABEL_TEXT_COLORS.get(true_label, "#555")

    # Trial info row
    info_col, _, _ = st.columns([2, 1, 1])
    with info_col:
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:4px;">'
            f'<span style="font-size:15px;opacity:0.5;">Trial {trial_idx}</span>'
            f'<span style="background:{bg};color:{fg};padding:3px 12px;'
            f'border-radius:20px;font-size:13px;font-weight:500;">'
            f'Label {true_label} · {label_name}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    render_eeg_chart(trial, num_channels, trial_idx, true_label)

    st.divider()
    st.markdown("#### About this view")
    st.markdown(
        "Each trace is one EEG channel offset vertically so multiple signals "
        "can be compared at a glance.  \n"
        "**Label 0** = left-hand motor imagery · **Label 1** = right-hand motor imagery."
    )

    # ── Power Spectral Density ───────────────────────────────────────────────
    st.divider()
    st.markdown("#### Power Spectral Density")
    st.markdown(
        "The **Power Spectral Density (PSD)** shows how signal energy is "
        "distributed across frequencies for a single EEG channel. "
        "In motor imagery BCI, the most important features are in the "
        "**alpha (8–13 Hz)** and **beta (13–30 Hz)** bands: imagining a "
        "hand movement suppresses power in these bands over the contralateral "
        "motor cortex — a phenomenon called Event-Related Desynchronisation (ERD). "
        "The PSD is estimated using Welch's method (overlapping Hann-windowed segments)."
    )

    # Sampling frequency: PhysioNet EEGMMI dataset is recorded at 160 Hz
    _FS = 160.0
    _MAX_HZ = 40.0

    n_channels = trial.shape[0]
    psd_channel = st.selectbox(
        "Channel for PSD",
        options=list(range(n_channels)),
        format_func=lambda i: f"Channel {i}",
        index=0,
        key="psd_channel",
    )

    signal = trial[psd_channel]
    freqs, power = welch(
        signal,
        fs=_FS,
        nperseg=min(256, len(signal)),
    )

    # Restrict to 0–40 Hz
    mask = freqs <= _MAX_HZ
    freqs_plot = freqs[mask]
    power_plot = power[mask]

    # Frequency band definitions: (label, f_low, f_high, color)
    _BANDS = [
        ("Delta\n0.5–4 Hz",  0.5,  4.0,  "rgba(99,102,241,0.08)"),
        ("Theta\n4–8 Hz",    4.0,  8.0,  "rgba(16,185,129,0.08)"),
        ("Alpha\n8–13 Hz",   8.0,  13.0, "rgba(245,158,11,0.10)"),
        ("Beta\n13–30 Hz",   13.0, 30.0, "rgba(239,68,68,0.08)"),
    ]

    psd_fig = go.Figure()

    # Band shading (drawn first so trace appears on top)
    for band_label, f_lo, f_hi, band_color in _BANDS:
        psd_fig.add_vrect(
            x0=f_lo, x1=f_hi,
            fillcolor=band_color,
            layer="below",
            line_width=0,
            annotation_text=band_label,
            annotation_position="top left",
            annotation_font_size=9,
            annotation_font_color="rgba(0,0,0,0.35)",
        )

    psd_fig.add_trace(go.Scatter(
        x=freqs_plot,
        y=power_plot,
        mode="lines",
        name=f"Channel {psd_channel}",
        line=dict(color="#6366f1", width=1.8),
        fill="tozeroy",
        fillcolor="rgba(99,102,241,0.06)",
    ))

    psd_fig.update_layout(
        title=f"PSD — Trial {trial_idx}, Channel {psd_channel} "
              f"({label_name})",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Power Spectral Density (μV²/Hz)",
        xaxis=dict(range=[0, _MAX_HZ]),
        height=380,
        hovermode="x unified",
        plot_bgcolor="#f9f9f9",
        showlegend=False,
        margin=dict(l=60, r=40, t=50, b=50),
    )
    st.plotly_chart(psd_fig, use_container_width=True)


render()
