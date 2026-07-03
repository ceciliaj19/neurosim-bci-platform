"""NeuroLab — EEG Studio page: raw EEG trial visualisation."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.signal import spectrogram, welch

from components.eeg_chart import render_eeg_chart
from components.exporter import download_buttons
from components.page_header import render_page_header
from components.ui import PLOTLY_LAYOUT, section_header
from neurosim.datasets import DatasetLoader

_LABEL_NAMES = {0: "Left-hand imagery", 1: "Right-hand imagery"}
_LABEL_COLORS = {0: "rgba(25,103,210,0.12)", 1: "rgba(197,34,31,0.12)"}
_LABEL_TEXT_COLORS = {0: "#1967d2", 1: "#c5221f"}

# Sampling frequency: PhysioNet EEGMMI dataset is recorded at 160 Hz
_FS = 160.0
_MAX_HZ = 40.0

# Canonical EEG frequency bands: (display_label, f_low, f_high, fill_color, bar_color, description)
_BANDS = [
    ("Delta",  0.5,  4.0,  "rgba(99,102,241,0.08)",  "#6366f1",
     "Deep sleep, high-amplitude slow waves; typically suppressed during active cognition."),
    ("Theta",  4.0,  8.0,  "rgba(16,185,129,0.08)",  "#10b981",
     "Drowsiness, memory encoding, and navigation; elevated during focused internal tasks."),
    ("Alpha",  8.0,  13.0, "rgba(245,158,11,0.10)",  "#f59e0b",
     "Relaxed wakefulness and cortical idling. Suppressed over motor cortex during motor imagery (ERD)."),
    ("Beta",   13.0, 30.0, "rgba(239,68,68,0.08)",   "#ef4444",
     "Active concentration, motor planning, and sensorimotor processing. Key band for BCI decoding."),
    ("Gamma",  30.0, 40.0, "rgba(168,85,247,0.08)",  "#a855f7",
     "High-level sensory binding and attention. Relevant range here limited to 30–40 Hz."),
]


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
    section_header("About This View")
    st.markdown(
        "Each trace is one EEG channel offset vertically so multiple signals "
        "can be compared at a glance.  \n"
        "**Label 0** = left-hand motor imagery · **Label 1** = right-hand motor imagery."
    )

    # ── Power Spectral Density ───────────────────────────────────────────────
    st.divider()
    section_header("Power Spectral Density")
    st.markdown(
        "The **Power Spectral Density (PSD)** shows how signal energy is "
        "distributed across frequencies for a single EEG channel. "
        "In motor imagery BCI, the most important features are in the "
        "**alpha (8–13 Hz)** and **beta (13–30 Hz)** bands: imagining a "
        "hand movement suppresses power in these bands over the contralateral "
        "motor cortex — a phenomenon called Event-Related Desynchronisation (ERD). "
        "The PSD is estimated using Welch's method (overlapping Hann-windowed segments)."
    )

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

    psd_fig = go.Figure()

    # Band shading (drawn first so trace appears on top)
    for name, f_lo, f_hi, fill_color, _, _desc in _BANDS:
        psd_fig.add_vrect(
            x0=f_lo, x1=f_hi,
            fillcolor=fill_color,
            layer="below",
            line_width=0,
            annotation_text=f"{name}\n{f_lo}–{f_hi} Hz",
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
        **PLOTLY_LAYOUT,
        title=f"PSD — Trial {trial_idx}, Channel {psd_channel} "
              f"({label_name})",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Power Spectral Density (μV²/Hz)",
        xaxis=dict(range=[0, _MAX_HZ]),
        height=380,
        hovermode="x unified",
        showlegend=False,
    )
    st.plotly_chart(psd_fig, use_container_width=True)
    download_buttons(
        "Power Spectral Density",
        f"psd_trial{trial_idx}_ch{psd_channel}",
        df=pd.DataFrame({"frequency_hz": freqs_plot, "power_uv2_per_hz": power_plot}),
        extra_meta={"trial_index": trial_idx, "channel": psd_channel,
                    "label": label_name, "fs_hz": _FS},
    )

    # ── Spectrogram ──────────────────────────────────────────────────────────
    st.divider()
    section_header("Spectrogram")
    st.markdown(
        "A **spectrogram** shows how the frequency content of a signal changes "
        "over time. Unlike the PSD (which averages across the whole trial), the "
        "spectrogram reveals *when* specific frequencies are active. "
        "In motor imagery EEG, the spectrogram can show transient suppression of "
        "alpha and beta power — Event-Related Desynchronisation (ERD) — during "
        "the imagery period. Power is shown on a logarithmic (dB) scale to reveal "
        "structure across a wide dynamic range."
    )

    spec_channel = st.selectbox(
        "Channel for spectrogram",
        options=list(range(n_channels)),
        format_func=lambda i: f"Channel {i}",
        index=0,
        key="spec_channel",
    )

    spec_signal = trial[spec_channel]
    f, t, Sxx = spectrogram(
        spec_signal,
        fs=_FS,
        nperseg=64,
        noverlap=48,
    )

    # Restrict to 0–40 Hz and convert to dB
    freq_mask = f <= _MAX_HZ
    f_plot = f[freq_mask]
    Sxx_db = 10.0 * np.log10(Sxx[freq_mask, :] + 1e-10)

    spec_fig = go.Figure(go.Heatmap(
        x=t,
        y=f_plot,
        z=Sxx_db,
        colorscale="Viridis",
        colorbar=dict(title="dB", thickness=14, len=0.85),
        hovertemplate="Time: %{x:.2f} s<br>Freq: %{y:.1f} Hz<br>Power: %{z:.1f} dB<extra></extra>",
    ))

    spec_fig.update_layout(
        **PLOTLY_LAYOUT,
        title=f"Spectrogram — Trial {trial_idx}, Channel {spec_channel} "
              f"({label_name})",
        xaxis_title="Time (s)",
        yaxis_title="Frequency (Hz)",
        yaxis=dict(range=[0, _MAX_HZ]),
        height=380,
    )
    st.plotly_chart(spec_fig, use_container_width=True)

    # ── Frequency Band Power ─────────────────────────────────────────────────
    st.divider()
    section_header("Frequency Band Power")
    st.markdown(
        "Band power summarises the PSD into five physiologically meaningful "
        "ranges. Power in each band is computed by integrating the PSD "
        "(trapezoidal rule) over the band's frequency limits. "
        "For motor imagery BCI, the **alpha** and **beta** bands are the most "
        "discriminative: left-hand imagery suppresses alpha/beta power over "
        "the right motor cortex, and vice versa."
    )

    bp_channel = st.selectbox(
        "Channel for band power",
        options=list(range(n_channels)),
        format_func=lambda i: f"Channel {i}",
        index=0,
        key="bp_channel",
    )

    bp_signal = trial[bp_channel]
    bp_freqs, bp_power = welch(bp_signal, fs=_FS, nperseg=min(256, len(bp_signal)))

    band_names: list[str] = []
    band_powers: list[float] = []
    band_ranges: list[str] = []
    band_bar_colors: list[str] = []
    band_descriptions: list[str] = []

    for name, f_lo, f_hi, _fill, bar_color, desc in _BANDS:
        mask = (bp_freqs >= f_lo) & (bp_freqs <= f_hi)
        power_val = float(np.trapezoid(bp_power[mask], bp_freqs[mask])) if mask.sum() > 1 else 0.0
        band_names.append(name)
        band_powers.append(power_val)
        band_ranges.append(f"{f_lo}–{f_hi} Hz")
        band_bar_colors.append(bar_color)
        band_descriptions.append(desc)

    bp_fig = go.Figure(go.Bar(
        x=band_names,
        y=band_powers,
        marker_color=band_bar_colors,
        text=[f"{p:.2e}" for p in band_powers],
        textposition="outside",
        hovertemplate="%{x}<br>Power: %{y:.4e} μV²<extra></extra>",
    ))
    bp_fig.update_layout(
        **PLOTLY_LAYOUT,
        title=f"Band Power — Trial {trial_idx}, Channel {bp_channel} ({label_name})",
        xaxis_title="Frequency Band",
        yaxis_title="Power (μV²)",
        height=360,
        showlegend=False,
    )
    st.plotly_chart(bp_fig, use_container_width=True)

    bp_df = pd.DataFrame({
        "Band": band_names,
        "Freq. Range": band_ranges,
        "Power (μV²)": [f"{p:.4e}" for p in band_powers],
        "Description": band_descriptions,
    })
    st.dataframe(bp_df, use_container_width=True, hide_index=True)
    download_buttons(
        "Frequency Band Power",
        f"band_power_trial{trial_idx}_ch{bp_channel}",
        df=pd.DataFrame({"band": band_names, "freq_range_hz": band_ranges,
                         "power_uv2": band_powers}),
        extra_meta={"trial_index": trial_idx, "channel": bp_channel,
                    "label": label_name, "fs_hz": _FS},
    )


render()
