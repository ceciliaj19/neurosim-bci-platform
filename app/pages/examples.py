"""NeuroLab — Example Experiments page.

Ten predefined experiments that run on demand and display results inline.
All simulation logic is delegated to existing NeuroLab modules — no
duplicate code.
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.signal import spectrogram, welch

from components.page_header import render_page_header
from components.ui import COLORS, PLOTLY_LAYOUT, section_header
from neurosim.analysis import compute_fi_curve, parameter_sweep
from neurosim.bci import CursorController
from neurosim.datasets import DatasetLoader
from neurosim.neurons import (
    IzhikevichNeuron,
    LIFNeuron,
    get_izhikevich_preset,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_FS = 160.0          # PhysioNet EEGMMI sampling rate (Hz)
_MAX_HZ = 40.0       # frequency ceiling for EEG plots

_FONT = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, sans-serif"


# ---------------------------------------------------------------------------
# Dataset loader (cached — shared with EEG Studio and Decoder Lab)
# ---------------------------------------------------------------------------

@st.cache_data
def _load_eeg() -> tuple[np.ndarray, np.ndarray] | None:
    try:
        return DatasetLoader().load_motor_imagery()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Example dataclass
# ---------------------------------------------------------------------------

@dataclass
class Example:
    id: str
    title: str
    icon: str
    description: str
    objective: str
    category: str
    run: Callable[[], go.Figure] = field(repr=False)


# ---------------------------------------------------------------------------
# Individual run functions — each returns a go.Figure
# ---------------------------------------------------------------------------

def _run_lif_current_sweep() -> go.Figure:
    """LIF membrane potential under default current (I=2.0)."""
    neuron = LIFNeuron()
    result = neuron.simulate(current=2.0, t_max=500.0, dt=0.1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=result.time, y=result.voltage,
        mode="lines", name="Membrane potential",
        line=dict(color=COLORS["left_class"], width=1.5),
    ))
    for t in result.spike_times:
        fig.add_vline(x=t, line_color=COLORS["spike"],
                      line_width=0.8, line_dash="dot", opacity=0.5)
    fig.add_hline(y=neuron.v_threshold, line_dash="dash",
                  line_color=COLORS["warning"], line_width=1.2,
                  annotation_text="Threshold",
                  annotation_font_size=10,
                  annotation_font_color=COLORS["warning"])
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=f"LIF Current Sweep — I=2.0, {len(result.spike_times)} spikes",
        xaxis_title="Time (ms)", yaxis_title="Voltage (mV)",
        height=380, hovermode="x unified",
    )
    return fig


def _run_lif_fi_curve() -> go.Figure:
    """F-I curve: firing rate vs. injected current for a LIF neuron."""
    neuron = LIFNeuron()
    currents = np.linspace(0.0, 5.0, 40)
    df = compute_fi_curve(neuron, currents, t_max=500.0, dt=0.1)

    fig = go.Figure(go.Scatter(
        x=df["current"], y=df["firing_rate"],
        mode="lines+markers",
        line=dict(color=COLORS["primary"], width=2),
        marker=dict(size=4, color=COLORS["primary"]),
        hovertemplate="I=%{x:.2f}<br>Rate=%{y:.1f} Hz<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="LIF F-I Curve — Firing Rate vs. Input Current",
        xaxis_title="Input Current (a.u.)", yaxis_title="Firing Rate (Hz)",
        height=360,
    )
    return fig


def _run_lif_parameter_sweep() -> go.Figure:
    """Parameter sweep: effect of membrane time constant τm on firing rate."""
    neuron = LIFNeuron(tau_m=20.0)
    tau_values = np.linspace(5.0, 50.0, 30)
    df = parameter_sweep(neuron, "tau_m", tau_values, current=2.0, t_max=500.0)

    fig = go.Figure(go.Scatter(
        x=df["parameter_value"], y=df["firing_rate"],
        mode="lines+markers",
        line=dict(color=COLORS["success"], width=2),
        marker=dict(size=4, color=COLORS["success"]),
        hovertemplate="τm=%{x:.1f} ms<br>Rate=%{y:.1f} Hz<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="LIF Parameter Sweep — τm vs. Firing Rate (I=2.0)",
        xaxis_title="Membrane Time Constant τm (ms)",
        yaxis_title="Firing Rate (Hz)",
        height=360,
    )
    return fig


def _run_izh_regular_spiking() -> go.Figure:
    """Izhikevich Regular Spiking — typical excitatory cortical pyramidal neuron."""
    preset = get_izhikevich_preset("Regular Spiking")
    neuron = IzhikevichNeuron(a=preset.a, b=preset.b, c=preset.c, d=preset.d)
    result = neuron.simulate(current=10.0, duration=500.0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=result.time, y=result.voltage,
        mode="lines", name="Membrane potential",
        line=dict(color=COLORS["left_class"], width=1.5),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=f"Izhikevich Regular Spiking — {len(result.spike_times)} spikes  |  "
              f"a={preset.a}, b={preset.b}, c={preset.c}, d={preset.d}",
        xaxis_title="Time (ms)", yaxis_title="Voltage (mV)",
        height=380, hovermode="x unified",
    )
    return fig


def _run_izh_fast_spiking() -> go.Figure:
    """Izhikevich Fast Spiking — inhibitory interneuron firing pattern."""
    preset = get_izhikevich_preset("Fast Spiking")
    neuron = IzhikevichNeuron(a=preset.a, b=preset.b, c=preset.c, d=preset.d)
    result = neuron.simulate(current=10.0, duration=500.0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=result.time, y=result.voltage,
        mode="lines", name="Membrane potential",
        line=dict(color=COLORS["right_class"], width=1.5),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=f"Izhikevich Fast Spiking — {len(result.spike_times)} spikes  |  "
              f"a={preset.a}, b={preset.b}, c={preset.c}, d={preset.d}",
        xaxis_title="Time (ms)", yaxis_title="Voltage (mV)",
        height=380, hovermode="x unified",
    )
    return fig


def _run_eeg_trial() -> go.Figure:
    """Multi-channel EEG trial waterfall plot (first 8 channels, trial 0)."""
    data = _load_eeg()
    if data is None:
        fig = go.Figure()
        fig.add_annotation(text="EEG dataset not available",
                           showarrow=False, font=dict(size=16))
        return fig

    X, y = data
    trial = X[0]
    n_ch = min(8, trial.shape[0])
    n_t = trial.shape[1]
    time = np.arange(n_t) / _FS * 1000  # → ms
    offset_scale = float(np.std(trial[:n_ch])) * 3.0

    label_name = "Left-hand imagery" if int(y[0]) == 0 else "Right-hand imagery"
    fig = go.Figure()
    for ch in range(n_ch):
        offset = (n_ch - ch - 1) * offset_scale
        fig.add_trace(go.Scatter(
            x=time,
            y=trial[ch] + offset,
            mode="lines",
            name=f"Ch {ch}",
            line=dict(width=0.9,
                      color=COLORS["left_class"] if int(y[0]) == 0
                      else COLORS["right_class"]),
            hovertemplate=f"Ch {ch} | t=%{{x:.0f}} ms | v=%{{y:.1f}}<extra></extra>",
        ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=f"EEG Trial 0 — {n_ch} channels  |  {label_name}",
        xaxis_title="Time (ms)", yaxis_title="Amplitude + offset",
        showlegend=False, height=420, hovermode="x unified",
    )
    return fig


def _run_eeg_psd() -> go.Figure:
    """Power Spectral Density for EEG trial 0, channel 0 (Welch method)."""
    data = _load_eeg()
    if data is None:
        fig = go.Figure()
        fig.add_annotation(text="EEG dataset not available",
                           showarrow=False, font=dict(size=16))
        return fig

    X, y = data
    signal = X[0, 0, :]
    freqs, power = welch(signal, fs=_FS, nperseg=min(256, len(signal)))
    mask = freqs <= _MAX_HZ

    _BANDS = [
        ("Delta", 0.5, 4.0, "rgba(99,102,241,0.08)"),
        ("Theta", 4.0, 8.0, "rgba(16,185,129,0.08)"),
        ("Alpha", 8.0, 13.0, "rgba(245,158,11,0.10)"),
        ("Beta", 13.0, 30.0, "rgba(239,68,68,0.08)"),
        ("Gamma", 30.0, 40.0, "rgba(168,85,247,0.08)"),
    ]

    fig = go.Figure()
    for name, f_lo, f_hi, fill in _BANDS:
        fig.add_vrect(x0=f_lo, x1=f_hi, fillcolor=fill, layer="below",
                      line_width=0, annotation_text=name,
                      annotation_position="top left",
                      annotation_font_size=9,
                      annotation_font_color="rgba(0,0,0,0.35)")
    fig.add_trace(go.Scatter(
        x=freqs[mask], y=power[mask],
        mode="lines", name="PSD",
        line=dict(color=COLORS["primary"], width=1.8),
        fill="tozeroy", fillcolor="rgba(99,102,241,0.06)",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="EEG Power Spectral Density — Trial 0, Channel 0",
        xaxis_title="Frequency (Hz)", yaxis_title="PSD (μV²/Hz)",
        xaxis=dict(range=[0, _MAX_HZ]),
        height=380, showlegend=False, hovermode="x unified",
    )
    return fig


def _run_eeg_spectrogram() -> go.Figure:
    """Time-frequency spectrogram for EEG trial 0, channel 0."""
    data = _load_eeg()
    if data is None:
        fig = go.Figure()
        fig.add_annotation(text="EEG dataset not available",
                           showarrow=False, font=dict(size=16))
        return fig

    X, _y = data
    signal = X[0, 0, :]
    f, t, Sxx = spectrogram(signal, fs=_FS, nperseg=64, noverlap=48)
    freq_mask = f <= _MAX_HZ
    Sxx_db = 10.0 * np.log10(Sxx[freq_mask, :] + 1e-10)

    fig = go.Figure(go.Heatmap(
        x=t, y=f[freq_mask], z=Sxx_db,
        colorscale="Viridis",
        colorbar=dict(title="dB", thickness=14, len=0.85),
        hovertemplate="t=%{x:.2f}s | f=%{y:.1f}Hz | %{z:.1f}dB<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="EEG Spectrogram — Trial 0, Channel 0",
        xaxis_title="Time (s)", yaxis_title="Frequency (Hz)",
        yaxis=dict(range=[0, _MAX_HZ]),
        height=380,
    )
    return fig


def _run_decoder_demo() -> go.Figure:
    """Mock decoder probability bar for a random EEG trial (no model needed)."""
    data = _load_eeg()
    if data is None:
        # Fully synthetic fallback
        true_label = 0
        probs = [0.72, 0.28]
    else:
        X, y = data
        rng = random.Random(7)
        idx = rng.randint(0, X.shape[0] - 1)
        true_label = int(y[idx])
        # Simulate realistic softmax output: correct class gets higher probability
        correct_prob = rng.uniform(0.60, 0.92)
        probs = ([correct_prob, 1.0 - correct_prob] if true_label == 0
                 else [1.0 - correct_prob, correct_prob])

    predicted = int(np.argmax(probs))
    labels = ["Class 0 (left)", "Class 1 (right)"]

    fig = go.Figure(go.Bar(
        x=labels,
        y=[p * 100 for p in probs],
        marker_color=[COLORS["left_class"], COLORS["right_class"]],
        text=[f"{p * 100:.1f}%" for p in probs],
        textposition="outside",
    ))
    correct = predicted == true_label
    outcome = "✓ Correct" if correct else "✗ Incorrect"
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=(f"Decoder Demo — True: Class {true_label}  |  "
               f"Predicted: Class {predicted}  |  {outcome}"),
        yaxis=dict(title="Probability (%)", range=[0, 115]),
        xaxis_title=None,
        height=340, showlegend=False,
    )
    return fig


def _run_closed_loop_demo() -> go.Figure:
    """Simulate a 12-trial closed-loop BCI session at 75 % decoder accuracy."""
    rng = random.Random(42)
    n_trials = 12
    accuracy = 0.75

    # Balanced cue list
    half = n_trials // 2
    cues = ["left"] * half + ["right"] * half
    rng.shuffle(cues)

    cursor = CursorController(step_size=1.0)
    traj: list[tuple[float, float]] = [(0.0, 0.0)]
    results: list[bool] = []

    for cue in cues:
        correct = rng.random() < accuracy
        command = cue if correct else ("right" if cue == "left" else "left")
        pos = cursor.move(command)
        traj.append(pos)
        results.append(correct)

    xs = [p[0] for p in traj]
    ys = [p[1] for p in traj]

    fig = go.Figure()
    # Trajectory line
    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        mode="lines+markers",
        line=dict(color=COLORS["left_class"], width=2),
        marker=dict(size=6, color=COLORS["left_class"]),
        name="Trajectory",
    ))
    # Start marker
    fig.add_trace(go.Scatter(
        x=[xs[0]], y=[ys[0]],
        mode="markers",
        marker=dict(size=14, color=COLORS["success"], symbol="circle",
                    line=dict(width=2, color="white")),
        name="Start",
    ))
    # End marker
    fig.add_trace(go.Scatter(
        x=[xs[-1]], y=[ys[-1]],
        mode="markers",
        marker=dict(size=14, color=COLORS["spike"], symbol="circle",
                    line=dict(width=2, color="white")),
        name="End",
    ))

    acc = sum(results) / len(results) * 100
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=(f"Closed-Loop Cursor Demo — {n_trials} trials  |  "
               f"Accuracy {acc:.0f}%  |  "
               f"Final pos ({xs[-1]:.1f}, {ys[-1]:.1f})"),
        xaxis_title="X position", yaxis_title="Y position",
        xaxis=dict(zeroline=True, zerolinewidth=1,
                   zerolinecolor="rgba(0,0,0,0.15)"),
        yaxis=dict(zeroline=True, zerolinewidth=1,
                   zerolinecolor="rgba(0,0,0,0.15)"),
        height=420, hovermode="closest",
    )
    return fig


# ---------------------------------------------------------------------------
# Example registry — ordered list
# ---------------------------------------------------------------------------

EXAMPLES: list[Example] = [
    Example(
        id="lif_current_sweep",
        title="LIF Current Sweep",
        icon="⚡",
        category="Neuron Models",
        description=(
            "Simulate a Leaky Integrate-and-Fire (LIF) neuron under a "
            "constant input current and visualise the membrane potential trace."
        ),
        objective=(
            "Observe how a constant supra-threshold current drives regular "
            "spiking, and identify the refractory period between spikes."
        ),
        run=_run_lif_current_sweep,
    ),
    Example(
        id="lif_fi_curve",
        title="LIF F-I Curve",
        icon="📈",
        category="Neuron Models",
        description=(
            "Sweep input current from 0 to 5 a.u. and measure the steady-state "
            "firing rate at each step to produce a frequency-current (F-I) curve."
        ),
        objective=(
            "Demonstrate the linear gain relationship between injected current "
            "and firing rate in the LIF model above rheobase."
        ),
        run=_run_lif_fi_curve,
    ),
    Example(
        id="lif_parameter_sweep",
        title="LIF Parameter Sweep",
        icon="🔧",
        category="Neuron Models",
        description=(
            "Vary the membrane time constant τm from 5 ms to 50 ms and "
            "measure the resulting change in firing rate at fixed I=2.0."
        ),
        objective=(
            "Understand how τm controls the integration window: faster "
            "membranes (small τm) integrate less and fire more slowly."
        ),
        run=_run_lif_parameter_sweep,
    ),
    Example(
        id="izh_regular_spiking",
        title="Izhikevich Regular Spiking",
        icon="🧠",
        category="Neuron Models",
        description=(
            "Simulate an Izhikevich neuron with Regular Spiking parameters "
            "(a=0.02, b=0.2, c=−65, d=8) — the canonical excitatory pyramidal "
            "cortical neuron."
        ),
        objective=(
            "Compare the richer spike-rate adaptation of the Izhikevich model "
            "with the idealized LIF trace, and observe initial burst followed "
            "by frequency adaptation."
        ),
        run=_run_izh_regular_spiking,
    ),
    Example(
        id="izh_fast_spiking",
        title="Izhikevich Fast Spiking",
        icon="⚡",
        category="Neuron Models",
        description=(
            "Simulate an Izhikevich neuron with Fast Spiking parameters "
            "(a=0.10, b=0.2, c=−65, d=2) — typical of inhibitory cortical "
            "interneurons."
        ),
        objective=(
            "See how increasing the recovery time scale `a` produces sustained "
            "high-frequency firing without adaptation, characteristic of "
            "parvalbumin-positive interneurons."
        ),
        run=_run_izh_fast_spiking,
    ),
    Example(
        id="eeg_trial",
        title="EEG Trial Exploration",
        icon="📡",
        category="EEG Analysis",
        description=(
            "Display the first 8 channels of EEG trial 0 from the PhysioNet "
            "Motor Movement/Imagery Dataset as a vertically-offset waterfall plot."
        ),
        objective=(
            "Inspect raw multi-channel EEG structure: visualise amplitude "
            "variation across channels and identify any obvious artefacts or "
            "label-correlated waveforms."
        ),
        run=_run_eeg_trial,
    ),
    Example(
        id="eeg_psd",
        title="EEG Power Spectrum",
        icon="〰️",
        category="EEG Analysis",
        description=(
            "Compute and plot the Power Spectral Density (PSD) of channel 0 "
            "from EEG trial 0 using Welch's method, with canonical frequency "
            "band annotations."
        ),
        objective=(
            "Locate alpha (8–13 Hz) and beta (13–30 Hz) power peaks that "
            "form the basis of motor imagery BCI decoding via Event-Related "
            "Desynchronisation (ERD)."
        ),
        run=_run_eeg_psd,
    ),
    Example(
        id="eeg_spectrogram",
        title="EEG Spectrogram",
        icon="🌈",
        category="EEG Analysis",
        description=(
            "Generate a time-frequency spectrogram of EEG trial 0, channel 0, "
            "using a short-time Fourier transform (Hann window, 64-sample "
            "segments, 75 % overlap)."
        ),
        objective=(
            "Observe how spectral power evolves over the trial duration; "
            "transient alpha/beta suppression appears as darker bands during "
            "the imagery period."
        ),
        run=_run_eeg_spectrogram,
    ),
    Example(
        id="decoder_demo",
        title="Decoder Prediction Demo",
        icon="🤖",
        category="BCI Decoding",
        description=(
            "Demonstrate EEGNet-style decoding with a probabilistic mock "
            "classifier. Shows the softmax probability bar chart for a randomly "
            "selected trial. No trained model required."
        ),
        objective=(
            "Understand the output format of a neural decoder: two softmax "
            "probabilities summing to 1.0, with the argmax determining the "
            "decoded class (left or right)."
        ),
        run=_run_decoder_demo,
    ),
    Example(
        id="closed_loop_demo",
        title="Closed-Loop Cursor Demo",
        icon="🎮",
        category="BCI Decoding",
        description=(
            "Run a 12-trial simulated closed-loop BCI session at 75 % decoder "
            "accuracy and plot the 2D cursor trajectory driven by decoded "
            "left/right motor imagery."
        ),
        objective=(
            "Visualise how decoder accuracy affects cursor displacement: at "
            "75 % accuracy the cursor drifts toward the target; at 50 % "
            "the trajectory is a random walk."
        ),
        run=_run_closed_loop_demo,
    ),
]

# Group examples by category for rendering
_CATEGORY_ORDER = ["Neuron Models", "EEG Analysis", "BCI Decoding"]


# ---------------------------------------------------------------------------
# Card renderer
# ---------------------------------------------------------------------------

def _render_card(ex: Example, col) -> None:
    """Render one experiment card inside *col*."""
    with col:
        with st.container(border=True):
            # Header row
            st.markdown(
                f"<div style='font-size:22px;margin-bottom:2px;'>{ex.icon}</div>"
                f"<div style='font-size:15px;font-weight:700;"
                f"letter-spacing:-.01em;margin-bottom:4px;'>{ex.title}</div>"
                f"<div style='font-size:11px;opacity:0.45;font-weight:600;"
                f"text-transform:uppercase;letter-spacing:.06em;"
                f"margin-bottom:8px;'>{ex.category}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(ex.description)

            with st.expander("Scientific objective"):
                st.markdown(ex.objective)

            key_btn = f"_ex_run_{ex.id}"
            key_result = f"_ex_result_{ex.id}"

            col_btn, col_clear = st.columns([3, 1])

            with col_btn:
                if st.button(
                    "▶ Run Example", key=key_btn, type="primary",
                    use_container_width=True,
                ):
                    with st.spinner("Running…"):
                        try:
                            fig = ex.run()
                            st.session_state[key_result] = fig
                        except Exception as exc:
                            st.session_state[key_result] = exc

            with col_clear:
                if st.button("✕", key=f"_ex_clear_{ex.id}",
                             use_container_width=True,
                             help="Clear result"):
                    st.session_state.pop(key_result, None)
                    st.rerun()

            result = st.session_state.get(key_result)
            if isinstance(result, go.Figure):
                st.plotly_chart(result, use_container_width=True)
            elif isinstance(result, Exception):
                st.error(f"Error: {result}", icon="❌")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

def render() -> None:
    render_page_header(
        "🧪", "Example Experiments",
        "Predefined experiments — click Run Example to execute and display results",
    )

    st.markdown(
        "Each card runs a complete experiment using NeuroLab's core simulation "
        "and analysis code. Results appear inline below each card. "
        "Use the **✕** button to clear a result."
    )

    # Sidebar: quick-jump links
    with st.sidebar:
        st.markdown("**Jump to category**")
        for cat in _CATEGORY_ORDER:
            count = sum(1 for e in EXAMPLES if e.category == cat)
            st.markdown(f"[{cat}](#{cat.lower().replace(' ', '-')}) — {count} examples")

        st.divider()
        n_run = sum(
            1 for e in EXAMPLES
            if st.session_state.get(f"_ex_result_{e.id}") is not None
        )
        st.metric("Examples run this session", n_run)

        if st.button("Clear all results", use_container_width=True,
                     key="_ex_clear_all"):
            for e in EXAMPLES:
                st.session_state.pop(f"_ex_result_{e.id}", None)
            st.rerun()

    # Render by category, 2-column grid
    for cat in _CATEGORY_ORDER:
        cat_examples = [e for e in EXAMPLES if e.category == cat]
        if not cat_examples:
            continue

        st.divider()
        section_header(cat)

        for i in range(0, len(cat_examples), 2):
            left_col, right_col = st.columns(2, gap="medium")
            _render_card(cat_examples[i], left_col)
            if i + 1 < len(cat_examples):
                _render_card(cat_examples[i + 1], right_col)


render()
