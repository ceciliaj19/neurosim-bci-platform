"""NeuroLab — Experiment Dashboard: unified view of all platform modules."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.page_header import render_page_header
from neurosim.datasets import DatasetLoader

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent.parent
_METRICS_PATH = _ROOT / "results" / "metrics.json"
_CM_PATH = _ROOT / "results" / "confusion_matrix.json"
_HISTORY_PATH = _ROOT / "results" / "training_history.json"

_CLASS_NAMES = {0: "left", 1: "right"}


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_data
def _load_metrics() -> dict | None:
    if not _METRICS_PATH.is_file():
        return None
    try:
        return json.loads(_METRICS_PATH.read_text())
    except json.JSONDecodeError:
        return None


@st.cache_data
def _load_confusion_matrix() -> list[list[int]] | None:
    if not _CM_PATH.is_file():
        return None
    try:
        payload = json.loads(_CM_PATH.read_text())
        return payload.get("matrix")
    except (json.JSONDecodeError, AttributeError):
        return None


@st.cache_data
def _load_training_history() -> dict | None:
    if not _HISTORY_PATH.is_file():
        return None
    try:
        data = json.loads(_HISTORY_PATH.read_text())
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


@st.cache_data
def _load_dataset_shape() -> tuple[int, int, int] | None:
    try:
        X, _ = DatasetLoader().load_motor_imagery()
        return X.shape  # (trials, channels, timepoints)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Small chart builders
# ---------------------------------------------------------------------------

def _training_curve_fig(history: dict) -> go.Figure:
    """Mini val_acc + val_loss dual-axis sparkline."""
    epochs = list(range(1, len(history["val_acc"]) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epochs, y=[v * 100 for v in history["val_acc"]],
        mode="lines", name="Val Acc (%)",
        line=dict(color="#6366f1", width=2),
        yaxis="y1",
    ))
    fig.add_trace(go.Scatter(
        x=epochs, y=history["val_loss"],
        mode="lines", name="Val Loss",
        line=dict(color="#f59e0b", width=1.5, dash="dot"),
        yaxis="y2",
    ))
    fig.update_layout(
        height=200,
        margin=dict(l=36, r=36, t=10, b=30),
        plot_bgcolor="#f9f9f9",
        legend=dict(orientation="h", y=-0.28, x=0.0, font=dict(size=11)),
        xaxis=dict(title="Epoch", tickfont=dict(size=10)),
        yaxis=dict(
            title=dict(text="Acc (%)", font=dict(size=11)),
            tickfont=dict(size=10),
        ),
        yaxis2=dict(
            title=dict(text="Loss", font=dict(size=11)),
            overlaying="y",
            side="right",
            tickfont=dict(size=10),
        ),
        hovermode="x unified",
    )
    return fig


def _confusion_matrix_fig(cm_data: list[list[int]]) -> go.Figure:
    cm = np.array(cm_data)
    n = cm.shape[0]
    labels = [f"Class {i} ({_CLASS_NAMES[i]})" for i in range(n)]
    cell_text = [[str(cm[r, c]) for c in range(n)] for r in range(n)]
    fig = go.Figure(go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        text=cell_text,
        texttemplate="%{text}",
        textfont=dict(size=14, color="white"),
        colorscale="Blues",
        showscale=False,
        hovertemplate="True: %{y}<br>Pred: %{x}<br>Count: %{z}<extra></extra>",
    ))
    fig.update_layout(
        height=220,
        margin=dict(l=60, r=20, t=10, b=50),
        xaxis_title="Predicted",
        yaxis_title="True",
        xaxis=dict(tickfont=dict(size=10)),
        yaxis=dict(tickfont=dict(size=10)),
    )
    return fig


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------

def _section_neurosim() -> None:
    with st.container(border=True):
        st.markdown("### 🧪 NeuroSim")
        st.caption("Computational neuron simulation and analysis")
        st.markdown(
            "NeuroSim provides biophysically grounded single-neuron simulation "
            "and a suite of analysis tools for characterising model behaviour."
        )
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Neuron models**")
            st.markdown(
                "- Leaky Integrate-and-Fire (LIF)\n"
                "- Izhikevich (6 published presets)\n"
                "- Extensible `BaseNeuron` protocol"
            )
        with col_b:
            st.markdown("**Analysis tools**")
            st.markdown(
                "- F-I curve (firing rate vs. current)\n"
                "- Parameter sweep\n"
                "- Multi-model comparison\n"
                "- Scientific validation suite"
            )
        st.markdown(
            "<div style='font-size:12px;opacity:0.45;margin-top:4px;'>"
            "Navigate to <b>NeuroSim</b> to run experiments interactively."
            "</div>",
            unsafe_allow_html=True,
        )


def _section_eeg(shape: tuple | None) -> None:
    with st.container(border=True):
        st.markdown("### 📡 EEG Analysis")
        st.caption("PhysioNet EEGMMI — 64-channel motor imagery EEG")
        st.markdown(
            "EEG Studio provides waveform inspection and spectral analysis "
            "of the motor imagery dataset used to train the neural decoder."
        )
        if shape is not None:
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Trials", shape[0])
            mc2.metric("Channels", shape[1])
            mc3.metric("Timepoints", shape[2])
        else:
            st.info("Dataset not yet loaded — open EEG Studio to initialise.",
                    icon="📂")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Spectral tools**")
            st.markdown(
                "- Power Spectral Density (Welch)\n"
                "- Spectrogram (STFT, Viridis colormap)\n"
                "- Frequency band power (δ θ α β γ)"
            )
        with col_b:
            st.markdown("**Key bands**")
            st.markdown(
                "- **Alpha (8–13 Hz)** — cortical idling / ERD\n"
                "- **Beta (13–30 Hz)** — motor planning / ERS\n"
                "- Sampling rate: 160 Hz"
            )
        st.markdown(
            "<div style='font-size:12px;opacity:0.45;margin-top:4px;'>"
            "Navigate to <b>EEG Studio</b> to explore individual trials."
            "</div>",
            unsafe_allow_html=True,
        )


def _section_decoder(metrics: dict | None, cm_data: list | None,
                     history: dict | None) -> None:
    with st.container(border=True):
        st.markdown("### 🤖 Decoder Performance")
        st.caption("EEGNet — convolutional neural network for motor imagery BCI")
        st.markdown(
            "EEGNet is a compact EEG-specific CNN trained on the PhysioNet "
            "EEGMMI dataset. It decodes left- vs. right-hand motor imagery "
            "from raw 64-channel EEG."
        )

        if metrics is not None:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy",  f"{metrics.get('accuracy', 0) * 100:.1f}%")
            m2.metric("F1 (macro)", f"{metrics.get('f1_macro', 0) * 100:.1f}%")
            m3.metric("Precision", f"{metrics.get('precision_macro', 0) * 100:.1f}%")
            m4.metric("Recall",    f"{metrics.get('recall_macro', 0) * 100:.1f}%")

            cfg1, cfg2, cfg3 = st.columns(3)
            cfg1.metric("Epochs", metrics.get("n_epochs", "—"))
            cfg2.metric("Batch",  metrics.get("batch_size", "—"))
            cfg3.metric("LR",     metrics.get("lr", "—"))
        else:
            st.info(
                "No metrics found. Train with `python scripts/train_eegnet.py`.",
                icon="🔲",
            )

        chart_col, cm_col = st.columns([3, 2], gap="medium")

        with chart_col:
            if history is not None:
                st.markdown("**Validation accuracy & loss over training**")
                st.plotly_chart(_training_curve_fig(history),
                                use_container_width=True)
            else:
                st.caption("Training history not available.")

        with cm_col:
            if cm_data is not None:
                st.markdown("**Confusion matrix (test set)**")
                st.plotly_chart(_confusion_matrix_fig(cm_data),
                                use_container_width=True)
            else:
                st.caption("Confusion matrix not available.")

        st.markdown(
            "<div style='font-size:12px;opacity:0.45;margin-top:4px;'>"
            "Navigate to <b>Decoder Lab</b> for per-trial inference and full "
            "evaluation detail."
            "</div>",
            unsafe_allow_html=True,
        )


def _section_closed_loop() -> None:
    with st.container(border=True):
        st.markdown("### 🎮 Closed-Loop BCI Session")
        st.caption("Motor imagery cue → mock decoder → virtual cursor")
        st.markdown(
            "The Closed-Loop BCI page runs a structured session: the participant "
            "sees a left/right cue, the mock decoder produces a classification "
            "and confidence score, and the virtual 2D cursor moves accordingly. "
            "Session accuracy is tracked across all trials."
        )

        # Read live session state from cl_* keys (populated by closed_loop.py)
        history: list[dict] = st.session_state.get("cl_history", [])
        active: bool = st.session_state.get("cl_active", False)
        cues: list[str] = st.session_state.get("cl_cues", [])

        if not history:
            st.info(
                "No session data yet. Open **Closed-Loop BCI** and run a session "
                "to see live results here.",
                icon="🎮",
            )
        else:
            n_done = len(history)
            n_correct = sum(t["correct"] for t in history)
            n_total = len(cues) if cues else n_done
            acc = n_correct / n_done * 100 if n_done > 0 else 0.0

            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("Trials Completed", f"{n_done} / {n_total}")
            sc2.metric("Correct",  n_correct)
            sc3.metric("Incorrect", n_done - n_correct)
            sc4.metric("Session Accuracy", f"{acc:.0f}%")

            # Per-trial result bar
            results = [1 if t["correct"] else 0 for t in history]
            fig = go.Figure(go.Bar(
                x=[f"T{t['trial']}" for t in history],
                y=results,
                marker_color=["#10b981" if r else "#ef4444" for r in results],
                text=["✓" if r else "✗" for r in results],
                textposition="inside",
                hovertemplate=(
                    "Trial %{x}<br>"
                    "Cue: %{customdata[0]}<br>"
                    "Pred: %{customdata[1]}<br>"
                    "Confidence: %{customdata[2]:.0%}<extra></extra>"
                ),
                customdata=[
                    [t["cue"].upper(), t["prediction"].upper(), t["confidence"]]
                    for t in history
                ],
            ))
            fig.update_layout(
                height=160,
                yaxis=dict(tickvals=[0, 1], ticktext=["✗", "✓"],
                           range=[-0.2, 1.4]),
                xaxis_title=None,
                margin=dict(l=30, r=10, t=8, b=30),
                plot_bgcolor="#f9f9f9",
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

            if active and n_done < n_total:
                st.caption(
                    f"Session in progress — {n_total - n_done} trial(s) remaining."
                )
            elif n_done >= n_total and n_total > 0:
                st.success(f"Session complete — {acc:.0f}% accuracy", icon="🏁")

        st.markdown(
            "<div style='font-size:12px;opacity:0.45;margin-top:4px;'>"
            "Navigate to <b>Closed-Loop BCI</b> to run or continue a session."
            "</div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

def render() -> None:
    render_page_header(
        "📊", "Experiment Dashboard",
        "Unified overview of all NeuroLab modules and results",
    )

    metrics = _load_metrics()
    cm_data = _load_confusion_matrix()
    history = _load_training_history()
    shape = _load_dataset_shape()

    # -- Top KPI row -----------------------------------------------------------
    st.markdown("#### Key Results")
    if metrics is not None:
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Decoder Accuracy",  f"{metrics.get('accuracy', 0) * 100:.1f}%")
        k2.metric("F1 Score (macro)",  f"{metrics.get('f1_macro', 0) * 100:.1f}%")
        k3.metric("Precision (macro)", f"{metrics.get('precision_macro', 0) * 100:.1f}%")
        k4.metric("Recall (macro)",    f"{metrics.get('recall_macro', 0) * 100:.1f}%")
        k5.metric("Training Epochs",   metrics.get("n_epochs", "—"))
    else:
        st.info(
            "No evaluation results found. "
            "Run `python scripts/train_eegnet.py` to generate metrics.",
            icon="📂",
        )

    st.divider()

    # -- Module cards: two columns ---------------------------------------------
    left_col, right_col = st.columns(2, gap="large")

    with left_col:
        _section_neurosim()
        st.markdown("")
        _section_eeg(shape)

    with right_col:
        _section_decoder(metrics, cm_data, history)
        st.markdown("")
        _section_closed_loop()

    # -- Footer ----------------------------------------------------------------
    st.divider()
    st.markdown(
        "<div style='text-align:center;font-size:12px;opacity:0.35;'>"
        "NeuroLab · open-source BCI platform · "
        "metrics from <code>results/</code> · "
        "session state from current app instance"
        "</div>",
        unsafe_allow_html=True,
    )


render()
