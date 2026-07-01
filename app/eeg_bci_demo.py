"""EEG Motor Imagery BCI Demo — Streamlit application.

Loads the motor imagery dataset, runs EEGNet inference on the selected
trial, moves a virtual cursor based on the prediction, and visualises
the raw EEG trace.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from neurosim.bci import CursorController
from neurosim.datasets import DatasetLoader

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent
_MODEL_PATH = _ROOT / "models" / "eegnet_motor_imagery_v1.pth"
_METRICS_PATH = _ROOT / "results" / "metrics.json"

# Fixed hyperparameters — must match the values used during training
_N_CLASSES = 2
_N_CHANNELS = 64
_N_TIMEPOINTS = 513

# Map predicted class index → cursor command
_CLASS_TO_COMMAND = {0: "left", 1: "right"}

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="EEG BCI Demo",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 EEG Motor Imagery BCI Demo")
st.subheader("Neural decoding with EEGNet — closed-loop cursor control")

st.markdown(
    """
    Select an EEG trial from the sidebar. The trained **EEGNet** model predicts
    the imagined movement class and moves a virtual cursor accordingly.
    """
)

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_data
def load_data() -> tuple[np.ndarray, np.ndarray]:
    """Load the motor imagery dataset (cached across reruns)."""
    loader = DatasetLoader()
    return loader.load_motor_imagery()


@st.cache_data
def load_metrics() -> dict | None:
    """Load results/metrics.json if it exists.

    Returns
    -------
    dict or None
        Parsed metrics dict, or ``None`` when the file is absent or unreadable.
    """
    if not _METRICS_PATH.is_file():
        return None
    try:
        return json.loads(_METRICS_PATH.read_text())
    except json.JSONDecodeError:
        return None


@st.cache_resource
def load_model():
    """Load the trained EEGNet checkpoint (singleton per server process).

    Returns
    -------
    model : torch.nn.Module or None
        The model in eval mode, or ``None`` if PyTorch is unavailable or the
        checkpoint is missing. The caller is responsible for checking.
    error : str or None
        Human-readable error message when the model cannot be loaded.
    """
    try:
        import torch
        from neurosim.networks import EEGNet
    except ImportError:
        return None, (
            "PyTorch is not installed. "
            "Run `pip install neurosim[training]` to enable inference."
        )

    if not _MODEL_PATH.is_file():
        return None, (
            f"Model checkpoint not found at `{_MODEL_PATH.relative_to(_ROOT)}`.\n\n"
            "Train the model first with:\n"
            "```\npython scripts/train_eegnet.py\n```"
        )

    model = EEGNet(
        n_classes=_N_CLASSES,
        n_channels=_N_CHANNELS,
        n_timepoints=_N_TIMEPOINTS,
    )
    model.load_state_dict(
        torch.load(_MODEL_PATH, map_location="cpu", weights_only=True)
    )
    model.eval()
    return model, None


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

X, y = load_data()
model, model_error = load_model()
metrics = load_metrics()

# ---------------------------------------------------------------------------
# Sidebar — dataset controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Dataset")
    st.write(f"Trials: {X.shape[0]}")
    st.write(f"Channels: {X.shape[1]}")
    st.write(f"Timepoints: {X.shape[2]}")

    st.header("Trial Selection")
    trial_idx = st.slider(
        "Select Trial",
        min_value=0,
        max_value=X.shape[0] - 1,
        value=0,
        step=1,
    )

    num_channels = st.slider(
        "Channels to Display",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
    )

    st.header("Cursor")
    if st.button("Reset Cursor"):
        if "cursor" in st.session_state:
            st.session_state.cursor.reset()
            st.session_state.trajectory = [(0.0, 0.0)]

    st.header("Model Performance")
    if metrics:
        st.metric("Accuracy", f"{metrics['accuracy'] * 100:.1f}%")
        st.metric("Precision (macro)", f"{metrics['precision_macro'] * 100:.1f}%")
        st.metric("Recall (macro)", f"{metrics['recall_macro'] * 100:.1f}%")
        st.metric("F1 (macro)", f"{metrics['f1_macro'] * 100:.1f}%")
        st.caption(
            f"Trained for {metrics.get('n_epochs', '?')} epochs  |  "
            f"lr={metrics.get('lr', '?')}"
        )
    else:
        st.info("Train the model to see metrics here.")

trial = X[trial_idx]
true_label = int(y[trial_idx])

# ---------------------------------------------------------------------------
# Inference section
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("Neural Decoding")

if model_error:
    st.warning(model_error)
else:
    # Initialise persistent cursor and trajectory in session state
    if "cursor" not in st.session_state:
        st.session_state.cursor = CursorController(step_size=1.0)
        st.session_state.trajectory = [(0.0, 0.0)]

    inf_col, cursor_col = st.columns([1, 2])

    with inf_col:
        st.markdown("#### Prediction")

        run_inference = st.button("Run Inference & Move Cursor")

        if run_inference:
            import torch

            with torch.no_grad():
                # (C, T) → (1, C, T) batch tensor
                x_tensor = torch.from_numpy(trial).float().unsqueeze(0)
                logits = model(x_tensor)
                probs = torch.softmax(logits, dim=1).squeeze(0).numpy()

            predicted_class = int(np.argmax(probs))
            confidence = float(probs[predicted_class])
            correct = predicted_class == true_label

            # Store result so it persists across reruns until next click
            st.session_state.last_prediction = {
                "class": predicted_class,
                "confidence": confidence,
                "correct": correct,
                "probs": probs.tolist(),
            }

            # Move cursor based on predicted class
            command = _CLASS_TO_COMMAND[predicted_class]
            new_pos = st.session_state.cursor.update(command)
            st.session_state.trajectory.append(new_pos)

        # Display last prediction if available
        pred = st.session_state.get("last_prediction")
        if pred:
            m1, m2, m3 = st.columns(3)
            m1.metric("True Label", true_label)
            m2.metric("Predicted", pred["class"])
            m3.metric("Confidence", f"{pred['confidence'] * 100:.1f}%")

            if pred["correct"]:
                st.success("Correct prediction")
            else:
                st.error("Incorrect prediction")

            st.markdown("**Class probabilities**")
            probs = pred["probs"]
            bar_colors = ["#1f77b4", "#ff7f0e"]
            prob_fig = go.Figure(go.Bar(
                x=[f"Class {i} ({_CLASS_TO_COMMAND[i]})" for i in range(len(probs))],
                y=[p * 100 for p in probs],
                marker_color=bar_colors,
                text=[f"{p * 100:.1f}%" for p in probs],
                textposition="outside",
            ))
            prob_fig.update_layout(
                yaxis=dict(title="Probability (%)", range=[0, 110]),
                xaxis_title="Class",
                height=260,
                margin=dict(l=20, r=20, t=20, b=40),
                plot_bgcolor="#f9f9f9",
                showlegend=False,
            )
            st.plotly_chart(prob_fig, use_container_width=True)
        else:
            st.info("Press **Run Inference & Move Cursor** to decode this trial.")

    with cursor_col:
        st.markdown("#### Cursor Trajectory")

        trajectory = st.session_state.get("trajectory", [(0.0, 0.0)])
        x_vals = [p[0] for p in trajectory]
        y_vals = [p[1] for p in trajectory]

        cursor_fig = go.Figure()

        cursor_fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="lines+markers",
            name="Trajectory",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=6),
        ))

        # Highlight current position
        cursor_fig.add_trace(go.Scatter(
            x=[x_vals[-1]],
            y=[y_vals[-1]],
            mode="markers",
            name="Current",
            marker=dict(size=14, color="#d62728", symbol="circle"),
        ))

        cursor_fig.update_layout(
            xaxis_title="X position",
            yaxis_title="Y position",
            height=350,
            xaxis=dict(range=[-15, 15]),
            yaxis=dict(range=[-15, 15]),
            legend=dict(x=1, y=1, xanchor="right"),
            margin=dict(l=40, r=20, t=20, b=40),
            plot_bgcolor="#f9f9f9",
        )

        st.plotly_chart(cursor_fig, use_container_width=True)

        cursor = st.session_state.get("cursor")
        if cursor:
            st.caption(
                f"Position: ({cursor.x:.1f}, {cursor.y:.1f})  |  "
                f"Steps: {len(trajectory) - 1}"
            )

# ---------------------------------------------------------------------------
# EEG visualisation (unchanged)
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader(f"EEG Trial {trial_idx}  —  True label: {true_label}")

eeg_fig = go.Figure()

for ch in range(num_channels):
    eeg_fig.add_trace(go.Scatter(
        y=trial[ch] + ch * 5,
        mode="lines",
        name=f"Channel {ch}",
    ))

eeg_fig.update_layout(
    xaxis_title="Timepoints",
    yaxis_title="Amplitude + channel offset",
    height=600,
    hovermode="x unified",
)

st.plotly_chart(eeg_fig, use_container_width=True)

st.markdown(
    """
    ### What this shows

    Each line is one EEG channel from a single motor imagery trial.
    The signals are offset vertically so multiple channels can be viewed at once.
    """
)
