"""NeuroLab — Decoder Lab page: EEGNet inference on motor imagery trials."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.cursor_chart import render_cursor_chart
from components.eeg_chart import render_eeg_chart
from components.exporter import download_buttons
from components.model_metrics import render_model_metrics
from components.page_header import render_page_header
from components.ui import COLORS, PLOTLY_LAYOUT, section_header
from neurosim.bci import CursorController
from neurosim.datasets import DatasetLoader

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent.parent
_MODEL_PATH = _ROOT / "models"   / "eegnet_motor_imagery_v1.pth"
_METRICS_PATH = _ROOT / "results" / "metrics.json"
_CM_PATH      = _ROOT / "results" / "confusion_matrix.json"

_N_CLASSES = 2
_N_CHANNELS = 64
_N_TIMEPOINTS = 513
_CLASS_TO_COMMAND: dict[int, str] = {0: "left", 1: "right"}

_KEY_CURSOR = "dl_cursor"
_KEY_TRAJ = "dl_trajectory"
_KEY_PRED = "dl_last_prediction"

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------


@st.cache_data
def _load_data() -> tuple[np.ndarray, np.ndarray]:
    return DatasetLoader().load_motor_imagery()


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
    """Load the raw confusion matrix from results/confusion_matrix.json.

    Returns the ``matrix`` field (a nested list) if the file exists and is
    valid, otherwise ``None``.
    """
    if not _CM_PATH.is_file():
        return None
    try:
        payload = json.loads(_CM_PATH.read_text())
        return payload.get("matrix")
    except (json.JSONDecodeError, AttributeError):
        return None


@st.cache_resource
def _load_model():
    try:
        import torch  # noqa: F401
        from neurosim.networks import EEGNet  # noqa: F401
    except ImportError:
        return None, (
            "PyTorch is not installed. "
            "Run `pip install neurosim[training]` to enable inference."
        )

    if not _MODEL_PATH.is_file():
        return None, (
            f"Checkpoint not found at `{_MODEL_PATH.relative_to(_ROOT)}`.\n\n"
            "Train first with:\n```\npython scripts/train_eegnet.py\n```"
        )

    import torch
    from neurosim.networks import EEGNet

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
# Page
# ---------------------------------------------------------------------------


def render() -> None:
    render_page_header(
        "🤖", "Decoder Lab",
        "EEGNet inference — motor imagery decoding and cursor control",
    )

    X, y = _load_data()
    model, model_error = _load_model()
    metrics = _load_metrics()

    # -- Sidebar -------------------------------------------------------------
    with st.sidebar:
        st.markdown("**Dataset**")
        col_a, col_b = st.columns(2)
        col_a.metric("Trials", X.shape[0])
        col_b.metric("Channels", X.shape[1])

        st.divider()
        st.markdown("**Trial Selection**")
        trial_idx = st.slider("Trial index", 0, X.shape[0] - 1, 0)
        num_channels = st.slider("Channels to display", 1, 10, 5)

        st.divider()
        st.markdown("**Cursor**")
        if st.button("Reset Cursor", key="dl_reset"):
            if _KEY_CURSOR in st.session_state:
                st.session_state[_KEY_CURSOR].reset()
                st.session_state[_KEY_TRAJ] = [(0.0, 0.0)]

        render_model_metrics(metrics)

    trial = X[trial_idx]
    true_label = int(y[trial_idx])

    # -- Inference panel -----------------------------------------------------
    section_header("Neural Decoding")

    if model_error:
        st.warning(model_error)
    else:
        if _KEY_CURSOR not in st.session_state:
            st.session_state[_KEY_CURSOR] = CursorController(step_size=1.0)
            st.session_state[_KEY_TRAJ] = [(0.0, 0.0)]

        inf_col, cursor_col = st.columns([1, 2])

        with inf_col:
            if st.button("▶ Run Inference & Move Cursor", key="dl_infer", type="primary"):
                import torch

                with torch.no_grad():
                    x_tensor = torch.from_numpy(trial).float().unsqueeze(0)
                    logits = model(x_tensor)
                    probs = torch.softmax(logits, dim=1).squeeze(0).numpy()

                predicted_class = int(np.argmax(probs))
                confidence = float(probs[predicted_class])
                correct = predicted_class == true_label

                st.session_state[_KEY_PRED] = {
                    "class": predicted_class,
                    "confidence": confidence,
                    "correct": correct,
                    "probs": probs.tolist(),
                }

                command = _CLASS_TO_COMMAND[predicted_class]
                new_pos = st.session_state[_KEY_CURSOR].update(command)
                st.session_state[_KEY_TRAJ].append(new_pos)

            pred = st.session_state.get(_KEY_PRED)
            if pred:
                m1, m2, m3 = st.columns(3)
                m1.metric("True", true_label)
                m2.metric("Predicted", pred["class"])
                m3.metric("Confidence", f"{pred['confidence'] * 100:.0f}%")

                if pred["correct"]:
                    st.success("Correct prediction", icon="✅")
                else:
                    st.error("Incorrect prediction", icon="❌")

                st.markdown("**Class probabilities**")
                probs_list = pred["probs"]
                prob_fig = go.Figure(go.Bar(
                    x=[f"Class {i} ({_CLASS_TO_COMMAND[i]})"
                       for i in range(len(probs_list))],
                    y=[p * 100 for p in probs_list],
                    marker_color=[COLORS["left_class"], COLORS["right_class"]],
                    text=[f"{p * 100:.1f}%" for p in probs_list],
                    textposition="outside",
                ))
                prob_fig.update_layout(
                    **PLOTLY_LAYOUT,
                    yaxis=dict(title="Probability (%)", range=[0, 115]),
                    xaxis_title=None,
                    height=220,
                    showlegend=False,
                )
                st.plotly_chart(prob_fig, use_container_width=True)
                download_buttons(
                    "Decoder Prediction",
                    f"prediction_trial{trial_idx}",
                    record={
                        "trial_index": trial_idx,
                        "true_label": true_label,
                        "true_command": _CLASS_TO_COMMAND[true_label],
                        "predicted_label": pred["class"],
                        "predicted_command": _CLASS_TO_COMMAND[pred["class"]],
                        "confidence": pred["confidence"],
                        "correct": pred["correct"],
                        "class_probabilities": {
                            _CLASS_TO_COMMAND[i]: pred["probs"][i]
                            for i in range(len(pred["probs"]))
                        },
                    },
                    extra_meta={"model": "EEGNet", "n_classes": _N_CLASSES},
                )
            else:
                st.info(
                    "Select a trial and press **Run Inference** to decode it.",
                    icon="💡",
                )

        with cursor_col:
            st.markdown("**Cursor trajectory**")
            trajectory = st.session_state.get(_KEY_TRAJ, [(0.0, 0.0)])
            render_cursor_chart(trajectory)

            cursor = st.session_state.get(_KEY_CURSOR)
            if cursor:
                st.caption(
                    f"Position ({cursor.x:.1f}, {cursor.y:.1f})  ·  "
                    f"{len(trajectory) - 1} step(s)"
                )

    # -- EEG signal ----------------------------------------------------------
    st.divider()
    section_header("Raw EEG Signal")
    render_eeg_chart(trial, num_channels, trial_idx, true_label)

    # -- Model performance ---------------------------------------------------
    st.divider()
    section_header("Model Performance")

    if metrics is None:
        st.info(
            "No evaluation metrics found. "
            f"Expected at `{_METRICS_PATH.relative_to(_ROOT)}`.  \n"
            "Generate them by running:  \n"
            "```\npython scripts/train_eegnet.py\n```",
            icon="📂",
        )
    else:
        st.caption(
            "Metrics are computed on the hold-out test set (20 % stratified split, "
            f"seed 42).  Trained for {metrics.get('n_epochs', '—')} epochs."
        )

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Accuracy",        f"{metrics.get('accuracy', 0) * 100:.1f}%")
        mc2.metric("Precision (macro)", f"{metrics.get('precision_macro', 0) * 100:.1f}%")
        mc3.metric("Recall (macro)",  f"{metrics.get('recall_macro', 0) * 100:.1f}%")
        mc4.metric("F1 (macro)",      f"{metrics.get('f1_macro', 0) * 100:.1f}%")

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        ht1, ht2, ht3 = st.columns(3)
        ht1.metric("Epochs",      metrics.get("n_epochs", "—"))
        ht2.metric("Batch Size",  metrics.get("batch_size", "—"))
        ht3.metric("Learning Rate", metrics.get("lr", "—"))

    # -- Confusion matrix ----------------------------------------------------
    st.divider()
    section_header("Confusion Matrix")
    st.markdown(
        "A **confusion matrix** shows how often the model's predictions match "
        "the true class labels across the test set. Each row represents the "
        "**true class**; each column represents the **predicted class**. "
        "The diagonal cells (top-left, bottom-right) are correct predictions — "
        "true left decoded as left, true right decoded as right. "
        "Off-diagonal cells are errors: the top-right cell counts left-hand trials "
        "mistakenly decoded as right, and the bottom-left counts the reverse."
    )

    cm_data = _load_confusion_matrix()

    if cm_data is None:
        st.info(
            f"Confusion matrix not found at `{_CM_PATH.relative_to(_ROOT)}`.  \n"
            "Generate it by running:  \n"
            "```\npython scripts/train_eegnet.py\n```",
            icon="🔲",
        )
    else:
        cm = np.array(cm_data)
        class_labels = [f"Class {i} ({_CLASS_TO_COMMAND[i]})" for i in range(_N_CLASSES)]

        # Annotate each cell with its count
        cell_text = [[str(cm[r, c]) for c in range(cm.shape[1])]
                     for r in range(cm.shape[0])]

        cm_fig = go.Figure(go.Heatmap(
            z=cm,
            x=class_labels,
            y=class_labels,
            text=cell_text,
            texttemplate="%{text}",
            textfont=dict(size=18, color="white"),
            colorscale="Blues",
            showscale=True,
            colorbar=dict(title="Count", thickness=14),
            hovertemplate=(
                "True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>"
            ),
        ))
        cm_fig.update_layout(
            **PLOTLY_LAYOUT,
            title="Confusion Matrix — EEGNet on Motor Imagery Test Set",
            xaxis_title="Predicted Class",
            yaxis_title="True Class",
            height=380,
        )
        st.plotly_chart(cm_fig, use_container_width=True)
        download_buttons(
            "Confusion Matrix",
            "confusion_matrix",
            record={
                "labels": [f"Class {i} ({_CLASS_TO_COMMAND[i]})"
                           for i in range(_N_CLASSES)],
                "matrix": cm_data,
                "metrics": {
                    "TN": cm_data[0][0], "FP": cm_data[0][1],
                    "FN": cm_data[1][0], "TP": cm_data[1][1],
                },
            },
            extra_meta={"model": "EEGNet", "dataset": "PhysioNet EEGMMI"},
        )


render()
