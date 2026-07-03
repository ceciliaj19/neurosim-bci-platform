"""Reusable sidebar model-metrics display component."""

from __future__ import annotations

import streamlit as st


def render_model_metrics(metrics: dict | None) -> None:
    """Render EEGNet training metrics in the Streamlit sidebar.

    Displays accuracy, precision, recall, and F1 as ``st.metric`` tiles.
    Shows an informational hint when *metrics* is ``None``.

    Parameters
    ----------
    metrics : dict or None
        Parsed ``results/metrics.json`` dict, or ``None`` when the file
        is absent or unreadable.
    """
    st.header("Model Performance")

    if metrics is None:
        st.info("Train the model to see metrics here.")
        return

    st.metric("Accuracy", f"{metrics['accuracy'] * 100:.1f}%")
    st.metric("Precision (macro)", f"{metrics['precision_macro'] * 100:.1f}%")
    st.metric("Recall (macro)", f"{metrics['recall_macro'] * 100:.1f}%")
    st.metric("F1 (macro)", f"{metrics['f1_macro'] * 100:.1f}%")
    st.caption(
        f"Trained for {metrics.get('n_epochs', '?')} epochs  |  "
        f"lr={metrics.get('lr', '?')}"
    )
