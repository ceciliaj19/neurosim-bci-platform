"""Reusable EEG multichannel trial chart component."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.ui import PLOTLY_LAYOUT


def render_eeg_chart(
    trial: np.ndarray,
    num_channels: int,
    trial_idx: int,
    true_label: int | None = None,
) -> None:
    """Render a stacked multichannel EEG trace using Plotly.

    Parameters
    ----------
    trial : np.ndarray
        Single EEG trial, shape ``(n_channels, n_timepoints)``.
    num_channels : int
        Number of channels to display (from index 0).
    trial_idx : int
        Trial index shown in the plot title.
    true_label : int or None, optional
        If provided, appended to the plot title for context.
    """
    title = f"EEG Trial {trial_idx}"
    if true_label is not None:
        title += f"  —  True label: {true_label}"

    fig = go.Figure()

    for ch in range(min(num_channels, trial.shape[0])):
        fig.add_trace(go.Scatter(
            y=trial[ch] + ch * 5,
            mode="lines",
            name=f"Channel {ch}",
        ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=title,
        xaxis_title="Timepoints",
        yaxis_title="Amplitude + channel offset",
        height=500,
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)
