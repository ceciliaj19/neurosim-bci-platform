"""NeuroLab — Tutorials page."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from components.page_header import render_page_header

_TRACKS = [
    {
        "number": "01",
        "title": "Neural Modelling Basics",
        "icon": "🔬",
        "color": "rgba(25,103,210,0.10)",
        "accent": "#1967d2",
        "items": [
            "What is a Leaky Integrate-and-Fire neuron?",
            "Tuning membrane time constant and threshold",
            "Interpreting spike trains and firing rate",
            "Beyond LIF: Izhikevich and Hodgkin-Huxley models",
        ],
    },
    {
        "number": "02",
        "title": "EEG Signal Processing",
        "icon": "📡",
        "color": "rgba(19,115,51,0.10)",
        "accent": "#137333",
        "items": [
            "What does an EEG trial contain?",
            "Bandpass filtering and artefact removal",
            "Common Spatial Patterns (CSP)",
            "Epoching and baseline correction",
        ],
    },
    {
        "number": "03",
        "title": "Training an EEG Classifier",
        "icon": "🤖",
        "color": "rgba(197,34,31,0.10)",
        "accent": "#c5221f",
        "items": [
            "EEGNet architecture overview",
            "Preparing the motor imagery dataset",
            "Train/test split and cross-validation",
            "Running `scripts/train_eegnet.py`",
            "Interpreting training curves and metrics",
        ],
    },
    {
        "number": "04",
        "title": "Building a Closed-Loop BCI",
        "icon": "🎮",
        "color": "rgba(230,108,0,0.10)",
        "accent": "#e66c00",
        "items": [
            "The BCI pipeline: signal → features → decoder → output device",
            "Using `CursorController` and `OutputDevice`",
            "Designing experiment protocols with `Protocol` and `Session`",
            "Latency and real-time constraints",
        ],
    },
    {
        "number": "05",
        "title": "Extending NeuroSim",
        "icon": "📦",
        "color": "rgba(100,70,200,0.10)",
        "accent": "#5a2de0",
        "items": [
            "Implementing a custom neuron with `BaseNeuron`",
            "Adding a new output device with `OutputDevice`",
            "Writing a custom `DatasetLoader` method",
            "Contributing to the NeuroSim open-source package",
        ],
    },
]


def _track_card(track: dict) -> str:
    items_html = "".join(
        f'<li style="margin-bottom:4px;opacity:0.7;">{item}</li>'
        for item in track["items"]
    )
    return (
        f'<div style="border:1px solid rgba(120,120,120,0.18);border-radius:12px;'
        f'padding:20px 22px;margin-bottom:14px;'
        f'border-left:4px solid {track["accent"]};">'
        f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">'
        f'<span style="font-size:20px;">{track["icon"]}</span>'
        f'<span style="font-size:12px;font-weight:600;letter-spacing:.06em;'
        f'opacity:0.4;text-transform:uppercase;">{track["number"]}</span>'
        f'<span style="font-weight:600;font-size:16px;">{track["title"]}</span>'
        f'</div>'
        f'<ul style="margin:0;padding-left:18px;font-size:14px;line-height:1.6;">'
        f'{items_html}</ul>'
        f'</div>'
    )


def render() -> None:
    render_page_header(
        "📚", "Tutorials",
        "Step-by-step guides for computational neuroscience and BCI research",
    )

    st.info(
        "Tutorials are coming soon. The tracks below outline the planned curriculum.",
        icon="🚧",
    )

    st.divider()

    left_col, right_col = st.columns(2, gap="large")

    for i, track in enumerate(_TRACKS):
        col = left_col if i % 2 == 0 else right_col
        with col:
            st.markdown(_track_card(track), unsafe_allow_html=True)

    st.divider()
    st.markdown(
        "Want to request a topic? "
        "[Open an issue on GitHub →](https://github.com/)"
    )


render()
