import streamlit as st
import plotly.graph_objects as go

st.write("App started")

from neurosim.datasets import DatasetLoader


st.set_page_config(
    page_title="EEG BCI Demo",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 EEG Motor Imagery BCI Demo")
st.subheader("Inspect real EEG trials before neural decoding")

st.markdown(
    """
    This page loads real motor imagery EEG trials and visualizes the neural signal.
    Later, this will connect to a decoder that predicts imagined movement and controls
    a virtual cursor.
    """
)

@st.cache_data
def load_data():
    loader = DatasetLoader()
    return loader.load_motor_imagery()

st.write("Loading dataset...")

X, y = load_data()

st.sidebar.header("Dataset")
st.sidebar.write(f"Trials: {X.shape[0]}")
st.sidebar.write(f"Channels: {X.shape[1]}")
st.sidebar.write(f"Timepoints: {X.shape[2]}")

trial_idx = st.sidebar.slider(
    "Select Trial",
    min_value=0,
    max_value=X.shape[0] - 1,
    value=0,
    step=1,
)

num_channels = st.sidebar.slider(
    "Channels to Display",
    min_value=1,
    max_value=10,
    value=5,
    step=1,
)

trial = X[trial_idx]
label = y[trial_idx]

st.metric("True Label", int(label))

fig = go.Figure()

for ch in range(num_channels):
    fig.add_trace(
        go.Scatter(
            y=trial[ch] + ch * 5,
            mode="lines",
            name=f"Channel {ch}",
        )
    )

fig.update_layout(
    title=f"EEG Trial {trial_idx}",
    xaxis_title="Timepoints",
    yaxis_title="Amplitude + channel offset",
    height=600,
    hovermode="x unified",
)

st.plotly_chart(fig, width="stretch")

st.markdown(
    """
    ### What this shows

    Each line is one EEG channel from a single motor imagery trial.
    The signals are offset vertically so multiple channels can be viewed at once.
    """
)