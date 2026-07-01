import streamlit as st
import plotly.graph_objects as go
import random

from neurosim.bci import CursorController


st.title("🎮 Closed-Loop BCI Demo")
st.subheader("Mock neural decoder controlling a virtual cursor")

st.markdown(
    """
    This demo represents the closed-loop BCI workflow:

    **neural signal → decoder → predicted intent → output device**
    """
)

if "cursor" not in st.session_state:
    st.session_state.cursor = CursorController(step_size=1.0)

if "trajectory" not in st.session_state:
    st.session_state.trajectory = [(0.0, 0.0)]

commands = ["left", "right", "up", "down"]

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### Decoder")

    if st.button("Generate Mock Prediction"):
        intent = random.choice(commands)
        confidence = round(random.uniform(0.65, 0.98), 2)

        position = st.session_state.cursor.move(intent)
        st.session_state.trajectory.append(position)

        st.session_state.intent = intent
        st.session_state.confidence = confidence

    if st.button("Reset Cursor"):
        st.session_state.cursor.reset()
        st.session_state.trajectory = [(0.0, 0.0)]
        st.session_state.intent = None
        st.session_state.confidence = None

    st.markdown("### Prediction")

    intent = st.session_state.get("intent", None)
    confidence = st.session_state.get("confidence", None)

    if intent is not None:
        st.metric("Decoded Intent", intent.upper())
        st.metric("Confidence", f"{confidence * 100:.1f}%")
    else:
        st.write("No prediction yet.")

with col2:
    trajectory = st.session_state.trajectory
    x_vals = [p[0] for p in trajectory]
    y_vals = [p[1] for p in trajectory]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="lines+markers",
            name="Cursor trajectory",
        )
    )

    fig.update_layout(
        title="Virtual Cursor Position",
        xaxis_title="X position",
        yaxis_title="Y position",
        height=500,
        xaxis=dict(range=[-10, 10]),
        yaxis=dict(range=[-10, 10]),
    )

    st.plotly_chart(fig, width="stretch")