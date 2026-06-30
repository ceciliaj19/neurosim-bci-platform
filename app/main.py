import streamlit as st
import plotly.graph_objects as go

from neurosim.neurons import LIFNeuron


st.set_page_config(
    page_title="NeuroSim",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 NeuroSim")
st.subheader("Interactive Computational Neuroscience Simulator")

st.markdown(
    """
    This demo simulates a **Leaky Integrate-and-Fire neuron**, one of the simplest
    models used in computational neuroscience to study how neurons transform input
    current into spikes.
    """
)

with st.sidebar:
    st.header("Neuron Parameters")

    current = st.slider("Input Current", 0.0, 5.0, 2.0, 0.1)
    tau_m = st.slider("Membrane Time Constant τm (ms)", 5.0, 50.0, 20.0, 1.0)
    resistance = st.slider("Membrane Resistance", 1.0, 20.0, 10.0, 0.5)
    v_threshold = st.slider("Spike Threshold (mV)", -60.0, -40.0, -50.0, 1.0)
    refractory_period = st.slider("Refractory Period (ms)", 0.5, 10.0, 2.0, 0.5)

    st.header("Simulation Settings")
    t_max = st.slider("Simulation Time (ms)", 100.0, 1000.0, 500.0, 50.0)
    dt = st.selectbox("Time Step (ms)", [0.05, 0.1, 0.2, 0.5], index=1)

neuron = LIFNeuron(
    tau_m=tau_m,
    resistance=resistance,
    v_threshold=v_threshold,
    refractory_period=refractory_period,
)

result = neuron.simulate(
    current=current,
    t_max=t_max,
    dt=dt,
)

col1, col2 = st.columns([3, 1])

with col1:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=result.time,
            y=result.voltage,
            mode="lines",
            name="Membrane Potential",
        )
    )

    fig.update_layout(
        title="Membrane Potential Over Time",
        xaxis_title="Time (ms)",
        yaxis_title="Voltage (mV)",
        height=500,
    )

    st.plotly_chart(fig, width="stretch")

with col2:
    st.metric("Spike Count", result.spike_count)
    st.metric("Firing Rate", f"{result.firing_rate:.2f} Hz")

    st.markdown("### Interpretation")

    if result.spike_count == 0:
        st.write(
            "The input current is too low for the neuron to reach threshold, "
            "so no spikes are generated."
        )
    elif result.firing_rate < 50:
        st.write(
            "The neuron is firing at a moderate rate. Increasing input current "
            "or lowering the threshold will usually increase firing rate."
        )
    else:
        st.write(
            "The neuron is firing rapidly. This suggests the input current is "
            "strong relative to the threshold and membrane properties."
        )

st.markdown("---")

st.markdown(
    """
    ### Model Equation

    The Leaky Integrate-and-Fire model approximates membrane voltage dynamics as:

    $$
    \\tau_m \\frac{dV}{dt} = -(V - V_{rest}) + RI
    $$

    When the membrane voltage reaches threshold, the neuron emits a spike and resets.
    """
)