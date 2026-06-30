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
            line=dict(color="#1f77b4", width=1.5),
        )
    )

    fig.add_hline(
        y=v_threshold,
        line_dash="dash",
        line_color="#ff7f0e",
        line_width=1.5,
        annotation_text="Threshold",
        annotation_position="top right",
        annotation_font_size=11,
        annotation_font_color="#ff7f0e",
    )

    fig.add_hline(
        y=neuron.v_rest,
        line_dash="dot",
        line_color="#2ca02c",
        line_width=1.5,
        annotation_text="V_rest",
        annotation_position="bottom right",
        annotation_font_size=11,
        annotation_font_color="#2ca02c",
    )

    if result.spike_count > 0:
        fig.add_trace(
            go.Scatter(
                x=result.spike_times,
                y=[neuron.spike_voltage] * result.spike_count,
                mode="markers",
                name="Spikes",
                marker=dict(size=10, symbol="triangle-up", color="#d62728"),
            )
        )

    fig.update_layout(
        title="Membrane Potential Over Time",
        xaxis_title="Time (ms)",
        yaxis_title="Voltage (mV)",
        height=500,
        hovermode="x unified",
        plot_bgcolor="#f9f9f9",
        legend=dict(
            x=1,
            y=1,
            xanchor="right",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.75)",
            bordercolor="lightgrey",
            borderwidth=1,
        ),
        margin=dict(l=60, r=40, t=60, b=60),
    )

    fig.update_yaxes(range=[neuron.v_rest - 10, neuron.spike_voltage + 10])

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