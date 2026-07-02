"""NeuroLab — NeuroSim page: Leaky Integrate-and-Fire neuron simulator."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.page_header import render_page_header
from neurosim.analysis import compare_neuron_models, compute_fi_curve, parameter_sweep
from neurosim.neurons import (
    IzhikevichNeuron,
    LIFNeuron,
    get_izhikevich_preset,
    list_presets,
)

_FI_KEY = "ns_fi_result"
_PS_KEY = "ns_ps_result"
_CMP_KEY = "ns_cmp_result"

# Natural ranges and defaults for each sweepable LIF parameter
_SWEEP_CONFIG: dict[str, dict] = {
    "tau_m": {
        "label": "Membrane Time Constant τm (ms)",
        "min": 5.0, "max": 50.0, "default_min": 5.0, "default_max": 50.0, "step": 1.0,
    },
    "resistance": {
        "label": "Membrane Resistance",
        "min": 1.0, "max": 20.0, "default_min": 1.0, "default_max": 20.0, "step": 0.5,
    },
    "v_threshold": {
        "label": "Spike Threshold (mV)",
        "min": -60.0, "max": -40.0, "default_min": -60.0, "default_max": -40.0, "step": 1.0,
    },
}


def render() -> None:
    render_page_header(
        "🔬", "NeuroSim",
        "Interactive Leaky Integrate-and-Fire neuron simulator",
    )

    with st.sidebar:
        st.markdown("**Neuron Parameters**")
        current = st.slider("Input Current", 0.0, 5.0, 2.0, 0.1)
        tau_m = st.slider("Membrane Time Constant τm (ms)", 5.0, 50.0, 20.0, 1.0)
        resistance = st.slider("Membrane Resistance", 1.0, 20.0, 10.0, 0.5)
        v_threshold = st.slider("Spike Threshold (mV)", -60.0, -40.0, -50.0, 1.0)
        refractory_period = st.slider("Refractory Period (ms)", 0.5, 10.0, 2.0, 0.5)

        st.divider()
        st.markdown("**Simulation Settings**")
        t_max = st.slider("Simulation Time (ms)", 100.0, 1000.0, 500.0, 50.0)
        dt = st.selectbox("Time Step (ms)", [0.05, 0.1, 0.2, 0.5], index=1)

        st.divider()
        st.markdown("**F-I Curve**")
        fi_i_min = st.slider("Min Current", 0.0, 5.0, 0.0, 0.1, key="fi_i_min")
        fi_i_max = st.slider("Max Current", 0.0, 10.0, 5.0, 0.1, key="fi_i_max")
        fi_steps = st.slider("Steps", 5, 100, 30, 5, key="fi_steps")

    neuron = LIFNeuron(
        tau_m=tau_m,
        resistance=resistance,
        v_threshold=v_threshold,
        refractory_period=refractory_period,
    )

    result = neuron.simulate(current=current, t_max=t_max, dt=dt)

    chart_col, stats_col = st.columns([3, 1])

    with chart_col:
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=result.time,
            y=result.voltage,
            mode="lines",
            name="Membrane Potential",
            line=dict(color="#1f77b4", width=1.5),
        ))

        fig.add_hline(
            y=v_threshold,
            line_dash="dash", line_color="#ff7f0e", line_width=1.5,
            annotation_text="Threshold", annotation_position="top right",
            annotation_font_size=11, annotation_font_color="#ff7f0e",
        )

        fig.add_hline(
            y=neuron.v_rest,
            line_dash="dot", line_color="#2ca02c", line_width=1.5,
            annotation_text="V_rest", annotation_position="bottom right",
            annotation_font_size=11, annotation_font_color="#2ca02c",
        )

        if result.spike_count > 0:
            fig.add_trace(go.Scatter(
                x=result.spike_times,
                y=[neuron.spike_voltage] * result.spike_count,
                mode="markers",
                name="Spikes",
                marker=dict(size=10, symbol="triangle-up", color="#d62728"),
            ))

        fig.update_layout(
            title="Membrane Potential Over Time",
            xaxis_title="Time (ms)",
            yaxis_title="Voltage (mV)",
            height=480,
            hovermode="x unified",
            plot_bgcolor="#f9f9f9",
            legend=dict(
                x=1, y=1, xanchor="right", yanchor="top",
                bgcolor="rgba(255,255,255,0.75)",
                bordercolor="lightgrey", borderwidth=1,
            ),
            margin=dict(l=60, r=40, t=50, b=50),
        )

        fig.update_yaxes(range=[neuron.v_rest - 10, neuron.spike_voltage + 10])
        st.plotly_chart(fig, use_container_width=True)

    with stats_col:
        st.markdown("#### Output")
        st.metric("Spike Count", result.spike_count)
        st.metric("Firing Rate", f"{result.firing_rate:.2f} Hz")

        st.divider()
        st.markdown("#### Interpretation")
        if result.spike_count == 0:
            st.info(
                "Current is too low to reach threshold — no spikes generated.",
                icon="💤",
            )
        elif result.firing_rate < 50:
            st.success(
                "Moderate firing rate. Try increasing current or lowering "
                "the threshold to speed it up.",
                icon="📈",
            )
        else:
            st.warning(
                "High firing rate — the neuron is saturating.",
                icon="⚡",
            )

    st.divider()
    st.markdown("#### F-I Curve Experiment")
    st.markdown(
        "Sweep input current from **I_min** to **I_max** and measure the "
        "steady-state firing rate. Parameters match the neuron configured above."
    )

    if st.button("▶ Run F-I Sweep", key="fi_run"):
        if fi_i_min >= fi_i_max:
            st.warning("Min Current must be less than Max Current.", icon="⚠️")
        else:
            currents = np.linspace(fi_i_min, fi_i_max, fi_steps)
            with st.spinner("Running sweep…"):
                st.session_state[_FI_KEY] = compute_fi_curve(
                    neuron, currents, t_max=t_max, dt=dt
                )

    fi_df = st.session_state.get(_FI_KEY)
    if fi_df is not None:
        fi_fig = go.Figure()
        fi_fig.add_trace(go.Scatter(
            x=fi_df["current"],
            y=fi_df["firing_rate"],
            mode="lines+markers",
            name="Firing Rate",
            line=dict(color="#6366f1", width=2),
            marker=dict(size=5),
        ))

        # Mark the currently selected current with a vertical line
        fi_fig.add_vline(
            x=current,
            line_dash="dash", line_color="#ff7f0e", line_width=1.5,
            annotation_text=f"Current = {current}",
            annotation_position="top right",
            annotation_font_size=11,
            annotation_font_color="#ff7f0e",
        )

        fi_fig.update_layout(
            title="F-I Curve — Firing Rate vs. Input Current",
            xaxis_title="Input Current",
            yaxis_title="Firing Rate (Hz)",
            height=380,
            hovermode="x unified",
            plot_bgcolor="#f9f9f9",
            margin=dict(l=60, r=40, t=50, b=50),
        )
        st.plotly_chart(fi_fig, use_container_width=True)

        if st.checkbox("Show data table", key="fi_show_table"):
            st.dataframe(
                fi_df.style.format({"current": "{:.2f}", "firing_rate": "{:.1f} Hz",
                                    "spike_count": "{:.0f}"}),
                use_container_width=True,
            )

    st.divider()
    st.markdown("#### Parameter Sweep Experiment")
    st.markdown(
        "A parameter sweep holds all neuron settings fixed except one, then "
        "runs a separate simulation for each value in a chosen range. "
        "This reveals how sensitive the firing rate is to a single biophysical "
        "parameter — useful for understanding model behaviour and tuning."
    )

    with st.expander("⚙️ Sweep settings"):
        ps_col1, ps_col2 = st.columns([1, 2])
        with ps_col1:
            ps_param = st.selectbox(
                "Parameter to sweep",
                options=list(_SWEEP_CONFIG.keys()),
                format_func=lambda k: _SWEEP_CONFIG[k]["label"],
                key="ps_param",
            )
        cfg = _SWEEP_CONFIG[ps_param]
        with ps_col2:
            ps_range_col1, ps_range_col2, ps_steps_col = st.columns(3)
            with ps_range_col1:
                ps_min = st.slider(
                    "Min value",
                    min_value=cfg["min"], max_value=cfg["max"],
                    value=cfg["default_min"], step=cfg["step"],
                    key=f"ps_min_{ps_param}",
                )
            with ps_range_col2:
                ps_max = st.slider(
                    "Max value",
                    min_value=cfg["min"], max_value=cfg["max"],
                    value=cfg["default_max"], step=cfg["step"],
                    key=f"ps_max_{ps_param}",
                )
            with ps_steps_col:
                ps_steps = st.slider(
                    "Steps", min_value=5, max_value=100, value=30, step=5,
                    key="ps_steps",
                )

    if st.button("▶ Run Parameter Sweep", key="ps_run"):
        if ps_min >= ps_max:
            st.warning("Min value must be less than Max value.", icon="⚠️")
        else:
            values = np.linspace(ps_min, ps_max, ps_steps)
            with st.spinner(f"Sweeping {ps_param}…"):
                ps_df = parameter_sweep(
                    neuron, ps_param, values, current=current, t_max=t_max, dt=dt,
                )
            st.session_state[_PS_KEY] = {"df": ps_df, "param": ps_param}

    ps_state = st.session_state.get(_PS_KEY)
    if ps_state is not None:
        ps_df = ps_state["df"]
        ps_param_name = ps_state["param"]
        ps_label = _SWEEP_CONFIG[ps_param_name]["label"]

        ps_fig = go.Figure()
        ps_fig.add_trace(go.Scatter(
            x=ps_df["parameter_value"],
            y=ps_df["firing_rate"],
            mode="lines+markers",
            name="Firing Rate",
            line=dict(color="#6366f1", width=2),
            marker=dict(size=5),
        ))

        # Mark the current sidebar value of the swept parameter
        current_param_value = getattr(neuron, ps_param_name, None)
        if current_param_value is not None:
            ps_fig.add_vline(
                x=current_param_value,
                line_dash="dash", line_color="#ff7f0e", line_width=1.5,
                annotation_text=f"Current: {current_param_value}",
                annotation_position="top right",
                annotation_font_size=11,
                annotation_font_color="#ff7f0e",
            )

        ps_fig.update_layout(
            title=f"Firing Rate vs. {ps_label}",
            xaxis_title=ps_label,
            yaxis_title="Firing Rate (Hz)",
            height=360,
            hovermode="x unified",
            plot_bgcolor="#f9f9f9",
            margin=dict(l=60, r=40, t=50, b=50),
        )
        st.plotly_chart(ps_fig, use_container_width=True)

        if st.checkbox("Show data table", key="ps_show_table"):
            st.dataframe(
                ps_df.style.format({
                    "parameter_value": "{:.3f}",
                    "firing_rate": "{:.1f}",
                    "spike_count": "{:.0f}",
                }),
                use_container_width=True,
            )

    st.divider()
    st.markdown("#### Neuron Model Comparison")
    st.markdown(
        "Run **LIF** and **Izhikevich** side-by-side under identical conditions "
        "to see how model choice affects spiking dynamics."
    )

    st.info(
        "**Current scale note:** LIF and Izhikevich use different current conventions. "
        "LIF typically fires around I = 2–5, while Izhikevich (regular spiking) needs "
        "I ≈ 8–15 to produce visible spiking. Try **I = 10** to see both models fire "
        "simultaneously — then compare spike counts and waveform shape.",
        icon="ℹ️",
    )

    with st.expander("⚙️ Comparison settings"):
        cmp_col1, cmp_col2, cmp_col3 = st.columns(3)
        with cmp_col1:
            cmp_current = st.slider(
                "Input Current", 0.0, 20.0, 10.0, 0.5, key="cmp_current",
            )
        with cmp_col2:
            cmp_t_max = st.slider(
                "Duration (ms)", 100.0, 1000.0, 500.0, 50.0, key="cmp_t_max",
            )
        with cmp_col3:
            cmp_dt = st.selectbox(
                "Time Step (ms)", [0.05, 0.1, 0.2, 0.5], index=1, key="cmp_dt",
            )

    if st.button("▶ Run Comparison", key="cmp_run"):
        neurons = {
            "LIF": LIFNeuron(
                tau_m=tau_m,
                resistance=resistance,
                v_threshold=v_threshold,
                refractory_period=refractory_period,
            ),
            "Izhikevich (RS)": IzhikevichNeuron(),
        }
        with st.spinner("Simulating both models…"):
            cmp_results, cmp_summary = compare_neuron_models(
                neurons, current=cmp_current, t_max=cmp_t_max, dt=cmp_dt,
            )
        st.session_state[_CMP_KEY] = {
            "results": cmp_results,
            "summary": cmp_summary,
            "current": cmp_current,
            "t_max": cmp_t_max,
        }

    cmp_state = st.session_state.get(_CMP_KEY)
    if cmp_state is not None:
        cmp_results = cmp_state["results"]
        cmp_summary = cmp_state["summary"]

        _TRACE_COLORS = {"LIF": "#1f77b4", "Izhikevich (RS)": "#ff7f0e"}

        cmp_fig = go.Figure()
        for model_name, sim_result in cmp_results.items():
            cmp_fig.add_trace(go.Scatter(
                x=sim_result.time,
                y=sim_result.voltage,
                mode="lines",
                name=model_name,
                line=dict(color=_TRACE_COLORS.get(model_name, "#888"), width=1.5),
            ))

        cmp_fig.update_layout(
            title=f"Voltage Traces — I = {cmp_state['current']}, "
                  f"T = {cmp_state['t_max']} ms",
            xaxis_title="Time (ms)",
            yaxis_title="Voltage (mV)",
            height=400,
            hovermode="x unified",
            plot_bgcolor="#f9f9f9",
            legend=dict(
                x=1, y=1, xanchor="right", yanchor="top",
                bgcolor="rgba(255,255,255,0.75)",
                bordercolor="lightgrey", borderwidth=1,
            ),
            margin=dict(l=60, r=40, t=50, b=50),
        )
        st.plotly_chart(cmp_fig, use_container_width=True)

        st.markdown("**Summary**")
        st.dataframe(
            cmp_summary.style.format(
                {"spike_count": "{:.0f}", "firing_rate": "{:.1f}"}
            ).set_properties(**{"text-align": "center"}),
            use_container_width=True,
            hide_index=True,
        )

    st.divider()
    st.markdown("#### Model Equation")
    st.markdown(
        r"""
        The membrane voltage evolves according to:

        $$
        \tau_m \frac{dV}{dt} = -(V - V_{\text{rest}}) + R \cdot I
        $$

        When $V$ reaches the threshold the neuron emits a spike, voltage is
        set to $V_{\text{spike}}$, then reset to $V_{\text{reset}}$ for the
        duration of the refractory period.
        """
    )

    # ── Izhikevich Preset Explorer ──────────────────────────────────────────
    st.divider()
    st.markdown("#### Izhikevich Preset Explorer")
    st.markdown(
        "Select a published parameter set from Izhikevich (2003) to simulate "
        "the corresponding cortical firing pattern."
    )
    st.caption(
        "Source: Izhikevich, E.M. (2003). Simple model of spiking neurons. "
        "*IEEE Transactions on Neural Networks*, 14(6), 1569–1572. "
        "https://doi.org/10.1109/TNN.2003.820440"
    )

    izh_col_sel, izh_col_cur, izh_col_dur = st.columns([2, 1, 1])
    with izh_col_sel:
        izh_preset_name = st.selectbox(
            "Firing pattern preset",
            options=list_presets(),
            key="izh_preset_name",
        )
    with izh_col_cur:
        izh_current = st.slider(
            "Input Current", 0.0, 20.0, 10.0, 0.5, key="izh_current",
        )
    with izh_col_dur:
        izh_t_max = st.slider(
            "Duration (ms)", 100.0, 1000.0, 500.0, 50.0, key="izh_t_max",
        )

    preset = get_izhikevich_preset(izh_preset_name)

    # Display preset parameters as read-only metrics
    pm1, pm2, pm3, pm4 = st.columns(4)
    pm1.metric("a", preset.a)
    pm2.metric("b", preset.b)
    pm3.metric("c (mV)", preset.c)
    pm4.metric("d", preset.d)
    st.caption(f"_{preset.description}_")

    # Simulate immediately on any control change (no button needed —
    # the preset + current are fully determined by the widgets above)
    izh_neuron = IzhikevichNeuron(a=preset.a, b=preset.b, c=preset.c, d=preset.d)
    izh_result = izh_neuron.simulate(izh_current, izh_t_max, 0.1)

    izh_fig = go.Figure()
    izh_fig.add_trace(go.Scatter(
        x=izh_result.time,
        y=izh_result.voltage,
        mode="lines",
        name=izh_preset_name,
        line=dict(color="#6366f1", width=1.5),
    ))
    izh_fig.add_hline(
        y=30.0,
        line_dash="dash", line_color="#ff7f0e", line_width=1.2,
        annotation_text="Spike threshold (30 mV)",
        annotation_position="top right",
        annotation_font_size=11,
        annotation_font_color="#ff7f0e",
    )
    izh_fig.update_layout(
        title=f"{izh_preset_name} — I = {izh_current}, T = {izh_t_max} ms",
        xaxis_title="Time (ms)",
        yaxis_title="Membrane Potential v (mV)",
        height=400,
        hovermode="x unified",
        plot_bgcolor="#f9f9f9",
        margin=dict(l=60, r=40, t=50, b=50),
    )
    st.plotly_chart(izh_fig, use_container_width=True)

    izh_out1, izh_out2 = st.columns(2)
    izh_out1.metric("Spike Count", int(len(izh_result.spike_times)))
    izh_out2.metric(
        "Firing Rate",
        f"{len(izh_result.spike_times) / (izh_t_max / 1000.0):.1f} Hz",
    )


render()
