"""Leaky Integrate-and-Fire neuron model."""

import numpy as np
from numpy.typing import NDArray

from neurosim.core import BaseNeuron, SimulationResult


class LIFNeuron(BaseNeuron):
    """Leaky Integrate-and-Fire (LIF) neuron model.

    Approximates membrane voltage dynamics with the equation:

        τ_m dV/dt = -(V - V_rest) + R * I

    When voltage reaches ``v_threshold`` a spike is recorded, voltage is
    clamped to ``spike_voltage`` for one step, then reset to ``v_reset``
    for ``refractory_period`` milliseconds.

    Parameters
    ----------
    tau_m : float, optional
        Membrane time constant in milliseconds. Default ``20.0``.
    resistance : float, optional
        Membrane resistance in MΩ. Default ``10.0``.
    v_rest : float, optional
        Resting membrane potential in mV. Default ``-65.0``.
    v_reset : float, optional
        Post-spike reset potential in mV. Default ``-65.0``.
    v_threshold : float, optional
        Spike threshold in mV. Default ``-50.0``.
    spike_voltage : float, optional
        Voltage assigned at the moment of a spike in mV. Default ``30.0``.
    refractory_period : float, optional
        Absolute refractory period in milliseconds. Default ``2.0``.
    dt : float, optional
        Default simulation time step in milliseconds, used by
        :class:`~neurosim.core.BaseNeuron` utilities. Default ``0.1``.

    Attributes
    ----------
    tau_m : float
    resistance : float
    v_rest : float
    v_reset : float
    v_threshold : float
    spike_voltage : float
    refractory_period : float
    dt : float
        Inherited from :class:`~neurosim.core.BaseNeuron`.
    """

    def __init__(
        self,
        tau_m: float = 20.0,
        resistance: float = 10.0,
        v_rest: float = -65.0,
        v_reset: float = -65.0,
        v_threshold: float = -50.0,
        spike_voltage: float = 30.0,
        refractory_period: float = 2.0,
        dt: float = 0.1,
    ) -> None:
        super().__init__(dt)
        self.tau_m = tau_m
        self.resistance = resistance
        self.v_rest = v_rest
        self.v_reset = v_reset
        self.v_threshold = v_threshold
        self.spike_voltage = spike_voltage
        self.refractory_period = refractory_period

    def simulate(
        self,
        current: float = 2.0,
        t_max: float = 500.0,
        dt: float = 0.1,
    ) -> SimulationResult:
        """Simulate the LIF neuron under a constant input current.

        Parameters
        ----------
        current : float, optional
            Constant external input current in arbitrary units. Default ``2.0``.
        t_max : float, optional
            Total simulation duration in milliseconds. Default ``500.0``.
        dt : float, optional
            Simulation time step in milliseconds. Default ``0.1``.

        Returns
        -------
        SimulationResult
            Container holding the time vector, voltage trace, spike times,
            model name, and parameter snapshot.
        """
        time = np.arange(0, t_max, dt)
        voltage = np.zeros_like(time)
        voltage[0] = self.v_rest

        spike_times: list[float] = []
        refractory_steps = int(self.refractory_period / dt)
        refractory_counter = 0

        for i in range(1, len(time)):
            if refractory_counter > 0:
                voltage[i] = self.v_reset
                refractory_counter -= 1
                continue

            dv = (-(voltage[i - 1] - self.v_rest) + self.resistance * current) / self.tau_m
            voltage[i] = voltage[i - 1] + dv * dt

            if voltage[i] >= self.v_threshold:
                voltage[i] = self.spike_voltage
                spike_times.append(time[i])
                refractory_counter = refractory_steps

        spike_times_arr = np.array(spike_times)

        return SimulationResult(
            time=time,
            voltage=voltage,
            spike_times=spike_times_arr,
            model_name="Leaky Integrate-and-Fire",
            parameters={
                "tau_m": self.tau_m,
                "resistance": self.resistance,
                "v_rest": self.v_rest,
                "v_reset": self.v_reset,
                "v_threshold": self.v_threshold,
                "spike_voltage": self.spike_voltage,
                "refractory_period": self.refractory_period,
                "current": current,
                "t_max": t_max,
                "dt": dt,
            },
        )
