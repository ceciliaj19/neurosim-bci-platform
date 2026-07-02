"""Izhikevich neuron model (Izhikevich 2003)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from neurosim.core import BaseNeuron, SimulationResult


class IzhikevichNeuron(BaseNeuron):
    """Two-variable spiking neuron model (Izhikevich 2003).

    Reproduces a wide range of cortical firing patterns using the equations::

        dv/dt = 0.04 v² + 5v + 140 - u + I
        du/dt = a (b v - u)

    A spike is emitted when ``v >= 30`` mV, after which the state is reset::

        v ← c
        u ← u + d

    Parameters
    ----------
    a : float
        Time scale of the recovery variable ``u``.  Smaller values give
        slower recovery (default ``0.02``).
    b : float
        Sensitivity of ``u`` to subthreshold membrane oscillations
        (default ``0.2``).
    c : float
        After-spike reset value of ``v`` in mV (default ``-65.0``).
    d : float
        After-spike increment applied to ``u`` (default ``8.0``).
    v0 : float
        Initial membrane potential in mV (default ``-65.0``).
    dt : float
        Default simulation time step in ms (default ``0.1``).

    Notes
    -----
    Default parameters (a=0.02, b=0.2, c=-65, d=8) produce **regular
    spiking** behaviour typical of excitatory cortical pyramidal neurons.

    Common parameter presets:

    ==================  ======  =====  =====  ===
    Firing pattern      a       b      c      d
    ==================  ======  =====  =====  ===
    Regular spiking     0.02    0.2    -65    8
    Fast spiking        0.10    0.2    -65    2
    Intrinsically burst 0.02    0.2    -55    4
    Chattering          0.02    0.2    -50    2
    ==================  ======  =====  =====  ===

    References
    ----------
    Izhikevich, E.M. (2003). Simple model of spiking neurons.
    *IEEE Transactions on Neural Networks*, 14(6), 1569–1572.
    """

    def __init__(
        self,
        a: float = 0.02,
        b: float = 0.2,
        c: float = -65.0,
        d: float = 8.0,
        v0: float = -65.0,
        dt: float = 0.1,
    ) -> None:
        super().__init__(dt)
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.v0 = v0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def simulate(
        self,
        current: float = 10.0,
        duration: float = 500.0,
        dt: float | None = None,
    ) -> SimulationResult:
        """Run a fixed-current simulation and return the result.

        Parameters
        ----------
        current : float
            Constant injected current ``I`` in arbitrary units
            (default ``10.0``).
        duration : float
            Total simulation time in ms (default ``500.0``).
        dt : float or None
            Time step in ms.  Defaults to ``self.dt`` when *None*.

        Returns
        -------
        SimulationResult
            Container holding the time vector, voltage trace, spike times,
            and a snapshot of the parameters used.

        Examples
        --------
        >>> neuron = IzhikevichNeuron()
        >>> result = neuron.simulate(current=10.0, duration=200.0)
        >>> result.spike_times
        array([...])
        """
        step = dt if dt is not None else self.dt
        n = int(duration / step)

        time: NDArray[np.float64] = np.arange(n) * step
        voltage: NDArray[np.float64] = np.empty(n)
        spike_times: list[float] = []

        v = self.v0
        u = self.b * self.v0

        for i in range(n):
            if v >= 30.0:
                spike_times.append(time[i])
                # Store spike peak before reset so plots show action potentials
                voltage[i] = 30.0
                v = self.c
                u = u + self.d
            else:
                voltage[i] = v

            dv = 0.04 * v * v + 5.0 * v + 140.0 - u + current
            du = self.a * (self.b * v - u)
            v += step * dv
            u += step * du

        return SimulationResult(
            time=time,
            voltage=voltage,
            spike_times=np.asarray(spike_times),
            model_name="IzhikevichNeuron",
            parameters={
                "a": self.a,
                "b": self.b,
                "c": self.c,
                "d": self.d,
                "v0": self.v0,
                "current": current,
                "duration": duration,
                "dt": step,
                "v_threshold": 30.0,
                "v_rest": self.c,
            },
        )

    def __repr__(self) -> str:
        return (
            f"IzhikevichNeuron(a={self.a}, b={self.b}, "
            f"c={self.c}, d={self.d}, dt={self.dt})"
        )
