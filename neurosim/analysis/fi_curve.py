"""F-I (firing rate vs. input current) curve analysis."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from neurosim.core import SimulationResult

from ._types import _Simulatable


def compute_fi_curve(
    neuron: _Simulatable,
    currents: Sequence[float],
    t_max: float = 500.0,
    dt: float = 0.1,
) -> pd.DataFrame:
    """Compute the firing rate vs. input current (F-I) curve for a neuron model.

    Runs one simulation per entry in *currents*, collects the spike count from
    each :class:`~neurosim.core.SimulationResult`, and returns the results as a
    tidy :class:`pandas.DataFrame` ready for plotting or further analysis.

    Parameters
    ----------
    neuron : _Simulatable
        Any object with a ``simulate(current, duration, dt)`` method that
        returns a :class:`~neurosim.core.SimulationResult`.  Both
        :class:`~neurosim.neurons.LIFNeuron` and
        :class:`~neurosim.neurons.IzhikevichNeuron` satisfy this interface.
    currents : Sequence[float]
        Sequence of input current values to sweep (arbitrary units matching
        the neuron model's convention).  Accepts a Python list,
        ``numpy.linspace``, or any other sequence.
    t_max : float, optional
        Duration of each simulation in milliseconds (default ``500.0``).
    dt : float, optional
        Integration time step in milliseconds (default ``0.1``).

    Returns
    -------
    pandas.DataFrame
        Tidy DataFrame with one row per current value and three columns:

        ``current``
            The injected current value (same units as *currents*).
        ``firing_rate``
            Mean firing rate in Hz (spikes per second): ``spike_count /
            (t_max / 1000)``.
        ``spike_count``
            Raw number of spikes detected during the simulation.

    Raises
    ------
    TypeError
        If *neuron* does not have a callable ``simulate`` attribute.
    ValueError
        If *currents* is empty.

    Notes
    -----
    ``simulate`` is called with positional arguments ``(current, t_max, dt)``
    to remain compatible with models that name the duration argument
    differently (e.g. ``t_max`` in :class:`~neurosim.neurons.LIFNeuron` vs.
    ``duration`` in :class:`~neurosim.neurons.IzhikevichNeuron`).

    Examples
    --------
    >>> import numpy as np
    >>> from neurosim.neurons import LIFNeuron
    >>> from neurosim.analysis import compute_fi_curve
    >>> neuron = LIFNeuron()
    >>> currents = np.linspace(0.0, 5.0, 20)
    >>> df = compute_fi_curve(neuron, currents, t_max=500.0)
    >>> df.columns.tolist()
    ['current', 'firing_rate', 'spike_count']
    >>> df[df["firing_rate"] > 0].head()
       current  firing_rate  spike_count
    ...
    """
    if not isinstance(neuron, _Simulatable):
        raise TypeError(
            f"neuron must have a callable 'simulate' method, got {type(neuron)!r}"
        )

    currents_seq = list(currents)
    if not currents_seq:
        raise ValueError("currents must contain at least one value")

    duration_s = t_max / 1000.0  # convert ms → s for Hz calculation

    rows: list[dict[str, float]] = []
    for c in currents_seq:
        result: SimulationResult = neuron.simulate(c, t_max, dt)
        spike_count = int(len(result.spike_times))
        firing_rate = spike_count / duration_s
        rows.append(
            {
                "current": float(c),
                "firing_rate": firing_rate,
                "spike_count": float(spike_count),
            }
        )

    return pd.DataFrame(rows, columns=["current", "firing_rate", "spike_count"])
