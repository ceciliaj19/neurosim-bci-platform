"""Generic single-parameter sweep analysis for neuron models."""

from __future__ import annotations

import copy
from typing import Sequence

import pandas as pd

from ._types import _Simulatable


def parameter_sweep(
    neuron: _Simulatable,
    parameter_name: str,
    values: Sequence[float],
    current: float = 2.0,
    t_max: float = 500.0,
    dt: float = 0.1,
) -> pd.DataFrame:
    """Sweep a single neuron parameter and measure the effect on firing.

    For each value in *values*, the function creates an independent copy of
    *neuron*, sets ``neuron.<parameter_name>`` to that value, runs a
    simulation with the given *current*, and records the resulting spike
    count and firing rate.  The original *neuron* object is never mutated.

    Parameters
    ----------
    neuron : _Simulatable
        Any object with a ``simulate(current, duration, dt)`` method that
        returns a :class:`~neurosim.core.SimulationResult`.  Must expose
        *parameter_name* as an instance attribute.
    parameter_name : str
        Name of the attribute to sweep.  Must already exist on *neuron*
        (e.g. ``"tau_m"``, ``"resistance"``, ``"a"``, ``"v_threshold"``).
    values : Sequence[float]
        Ordered sequence of values to assign to *parameter_name* in
        successive simulations.  Accepts a Python list, ``numpy.linspace``,
        or any iterable that converts to a list.
    current : float, optional
        Constant injected current for every simulation (default ``2.0``).
    t_max : float, optional
        Duration of each simulation in milliseconds (default ``500.0``).
    dt : float, optional
        Integration time step in milliseconds (default ``0.1``).

    Returns
    -------
    pandas.DataFrame
        One row per value in *values* with three columns:

        ``parameter_value``
            The value assigned to *parameter_name* for that run.
        ``spike_count``
            Number of spikes emitted during the simulation.
        ``firing_rate``
            Mean firing rate in Hz: ``spike_count / (t_max / 1000)``.

    Raises
    ------
    AttributeError
        If *parameter_name* is not an existing attribute of *neuron*.
    TypeError
        If *neuron* does not have a callable ``simulate`` attribute.
    ValueError
        If *values* is empty.

    Notes
    -----
    Each neuron clone is created with :func:`copy.copy` (shallow copy).
    This is safe for all built-in NeuroLab neuron models because their state
    consists entirely of scalar floats.  If a custom neuron stores mutable
    objects as attributes, pass a pre-deepcopied neuron or subclass and
    override ``__copy__``.

    ``simulate`` is called positionally as ``clone.simulate(current, t_max,
    dt)`` to remain compatible with models that name the duration argument
    differently (e.g. ``t_max`` in :class:`~neurosim.neurons.LIFNeuron` vs.
    ``duration`` in :class:`~neurosim.neurons.IzhikevichNeuron`).

    Examples
    --------
    Sweep membrane time constant for a LIF neuron:

    >>> import numpy as np
    >>> from neurosim.neurons import LIFNeuron
    >>> from neurosim.analysis import parameter_sweep
    >>> neuron = LIFNeuron()
    >>> tau_values = np.linspace(5.0, 50.0, 10)
    >>> df = parameter_sweep(neuron, "tau_m", tau_values, current=2.0)
    >>> df.columns.tolist()
    ['parameter_value', 'spike_count', 'firing_rate']

    Sweep the recovery time scale for an Izhikevich neuron:

    >>> from neurosim.neurons import IzhikevichNeuron
    >>> neuron = IzhikevichNeuron()
    >>> df = parameter_sweep(neuron, "a", [0.02, 0.05, 0.10], current=10.0)
    >>> len(df)
    3
    """
    if not isinstance(neuron, _Simulatable):
        raise TypeError(
            f"neuron must have a callable 'simulate' method, got {type(neuron)!r}"
        )

    if not hasattr(neuron, parameter_name):
        raise AttributeError(
            f"{type(neuron).__name__!r} has no attribute {parameter_name!r}. "
            f"Check the parameter name or use vars(neuron) to list available attributes."
        )

    values_list = list(values)
    if not values_list:
        raise ValueError("values must contain at least one entry")

    duration_s = t_max / 1000.0  # ms → s for Hz
    rows: list[dict[str, float]] = []

    for value in values_list:
        clone = copy.copy(neuron)
        setattr(clone, parameter_name, value)
        result = clone.simulate(current, t_max, dt)
        spike_count = int(len(result.spike_times))
        rows.append(
            {
                "parameter_value": float(value),
                "spike_count": float(spike_count),
                "firing_rate": spike_count / duration_s,
            }
        )

    return pd.DataFrame(rows, columns=["parameter_value", "spike_count", "firing_rate"])
