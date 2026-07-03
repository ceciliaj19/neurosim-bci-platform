"""Side-by-side neuron model comparison."""

from __future__ import annotations

import pandas as pd

from neurosim.core import SimulationResult

from ._types import _Simulatable


def compare_neuron_models(
    neurons: dict[str, _Simulatable],
    current: float = 2.0,
    t_max: float = 500.0,
    dt: float = 0.1,
) -> tuple[dict[str, SimulationResult], pd.DataFrame]:
    """Simulate multiple neuron models under identical conditions and summarise.

    Each model in *neurons* is run with the same *current*, *t_max*, and *dt*,
    producing a :class:`~neurosim.core.SimulationResult` per model.  The raw
    results are returned alongside a tidy :class:`pandas.DataFrame` summary so
    callers can plot voltage traces or compare aggregate statistics without
    re-simulating.

    Parameters
    ----------
    neurons : dict[str, _Simulatable]
        Mapping of human-readable model names to neuron objects.  Any object
        with a ``simulate(current, duration, dt)`` method returning a
        :class:`~neurosim.core.SimulationResult` is accepted.  Row order in
        the returned DataFrame follows insertion order.

        Example::

            {
                "LIF": LIFNeuron(),
                "Izhikevich": IzhikevichNeuron(),
            }

    current : float, optional
        Constant injected current applied to every model (default ``2.0``).
        Units are model-dependent — ensure the value is meaningful for all
        models in *neurons*.
    t_max : float, optional
        Duration of each simulation in milliseconds (default ``500.0``).
    dt : float, optional
        Integration time step in milliseconds (default ``0.1``).

    Returns
    -------
    results : dict[str, SimulationResult]
        Raw simulation results keyed by the same names as *neurons*.  Use
        these to plot voltage traces or access spike times per model.
    summary : pandas.DataFrame
        One row per model with three columns:

        ``model``
            Model name as supplied in *neurons*.
        ``spike_count``
            Number of spikes emitted during the simulation.
        ``firing_rate``
            Mean firing rate in Hz: ``spike_count / (t_max / 1000)``.

    Raises
    ------
    TypeError
        If any value in *neurons* does not have a callable ``simulate``
        attribute.
    ValueError
        If *neurons* is empty.

    Notes
    -----
    ``simulate`` is called positionally as ``neuron.simulate(current, t_max,
    dt)`` to remain compatible with models that name the duration argument
    differently (e.g. ``t_max`` in :class:`~neurosim.neurons.LIFNeuron` vs.
    ``duration`` in :class:`~neurosim.neurons.IzhikevichNeuron`).

    Examples
    --------
    >>> from neurosim.neurons import IzhikevichNeuron, LIFNeuron
    >>> from neurosim.analysis import compare_neuron_models
    >>> neurons = {"LIF": LIFNeuron(), "Izhikevich": IzhikevichNeuron()}
    >>> results, summary = compare_neuron_models(neurons, current=5.0, t_max=500.0)
    >>> summary.columns.tolist()
    ['model', 'spike_count', 'firing_rate']
    >>> set(results.keys()) == {"LIF", "Izhikevich"}
    True
    """
    if not neurons:
        raise ValueError("neurons must contain at least one model")

    for name, neuron in neurons.items():
        if not isinstance(neuron, _Simulatable):
            raise TypeError(
                f"neurons[{name!r}] must have a callable 'simulate' method, "
                f"got {type(neuron)!r}"
            )

    duration_s = t_max / 1000.0  # ms → s for Hz
    results: dict[str, SimulationResult] = {}
    rows: list[dict[str, object]] = []

    for name, neuron in neurons.items():
        result = neuron.simulate(current, t_max, dt)
        results[name] = result

        spike_count = int(len(result.spike_times))
        rows.append(
            {
                "model": name,
                "spike_count": spike_count,
                "firing_rate": spike_count / duration_s,
            }
        )

    summary = pd.DataFrame(rows, columns=["model", "spike_count", "firing_rate"])
    return results, summary
