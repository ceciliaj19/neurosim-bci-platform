"""Scientific validation tests for the Izhikevich neuron model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from neurosim.neurons import IzhikevichNeuron


@dataclass(frozen=True)
class _TestResult:
    test: str
    expected_behavior: str
    observed_behavior: str
    passed: bool


def validate_izhikevich() -> pd.DataFrame:
    """Run scientific validation tests for the Izhikevich neuron model.

    Creates an :class:`~neurosim.neurons.IzhikevichNeuron` with default
    regular-spiking (RS) parameters and verifies that it exhibits the
    expected biophysical behaviour across a set of named tests.

    Returns
    -------
    pandas.DataFrame
        One row per validation test with four columns:

        ``test``
            Short name identifying the validation test.
        ``expected_behavior``
            Prose description of what a correct Izhikevich model should do.
        ``observed_behavior``
            Summary of the measured result from the actual simulation.
        ``passed``
            ``True`` if the observed behaviour matches the expectation.

    Notes
    -----
    All tests use an :class:`~neurosim.neurons.IzhikevichNeuron` constructed
    with default regular-spiking parameters (``a=0.02``, ``b=0.2``,
    ``c=-65.0``, ``d=8.0``, ``v0=-65.0``).  Simulation duration is 500 ms
    with ``dt=0.1``.

    The Izhikevich model requires a larger injected current than the LIF
    model to produce spiking.  The firing-rate sweep uses currents in the
    range ``[0, 5, 10, 15]``, where rheobase lies between 5 and 10 for
    the default RS parameters.

    Examples
    --------
    >>> from neurosim.validation import validate_izhikevich
    >>> df = validate_izhikevich()
    >>> df.columns.tolist()
    ['test', 'expected_behavior', 'observed_behavior', 'passed']
    >>> df["passed"].all()
    True
    """
    neuron = IzhikevichNeuron()
    t_max = 500.0
    dt = 0.1
    results: list[_TestResult] = []

    # ------------------------------------------------------------------
    # Test 1: Regular spiking parameters produce spikes
    # ------------------------------------------------------------------
    rs_result = neuron.simulate(10.0, t_max, dt)
    rs_spike_count = int(len(rs_result.spike_times))
    rs_fires = rs_spike_count > 0
    results.append(_TestResult(
        test="Regular spiking parameters produce spikes",
        expected_behavior=(
            "Default RS parameters (a=0.02, b=0.2, c=-65, d=8) should "
            "produce repetitive spiking under a suprathreshold current of I=10."
        ),
        observed_behavior=(
            f"At I=10.0, t_max={t_max} ms: {rs_spike_count} spike(s) detected — "
            f"{'spikes present ✓' if rs_fires else 'no spikes ✗'}."
        ),
        passed=rs_fires,
    ))

    # ------------------------------------------------------------------
    # Test 2: Firing rate increases with larger injected current
    # ------------------------------------------------------------------
    sweep_currents = [0.0, 5.0, 10.0, 15.0]
    rates = [
        len(neuron.simulate(c, t_max, dt).spike_times) / (t_max / 1000.0)
        for c in sweep_currents
    ]
    is_nondecreasing = all(rates[i] <= rates[i + 1] for i in range(len(rates) - 1))
    results.append(_TestResult(
        test="Firing rate increases with larger injected current",
        expected_behavior=(
            "Firing rate should be non-decreasing as current rises from 0 to 15 "
            "(the model must be in its regular-spiking regime, not bursting)."
        ),
        observed_behavior=(
            f"Rates at I={sweep_currents}: "
            f"{[f'{r:.1f}' for r in rates]} Hz — "
            f"{'non-decreasing ✓' if is_nondecreasing else 'NOT non-decreasing ✗'}."
        ),
        passed=is_nondecreasing,
    ))

    # ------------------------------------------------------------------
    # Test 3: Voltage trace contains no NaN values
    # ------------------------------------------------------------------
    voltage = neuron.simulate(10.0, t_max, dt).voltage
    has_nan = bool(np.any(np.isnan(voltage)))
    results.append(_TestResult(
        test="Voltage trace contains no NaN values",
        expected_behavior=(
            "The membrane potential trace must be finite at every time step. "
            "NaN propagation indicates numerical blow-up of the quadratic ODE."
        ),
        observed_behavior=(
            f"Trace length {len(voltage)} at I=10.0 — "
            f"{'no NaN values found ✓' if not has_nan else 'NaN values detected ✗'}."
        ),
        passed=not has_nan,
    ))

    return pd.DataFrame(
        [
            {
                "test": r.test,
                "expected_behavior": r.expected_behavior,
                "observed_behavior": r.observed_behavior,
                "passed": r.passed,
            }
            for r in results
        ],
        columns=["test", "expected_behavior", "observed_behavior", "passed"],
    )
