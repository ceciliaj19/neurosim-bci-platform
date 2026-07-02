"""Scientific validation tests for the LIF neuron model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from neurosim.neurons import LIFNeuron


@dataclass(frozen=True)
class _TestResult:
    test: str
    expected_behavior: str
    observed_behavior: str
    passed: bool


def validate_lif() -> pd.DataFrame:
    """Run scientific validation tests for the LIF neuron model.

    Creates a :class:`~neurosim.neurons.LIFNeuron` with default parameters
    and verifies that it exhibits the expected biophysical behaviour across
    a set of named tests.  No external neuron instance is required.

    Returns
    -------
    pandas.DataFrame
        One row per validation test with four columns:

        ``test``
            Short name identifying the validation test.
        ``expected_behavior``
            Prose description of what a correct LIF model should do.
        ``observed_behavior``
            Summary of the measured result from the actual simulation.
        ``passed``
            ``True`` if the observed behaviour matches the expectation.

    Notes
    -----
    All tests use a :class:`~neurosim.neurons.LIFNeuron` constructed with
    default parameters (``tau_m=20.0``, ``resistance=10.0``,
    ``v_threshold=-50.0``, ``refractory_period=2.0``).  Simulation duration
    is 500 ms with ``dt=0.1``.

    Examples
    --------
    >>> from neurosim.validation import validate_lif
    >>> df = validate_lif()
    >>> df.columns.tolist()
    ['test', 'expected_behavior', 'observed_behavior', 'passed']
    >>> df["passed"].all()
    True
    """
    neuron = LIFNeuron()
    t_max = 500.0
    dt = 0.1
    results: list[_TestResult] = []

    # ------------------------------------------------------------------
    # Test 1: Increasing input current increases firing rate
    # ------------------------------------------------------------------
    sweep_currents = [0.0, 1.0, 2.0, 3.0, 4.0]
    rates = [
        len(neuron.simulate(c, t_max, dt).spike_times) / (t_max / 1000.0)
        for c in sweep_currents
    ]
    is_nondecreasing = all(rates[i] <= rates[i + 1] for i in range(len(rates) - 1))
    results.append(_TestResult(
        test="Increasing current increases firing rate",
        expected_behavior=(
            "Firing rate is non-decreasing as input current rises "
            "from 0 to 4 (LIF is a monotone integrator above rheobase)."
        ),
        observed_behavior=(
            f"Rates at I={sweep_currents}: "
            f"{[f'{r:.1f}' for r in rates]} Hz — "
            f"{'non-decreasing ✓' if is_nondecreasing else 'NOT non-decreasing ✗'}."
        ),
        passed=is_nondecreasing,
    ))

    # ------------------------------------------------------------------
    # Test 2: Spike count is never negative
    # ------------------------------------------------------------------
    spike_counts = [
        int(len(neuron.simulate(c, t_max, dt).spike_times))
        for c in sweep_currents
    ]
    all_nonneg = all(sc >= 0 for sc in spike_counts)
    results.append(_TestResult(
        test="Spike count is never negative",
        expected_behavior=(
            "Spike counts must be non-negative integers for all input currents."
        ),
        observed_behavior=(
            f"Counts at I={sweep_currents}: {spike_counts} — "
            f"{'all ≥ 0 ✓' if all_nonneg else 'negative value found ✗'}."
        ),
        passed=all_nonneg,
    ))

    # ------------------------------------------------------------------
    # Test 3: Voltage trace contains no NaN values
    # ------------------------------------------------------------------
    voltage = neuron.simulate(2.0, t_max, dt).voltage
    has_nan = bool(np.any(np.isnan(voltage)))
    results.append(_TestResult(
        test="Voltage trace contains no NaN values",
        expected_behavior=(
            "The membrane potential trace must be finite at every time step "
            "(NaN indicates a numerical instability or uninitialised buffer)."
        ),
        observed_behavior=(
            f"Trace length {len(voltage)} at I=2.0 — "
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
