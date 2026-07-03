"""Shared structural types for the neurosim.analysis package."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from neurosim.core import SimulationResult


@runtime_checkable
class _Simulatable(Protocol):
    """Structural type for any neuron model used in analysis functions.

    Any object that exposes a ``simulate`` method whose first positional
    argument is *current* and that returns a
    :class:`~neurosim.core.SimulationResult` satisfies this protocol.
    No base class or explicit registration is required.
    """

    def simulate(
        self,
        current: float,
        /,
        *args: object,
        **kwargs: object,
    ) -> SimulationResult:
        """Run a single simulation and return the result."""
        ...
