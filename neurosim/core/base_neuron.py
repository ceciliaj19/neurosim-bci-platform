"""Abstract base class for all neuron models in NeuroSim."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray


class BaseNeuron(ABC):
    """Abstract base class that every neuron model must subclass.

    Parameters
    ----------
    dt : float
        Simulation time step in seconds.

    Attributes
    ----------
    dt : float
        Simulation time step in seconds.
    """

    def __init__(self, dt: float) -> None:
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        self.dt: float = dt

    @abstractmethod
    def simulate(
        self,
        duration: float,
        I_ext: NDArray[np.floating[Any]],
    ) -> Any:
        """Run the neuron simulation for a given duration.

        Parameters
        ----------
        duration : float
            Total simulation time in seconds.
        I_ext : NDArray[np.floating]
            External input current array with one value per time step,
            shape ``(n_steps,)``.

        Returns
        -------
        Any
            Simulation output. Concrete subclasses define the exact type
            (e.g. ``SimulationResult``).

        Raises
        ------
        NotImplementedError
            Always, because this method is abstract.
        """

    def n_steps(self, duration: float) -> int:
        """Return the number of time steps for a given duration.

        Parameters
        ----------
        duration : float
            Total simulation time in seconds.

        Returns
        -------
        int
            Number of discrete time steps, ``round(duration / dt)``.
        """
        return round(duration / self.dt)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(dt={self.dt})"
