from __future__ import annotations

from pathlib import Path
from typing import Union

from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.figure


@dataclass
class SimulationResult:
    """Container for the output of a single neuron simulation run.

    Parameters
    ----------
    time : np.ndarray
        Time vector in milliseconds, shape ``(n_samples,)``.
    voltage : np.ndarray
        Membrane potential trace in mV, shape ``(n_samples,)``.
    spike_times : np.ndarray
        Times at which spikes occurred in milliseconds.
    model_name : str
        Human-readable name of the neuron model that produced this result.
    parameters : dict
        Snapshot of the model and simulation parameters used for this run.

    Attributes
    ----------
    spike_count : int
        Number of spikes (read-only property).
    firing_rate : float
        Mean firing rate in Hz (read-only property).
    """

    time: np.ndarray
    voltage: np.ndarray
    spike_times: np.ndarray
    model_name: str
    parameters: dict

    # ------------------------------------------------------------------
    # Existing properties — unchanged
    # ------------------------------------------------------------------

    @property
    def spike_count(self) -> int:
        """Number of spikes in the simulation.

        Returns
        -------
        int
            Length of ``spike_times``.
        """
        return len(self.spike_times)

    @property
    def firing_rate(self) -> float:
        """Mean firing rate over the simulation duration.

        Returns
        -------
        float
            Spikes per second (Hz). Returns ``0.0`` if the duration is
            zero or negative.
        """
        duration_seconds = (self.time[-1] - self.time[0]) / 1000.0
        if duration_seconds <= 0:
            return 0.0
        return self.spike_count / duration_seconds

    # ------------------------------------------------------------------
    # Existing methods — unchanged
    # ------------------------------------------------------------------

    def inter_spike_intervals(self) -> np.ndarray:
        """Compute inter-spike intervals (ISIs).

        Returns
        -------
        np.ndarray
            Differences between consecutive spike times in milliseconds.
            Empty array when fewer than two spikes exist.
        """
        return np.diff(self.spike_times)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the voltage trace to a tidy DataFrame.

        Returns
        -------
        pd.DataFrame
            Two-column frame with ``time_ms`` and ``voltage_mv``.
        """
        return pd.DataFrame(
            {
                "time_ms": self.time,
                "voltage_mv": self.voltage,
            }
        )

    # ------------------------------------------------------------------
    # New methods
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a formatted summary of key simulation metrics.

        Returns
        -------
        str
            Multiline string containing model name, duration, spike count,
            firing rate, and number of time samples.

        Examples
        --------
        >>> print(result.summary())
        Model           : Leaky Integrate-and-Fire
        Duration        : 500.00 ms
        Samples         : 5000
        Spike count     : 16
        Firing rate     : 32.06 Hz
        """
        duration_ms = float(self.time[-1] - self.time[0])
        lines = [
            f"{'Model':<16}: {self.model_name}",
            f"{'Duration':<16}: {duration_ms:.2f} ms",
            f"{'Samples':<16}: {len(self.time)}",
            f"{'Spike count':<16}: {self.spike_count}",
            f"{'Firing rate':<16}: {self.firing_rate:.2f} Hz",
        ]
        return "\n".join(lines)

    def export_csv(self, path: Union[str, Path]) -> Path:
        """Save the voltage trace to a CSV file.

        Parameters
        ----------
        path : str or Path
            Destination file path. Parent directories must already exist.

        Returns
        -------
        Path
            Resolved absolute path of the written file.

        Examples
        --------
        >>> result.export_csv("output/trace.csv")
        PosixPath('/abs/path/output/trace.csv')
        """
        resolved = Path(path).resolve()
        self.to_dataframe().to_csv(resolved, index=False)
        return resolved

    def plot_voltage(
        self,
        show: bool = True,
    ) -> matplotlib.figure.Figure:
        """Plot the membrane potential trace.

        Overlays dashed reference lines for spike threshold and resting
        potential when those keys are present in ``parameters``.

        Parameters
        ----------
        show : bool, optional
            Call ``plt.show()`` before returning. Set to ``False`` when
            embedding in notebooks or saving programmatically. Default ``True``.

        Returns
        -------
        matplotlib.figure.Figure
            The figure object, so callers can save or further customise it.

        Examples
        --------
        >>> fig = result.plot_voltage(show=False)
        >>> fig.savefig("voltage.png", dpi=150)
        """
        fig, ax = plt.subplots(figsize=(12, 4))

        ax.plot(self.time, self.voltage, color="#1f77b4", linewidth=1.2,
                label="Membrane potential")

        v_threshold = self.parameters.get("v_threshold")
        if v_threshold is not None:
            ax.axhline(v_threshold, linestyle="--", color="#ff7f0e",
                       linewidth=1.2, label=f"Threshold ({v_threshold} mV)")

        v_rest = self.parameters.get("v_rest")
        if v_rest is not None:
            ax.axhline(v_rest, linestyle=":", color="#2ca02c",
                       linewidth=1.2, label=f"V_rest ({v_rest} mV)")

        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Membrane potential (mV)")
        ax.set_title(f"{self.model_name} — Voltage Trace")
        ax.legend(loc="upper right", fontsize=9)
        fig.tight_layout()

        if show:
            plt.show()

        return fig

    def plot_spikes(
        self,
        show: bool = True,
    ) -> matplotlib.figure.Figure:
        """Plot a spike raster for the simulation.

        Each spike is rendered as a vertical tick via
        ``matplotlib.axes.Axes.eventplot``.

        Parameters
        ----------
        show : bool, optional
            Call ``plt.show()`` before returning. Default ``True``.

        Returns
        -------
        matplotlib.figure.Figure
            The figure object.

        Examples
        --------
        >>> fig = result.plot_spikes(show=False)
        >>> fig.savefig("raster.png", dpi=150)
        """
        fig, ax = plt.subplots(figsize=(12, 2))

        if self.spike_count > 0:
            ax.eventplot(
                self.spike_times,
                orientation="horizontal",
                lineoffsets=0,
                linelengths=0.8,
                colors="#d62728",
            )
        else:
            ax.text(
                0.5, 0.5, "No spikes",
                ha="center", va="center",
                transform=ax.transAxes,
                color="grey",
            )

        ax.set_xlim(self.time[0], self.time[-1])
        ax.set_yticks([])
        ax.set_xlabel("Time (ms)")
        ax.set_title(
            f"{self.model_name} — Spike Raster  "
            f"({self.spike_count} spikes, {self.firing_rate:.1f} Hz)"
        )
        fig.tight_layout()

        if show:
            plt.show()

        return fig
