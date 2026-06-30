from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class SimulationResult:
    time: np.ndarray
    voltage: np.ndarray
    spike_times: np.ndarray
    model_name: str
    parameters: dict

    @property
    def spike_count(self) -> int:
        return len(self.spike_times)

    @property
    def firing_rate(self) -> float:
        duration_seconds = (self.time[-1] - self.time[0]) / 1000.0
        if duration_seconds <= 0:
            return 0.0
        return self.spike_count / duration_seconds

    def inter_spike_intervals(self) -> np.ndarray:
        return np.diff(self.spike_times)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "time_ms": self.time,
                "voltage_mv": self.voltage,
            }
        )