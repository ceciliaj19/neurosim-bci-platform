"""Dataset loader utilities for NeuroSim."""

from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray

# Project root: neurosim/datasets/loader.py -> neurosim/datasets -> neurosim -> root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class DatasetLoader:
    """Locate and load NeuroSim datasets relative to the project root.

    All dataset paths are resolved relative to a *root* directory so no
    hardcoded absolute paths are ever used. The default root is inferred
    from the location of this module, making the loader work correctly
    whether the package is installed in editable mode or as a wheel.

    Parameters
    ----------
    root : Path or str, optional
        Project root directory. Defaults to the directory that contains
        the ``neurosim`` package. Override in tests to point at a fixture
        directory.

    Examples
    --------
    >>> loader = DatasetLoader()
    >>> X, y = loader.load_motor_imagery()
    >>> X.shape
    (4897, 64, 513)
    """

    def __init__(self, root: Optional[Path | str] = None) -> None:
        self.root: Path = Path(root).resolve() if root is not None else _PROJECT_ROOT

    # ------------------------------------------------------------------
    # Public dataset methods
    # ------------------------------------------------------------------

    def load_motor_imagery(self) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
        """Load the multi-subject motor imagery dataset.

        Reads ``data/motor_imagery/multisubject_motor_imagery.npz``
        relative to the project root and returns the EEG trial array and
        corresponding class labels.

        Returns
        -------
        X : NDArray[np.float32]
            EEG trials of shape ``(n_trials, n_channels, n_times)``.
        y : NDArray[np.int64]
            Integer class labels of shape ``(n_trials,)``.

        Raises
        ------
        FileNotFoundError
            If the ``.npz`` file cannot be found at the expected path.

        Examples
        --------
        >>> loader = DatasetLoader()
        >>> X, y = loader.load_motor_imagery()
        >>> X.shape, y.shape
        ((4897, 64, 513), (4897,))
        """
        data = self._load_npz(
            Path("data") / "motor_imagery" / "multisubject_motor_imagery.npz"
        )
        return data["X"], data["y"]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_npz(self, relative_path: Path) -> np.lib.npyio.NpzFile:
        """Resolve *relative_path* against the project root and load it.

        Parameters
        ----------
        relative_path : Path
            Path to the ``.npz`` file, relative to ``self.root``.

        Returns
        -------
        np.lib.npyio.NpzFile
            Lazy-loaded archive whose keys can be accessed like a dict.

        Raises
        ------
        FileNotFoundError
            If the resolved path does not exist, with the absolute path
            included in the message for easy diagnosis.
        """
        full_path = (self.root / relative_path).resolve()
        if not full_path.is_file():
            raise FileNotFoundError(
                f"Dataset not found: {full_path}\n"
                f"Expected relative to project root: {relative_path}"
            )
        return np.load(full_path)
