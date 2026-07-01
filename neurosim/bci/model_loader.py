"""Inference model loading utilities for NeuroLab."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    import torch


class ModelLoader:
    """Load trained models from disk for BCI inference.

    Designed to be extensible: additional ``load_*`` methods can be added
    for other serialisation formats (ONNX, scikit-learn pickles, etc.)
    without breaking existing call sites.

    Parameters
    ----------
    device : str, optional
        Default device string passed to ``map_location`` when loading
        PyTorch checkpoints. Common values are ``"cpu"`` and ``"cuda"``.
        Default ``"cpu"``.

    Examples
    --------
    >>> loader = ModelLoader()
    >>> model = loader.load_torch_model("checkpoints/eeg_classifier.pt")
    >>> model.eval()
    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = device

    def load_torch_model(
        self,
        model_path: Union[str, Path],
        map_location: Union[str, None] = None,
    ) -> "torch.nn.Module":
        """Load a serialised PyTorch model from disk.

        Expects a file saved with ``torch.save(model, path)`` (full model
        serialisation). For state-dict checkpoints the caller is responsible
        for constructing the architecture first.

        Parameters
        ----------
        model_path : str or Path
            Path to the ``.pt`` or ``.pth`` checkpoint file.
        map_location : str or None, optional
            Override the device mapping used when deserialising tensors.
            Defaults to ``self.device`` when ``None``.

        Returns
        -------
        torch.nn.Module
            The deserialised model, ready for ``.eval()`` and inference.

        Raises
        ------
        ImportError
            If PyTorch is not installed in the current environment.
        FileNotFoundError
            If *model_path* does not exist, with the resolved absolute
            path included in the message.

        Examples
        --------
        >>> loader = ModelLoader(device="cpu")
        >>> model = loader.load_torch_model("runs/best.pt")
        >>> model.eval()
        """
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "PyTorch is required for load_torch_model but is not installed.\n"
                "Install it with:  pip install torch"
            ) from exc

        resolved = Path(model_path).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(
                f"Model checkpoint not found: {resolved}"
            )

        target = map_location if map_location is not None else self.device
        model: torch.nn.Module = torch.load(resolved, map_location=target)
        return model
