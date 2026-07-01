"""EEGNet: compact CNN for EEG-based BCI classification.

Reference
---------
Lawhern et al. (2018). EEGNet: A Compact Convolutional Neural Network for
EEG-based Brain-Computer Interfaces. *Journal of Neural Engineering*, 15(5).
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PyTorch is required for EEGNet.\n"
        "Install it with:  pip install neurosim[training]"
    ) from exc


class EEGNet(nn.Module):
    """Compact convolutional neural network for EEG motor imagery decoding.

    Accepts raw EEG epochs of shape ``(batch, n_channels, n_timepoints)``
    and returns class logits of shape ``(batch, n_classes)``.

    The architecture follows three stages:

    1. **Temporal filtering** — a 2-D convolution across time only
       (kernel ``(1, kernel_temporal)``) learns frequency-band filters.
    2. **Spatial filtering** — a depthwise convolution across channels
       (kernel ``(n_channels, 1)``) mixes channel information within each
       temporal filter, implementing a learned spatial filter.
    3. **Separable convolution** — a depthwise-then-pointwise convolution
       across time compresses the feature map before classification.

    Parameters
    ----------
    n_classes : int, optional
        Number of output classes. Default ``2``.
    n_channels : int, optional
        Number of EEG channels (electrodes). Default ``64``.
    n_timepoints : int, optional
        Number of time samples per trial. Default ``513``.
    F1 : int, optional
        Number of temporal filters in Block 1. Default ``8``.
    D : int, optional
        Depth multiplier for the depthwise spatial convolution. Total
        spatial filters = ``F1 * D``. Default ``2``.
    F2 : int, optional
        Number of pointwise filters in Block 2. Default ``16``.
    dropout : float, optional
        Dropout probability applied after each block. Default ``0.5``.
    kernel_temporal : int, optional
        Length of the temporal convolution kernel. Default ``64``
        (approximately one second at 512 Hz).

    Attributes
    ----------
    block1 : nn.Sequential
        Temporal + depthwise spatial convolution block.
    block2 : nn.Sequential
        Separable convolution block.
    classifier : nn.Sequential
        Global average-pooling flatten + linear head.

    Examples
    --------
    >>> model = EEGNet(n_classes=2, n_channels=64, n_timepoints=513)
    >>> x = torch.randn(32, 64, 513)
    >>> model(x).shape
    torch.Size([32, 2])
    """

    def __init__(
        self,
        n_classes: int = 2,
        n_channels: int = 64,
        n_timepoints: int = 513,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        dropout: float = 0.5,
        kernel_temporal: int = 64,
    ) -> None:
        super().__init__()

        F2_actual = F1 * D  # noqa: N806 — matches paper notation
        pool1 = 4
        pool2 = 8

        # ------------------------------------------------------------------
        # Block 1 — temporal convolution + depthwise spatial convolution
        # ------------------------------------------------------------------
        self.block1 = nn.Sequential(
            # Temporal filter: (1, kernel_temporal) convolution across time
            nn.Conv2d(1, F1, kernel_size=(1, kernel_temporal),
                      padding=(0, kernel_temporal // 2), bias=False),
            nn.BatchNorm2d(F1),

            # Depthwise spatial filter: mixes information across channels
            nn.Conv2d(F1, F1 * D, kernel_size=(n_channels, 1),
                      groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, pool1)),
            nn.Dropout(dropout),
        )

        # ------------------------------------------------------------------
        # Block 2 — separable convolution
        # ------------------------------------------------------------------
        self.block2 = nn.Sequential(
            # Depthwise convolution across time
            nn.Conv2d(F2_actual, F2_actual, kernel_size=(1, 16),
                      padding=(0, 8), groups=F2_actual, bias=False),
            # Pointwise convolution mixing filters
            nn.Conv2d(F2_actual, F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, pool2)),
            nn.Dropout(dropout),
        )

        # ------------------------------------------------------------------
        # Compute flattened feature size dynamically to avoid hard-coding
        # ------------------------------------------------------------------
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_timepoints)
            out = self.block2(self.block1(dummy))
            flat_size = out.numel()

        # ------------------------------------------------------------------
        # Classifier head
        # ------------------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            EEG batch of shape ``(batch, n_channels, n_timepoints)``.

        Returns
        -------
        torch.Tensor
            Class logits of shape ``(batch, n_classes)``.
        """
        # Add singleton input-channel dimension: (N, C, T) → (N, 1, C, T)
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        return self.classifier(x)
