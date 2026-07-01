"""Trial dataclass for NeuroLab experiment sessions."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Trial:
    """A single experimental trial within a BCI session.

    A trial begins with a cue presented to the participant and ends when
    the system records a prediction and compares it to the true label.
    Fields that are not yet known at trial creation are initialised to
    ``None`` and filled in as the experiment progresses.

    Parameters
    ----------
    trial_id : int
        Zero-based index of this trial within its session.
    cue : str
        The stimulus or instruction shown to the participant (e.g.
        ``"left"``, ``"right"``).
    true_label : int or None, optional
        Ground-truth class label associated with the cue. ``None`` when
        the dataset label has not yet been assigned. Default ``None``.
    prediction : str or None, optional
        Decoded intent produced by the model. ``None`` until inference
        has been run. Default ``None``.
    confidence : float or None, optional
        Softmax probability of the predicted class, in ``[0, 1]``.
        ``None`` until inference has been run. Default ``None``.
    correct : bool or None, optional
        Whether *prediction* matched *cue* (or the label derived from
        it). ``None`` until the trial has been scored. Default ``None``.

    Examples
    --------
    >>> trial = Trial(trial_id=0, cue="left")
    >>> trial.prediction = "left"
    >>> trial.confidence = 0.91
    >>> trial.correct = True
    """

    trial_id: int
    cue: str
    true_label: int | None = field(default=None)
    prediction: str | None = field(default=None)
    confidence: float | None = field(default=None)
    correct: bool | None = field(default=None)
