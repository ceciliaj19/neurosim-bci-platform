"""Protocol dataclass for NeuroLab experiment design."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from neurosim.experiments.trial import Trial


@dataclass
class Protocol:
    """Specification for a BCI experiment protocol.

    A protocol defines *what* trials a session should contain: which cues
    are used, how many trials to run, and whether the order is randomised.
    It does not hold any runtime state; call :meth:`generate_trials` to
    produce a concrete trial list for a :class:`~neurosim.experiments.session.Session`.

    Parameters
    ----------
    name : str
        Human-readable name for this protocol (e.g.
        ``"Motor Imagery 2-Class"``).
    cues : list[str]
        Ordered list of unique cue labels (e.g. ``["left", "right"]``).
    n_trials : int
        Total number of trials to generate. Cues are tiled so each
        appears as evenly as possible.
    randomize : bool, optional
        If ``True`` (default), shuffle the trial order when generating.
        Set to ``False`` for fixed, repeatable sequences without a seed.

    Examples
    --------
    >>> proto = Protocol(name="MI-2", cues=["left", "right"], n_trials=10)
    >>> trials = proto.generate_trials(seed=42)
    >>> len(trials)
    10
    >>> {t.cue for t in trials} == {"left", "right"}
    True
    """

    name: str
    cues: list[str]
    n_trials: int
    randomize: bool = field(default=True)

    def generate_trials(self, seed: int | None = None) -> list[Trial]:
        """Generate an ordered list of :class:`Trial` objects for this protocol.

        Cues are tiled across ``n_trials`` so each cue appears as close to
        ``n_trials / len(cues)`` times as possible. When ``randomize`` is
        ``True`` the sequence is shuffled using an isolated
        :class:`random.Random` instance so the global random state is
        unaffected.

        Parameters
        ----------
        seed : int or None, optional
            Seed for the shuffle RNG. Use a fixed value for reproducible
            sequences in tests or offline analysis. ``None`` (default) lets
            the RNG seed itself from OS entropy.

        Returns
        -------
        list[Trial]
            A list of ``n_trials`` :class:`Trial` objects with ``trial_id``
            set to their zero-based position in the sequence.

        Raises
        ------
        ValueError
            If ``cues`` is empty or ``n_trials`` is not positive.

        Examples
        --------
        >>> proto = Protocol(name="Test", cues=["left", "right"], n_trials=6)
        >>> trials = proto.generate_trials(seed=0)
        >>> [t.cue for t in trials]  # order will vary
        [...]
        """
        if not self.cues:
            raise ValueError("Protocol.cues must not be empty.")
        if self.n_trials < 1:
            raise ValueError(f"n_trials must be >= 1, got {self.n_trials}.")

        # Tile cues to fill n_trials as evenly as possible
        n_repeats = math.ceil(self.n_trials / len(self.cues))
        cue_pool = (self.cues * n_repeats)[: self.n_trials]

        if self.randomize:
            rng = random.Random(seed)
            rng.shuffle(cue_pool)

        return [
            Trial(trial_id=i, cue=cue)
            for i, cue in enumerate(cue_pool)
        ]
