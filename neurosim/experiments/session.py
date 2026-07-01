"""Session class for managing a running BCI experiment."""

from __future__ import annotations

from neurosim.experiments.protocol import Protocol
from neurosim.experiments.trial import Trial


class Session:
    """A running instance of a BCI experiment protocol.

    ``Session`` couples a :class:`Protocol` with a concrete, ordered list
    of :class:`Trial` objects and keeps track of which trial the experiment
    is currently on. Trials are generated immediately at construction so
    the full sequence can be inspected or logged before data collection
    begins.

    Parameters
    ----------
    protocol : Protocol
        The protocol that defines the cues, trial count, and randomisation
        settings for this session.
    seed : int or None, optional
        RNG seed forwarded to :meth:`Protocol.generate_trials`. Use a
        fixed value for reproducible sessions. Default ``None``.

    Attributes
    ----------
    protocol : Protocol
        The protocol used by this session.
    trials : list[Trial]
        The full ordered list of trials for this session. Individual
        trials are mutable and are updated in place as the session runs.
    _index : int
        Zero-based index of the next trial to be returned by
        :meth:`next_trial`.

    Examples
    --------
    >>> from neurosim.experiments import Protocol, Session
    >>> proto = Protocol(name="MI-2", cues=["left", "right"], n_trials=4)
    >>> session = Session(proto, seed=0)
    >>> while not session.is_complete():
    ...     trial = session.next_trial()
    ...     trial.prediction = trial.cue   # perfect mock decoder
    ...     trial.correct = True
    >>> session.is_complete()
    True
    """

    def __init__(self, protocol: Protocol, seed: int | None = None) -> None:
        self.protocol: Protocol = protocol
        self.trials: list[Trial] = protocol.generate_trials(seed=seed)
        self._index: int = 0

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def next_trial(self) -> Trial:
        """Return the next trial and advance the internal cursor.

        Parameters
        ----------
        None

        Returns
        -------
        Trial
            The next :class:`Trial` in sequence.

        Raises
        ------
        StopIteration
            When all trials have been exhausted. Check :meth:`is_complete`
            first if you prefer a softer guard.

        Examples
        --------
        >>> trial = session.next_trial()
        >>> trial.trial_id
        0
        """
        if self.is_complete():
            raise StopIteration(
                f"Session '{self.protocol.name}' is complete "
                f"({len(self.trials)} trials exhausted)."
            )
        trial = self.trials[self._index]
        self._index += 1
        return trial

    def reset(self) -> None:
        """Reset the session to the beginning without regenerating trials.

        The trial sequence and any recorded predictions are preserved so
        a reset can be used to replay or review a completed session.
        Call :meth:`Protocol.generate_trials` directly and construct a
        new ``Session`` if you want a fresh randomised sequence.

        Returns
        -------
        None

        Examples
        --------
        >>> session.reset()
        >>> session.current_index
        0
        """
        self._index = 0

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def is_complete(self) -> bool:
        """Return ``True`` when all trials have been served.

        Returns
        -------
        bool

        Examples
        --------
        >>> session.is_complete()
        False
        """
        return self._index >= len(self.trials)

    @property
    def current_index(self) -> int:
        """Index of the next trial to be returned by :meth:`next_trial`.

        Returns
        -------
        int
        """
        return self._index

    @property
    def n_completed(self) -> int:
        """Number of trials that have been served so far.

        Returns
        -------
        int
        """
        return self._index

    @property
    def n_remaining(self) -> int:
        """Number of trials not yet served.

        Returns
        -------
        int
        """
        return len(self.trials) - self._index

    def __repr__(self) -> str:
        status = "complete" if self.is_complete() else f"{self.n_completed}/{len(self.trials)}"
        return f"Session(protocol={self.protocol.name!r}, progress={status})"
