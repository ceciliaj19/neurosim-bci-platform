"""Abstract base class for BCI output devices."""

from abc import ABC, abstractmethod
from typing import Any


class OutputDevice(ABC):
    """Abstract base class for all BCI output devices.

    An output device sits at the end of a BCI pipeline: it receives a
    decoded neural *intent* and translates it into an action (cursor
    movement, robotic-arm command, speech output, etc.).

    Subclasses must implement :meth:`update` and :meth:`reset`. No shared
    state is defined here — each device owns only the state it needs.

    Examples
    --------
    Defining a minimal concrete device::

        class LEDDisplay(OutputDevice):
            def update(self, intent: str) -> None:
                print(f"LED command: {intent}")

            def reset(self) -> None:
                print("LEDs cleared")
    """

    @abstractmethod
    def update(self, intent: Any) -> Any:
        """Apply a decoded neural intent to this device.

        The meaning of *intent* is device-specific (e.g. a direction
        string for a cursor, a joint-angle vector for a robotic arm).
        Subclasses must document the expected type and return value.

        Parameters
        ----------
        intent : Any
            Decoded neural command. The concrete type is defined by each
            subclass.

        Returns
        -------
        Any
            Device-specific feedback after the update. May be ``None``
            for fire-and-forget devices.

        Raises
        ------
        ValueError
            If *intent* is not recognised by the device (recommended
            convention for subclasses).
        """

    @abstractmethod
    def reset(self) -> Any:
        """Reset the device to its initial state.

        After calling this method the device must behave identically to
        a freshly constructed instance.

        Returns
        -------
        Any
            Device-specific state after reset (e.g. a position tuple).
            May be ``None`` for devices with no meaningful return value.
        """

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"
