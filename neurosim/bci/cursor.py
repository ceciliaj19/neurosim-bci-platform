"""Cursor control utilities for closed-loop BCI demos."""

from neurosim.bci.output_device import OutputDevice


class CursorController(OutputDevice):
    """Simple 2D cursor controller.

    Sits at the output stage of a BCI pipeline: decoded neural intent
    (a direction string) is translated into movement of a virtual cursor.

    Inherits from :class:`~neurosim.bci.output_device.OutputDevice` and
    satisfies the abstract contract via :meth:`update` and :meth:`reset`.

    Parameters
    ----------
    x : float, optional
        Initial x-position. Default ``0.0``.
    y : float, optional
        Initial y-position. Default ``0.0``.
    step_size : float, optional
        Distance the cursor moves per command. Default ``1.0``.

    Attributes
    ----------
    x : float
        Current x-position.
    y : float
        Current y-position.
    step_size : float
        Movement increment per command.
    """

    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        step_size: float = 1.0,
    ) -> None:
        self.x = x
        self.y = y
        self.step_size = step_size

    # ------------------------------------------------------------------
    # OutputDevice contract
    # ------------------------------------------------------------------

    def update(self, intent: str) -> tuple[float, float]:
        """Apply a decoded direction intent to the cursor.

        Delegates to :meth:`move`. Fulfils the :class:`OutputDevice`
        abstract method.

        Parameters
        ----------
        intent : str
            Direction command: ``"left"``, ``"right"``, ``"up"``, or
            ``"down"`` (case-insensitive).

        Returns
        -------
        tuple[float, float]
            Updated cursor position ``(x, y)``.

        Raises
        ------
        ValueError
            If *intent* is not one of the four supported directions.
        """
        return self.move(intent)

    def reset(self) -> tuple[float, float]:
        """Reset the cursor to the origin and return the new position.

        Returns
        -------
        tuple[float, float]
            ``(0.0, 0.0)`` after reset.
        """
        self.x = 0.0
        self.y = 0.0
        return self.position

    # ------------------------------------------------------------------
    # Domain-specific API
    # ------------------------------------------------------------------

    def move(self, command: str) -> tuple[float, float]:
        """Move the cursor based on a decoded command.

        Parameters
        ----------
        command : str
            Movement command. Supported commands are ``"left"``,
            ``"right"``, ``"up"``, and ``"down"`` (case-insensitive).

        Returns
        -------
        tuple[float, float]
            Updated cursor position ``(x, y)``.

        Raises
        ------
        ValueError
            If *command* is not one of the four supported directions.
        """
        command = command.lower()

        if command == "left":
            self.x -= self.step_size
        elif command == "right":
            self.x += self.step_size
        elif command == "up":
            self.y += self.step_size
        elif command == "down":
            self.y -= self.step_size
        else:
            raise ValueError(
                "Unsupported command. Use 'left', 'right', 'up', or 'down'."
            )

        return self.position

    @property
    def position(self) -> tuple[float, float]:
        """Current cursor position.

        Returns
        -------
        tuple[float, float]
            ``(x, y)`` coordinates.
        """
        return self.x, self.y

    def __repr__(self) -> str:
        return (
            f"CursorController(x={self.x}, y={self.y}, "
            f"step_size={self.step_size})"
        )
