"""Airbreathing propulsion: built-in turbojet cycle (pyCycle / TurboFlow are in propulsion/adapters)."""

from aerosim.propulsion.airbreathing import ideal_turbojet  # noqa: F401
from aerosim.propulsion.airbreathing.ideal_turbojet import Turbojet

__all__ = ["Turbojet"]
