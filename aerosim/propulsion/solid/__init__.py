"""Solid rocket motors: built-in BATES model (openMotor adapter is in propulsion/adapters)."""

from aerosim.propulsion.solid import bates  # noqa: F401
from aerosim.propulsion.solid.bates import BatesMotor, Propellant

__all__ = ["BatesMotor", "Propellant"]
