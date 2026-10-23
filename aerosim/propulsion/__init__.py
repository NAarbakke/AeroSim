"""Solid rocket motors: built-in BATES model and the optional openMotor backend."""

from aerosim.propulsion.solid import bates, openmotor  # noqa: F401
from aerosim.propulsion.solid.bates import BatesMotor, Propellant

__all__ = ["BatesMotor", "Propellant"]
