"""Combustion chemistry with NASA CEA: rocket chamber + nozzle, propellant names, solid recipes."""

from aerosim.thermo.combustion import nasa_cea, propellants, solid_propellants  # noqa: F401
from aerosim.thermo.combustion.nasa_cea import NasaCEAModel

__all__ = ["NasaCEAModel"]
