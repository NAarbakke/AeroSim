"""Cantera: gas mixtures along a gas-turbine flow path and finite-rate kinetics."""

from aerosim.thermo.gas import kinetics  # noqa: F401
from aerosim.thermo.gas.mixture import AIR, AirFuelGas

__all__ = ["AIR", "AirFuelGas"]
