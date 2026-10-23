"""Core layer: data contracts, interfaces, registry, units. Depends on nothing else in aerosim."""

from aerosim.core import registry
from aerosim.core.errors import AerosimError, OptionalDependencyError, require
from aerosim.core.types import AtmosphereState, EngineDeck, FlightResult, ThrustCurve

__all__ = [
    "AerosimError", "AtmosphereState", "EngineDeck", "FlightResult",
    "OptionalDependencyError", "ThrustCurve", "registry", "require",
]
