"""Core layer: data contracts, interfaces, registry, units, file formats. Depends on nothing else in aerosim."""

from aerosim.core import registry
from aerosim.core.eng import read_eng, write_eng
from aerosim.core.errors import AerosimError, OptionalDependencyError, require
from aerosim.core.types import AtmosphereState, EngineDeck, FlightResult, ThrustCurve

__all__ = [
    "AerosimError", "AtmosphereState", "EngineDeck", "FlightResult",
    "OptionalDependencyError", "ThrustCurve", "read_eng", "registry", "require", "write_eng",
]
