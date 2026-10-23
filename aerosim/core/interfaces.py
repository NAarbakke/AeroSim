"""Interfaces (structural Protocols) that every backend implements.

Adding a new backend = write a class satisfying one of these and register it
with ``aerosim.core.registry``. Nothing else in the codebase needs to change.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from aerosim.core.types import AtmosphereState, EngineDeck, FlightResult, ThrustCurve


@runtime_checkable
class CombustionModel(Protocol):
    """Chamber chemistry: propellants + conditions -> performance (thermo layer)."""

    def chamber(self, oxidizer: str, fuel: str, mixture_ratio: float,
                chamber_pressure: float, expansion_ratio: float) -> dict[str, float]:
        """Return at least: isp_vac, isp_sl, cstar, t_chamber, gamma, mw."""
        ...


@runtime_checkable
class RocketEngineModel(Protocol):
    """Solid or liquid engine model (propulsion layer)."""

    def thrust_curve(self) -> ThrustCurve: ...


@runtime_checkable
class AirbreathingEngineModel(Protocol):
    """Gas-turbine cycle model (propulsion layer)."""

    def operating_point(self, altitude: float, mach: float, power: float) -> dict[str, float]:
        """Return at least: thrust, fuel_flow, and any station data."""
        ...

    def deck(self, altitudes, machs, powers) -> EngineDeck: ...


@runtime_checkable
class AtmosphereModel(Protocol):
    """Atmosphere + wind provider (atmosphere layer)."""

    def at(self, altitude: float) -> AtmosphereState: ...


@runtime_checkable
class FlightSimulator(Protocol):
    """Trajectory integrator (flight layer)."""

    def run(self, vehicle: Any, propulsion: ThrustCurve | EngineDeck,
            atmosphere: AtmosphereModel, **options: Any) -> FlightResult: ...
