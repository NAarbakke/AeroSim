"""Shared data contracts. Every layer produces or consumes these, never another
layer's internal objects. All quantities are SI (m, kg, s, N, Pa, K)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ThrustCurve:
    """Time-domain engine output: what a rocket flight sim needs from a motor/engine."""

    time: np.ndarray            # s
    thrust: np.ndarray          # N
    mass_flow: np.ndarray       # kg/s, propellant consumed
    propellant_mass: float      # kg, total loaded
    dry_mass: float = 0.0       # kg, motor/engine hardware
    name: str = "unnamed"
    exit_area: float | None = None      # m^2, for back-pressure correction
    vacuum: bool = False                # True if thrust is vacuum thrust
    chamber_pressure: np.ndarray | None = field(default=None, compare=False)  # Pa, if known

    def __post_init__(self) -> None:
        n = len(self.time)
        if not (len(self.thrust) == len(self.mass_flow) == n):
            raise ValueError("time, thrust and mass_flow must have equal length")
        if n < 2 or np.any(np.diff(self.time) <= 0):
            raise ValueError("time must be strictly increasing with >= 2 samples")

    @property
    def burn_time(self) -> float:
        return float(self.time[-1] - self.time[0])

    @property
    def total_impulse(self) -> float:
        return float(np.trapezoid(self.thrust, self.time))

    @property
    def average_isp(self) -> float:
        """Impulse-averaged specific impulse, s."""
        return self.total_impulse / (self.propellant_mass * 9.80665)

    def thrust_at(self, t: float, ambient_pressure: float | None = None) -> float:
        f = float(np.interp(t, self.time, self.thrust, left=0.0, right=0.0))
        if self.vacuum and ambient_pressure is not None and self.exit_area:
            f = max(f - ambient_pressure * self.exit_area, 0.0) if f > 0 else 0.0
        return f


@dataclass(frozen=True)
class EngineDeck:
    """Operating-point table for airbreathing engines: thrust/fuel flow vs flight condition.

    Axes: altitude (m), Mach (-), power setting (0..1). Arrays shaped (n_alt, n_mach, n_pwr).
    """

    altitude: np.ndarray
    mach: np.ndarray
    power: np.ndarray
    thrust: np.ndarray          # N
    fuel_flow: np.ndarray       # kg/s
    name: str = "unnamed"

    def __post_init__(self) -> None:
        shape = (len(self.altitude), len(self.mach), len(self.power))
        if self.thrust.shape != shape or self.fuel_flow.shape != shape:
            raise ValueError(f"thrust/fuel_flow must have shape {shape}")

    def lookup(self, altitude: float, mach: float, power: float) -> tuple[float, float]:
        from scipy.interpolate import RegularGridInterpolator

        pt = [[altitude, mach, power]]
        kw = {"bounds_error": False, "fill_value": None}
        axes = (self.altitude, self.mach, self.power)
        f = RegularGridInterpolator(axes, self.thrust, **kw)(pt)[0]
        w = RegularGridInterpolator(axes, self.fuel_flow, **kw)(pt)[0]
        return float(f), float(w)


@dataclass(frozen=True)
class AtmosphereState:
    """Atmospheric state at a single point."""

    altitude: float             # m, geometric above MSL
    pressure: float             # Pa
    temperature: float          # K
    density: float              # kg/m^3
    speed_of_sound: float       # m/s
    wind_u: float = 0.0         # m/s, eastward
    wind_v: float = 0.0         # m/s, northward


@dataclass
class FlightResult:
    """Common trajectory output, regardless of which simulator produced it."""

    time: np.ndarray
    position: np.ndarray        # (n, 3) m, local ENU relative to launch
    velocity: np.ndarray        # (n, 3) m/s
    mass: np.ndarray            # kg
    events: dict[str, float] = field(default_factory=dict)   # e.g. {"burnout": 3.2}
    meta: dict = field(default_factory=dict)

    @property
    def apogee(self) -> float:
        return float(self.position[:, 2].max())

    @property
    def max_speed(self) -> float:
        return float(np.linalg.norm(self.velocity, axis=1).max())
