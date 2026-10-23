"""U.S. Standard Atmosphere 1976 (0-86 km geometric) with optional simple wind."""

from __future__ import annotations

import itertools
import math

from aerosim.core import AtmosphereState, registry

_R = 287.05287          # J/(kg K), dry air
_GAMMA = 1.4
_G0 = 9.80665
_RE = 6_356_766.0       # m, effective Earth radius used by US76

# (base geopotential altitude m, base temperature K, lapse rate K/m)
_LAYERS = [
    (0.0, 288.15, -0.0065),
    (11_000.0, 216.65, 0.0),
    (20_000.0, 216.65, 0.001),
    (32_000.0, 228.65, 0.0028),
    (47_000.0, 270.65, 0.0),
    (51_000.0, 270.65, -0.0028),
    (71_000.0, 214.65, -0.002),
    (84_852.0, 186.946, 0.0),
]


def _base_pressures() -> list[float]:
    p = [101_325.0]
    for (h0, t0, lr), (h1, _, _) in itertools.pairwise(_LAYERS):
        dh = h1 - h0
        if lr == 0.0:
            p.append(p[-1] * math.exp(-_G0 * dh / (_R * t0)))
        else:
            p.append(p[-1] * ((t0 + lr * dh) / t0) ** (-_G0 / (lr * _R)))
    return p


_P_BASE = _base_pressures()


def isa(altitude: float) -> tuple[float, float, float]:
    """Return (pressure Pa, temperature K, density kg/m^3) at geometric altitude (m)."""
    z = min(max(altitude, -5_000.0), 86_000.0)
    h = _RE * z / (_RE + z)  # geopotential
    i = max(k for k, layer in enumerate(_LAYERS) if layer[0] <= h) if h >= 0 else 0
    h0, t0, lr = _LAYERS[i]
    t = t0 + lr * (h - h0)
    if lr == 0.0:
        p = _P_BASE[i] * math.exp(-_G0 * (h - h0) / (_R * t0))
    else:
        p = _P_BASE[i] * (t / t0) ** (-_G0 / (lr * _R))
    return p, t, p / (_R * t)


@registry.register("atmosphere", "isa")
class StandardAtmosphere:
    """ISA with an optional power-law wind profile: speed(z) = ref_speed * (z/ref_alt)^alpha."""

    def __init__(self, wind_speed: float = 0.0, wind_from_deg: float = 270.0,
                 ref_altitude: float = 10.0, alpha: float = 1 / 7,
                 temperature_offset: float = 0.0) -> None:
        self.wind_speed = wind_speed
        self.wind_from = math.radians(wind_from_deg)
        self.ref_altitude = ref_altitude
        self.alpha = alpha
        self.dT = temperature_offset

    def at(self, altitude: float) -> AtmosphereState:
        p, t, _ = isa(altitude)
        t += self.dT
        rho = p / (_R * t)
        z = max(altitude, 0.0)
        spd = self.wind_speed * (max(z, 0.1) / self.ref_altitude) ** self.alpha if z > 0 else 0.0
        # meteorological "from" direction -> vector pointing where the wind goes
        u = -spd * math.sin(self.wind_from)
        v = -spd * math.cos(self.wind_from)
        return AtmosphereState(altitude, p, t, rho, math.sqrt(_GAMMA * _R * t), u, v)
