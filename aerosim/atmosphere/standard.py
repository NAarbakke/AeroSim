"""U.S. Standard Atmosphere 1976 (0-86 km geometric) with optional simple wind.

Above 86 km density follows the piecewise-exponential model of Vallado,
*Fundamentals of Astrodynamics and Applications*, Table 8-4 (scaled to join US76
continuously at 86 km), with temperature held at the 86 km value. Good enough for
ascent drag, fairing-jettison heat flux and short coasts; not for orbit lifetime.
"""

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


# (base altitude km, base density kg/m^3, scale height km), Vallado Table 8-4
_EXP = [(80, 1.905e-5, 5.927), (90, 3.396e-6, 5.533), (100, 5.297e-7, 5.877),
        (110, 9.661e-8, 7.263), (120, 2.438e-8, 9.473), (130, 8.484e-9, 12.636),
        (140, 3.845e-9, 16.149), (150, 2.070e-9, 22.523), (180, 5.464e-10, 29.740),
        (200, 2.789e-10, 37.105), (250, 7.248e-11, 45.546), (300, 2.418e-11, 53.628),
        (350, 9.518e-12, 53.298), (400, 3.725e-12, 58.515), (450, 1.585e-12, 60.828),
        (500, 6.967e-13, 63.822), (600, 1.454e-13, 71.835), (700, 3.614e-14, 88.667),
        (800, 1.170e-14, 124.64), (900, 5.245e-15, 181.05), (1000, 3.019e-15, 268.00)]


def _exp_density(z_km: float) -> float:
    h0, rho0, H = max((e for e in _EXP if e[0] <= z_km), key=lambda e: e[0])
    return rho0 * math.exp(-(z_km - h0) / H)


def isa(altitude: float) -> tuple[float, float, float]:
    """Return (pressure Pa, temperature K, density kg/m^3) at geometric altitude (m)."""
    if altitude > 86_000.0:
        _p86, t86, rho86 = isa(86_000.0)
        rho = rho86 * _exp_density(altitude / 1e3) / _exp_density(86.0)
        return rho * _R * t86, t86, rho
    z = max(altitude, -5_000.0)
    h = _RE * z / (_RE + z)  # geopotential
    i = max(k for k, layer in enumerate(_LAYERS) if layer[0] <= h) if h >= 0 else 0
    h0, t0, lr = _LAYERS[i]
    t = t0 + lr * (h - h0)
    if lr == 0.0:
        p = _P_BASE[i] * math.exp(-_G0 * (h - h0) / (_R * t0))
    else:
        p = _P_BASE[i] * (t / t0) ** (-_G0 / (lr * _R))
    return p, t, p / (_R * t)


def kinetic_temperature(altitude: float) -> float:
    """US76 kinetic temperature (K) at geometric altitude 0-1000 km, including the thermosphere
    above 86 km (where ``isa`` holds the 86 km value for its drag densities). For display:
    86-91 km isothermal, 91-110 km elliptical, 110-120 km linear, 120-1000 km exponential
    approach to the exospheric 1000 K (US Standard Atmosphere 1976, section 1.2.5)."""
    z = altitude / 1e3
    if z <= 86.0:
        return isa(altitude)[1]
    if z <= 91.0:
        return 186.8673
    if z <= 110.0:
        return 263.1905 - 76.3232 * math.sqrt(max(1.0 - ((z - 91.0) / 19.9429) ** 2, 0.0))
    if z <= 120.0:
        return 240.0 + 12.0 * (z - 110.0)
    re = _RE / 1e3
    xi = (z - 120.0) * (re + 120.0) / (re + z)
    return 1000.0 - (1000.0 - 360.0) * math.exp(-0.01875 * xi)


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
