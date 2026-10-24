"""Earth model and orbit geometry for launch-vehicle ascent (SI units, ECI frame).

ECI here is an Earth-centred frame fixed at the launch epoch: z through the north
pole, x through the launch site's meridian at t = 0. Earth rotates about z at
``OMEGA``. Altitudes are above the WGS-84 ellipsoid (geocentric-latitude
approximation, error < 0.1 km), orbit perigee/apogee altitudes are above the
equatorial radius, as launch-vehicle user guides quote them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

MU = 3.986004418e14          # m^3/s^2
RE = 6_378_137.0             # m, equatorial radius (WGS-84)
FLAT = 1 / 298.257223563
J2 = 1.08262668e-3
OMEGA = 7.2921159e-5         # rad/s
OMEGA_VEC = np.array([0.0, 0.0, OMEGA])


def gravity(r: np.ndarray) -> np.ndarray:
    """Point mass + J2 acceleration, m/s^2."""
    x, y, z = r
    rn = math.sqrt(x * x + y * y + z * z)
    k = 1.5 * J2 * MU * RE**2 / rn**5
    f = 5 * z * z / rn**2
    return -MU * r / rn**3 + k * np.array([x * (f - 1), y * (f - 1), z * (f - 3)])


def altitude(r: np.ndarray) -> float:
    """Height above the ellipsoid, m."""
    rn = float(np.linalg.norm(r))
    s = r[2] / rn
    return rn - RE * (1 - FLAT * s * s)


def site_state(lat_deg: float, lon_deg: float, height: float = 0.0):
    """ECI position and inertial velocity of a launch site at t = 0 (x on its meridian)."""
    lat = math.radians(lat_deg)
    rs = RE * (1 - FLAT * math.sin(lat) ** 2) + height
    r = rs * np.array([math.cos(lat), 0.0, math.sin(lat)])
    return r, np.cross(OMEGA_VEC, r)


def enu(r: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Local east, north, up unit vectors (geocentric) at ECI position r."""
    up = r / np.linalg.norm(r)
    east = np.cross([0.0, 0.0, 1.0], up)
    east /= np.linalg.norm(east)
    return east, np.cross(up, east), up


def inertial_azimuth(lat_deg: float, inclination_deg: float, ascending: bool = True) -> float:
    """Launch azimuth (deg from north) of a plane with the given inclination through the site.
    Retrograde planes (i > 90 deg) give a north-westerly azimuth."""
    c = math.cos(math.radians(inclination_deg)) / math.cos(math.radians(lat_deg))
    if abs(c) > 1:
        raise ValueError(f"inclination {inclination_deg} deg unreachable from latitude {lat_deg}")
    az = math.degrees(math.asin(c))
    return az if ascending else 180.0 - az


@dataclass(frozen=True)
class Orbit:
    perigee_alt: float        # m above RE
    apogee_alt: float         # m above RE (inf if hyperbolic)
    inclination: float        # deg
    semi_major_axis: float    # m
    eccentricity: float
    period: float             # s

    @classmethod
    def from_state(cls, r: np.ndarray, v: np.ndarray) -> Orbit:
        rn, vn = np.linalg.norm(r), np.linalg.norm(v)
        h = np.cross(r, v)
        e_vec = np.cross(v, h) / MU - r / rn
        e = float(np.linalg.norm(e_vec))
        energy = vn**2 / 2 - MU / rn
        a = -MU / (2 * energy) if energy < 0 else math.inf
        rp = a * (1 - e) if a != math.inf else float(np.dot(h, h)) / MU / (1 + e)
        ra = a * (1 + e) if a != math.inf else math.inf
        inc = math.degrees(math.acos(h[2] / np.linalg.norm(h)))
        T = 2 * math.pi * math.sqrt(a**3 / MU) if a != math.inf else math.inf
        return cls(rp - RE, ra - RE, inc, a, e, T)
