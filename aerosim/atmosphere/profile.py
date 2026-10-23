"""Tabulated atmosphere: the common target every weather source is converted into."""

from __future__ import annotations

import math

import numpy as np

from aerosim.atmosphere.standard import StandardAtmosphere
from aerosim.core import AtmosphereState, registry

_R = 287.05287


@registry.register("atmosphere", "profile")
class ProfileAtmosphere:
    """Vertical profile (e.g. from a GFS sounding). Above the top level, falls back to ISA
    blended to match the top pressure; winds are held at the top value."""

    def __init__(self, altitude, pressure, temperature, wind_u, wind_v, source: str = "") -> None:
        order = np.argsort(altitude)
        self.z = np.asarray(altitude, float)[order]
        self.p = np.asarray(pressure, float)[order]
        self.t = np.asarray(temperature, float)[order]
        self.u = np.asarray(wind_u, float)[order]
        self.v = np.asarray(wind_v, float)[order]
        self.source = source
        self._isa = StandardAtmosphere()

    def at(self, altitude: float) -> AtmosphereState:
        if altitude > self.z[-1]:
            s = self._isa.at(altitude)
            scale = self.p[-1] / self._isa.at(self.z[-1]).pressure
            p, t = s.pressure * scale, s.temperature
            u, v = self.u[-1], self.v[-1]
        else:
            # log-pressure interpolation is far better than linear across levels
            p = float(np.exp(np.interp(altitude, self.z, np.log(self.p))))
            t = float(np.interp(altitude, self.z, self.t))
            u = float(np.interp(altitude, self.z, self.u))
            v = float(np.interp(altitude, self.z, self.v))
        return AtmosphereState(altitude, p, t, p / (_R * t), math.sqrt(1.4 * _R * t), u, v)
