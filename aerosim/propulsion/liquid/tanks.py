"""Propellant tanks with real-fluid properties (CoolProp).

- ``PropellantTank``      : loaded mass and ullage for a tank at given conditions.
- ``SelfPressurisingTank``: equilibrium two-phase blowdown (N2O, CO2, propane),
  the feed model for hybrids and nitrous-fed liquids.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from aerosim.core import AerosimError, require
from aerosim.thermo.fluids import Fluid, coolprop_name


@dataclass
class PropellantTank:
    """Tank of liquid propellant.

    Cryogens (LOX, LCH4, LH2) default to saturated at ``pressure``; storables
    use ``temperature`` (default 293.15 K) as subcooled liquid at ``pressure``.
    """

    fluid: str
    volume: float                    # m^3, internal
    pressure: float                  # Pa, operating
    fill_fraction: float = 0.95      # liquid volume / tank volume
    temperature: float | None = None # K; None -> saturated (cryogen) or 293.15 K

    def __post_init__(self) -> None:
        self._f = Fluid(self.fluid)
        if self.temperature is None:
            boils_at_room = self._f.t_crit < 293.15 or self._f.saturation(T=293.15).pressure > self.pressure
            self.temperature = (self._f.saturation(P=min(self.pressure, 0.99 * self._f.p_crit)).temperature
                                if boils_at_room else 293.15)

    @property
    def liquid_density(self) -> float:
        sat_t = None
        if self.temperature < self._f.t_crit:
            sat = self._f.saturation(T=self.temperature)
            sat_t = sat if abs(sat.pressure - self.pressure) / self.pressure < 0.02 else None
        return sat_t.rho_liquid if sat_t else self._f.density(self.temperature, self.pressure)

    @property
    def propellant_mass(self) -> float:
        return self.liquid_density * self.volume * self.fill_fraction

    @staticmethod
    def volume_for(fluid: str, mass: float, pressure: float, fill_fraction: float = 0.95,
                   temperature: float | None = None) -> float:
        """Size a tank: volume needed to hold ``mass`` of ``fluid``."""
        probe = PropellantTank(fluid, 1.0, pressure, 1.0, temperature)
        return mass / (probe.liquid_density * fill_fraction)


@dataclass
class BlowdownResult:
    time: np.ndarray
    pressure: np.ndarray        # Pa
    temperature: np.ndarray     # K
    mass: np.ndarray            # kg remaining
    mass_flow: np.ndarray       # kg/s out
    quality: np.ndarray         # vapour mass fraction in tank
    liquid_depletion: float | None   # s, when liquid ran out


class SelfPressurisingTank:
    """Homogeneous-equilibrium two-phase tank draining through an injector.

    Conserves mass and internal energy: dm/dt = -mdot, dU/dt = -mdot * h_out.
    Outflow is liquid while any remains (dip-tube at the bottom), then vapour.
    Injector flow uses the single-phase incompressible (SPI) model,
    mdot = Cd * A * sqrt(2 * rho * dP), which is conservative for N2O
    (real flow flashes; swap in a Dyer/HEM model in ``_mass_flow`` if needed).
    """

    def __init__(self, fluid: str = "N2O", volume: float = 0.005, mass: float = 3.0,
                 temperature: float = 293.15, injector_cd: float = 0.7,
                 injector_area: float = 2.0e-5, downstream_pressure: float = 2.0e6) -> None:
        cp = require("CoolProp.CoolProp", "fluids")
        self._CP = cp
        self._AS = cp.AbstractState("HEOS", coolprop_name(fluid))
        self.fluid, self.volume, self.m0, self.T0 = fluid, volume, mass, temperature
        self.cda = injector_cd * injector_area
        self.p_down = downstream_pressure
        self._AS.update(cp.DmassT_INPUTS, mass / volume, temperature)
        if not 0.0 <= self._AS.Q() < 1.0:
            raise AerosimError("initial state is not two-phase; check mass, volume, temperature")
        self.u0 = self._AS.umass()

    def _state(self, m: float, U: float):
        self._AS.update(self._CP.DmassUmass_INPUTS, m / self.volume, U / m)
        return self._AS

    def _liquid_left(self, s, m: float) -> bool:
        q = s.Q()
        return 0.0 <= q < 1.0 and (1.0 - q) * m > 1e-4 * self.m0

    def _mass_flow(self, s, liquid: bool) -> tuple[float, float]:
        """(mdot, h_out) for current tank state; ``liquid`` = dip tube still covered."""
        dp = max(s.p() - self.p_down, 0.0)
        if liquid:
            rho = s.saturated_liquid_keyed_output(self._CP.iDmass)
            h = s.saturated_liquid_keyed_output(self._CP.iHmass)
        else:
            rho, h = s.rhomass(), s.hmass()
        return self.cda * math.sqrt(2.0 * rho * dp), h

    def run(self, dt: float = 0.01, t_max: float = 60.0, p_cutoff: float | None = None) -> BlowdownResult:
        """Explicit integration of the blowdown until pressure drops to the cutoff."""
        p_cut = p_cutoff if p_cutoff is not None else 1.05 * self.p_down
        m, U = self.m0, self.m0 * self.u0
        rows, t, t_liq = [], 0.0, None
        liquid = True          # latches False once the dip tube uncovers (never re-wets)
        while t < t_max:
            s = self._state(m, U)
            P, T, q = s.p(), s.T(), s.Q()
            if liquid and not self._liquid_left(s, m):
                liquid, t_liq = False, t
            mdot, h = self._mass_flow(s, liquid)
            rows.append((t, P, T, m, mdot, min(max(q, 0.0), 1.0) if 0 <= q <= 1 else 1.0))
            if P <= p_cut or mdot <= 0 or m <= 1e-3 * self.m0:
                break
            # Heun's method (2nd order) on (m, U)
            m1, U1 = m - dt * mdot, U - dt * mdot * h
            mdot2, h2 = self._mass_flow(self._state(m1, U1), liquid)
            m -= 0.5 * dt * (mdot + mdot2)
            U -= 0.5 * dt * (mdot * h + mdot2 * h2)
            t += dt
        a = np.asarray(rows)
        return BlowdownResult(a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4], a[:, 5], t_liq)
