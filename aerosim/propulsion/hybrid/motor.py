"""Hybrid rocket motor: liquid / gaseous oxidizer over a solid fuel grain, time-stepped.

    HybridMotor(fuel="HTPB", oxidizer="N2O", grain_length=0.5, port_radius=0.015,
                outer_radius=0.045, regression_a=..., regression_n=0.5, throat_area=...,
                tank={"volume": 0.012, "mass": 8.0, "temperature": 293.15}).simulate()

Each step:

- oxidizer flow: a self-pressurising N2O tank (``tank={...}``; homogeneous-equilibrium
  two-phase blowdown from ``liquid.tanks``, liquid out of the dip tube, then vapour) through
  the injector, mdot = Cd A sqrt(2 rho (p_tank - p_c)); or a regulated feed at a fixed
  ``oxidizer_mass_flow`` until ``oxidizer_mass`` is used
- fuel flow: regression rate rdot = a * G_ox^n with G_ox = mdot_ox / port area (classical
  space-averaged law; a and n are fitted to firings of the propellant pair and must be
  given: they are not universal), mdot_f = rho_f * rdot * port perimeter * length
- chamber pressure: the value at which nozzle outflow p_c A_t / c*(O/F, p_c) equals the
  oxidizer plus fuel flow (both depend on p_c), found by root bracketing each step
- thrust: vacuum Cf(O/F, p_c) from NASA CEA times p_c A_t, with c* and Cf efficiencies;
  O/F shifts during the burn as the port opens, which is the characteristic hybrid effect

CEA performance is tabulated over O/F and chamber pressure once per motor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq

from aerosim.core import AerosimError, ThrustCurve, registry
from aerosim.core.units import G0
from aerosim.thermo.combustion import solid_propellants as sp

FUEL_DENSITY = {"HTPB": 920.0, "Paraffin": 900.0}


@dataclass
class HybridHistory:
    time: np.ndarray
    thrust: np.ndarray          # N, vacuum
    chamber_pressure: np.ndarray
    of_ratio: np.ndarray
    mdot_ox: np.ndarray
    mdot_fuel: np.ndarray
    port_radius: np.ndarray
    tank_pressure: np.ndarray
    oxidizer_used: float
    fuel_used: float
    end_reason: str


@dataclass
class HybridMotor:
    fuel: str
    oxidizer: str
    grain_length: float              # m
    port_radius: float               # m, initial (per port)
    outer_radius: float              # m, web limit
    regression_a: float              # m/s per (kg/m^2/s)^n
    regression_n: float
    throat_area: float               # m^2
    expansion_ratio: float = 5.0
    ports: int = 1                   # identical circular ports (each at port_radius)
    fuel_density: float | None = None
    tank: dict | None = None         # SelfPressurisingTank kwargs (N2O); injector inside
    injector_cd: float = 0.65
    injector_area: float | None = None   # m^2, self-pressurising feed
    oxidizer_mass_flow: float | None = None   # kg/s, regulated feed
    oxidizer_mass: float | None = None        # kg, regulated feed
    cstar_efficiency: float = 0.92
    cf_efficiency: float = 0.97
    dt: float = 0.01
    t_max: float = 60.0
    min_chamber_pressure: float = 0.15e6
    dry_mass: float = 0.0
    name: str = "hybrid motor"
    _table: tuple = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.rho_f = self.fuel_density or FUEL_DENSITY.get(self.fuel)
        if self.rho_f is None:
            raise AerosimError(f"give fuel_density for fuel {self.fuel!r}")
        if (self.tank is None) == (self.oxidizer_mass_flow is None):
            raise AerosimError("give either tank={...} (self-pressurising) or oxidizer_mass_flow")
        if self.tank is not None and self.injector_area is None:
            raise AerosimError("a self-pressurising feed needs injector_area")
        if self.oxidizer_mass_flow is not None and self.oxidizer_mass is None:
            raise AerosimError("a regulated feed needs oxidizer_mass")

    # -- CEA table ---------------------------------------------------------------
    def performance_table(self, ofs=None, pcs=None):
        """Ideal equilibrium (c*, Cf_vac) tables over (O/F, p_c), with their axes."""
        if self._table is None:
            ofs = np.asarray(ofs if ofs is not None else np.r_[np.linspace(0.6, 4, 9), np.linspace(4.5, 16, 12)])
            pcs = np.asarray(pcs if pcs is not None else np.geomspace(0.2e6, 8e6, 6))
            prop = sp.SolidPropellant(f"{self.oxidizer}/{self.fuel}", {self.fuel: 1.0, self.oxidizer: 1.0}, 0, 0)
            cs, cf = np.zeros((len(ofs), len(pcs))), np.zeros((len(ofs), len(pcs)))
            for i, of in enumerate(ofs):
                for j, pc in enumerate(pcs):
                    try:
                        r = sp.performance(prop, pc, self.expansion_ratio, weights=[1.0, float(of)])
                        cs[i, j], cf[i, j] = r["cstar"], r["cf_vac"]
                    except AerosimError:          # very fuel-rich points can fail to converge
                        cs[i, j] = cf[i, j] = np.nan
            for a in (cs, cf):                    # fill failures from the nearest O/F that solved
                for j in range(len(pcs)):
                    ok = np.isfinite(a[:, j])
                    if not ok.any():
                        raise AerosimError(f"CEA failed for {prop.name} at every O/F")
                    a[~ok, j] = np.interp(ofs[~ok], ofs[ok], a[ok, j])
            self._table = (cs, cf, ofs, pcs)
        return self._table

    def _cstar_cf(self, of, pc):
        """Delivered (c*, Cf_vac): bilinear in O/F and log p_c, clamped to the table."""
        cs, cf, ofs, pcs = self.performance_table()
        lp = np.log(pcs)
        x = math.log(min(max(pc, pcs[0]), pcs[-1]))
        j = min(int(np.searchsorted(lp, x)), len(lp) - 1)
        i0, w = max(j - 1, 0), 0.0 if j == 0 else (x - lp[j - 1]) / (lp[j] - lp[j - 1])
        c = (1 - w) * np.interp(of, ofs, cs[:, i0]) + w * np.interp(of, ofs, cs[:, j])
        f = (1 - w) * np.interp(of, ofs, cf[:, i0]) + w * np.interp(of, ofs, cf[:, j])
        return float(c) * self.cstar_efficiency, float(f) * self.cf_efficiency

    # -- simulation --------------------------------------------------------------
    def simulate(self) -> HybridHistory:
        tank = None
        if self.tank is not None:
            from aerosim.propulsion.liquid.tanks import SelfPressurisingTank
            tank = SelfPressurisingTank(**{"fluid": "N2O", **self.tank, "injector_cd": self.injector_cd,
                                           "injector_area": self.injector_area})
            m, U = tank.m0, tank.m0 * tank.u0
            liquid = True
        ox_left = self.oxidizer_mass
        r = self.port_radius
        L, At, n_p = self.grain_length, self.throat_area, self.ports
        rows, t, reason = [], 0.0, "t_max"

        def flows(pc, mdot_ox):
            G = mdot_ox / (n_p * math.pi * r * r)
            rdot = self.regression_a * G**self.regression_n
            return rdot, self.rho_f * rdot * 2 * math.pi * r * L * n_p

        while t < self.t_max:
            if tank is not None:
                s = tank._state(m, U)
                p_tank = s.p()
                if liquid and not tank._liquid_left(s, m):
                    liquid = False
                if liquid:
                    rho = s.saturated_liquid_keyed_output(tank._CP.iDmass)
                    h = s.saturated_liquid_keyed_output(tank._CP.iHmass)
                else:
                    rho, h = s.rhomass(), s.hmass()
                ox_of = lambda pc, rho=rho, pt=p_tank: tank.cda * math.sqrt(2 * rho * max(pt - pc, 0.0))
            else:
                p_tank, h = float("nan"), 0.0
                ox_of = lambda pc, left=ox_left: self.oxidizer_mass_flow if left > 0 else 0.0

            def resid(pc, ox_of=ox_of):
                mo = ox_of(pc)
                if mo <= 0:
                    return -pc
                _, mf = flows(pc, mo)
                cs, _ = self._cstar_cf(mo / mf, pc)
                return (mo + mf) * cs / At - pc

            p_hi = p_tank * 0.999 if tank is not None else 50e6
            if resid(1e3) <= 0 or p_hi <= 1e3:
                reason = "oxidizer exhausted"
                break
            pc = brentq(resid, 1e3, p_hi, xtol=10.0)
            mo = ox_of(pc)
            rdot, mf = flows(pc, mo)
            of = mo / mf
            cf = self._cstar_cf(of, pc)[1]
            rows.append((t, cf * pc * At, pc, of, mo, mf, r, p_tank))
            if pc < self.min_chamber_pressure and t > 0.5:
                reason = "chamber pressure below cutoff"
                break
            # advance
            r += rdot * self.dt
            if tank is not None:
                m -= mo * self.dt
                U -= mo * h * self.dt
                if m <= 1e-3 * tank.m0:
                    reason = "oxidizer exhausted"
                    break
            else:
                ox_left -= mo * self.dt
            t += self.dt
            if r >= self.outer_radius:
                reason = "web burned through"
                break
        if len(rows) < 2:
            raise AerosimError("hybrid did not ignite: oxidizer flow cannot sustain chamber pressure")
        a = np.asarray(rows)
        dt = np.diff(a[:, 0], append=a[-1, 0] + self.dt)
        return HybridHistory(a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4], a[:, 5], a[:, 6], a[:, 7],
                             float(np.sum(a[:, 4] * dt)), float(np.sum(a[:, 5] * dt)), reason)

    def thrust_curve(self) -> ThrustCurve:
        h = self.simulate()
        t = np.r_[h.time, h.time[-1] + self.dt]
        F, md = np.r_[h.thrust, 0.0], np.r_[h.mdot_ox + h.mdot_fuel, 0.0]
        return ThrustCurve(t, F, md, h.oxidizer_used + h.fuel_used, self.dry_mass, self.name,
                           exit_area=self.expansion_ratio * self.throat_area, vacuum=True,
                           chamber_pressure=np.r_[h.chamber_pressure, 0.0])

    @property
    def average_isp(self) -> float:
        h = self.simulate()
        return float(np.trapezoid(h.thrust, h.time) / ((h.oxidizer_used + h.fuel_used) * G0))


@registry.register("rocket_engine", "hybrid")
def _factory(**kw) -> HybridMotor:
    return HybridMotor(**kw)
