"""0-D quasi-steady solid motor model for cylindrical BATES grains.

Propellant can be given two ways:

* by name / recipe (``"KNSB"``, ``"APCP"``, or a ``SolidPropellant``): c* and an
  effective gamma come from NASA CEA as functions of chamber pressure, and
  density / burn rate from the propellant library (needs ``aerosim[chem]``);
* as a manual ``Propellant`` with fixed delivered c* and gamma (no dependencies).

Chamber pressure at each web step is the root of the mass balance
``rho_p * Ab * r(Pc) = Pc * At / c*(Pc)``. For other grain shapes, grain
stacks and openMotor .ric files use GrainMotor (grain_motor.py).
"""

from __future__ import annotations

import math

import numpy as np

from aerosim.core import ThrustCurve, registry
from aerosim.propulsion.nozzle import thrust_coefficient
from aerosim.propulsion.solid.propellant import Propellant, make_model, solve_pressure

__all__ = ["BatesMotor", "Propellant"]


@registry.register("rocket_engine", "bates")
class BatesMotor:
    def __init__(self, grain_outer_diameter: float, grain_core_diameter: float,
                 grain_length: float, grain_count: int, throat_diameter: float,
                 expansion_ratio: float, propellant=None,
                 dry_mass: float = 0.0, ambient_pressure: float = 101_325.0,
                 nozzle_efficiency: float = 0.95, cstar_efficiency: float = 0.95,
                 name: str = "BATES motor") -> None:
        """
        propellant : None | str | SolidPropellant | dict | Propellant
            None -> manual KNSB-like default. str -> library name (CEA).
            dict with "ingredients" -> custom recipe (CEA). dict with "tabs" ->
            TabbedPropellant. Other dict -> manual Propellant.
        cstar_efficiency : applied to CEA c* only (manual c* is already delivered).
        """
        self.D = float(grain_outer_diameter)
        self.d0 = float(grain_core_diameter)
        self.L0 = float(grain_length)
        self.N = int(grain_count)
        self.At = math.pi / 4 * float(throat_diameter) ** 2
        self.eps = float(expansion_ratio)
        self.model = make_model(propellant, self.eps, cstar_efficiency)
        self.dry_mass = dry_mass
        self.pa = ambient_pressure
        self.eta = nozzle_efficiency
        self.name = name
        self.warnings: list[str] = []

    @property
    def density(self) -> float:
        return self.model.density

    def _burn_area(self, web: float) -> float:
        d = self.d0 + 2 * web
        L = self.L0 - 2 * web
        if d >= self.D or L <= 0:
            return 0.0
        return self.N * (math.pi * d * L + 2 * math.pi / 4 * (self.D**2 - d**2))

    @property
    def propellant_mass(self) -> float:
        vol = self.N * math.pi / 4 * (self.D**2 - self.d0**2) * self.L0
        return vol * self.density

    @property
    def kn_initial(self) -> float:
        """Klemmung: burning area / throat area at ignition."""
        return self._burn_area(0.0) / self.At

    def _pressure(self, Ab: float, pc_guess: float) -> float:
        return solve_pressure(self.model, Ab / self.At, pc_guess)

    def thrust_curve(self, web_steps: int = 400) -> ThrustCurve:
        m = self.model
        web_max = min((self.D - self.d0) / 2, self.L0 / 2)
        webs = np.linspace(0.0, web_max, web_steps + 1)
        t, F, mdot, P = [0.0], [], [], []
        pc = 2e6
        for i, w in enumerate(webs[:-1]):
            Ab = self._burn_area(w)
            pc = self._pressure(Ab, pc)
            r = m.rate(pc)
            _, gamma = m.cstar_gamma(pc)
            cf = thrust_coefficient(gamma, self.eps, pc, self.pa)
            F.append(max(self.eta * cf * pc * self.At, 0.0))
            mdot.append(m.density * Ab * r)
            P.append(pc)
            if i < web_steps - 1:
                t.append(t[-1] + (webs[i + 1] - w) / r)
        t.append(t[-1] + (webs[-1] - webs[-2]) / r)
        F.append(0.0)
        mdot.append(0.0)
        P.append(self.pa)
        t, F, mdot, P = map(np.asarray, (t, F, mdot, P))
        mdot *= self.propellant_mass / np.trapezoid(mdot, t)

        self.warnings.clear()
        rng = m.pressure_range
        if rng and (P[:-1].max() > rng[1] or P[:-1].min() < rng[0]):
            self.warnings.append(
                f"Pc {P[:-1].min()/1e6:.2f}-{P[:-1].max()/1e6:.2f} MPa leaves the burn-rate fit "
                f"range {rng[0]/1e6:.2f}-{rng[1]/1e6:.2f} MPa for {m.name}")
        return ThrustCurve(t, F, mdot, self.propellant_mass, self.dry_mass, self.name,
                           exit_area=self.eps * self.At, chamber_pressure=P)
