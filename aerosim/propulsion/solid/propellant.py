"""Solid propellant models used by the motor simulators (BATES, GrainMotor).

Every model exposes the same small interface:
``density``, ``name``, ``rate(pc)`` (m/s), ``cstar_gamma(pc)`` -> (c*, gamma),
and optionally ``pressure_range`` (Pa) for out-of-range warnings.

* ``Propellant``      manual: fixed delivered c* and gamma, one St. Robert's law
* ``CEAPropellant``   library / recipe name: c*(Pc), gamma(Pc) from NASA CEA
* ``TabbedPropellant`` piecewise St. Robert's law with per-range k, T, M
                      (the format openMotor .ric files store)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from aerosim.core.units import R_UNIVERSAL


def cstar_ideal(gamma: float, t_chamber: float, molar_mass: float) -> float:
    """Ideal characteristic velocity, m/s. molar_mass in kg/mol."""
    g = gamma
    return math.sqrt(R_UNIVERSAL / molar_mass * t_chamber / g) * ((g + 1) / 2) ** (
        (g + 1) / (2 * (g - 1)))


@dataclass
class Propellant:
    """Manual propellant: fixed, already *delivered* c* (no efficiency applied)."""

    density: float = 1800.0        # kg/m^3
    burn_rate_a: float = 8.26e-3 / (1e6 ** 0.319)   # m/s/Pa^n  (r = a * Pc^n)
    burn_rate_n: float = 0.319
    cstar: float = 885.0           # m/s, delivered
    gamma: float = 1.13


class ManualModel:
    def __init__(self, p: Propellant) -> None:
        self.p, self.name, self.density, self.eta_cstar = p, "manual", p.density, 1.0
        self.burn_rate_n = p.burn_rate_n
        self.pressure_range = None

    def rate(self, pc):
        return self.p.burn_rate_a * pc**self.p.burn_rate_n

    def cstar_gamma(self, pc):
        return self.p.cstar, self.p.gamma


class CEAPropellant:
    def __init__(self, prop, expansion_ratio: float, eta_cstar: float) -> None:
        from aerosim.thermo.combustion import solid_propellants as sp
        self.sp = sp.get(prop)
        self.name, self.density, self.eta_cstar = self.sp.name, self.sp.density, eta_cstar
        self.burn_rate_n = self.sp.burn_rate_n
        self.pressure_range = getattr(self.sp, "burn_rate_range", None)
        self.table = sp.PerformanceTable.build(self.sp, expansion_ratio)

    def rate(self, pc):
        return self.sp.burn_rate(pc)

    def cstar_gamma(self, pc):
        c, g = self.table.at(pc)
        return c * self.eta_cstar, g


@dataclass
class BurnRateTab:
    """One pressure range: r = a * Pc^n; combustion gas k (gamma), T (K), M (kg/mol)."""

    a: float                       # m/s/Pa^n
    n: float
    gamma: float
    t_chamber: float               # K
    molar_mass: float              # kg/mol
    min_pressure: float = 0.0      # Pa
    max_pressure: float = math.inf

    def __post_init__(self) -> None:
        # YAML 1.1 reads "10.0e6" (no exponent sign) as a string
        for f in ("a", "n", "gamma", "t_chamber", "molar_mass", "min_pressure", "max_pressure"):
            setattr(self, f, float(getattr(self, f)))


@dataclass
class TabbedPropellant:
    """Piecewise St. Robert's law. Outside every tab the nearest tab is used."""

    name: str
    density: float                 # kg/m^3
    tabs: list[BurnRateTab] = field(default_factory=list)
    eta_cstar: float = 1.0

    def __post_init__(self) -> None:
        self.tabs = [t if isinstance(t, BurnRateTab) else BurnRateTab(**t) for t in self.tabs]
        if not self.tabs:
            raise ValueError("TabbedPropellant needs at least one tab")
        self.pressure_range = (min(t.min_pressure for t in self.tabs),
                               max(t.max_pressure for t in self.tabs))

    def tab(self, pc: float) -> BurnRateTab:
        for t in self.tabs:
            if t.min_pressure <= pc <= t.max_pressure:
                return t
        return min(self.tabs, key=lambda t: min(abs(pc - t.min_pressure),
                                                abs(pc - t.max_pressure)))

    def rate(self, pc):
        t = self.tab(pc)
        return t.a * pc**t.n

    def cstar_gamma(self, pc):
        t = self.tab(pc)
        return cstar_ideal(t.gamma, t.t_chamber, t.molar_mass) * self.eta_cstar, t.gamma


def make_model(propellant, expansion_ratio: float, cstar_efficiency: float = 1.0):
    """None | Propellant | dict | str | SolidPropellant | TabbedPropellant -> model.

    dict with "tabs" -> TabbedPropellant; dict with "ingredients" -> CEA recipe;
    other dict -> manual Propellant; str / SolidPropellant -> CEA library propellant.
    """
    if propellant is None or isinstance(propellant, Propellant):
        return ManualModel(propellant or Propellant())
    if isinstance(propellant, TabbedPropellant):
        return propellant
    if isinstance(propellant, dict) and "tabs" in propellant:
        return TabbedPropellant(**propellant)
    if isinstance(propellant, dict) and "ingredients" not in propellant:
        return ManualModel(Propellant(**propellant))
    return CEAPropellant(propellant, expansion_ratio, cstar_efficiency)


def solve_pressure(model, kn: float, pc_guess: float = 2e6) -> float:
    """Steady chamber pressure for Klemmung ``kn`` = Ab/At.

    Mass balance rho * Ab * r(Pc) = Pc * At / c*(Pc). With the *local* burn-rate
    exponent n, r(Pc) = r(p) (Pc/p)^n, which gives the fixed-point update below;
    it is exact in one iteration for a single St. Robert's law and converges in a
    few when c* or the tab changes with pressure.
    """
    pc = pc_guess
    for _ in range(50):
        cstar, _ = model.cstar_gamma(pc)
        r = model.rate(pc)
        n = math.log(model.rate(pc * 1.001) / r) / math.log(1.001)
        new = (kn * model.density * cstar * r / pc**n) ** (1 / (1 - n))
        if abs(new - pc) < 1e-6 * pc:
            return new
        pc = new
    return pc
