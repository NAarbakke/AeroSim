"""Solid rocket propellants: recipes, burn-rate laws and NASA CEA performance.

A ``SolidPropellant`` is a mass-fraction recipe of ingredients plus a density
and a burn-rate law. ``performance()`` runs the CEA rocket problem on the
recipe (the whole grain is one premixed reactant, so there is no O/F); the
motor model uses the resulting c*, flame temperature and an effective gamma.

Ingredients not in CEA's thermo database (sugars, HTPB) are defined here as
custom reactants from formula + standard heat of formation. Condensed products
(K2CO3 for KN propellants, Al2O3 for aluminised APCP) are handled by CEA's
equilibrium; two-phase flow losses (particle lag) are NOT, so use a nozzle /
c* efficiency below 1 for metallised propellants.

Burn-rate coefficients are representative literature values over a limited
pressure range. Measure your own batch (strand burner) before trusting them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from aerosim.core import AerosimError, require
from aerosim.core.units import G0

_BAR = 1.0e5
_P_SL = 101_325.0


@dataclass(frozen=True)
class CustomIngredient:
    """Reactant not in CEA's database. Heat of formation in J/mol at 298.15 K."""

    formula: dict[str, float]
    heat_of_formation: float
    density: float          # kg/m^3 (for theoretical grain density)
    note: str = ""

    @property
    def molecular_weight(self) -> float:
        aw = {"C": 12.011, "H": 1.008, "O": 15.999, "N": 14.007, "S": 32.06, "Cl": 35.45,
              "K": 39.098, "AL": 26.982, "Al": 26.982, "Fe": 55.845}
        return sum(aw[e] * n for e, n in self.formula.items())


# Ingredient name -> CEA database species (density kg/m^3)
DATABASE_INGREDIENTS: dict[str, tuple[str, float]] = {
    "KNO3": ("KNO3(a)", 2109.0),
    "AP": ("NH4CLO4(I)", 1950.0),
    "AN": ("NH4NO3(IV)", 1725.0),
    "Al": ("AL(cr)", 2700.0),
    "Mg": ("Mg(cr)", 1740.0),
    "C": ("C(gr)", 2200.0),        # carbon black / graphite
    "S": ("S(a)", 2070.0),
    "Fe2O3": ("Fe2O3(cr)", 5240.0),  # burn-rate catalyst
}

CUSTOM_INGREDIENTS: dict[str, CustomIngredient] = {
    "Sorbitol": CustomIngredient({"C": 6, "H": 14, "O": 6}, -1353.7e3, 1489.0, "NIST / Nakka"),
    "Sucrose": CustomIngredient({"C": 12, "H": 22, "O": 11}, -2221.2e3, 1587.0, "NIST"),
    "Dextrose": CustomIngredient({"C": 6, "H": 12, "O": 6}, -1273.3e3, 1540.0, "NIST, anhydrous"),
    "Erythritol": CustomIngredient({"C": 4, "H": 10, "O": 4}, -900.1e3, 1451.0, "NIST"),
    # HTPB varies by supplier/cure; this lumped formula is a common approximation.
    "HTPB": CustomIngredient({"C": 10, "H": 15.4, "O": 0.07}, -51.9e3, 920.0,
                             "approximate; replace with supplier data"),
}


@dataclass(frozen=True)
class SolidPropellant:
    name: str
    ingredients: dict[str, float]          # name -> mass fraction
    burn_rate_a: float                     # m/s at 1 MPa
    burn_rate_n: float                     # pressure exponent
    density_ratio: float = 0.95            # achieved / theoretical density
    reference: str = ""
    burn_rate_range: tuple[float, float] = (0.5e6, 10e6)   # Pa, where the law was fitted
    extra: dict = field(default_factory=dict)

    @property
    def mass_fractions(self) -> dict[str, float]:
        tot = sum(self.ingredients.values())
        return {k: v / tot for k, v in self.ingredients.items()}

    @property
    def theoretical_density(self) -> float:
        inv = 0.0
        for k, w in self.mass_fractions.items():
            rho = (DATABASE_INGREDIENTS[k][1] if k in DATABASE_INGREDIENTS
                   else CUSTOM_INGREDIENTS[k].density)
            inv += w / rho
        return 1.0 / inv

    @property
    def density(self) -> float:
        return self.theoretical_density * self.density_ratio

    def burn_rate(self, pc: float) -> float:
        """r = a * (Pc / 1 MPa)^n  [m/s]."""
        return self.burn_rate_a * (pc / 1e6) ** self.burn_rate_n


# Representative propellants. Burn rates: Nakka (KN), generic AP composite.
PROPELLANTS: dict[str, SolidPropellant] = {
    "KNSB": SolidPropellant("KNSB", {"KNO3": 65, "Sorbitol": 35}, 8.26e-3 * 1.0**0.319, 0.319,
                            reference="R. Nakka, KNSB 65/35 (r = 8.26 mm/s at 1 MPa, 0.8-10 MPa)"),
    "KNSU": SolidPropellant("KNSU", {"KNO3": 65, "Sucrose": 35}, 8.26e-3, 0.319,
                            reference="R. Nakka, KNSU 65/35 (rate approx. as KNSB)"),
    "KNDX": SolidPropellant("KNDX", {"KNO3": 65, "Dextrose": 35}, 8.875e-3, 0.619,
                            reference="R. Nakka, KNDX 65/35 (one segment, 0.78-2.57 MPa)",
                            burn_rate_range=(0.78e6, 2.57e6)),
    "KNER": SolidPropellant("KNER", {"KNO3": 65, "Erythritol": 35}, 8.0e-3, 0.35,
                            reference="KNER 65/35, rate approximate"),
    "APCP": SolidPropellant("APCP", {"AP": 68, "Al": 18, "HTPB": 14}, 5.5e-3, 0.35,
                            density_ratio=0.98,
                            reference="generic aluminised AP/HTPB composite, rate approximate"),
    "APCP-NM": SolidPropellant("APCP-NM", {"AP": 86, "HTPB": 14}, 4.0e-3, 0.40,
                               density_ratio=0.98,
                               reference="non-metallised AP/HTPB, rate approximate"),
}


def get(propellant: str | SolidPropellant | dict) -> SolidPropellant:
    """Look up a propellant by name, or build one from a dict."""
    if isinstance(propellant, SolidPropellant):
        return propellant
    if isinstance(propellant, dict):
        return SolidPropellant(**propellant)
    try:
        return PROPELLANTS[propellant]
    except KeyError:
        raise AerosimError(f"unknown propellant {propellant!r}; known: {sorted(PROPELLANTS)}") from None


# -- CEA ---------------------------------------------------------------------------------
_MIX_CACHE: dict[tuple, tuple] = {}


def _mixture(prop: SolidPropellant):
    key = tuple(sorted(prop.mass_fractions.items()))
    if key in _MIX_CACHE:
        return _MIX_CACHE[key]
    cea = require("cea", "chem")
    reactants, weights = [], []
    for name, w in prop.mass_fractions.items():
        if name in DATABASE_INGREDIENTS:
            reactants.append(DATABASE_INGREDIENTS[name][0])
        elif name in CUSTOM_INGREDIENTS:
            ci = CUSTOM_INGREDIENTS[name]
            reactants.append(cea.Reactant(name=name, formula=dict(ci.formula),
                                          molecular_weight=ci.molecular_weight,
                                          enthalpy=ci.heat_of_formation, enthalpy_units="J/mol",
                                          temperature=298.15))
        else:
            raise AerosimError(f"unknown ingredient {name!r}")
        weights.append(w)
    reac = cea.Mixture(reactants)
    prod = cea.Mixture(reactants, products_from_reactants=True)
    solver = cea.RocketSolver(prod, reactants=reac)
    out = (cea, reac, solver, cea.RocketSolution(solver), np.array(weights))
    _MIX_CACHE[key] = out
    return out


def performance(propellant, chamber_pressure: float, expansion_ratio: float,
                ambient_pressure: float = _P_SL) -> dict[str, float]:
    """Ideal (equilibrium) motor performance from NASA CEA, SI units."""
    prop = get(propellant)
    cea, reac, solver, sol, w = _mixture(prop)
    hc = reac.calc_property(cea.ENTHALPY, w, np.full(len(w), 298.15)) / cea.R
    pc_bar = chamber_pressure / _BAR
    solver.solve(sol, w, pc_bar, [max(pc_bar * _BAR / _P_SL, 1.01)],
                 supar=[float(expansion_ratio)], iac=True, hc=hc)
    ivac = float(sol.Isp_vacuum[-1])
    if not sol.converged or not np.isfinite(ivac):
        raise AerosimError(f"CEA failed for {prop.name} at Pc={pc_bar:.2f} bar, eps={expansion_ratio}")
    cstar = float(sol.c_star[0])
    cf_vac = ivac / cstar
    return {
        "cstar": cstar, "t_chamber": float(sol.T[0]), "mw": float(sol.MW[0]),
        "gamma_chamber": float(sol.gamma_s[0]),
        "gamma_effective": effective_gamma(cf_vac, expansion_ratio),
        "cf_vac": cf_vac,
        "isp_vac": ivac / G0,
        "isp_ambient": (ivac - ambient_pressure * expansion_ratio * cstar / chamber_pressure) / G0,
        "condensed_fraction": float(sum(v[0] for k, v in sol.mass_fractions.items()
                                        if "(L)" in k or "(cr)" in k or "(a)" in k)),
    }


def effective_gamma(cf_vac: float, expansion_ratio: float) -> float:
    """Gamma that makes the ideal-gas nozzle reproduce CEA's vacuum Cf.

    Lets the simple isentropic nozzle model in propulsion/nozzle.py carry the
    equilibrium + condensed-phase effects CEA computed.
    """
    from scipy.optimize import brentq

    from aerosim.propulsion.nozzle import thrust_coefficient
    return brentq(lambda g: thrust_coefficient(g, expansion_ratio, 1.0, 0.0) - cf_vac, 1.01, 1.6)


@dataclass
class PerformanceTable:
    """c*, gamma_eff and Tc tabulated vs chamber pressure for fast motor simulation."""

    pressure: np.ndarray
    cstar: np.ndarray
    gamma: np.ndarray
    t_chamber: np.ndarray

    @classmethod
    def build(cls, propellant, expansion_ratio: float, p_min: float = 0.2e6,
              p_max: float = 20e6, n: int = 16) -> PerformanceTable:
        P = np.geomspace(p_min, p_max, n)
        rows = [performance(propellant, p, expansion_ratio) for p in P]
        return cls(P, np.array([r["cstar"] for r in rows]),
                   np.array([r["gamma_effective"] for r in rows]),
                   np.array([r["t_chamber"] for r in rows]))

    def at(self, pc: float) -> tuple[float, float]:
        lp = np.log(np.clip(pc, self.pressure[0], self.pressure[-1]))
        lP = np.log(self.pressure)
        return float(np.interp(lp, lP, self.cstar)), float(np.interp(lp, lP, self.gamma))
