"""Cantera chemistry for things NASA CEA cannot do, plus an equilibrium cross-check.

CEA answers "where does the chemistry *end up*?" (equilibrium, infinitely fast).
Cantera additionally answers "how *fast* does it get there?" (finite-rate
kinetics from a reaction mechanism), which matters for ignition, gas generators
and preburners (deliberately off-equilibrium), and combustor residence time.

Default mechanism GRI-Mech 3.0 covers H2, CH4, C2H6, C3H8 with O2/air.
Kerosene/ethanol kinetics need another mechanism file (pass ``mechanism=``).
"""

from __future__ import annotations

from aerosim.core import require


def _solution(mechanism: str):
    ct = require("cantera", "gas")
    return ct, ct.Solution(mechanism)


def chamber_equilibrium(fuel: str = "CH4", oxidizer: str = "O2", mixture_ratio: float = 3.2,
                        pressure: float = 1.0e6, T_reactants: float = 298.15,
                        mechanism: str = "gri30.yaml") -> dict[str, float]:
    """Adiabatic constant-pressure equilibrium (Cantera HP) of gaseous reactants.

    Same physics as CEA's chamber point, so the two must agree: it is the
    cross-check that keeps both backends honest.
    """
    _, g = _solution(mechanism)
    phi = _phi(g, fuel, oxidizer, mixture_ratio)   # uses g as scratch; set state after
    g.TP = T_reactants, pressure
    g.set_equivalence_ratio(phi, fuel, oxidizer)
    g.equilibrate("HP")
    return {"T": g.T, "MW": g.mean_molecular_weight, "gamma_frozen": g.cp_mass / g.cv_mass,
            "X": {s: x for s, x in zip(g.species_names, g.X) if x > 1e-4}}


def _phi(g, fuel: str, oxidizer: str, mixture_ratio: float) -> float:
    """Equivalence ratio from an O/F mass ratio."""
    g.TPY = 300.0, 101_325.0, {fuel: 1.0}
    g.set_equivalence_ratio(1.0, fuel, oxidizer)
    y = dict(zip(g.species_names, g.Y))
    of_stoich = y[oxidizer] / y[fuel]
    return of_stoich / mixture_ratio


def ignition_delay(fuel: str = "CH4", oxidizer: str = "O2:1.0, N2:3.76", phi: float = 1.0,
                   T: float = 1200.0, pressure: float = 1.0e6, t_max: float = 0.1,
                   mechanism: str = "gri30.yaml") -> float:
    """Constant-pressure autoignition delay (s): time to peak heat release.

    Finite-rate only; CEA has no notion of time. Use for igniter / gas-generator
    design and combustor residence-time margins.
    """
    ct, g = _solution(mechanism)
    g.TP = T, pressure
    g.set_equivalence_ratio(phi, fuel, oxidizer)
    r = ct.IdealGasConstPressureReactor(g, clone=False)
    net = ct.ReactorNet([r])
    t, T_prev, best, t_ign = 0.0, T, 0.0, float("nan")
    while t < t_max:
        t = net.step()
        dTdt = (r.T - T_prev)
        if dTdt > best:
            best, t_ign = dTdt, t
        T_prev = r.T
        if r.T > T + 400 and dTdt < 0.1 * best:   # past the ignition event
            break
    return t_ign
