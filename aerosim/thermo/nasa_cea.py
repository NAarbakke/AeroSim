"""NASA CEA backend (github.com/nasa/cea, ``pip install cea``) — the default.

Solves the infinite-area-combustor rocket problem in SI units:
chamber -> throat -> exit at the requested area ratio, with equilibrium or
frozen expansion. Validated against RP-1311 Example 8 in tests/test_nasa_cea.py.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from aerosim.core import AerosimError, registry, require
from aerosim.core.units import G0
from aerosim.thermo import propellants

_BAR = 1.0e5
_P_SL = 101_325.0


@registry.register("combustion", "nasa_cea")
class NasaCEAModel:
    """CombustionModel backed by NASA CEA.

    Parameters
    ----------
    frozen : None | "chamber" | "throat"
        None = shifting equilibrium expansion (default). "chamber" / "throat" =
        composition frozen from that station onward (conservative Isp).
    ambient_pressure : float
        Pa, used for the ``isp_sl`` figure (fixed-nozzle, no separation model).
    data_dir : str, optional
        Custom thermo/transport database directory (cea.init).
    """

    def __init__(self, frozen: str | None = None, ambient_pressure: float = _P_SL,
                 data_dir: str | None = None) -> None:
        self._cea = require("cea", "chem")
        if data_dir:
            self._cea.init(data_dir)
        if frozen not in (None, "chamber", "throat"):
            raise ValueError("frozen must be None, 'chamber' or 'throat'")
        self.frozen = frozen
        self.pa = ambient_pressure
        self._cache: dict[tuple, dict[str, Any]] = {}

    # -- setup -----------------------------------------------------------------
    def _problem(self, oxidizer, fuel) -> dict[str, Any]:
        k = (propellants.key(oxidizer), propellants.key(fuel))
        if k in self._cache:
            return self._cache[k]
        cea = self._cea
        ox, fu = propellants.resolve(oxidizer), propellants.resolve(fuel)
        comps = fu + ox
        names = [c.species for c in comps]
        if len(set(names)) != len(names):
            raise AerosimError("the same species appears in both fuel and oxidizer")
        try:
            reac = cea.Mixture(names)
            prod = cea.Mixture(names, products_from_reactants=True)
        except Exception as e:  # CEA raises on unknown species
            raise AerosimError(f"CEA could not build mixture {names}: {e}") from e
        solver = cea.RocketSolver(prod, reactants=reac)
        p = {
            "reac": reac, "solver": solver, "solution": cea.RocketSolution(solver),
            "fuel_w": np.array([c.mass_fraction for c in fu] + [0.0] * len(ox)),
            "ox_w": np.array([0.0] * len(fu) + [c.mass_fraction for c in ox]),
            "T": np.array([c.temperature for c in comps]),
        }
        self._cache[k] = p
        return p

    # -- CombustionModel -------------------------------------------------------
    def solve(self, oxidizer, fuel, mixture_ratio: float, chamber_pressure: float,
              expansion_ratio: float) -> dict[str, Any]:
        """Full station data. Returns arrays over [chamber, throat, exit]."""
        cea = self._cea
        p = self._problem(oxidizer, fuel)
        w = p["reac"].of_ratio_to_weights(p["ox_w"], p["fuel_w"], float(mixture_ratio))
        hc = p["reac"].calc_property(cea.ENTHALPY, w, p["T"]) / cea.R
        pc_bar = chamber_pressure / _BAR
        kw: dict[str, Any] = {"supar": [float(expansion_ratio)], "iac": True, "hc": hc}
        if self.frozen:
            kw["n_frz"] = 1 if self.frozen == "chamber" else 2
        sol = p["solution"]
        # pi_p is required by CEA; ask for chamber/sea-level ratio as an extra station
        p["solver"].solve(sol, w, pc_bar, [pc_bar * _BAR / _P_SL], **kw)
        # station order: chamber, throat, pi_p point, supar point -> keep chamber, throat, exit
        idx = [0, 1, sol.num_pts - 1]
        # CEA can report converged while the nozzle march fails; check the exit too
        if not sol.converged or not np.all(np.isfinite(np.asarray(sol.Isp_vacuum, float)[idx[1:]])):
            raise AerosimError(
                f"CEA found no valid solution (O/F={mixture_ratio}, Pc={pc_bar:.2f} bar, "
                f"eps={expansion_ratio}); check propellant order (oxidizer, fuel) and O/F range"
            )
        pick = lambda a: np.asarray(a, float)[idx]
        return {
            "P": pick(sol.P) * _BAR, "T": pick(sol.T), "MW": pick(sol.MW),
            "gamma": pick(sol.gamma_s), "sonic_velocity": pick(sol.sonic_velocity),
            "Mach": pick(sol.Mach), "density": pick(sol.density),
            "ae_at": pick(sol.ae_at), "cstar": float(sol.c_star[0]),
            "isp_vac": float(sol.Isp_vacuum[-1]),      # m/s
            "isp_exit": float(sol.Isp[-1]),            # m/s, matched to own exit pressure
            "mole_fractions": {k: float(v[0]) for k, v in sol.mole_fractions.items()
                               if v[0] > 1e-5},
        }

    def chamber(self, oxidizer, fuel, mixture_ratio, chamber_pressure, expansion_ratio):
        s = self.solve(oxidizer, fuel, mixture_ratio, chamber_pressure, expansion_ratio)
        # fixed-geometry ambient Isp:  Isp_amb = Isp_vac - pa * eps * c* / pc
        isp_sl = s["isp_vac"] - self.pa * expansion_ratio * s["cstar"] / chamber_pressure
        return {
            "isp_vac": s["isp_vac"] / G0,
            "isp_sl": isp_sl / G0,
            "cstar": s["cstar"],
            "cf_vac": s["isp_vac"] / s["cstar"],
            "t_chamber": float(s["T"][0]),
            "t_throat": float(s["T"][1]),
            "t_exit": float(s["T"][2]),
            "p_exit": float(s["P"][2]),
            "gamma": float(s["gamma"][0]),
            "gamma_throat": float(s["gamma"][1]),
            "mw": float(s["MW"][0]),
        }
