"""Ramjet and scramjet cycles with real-gas equilibrium thermodynamics (Cantera).

    Ramjet(fuel="JetA", capture_area=0.05).operating_point(altitude=18e3, mach=3.0, power=1.0)
    Ramjet(mode="scramjet", fuel="H2", capture_area=0.05).operating_point(30e3, 8.0, 1.0)

Stream-tube cycle per kg of captured air (station 0 free stream, 3 burner entry, 4 burner
exit, 9 nozzle exit expanded to ambient pressure), thrust F = mdot_0 [(1+f) V9 - V0]:

- ``ramjet``   subsonic combustion. Inlet total-pressure recovery from MIL-E-5007D times
               ``inlet_recovery``; burner at total conditions with a total-pressure ratio
               (flameholder and heat-addition loss); nozzle expands the total state.
- ``scramjet`` supersonic combustion (Heiser & Pratt, *Hypersonic Airbreathing
               Propulsion*, ch. 5): the inlet compresses to burner-entry temperature
               T3 = psi T0 with kinetic-energy efficiency eta_c; burner at constant pressure,
               momentum conserved (V4 = V3 / (1+f)); expansion efficiency eta_e on the nozzle.

Burner: fuel at the equivalence ratio (``power`` scales it, lean up to stoichiometric) mixed
by energy balance using the fuel's lower heating value, then brought to chemical equilibrium
at that enthalpy and pressure (GRI-Mech 3.0 species: CO, OH, NO, H, O... so dissociation at
2000-2800 K is included). Nozzle expansion is shifting equilibrium (``nozzle="frozen"`` to
freeze it). No inlet unstart, combustor choking or wall heat loss model: a negative or
missing burner-entry velocity is reported as no thrust.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq

from aerosim.atmosphere.standard import isa
from aerosim.core import EngineDeck, registry, require
from aerosim.core.units import G0
from aerosim.thermo.gas.mixture import AirFuelGas

# fuel -> (C atoms, H atoms, lower heating value J/kg)
FUELS = {"JetA": (12.0, 23.0, 43.1e6), "H2": (0.0, 2.0, 119.96e6), "CH4": (1.0, 4.0, 50.0e6)}


def mil_recovery(mach: float) -> float:
    """MIL-E-5007D inlet total-pressure recovery."""
    if mach <= 1.0:
        return 1.0
    if mach <= 5.0:
        return 1.0 - 0.075 * (mach - 1.0) ** 1.35
    return 800.0 / (mach**4 + 935.0)


@registry.register("airbreathing", "ramjet")
class Ramjet:
    def __init__(self, mode: str = "ramjet", fuel: str = "JetA", capture_area: float = 0.05,
                 equivalence_ratio: float = 1.0, inlet_recovery: float = 0.95,
                 burner_pressure_ratio: float = 0.95, nozzle_efficiency: float = 0.97,
                 burner_entry_temperature_ratio: float = 7.0, compression_efficiency: float = 0.90,
                 expansion_efficiency: float = 0.90, nozzle: str = "equilibrium",
                 name: str | None = None) -> None:
        """capture_area: free-stream tube captured by the inlet (m^2), sets mass flow.
        inlet_recovery: multiplies the MIL-E-5007D recovery (ramjet).
        burner_entry_temperature_ratio: psi = T3/T0 (scramjet)."""
        if mode not in ("ramjet", "scramjet"):
            raise ValueError("mode must be 'ramjet' or 'scramjet'")
        if fuel not in FUELS:
            raise ValueError(f"fuel must be one of {sorted(FUELS)}")
        require("cantera", "gas")
        c, h, lhv = FUELS[fuel]
        self.gas = AirFuelGas(fuel_c=c, fuel_h=h, lhv=lhv)
        self.mode, self.fuel, self.A0 = mode, fuel, capture_area
        self.phi_max, self.pi_d, self.pi_b = equivalence_ratio, inlet_recovery, burner_pressure_ratio
        self.eta_n, self.psi = nozzle_efficiency, burner_entry_temperature_ratio
        self.eta_c, self.eta_e, self.shifting = compression_efficiency, expansion_efficiency, nozzle != "frozen"
        self.name = name or f"{fuel} {mode}"

    # -- thermo helpers ----------------------------------------------------------
    def _burn(self, h_mix: float, p: float, far: float):
        """Equilibrium products at mixture enthalpy h_mix (J/kg) and pressure p."""
        g = self.gas.set(300.0, p, far)
        g.HP = h_mix, p
        if far > 0:                     # air alone needs no equilibrium step
            g.equilibrate("HP")
        return g.T, g.enthalpy_mass, g.entropy_mass, g.X.copy()

    def _temperature(self, h: float, p: float, X) -> float:
        g = self.gas.gas
        g.TPX = 1000.0, p, X
        g.HP = h, p
        return g.T

    def _expand(self, s: float, p: float, X) -> float:
        """Enthalpy after expansion to p along the isentrope s (shifting or frozen)."""
        g = self.gas.gas
        g.TPX = 1000.0, p, X
        g.SP = s, p
        if self.shifting and g.T > 300.0:       # Cantera's equilibrium data starts at 300 K
            g.equilibrate("SP")
        return g.enthalpy_mass

    # -- cycle -------------------------------------------------------------------
    def operating_point(self, altitude: float, mach: float, power: float = 1.0) -> dict[str, float]:
        p0, T0, rho0 = isa(altitude)
        h0, s0 = self.gas.state(T0, p0)
        g = self.gas.gas
        V0 = mach * math.sqrt(g.cp_mass / g.cv_mass * p0 / rho0)
        mdot0 = rho0 * V0 * self.A0
        ht0 = h0 + 0.5 * V0 * V0
        phi = float(np.clip(power, 0.0, 1.0)) * self.phi_max
        f = phi * self.gas.far_stoich
        hf = self.gas._h_fuel
        out = {"mach": mach, "altitude": altitude, "air_flow": mdot0, "far": f,
               "equivalence_ratio": phi}
        if self.mode == "ramjet":
            pt0 = self.gas.P_from_hs(ht0, s0, 0.0, p0, p0 * 1e4)
            pt4 = pt0 * mil_recovery(mach) * self.pi_d * self.pi_b
            T4, h4, s4, X = self._burn((ht0 + f * hf) / (1 + f), pt4, f)
            if pt4 <= p0:
                v9, t9 = 0.0, T4
            else:
                h9s = self._expand(s4, p0, X)
                v9 = math.sqrt(max(2 * self.eta_n * (h4 - h9s), 0.0))
                t9 = self._temperature(h4 - 0.5 * v9 * v9, p0, X)
            tt0 = self.gas.T_from_h(ht0, pt0)
            out |= {"t4": T4, "pt4": pt4, "burner_pressure": pt4, "t0": T0, "p0": p0, "tt0": tt0,
                    "pt0": pt0, "pt2": pt0 * mil_recovery(mach) * self.pi_d, "t9": t9}
        else:
            T3 = self.psi * T0
            h3 = self.gas.state(T3, p0)[0]
            if h3 >= ht0:
                return out | {"thrust": 0.0, "fuel_flow": mdot0 * f, "isp": 0.0, "t4": float("nan"),
                              "specific_thrust": 0.0, "note": "inlet cannot reach burner-entry T"}
            V3 = math.sqrt(2 * (ht0 - h3))
            # Heiser & Pratt kinetic-energy efficiency: from state 3 an isentrope back to p0
            # reaches h_x = h3 - eta_c (h3 - h0)
            hx = h3 - self.eta_c * (h3 - h0)
            g = self.gas.gas
            def resid(p3):
                g.TPX = T3, p3, self.gas.composition(0.0)
                return self.gas.isentropic_h(g.entropy_mass, p0) - hx
            p3 = brentq(resid, p0 * 1.0001, p0 * 1e5, xtol=1.0)
            V4 = V3 / (1 + f)
            h4 = (h3 + 0.5 * V3 * V3 + f * hf) / (1 + f) - 0.5 * V4 * V4
            T4, h4, s4, X = self._burn(h4, p3, f)
            ht4 = h4 + 0.5 * V4 * V4
            h9s = self._expand(s4, p0, X)
            h9 = h4 - self.eta_e * (h4 - h9s)
            v9 = math.sqrt(max(2 * (ht4 - h9), 0.0))
            out |= {"t3": T3, "p3": p3, "v3": V3, "t4": T4, "burner_pressure": p3, "t0": T0, "p0": p0,
                    "v0": V0, "v4": V4, "t9": self._temperature(h9, p0, X)}
        sf = (1 + f) * v9 - V0
        F = mdot0 * sf
        wf = mdot0 * f
        out |= {"thrust": F, "fuel_flow": wf, "specific_thrust": sf, "v9": v9,
                "isp": F / (wf * G0) if wf > 0 else 0.0, "tsfc": wf / F if F > 0 else float("nan")}
        return out

    def deck(self, altitudes, machs, powers) -> EngineDeck:
        alts, mns, pws = (np.asarray(a, float) for a in (altitudes, machs, powers))
        F = np.zeros((len(alts), len(mns), len(pws)))
        W = np.zeros_like(F)
        for i, h in enumerate(alts):
            for j, m in enumerate(mns):
                for k, p in enumerate(pws):
                    o = self.operating_point(h, m, p)
                    F[i, j, k], W[i, j, k] = o["thrust"], o["fuel_flow"]
        return EngineDeck(alts, mns, pws, F, W, name=self.name)
