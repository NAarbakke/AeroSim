"""Cantera-backed ideal-gas mixtures for gas-turbine cycles: air and burned
air/hydrocarbon products with temperature-dependent cp, gamma and enthalpy.

Why not CEA here? CEA is built around one-shot equilibrium problems (rocket,
HP, detonation). A cycle code needs thousands of cheap state evaluations of a
*known* mixture along a flow path (compress, burn, expand) - exactly what a
Cantera ``Solution`` object is for. Products are formed by complete combustion
of CxHy; at turbine-inlet temperatures (<2000 K, lean) dissociation is negligible.
"""

from __future__ import annotations

from scipy.optimize import brentq

from aerosim.core import require

AIR = {"O2": 0.2095, "N2": 0.7808, "AR": 0.0093}       # mole fractions, dry air
_MW = {"C": 12.011, "H": 1.008, "O2": 31.998, "CO2": 44.009, "H2O": 18.015}


class AirFuelGas:
    """Air plus the products of burning a CxHy fuel at fuel-air ratio ``far``.

    Fuel defaults to a kerosene surrogate C12H23 (Jet-A), LHV 43.1 MJ/kg.
    """

    def __init__(self, fuel_c: float = 12.0, fuel_h: float = 23.0, lhv: float = 43.1e6,
                 mechanism: str = "gri30.yaml") -> None:
        ct = require("cantera", "gas")
        self.gas = ct.Solution(mechanism)
        self.c, self.h, self.lhv = fuel_c, fuel_h, lhv
        self.mw_fuel = fuel_c * _MW["C"] + fuel_h * _MW["H"]
        # mass of air per mole of air (for converting FAR to moles of fuel)
        self.gas.TPX = 298.15, 101_325.0, AIR
        self._mw_air = self.gas.mean_molecular_weight
        self.far_stoich = self.mw_fuel / ((fuel_c + fuel_h / 4) / AIR["O2"] * self._mw_air)
        self._h_fuel = self._fuel_enthalpy()

    # -- composition -----------------------------------------------------------------
    def composition(self, far: float) -> dict[str, float]:
        """Mole amounts of products per mole of air for fuel-air mass ratio ``far`` (lean)."""
        if far > self.far_stoich:
            raise ValueError(f"far={far:.4f} is rich (stoich {self.far_stoich:.4f}); lean only")
        n_fuel = far * self._mw_air / self.mw_fuel
        x = dict(AIR)
        x["O2"] -= n_fuel * (self.c + self.h / 4)
        x["CO2"] = n_fuel * self.c
        x["H2O"] = n_fuel * self.h / 2
        return x

    def set(self, T: float, P: float, far: float = 0.0):
        self.gas.TPX = T, P, self.composition(far)
        return self.gas

    def _fuel_enthalpy(self) -> float:
        """Absolute enthalpy of the fuel (J/kg), consistent with Cantera's datum,
        back-computed from its LHV:  h_fuel = LHV + sum(h_products) - h_O2 (per kg fuel at 298 K)."""
        g = self.gas
        g.TP = 298.15, 101_325.0
        h = dict(zip(g.species_names, g.partial_molar_enthalpies))      # J/kmol
        per_mol = (self.c * h["CO2"] + self.h / 2 * h["H2O"] - (self.c + self.h / 4) * h["O2"])
        return self.lhv + per_mol / self.mw_fuel

    # -- process helpers ---------------------------------------------------------------
    def cp(self, T: float, P: float = 101_325.0, far: float = 0.0) -> float:
        return self.set(T, P, far).cp_mass

    def gamma(self, T: float, P: float = 101_325.0, far: float = 0.0) -> float:
        g = self.set(T, P, far)
        return g.cp_mass / g.cv_mass

    def state(self, T: float, P: float, far: float = 0.0) -> tuple[float, float]:
        """(h, s) per kg mixture."""
        g = self.set(T, P, far)
        return g.enthalpy_mass, g.entropy_mass

    def T_from_h(self, h: float, P: float, far: float = 0.0) -> float:
        g = self.set(300.0, P, far)
        g.HP = h, P
        return g.T

    def isentropic_h(self, s: float, P: float, far: float = 0.0) -> float:
        """Enthalpy after an isentropic change to pressure P."""
        g = self.set(300.0, P, far)
        g.SP = s, P
        return g.enthalpy_mass

    def P_from_hs(self, h: float, s: float, far: float, p_lo: float, p_hi: float) -> float:
        """Pressure on the isentrope s at which enthalpy equals h."""
        return brentq(lambda P: self.isentropic_h(s, P, far) - h, p_lo, p_hi, xtol=1e-3)

    def far_for_T(self, T_out: float, h_air_in: float, P: float, burner_eff: float = 1.0) -> float:
        """Fuel-air ratio that heats air of enthalpy h_air_in to T_out (adiabatic burner).

        Energy balance per kg air:  h_air + f*(h_fuel - (1-eta)*LHV) = (1+f) * h_products(T_out, f)
        """
        h_f = self._h_fuel - (1 - burner_eff) * self.lhv

        def balance(f):
            return h_air_in + f * h_f - (1 + f) * self.set(T_out, P, f).enthalpy_mass

        return brentq(balance, 0.0, 0.999 * self.far_stoich, xtol=1e-9)
