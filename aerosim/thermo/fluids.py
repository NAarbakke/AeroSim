"""Real-fluid properties via CoolProp: the *unburned* side of propulsion.

Use for propellant densities in tanks, vapour pressure / self-pressurisation
(N2O, CO2), pump work, and coolant properties in regenerative-cooling channels.
Burned gases are handled by NASA CEA (rocket chambers) and Cantera (gas turbines).
"""

from __future__ import annotations

from dataclasses import dataclass

from aerosim.core import AerosimError, require

# aerosim propellant alias -> (CoolProp fluid, is_surrogate)
COOLPROP_NAMES: dict[str, tuple[str, bool]] = {
    "LOX": ("Oxygen", False), "O2": ("Oxygen", False),
    "LH2": ("ParaHydrogen", False), "H2": ("Hydrogen", False),
    "LCH4": ("Methane", False), "Methane": ("Methane", False),
    "Ethanol": ("Ethanol", False), "Methanol": ("Methanol", False),
    "N2O": ("NitrousOxide", False), "Water": ("Water", False),
    "Ammonia": ("Ammonia", False), "Propane": ("n-Propane", False),
    "CO2": ("CarbonDioxide", False), "N2": ("Nitrogen", False), "He": ("Helium", False),
    # no RP-1 equation of state in CoolProp: n-dodecane is the usual surrogate
    "RP1": ("n-Dodecane", True), "RP-1": ("n-Dodecane", True),
    "Kerosene": ("n-Dodecane", True), "JetA": ("n-Dodecane", True),
}


def _cp():
    return require("CoolProp.CoolProp", "fluids")


def coolprop_name(name: str) -> str:
    return COOLPROP_NAMES.get(name, (name, False))[0]


def props(output: str, name1: str, value1: float, name2: str, value2: float, fluid: str) -> float:
    """Direct PropsSI access with aerosim aliases, e.g. props('D', 'T', 90, 'P', 3e5, 'LOX')."""
    try:
        return float(_cp().PropsSI(output, name1, value1, name2, value2, coolprop_name(fluid)))
    except ValueError as e:
        raise AerosimError(f"CoolProp: {e}") from e


@dataclass(frozen=True)
class Saturation:
    temperature: float          # K
    pressure: float             # Pa
    rho_liquid: float           # kg/m^3
    rho_vapor: float
    h_liquid: float             # J/kg
    h_vapor: float

    @property
    def latent_heat(self) -> float:
        return self.h_vapor - self.h_liquid


class Fluid:
    """A pure propellant / pressurant with real-gas properties."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.coolprop, self.surrogate = COOLPROP_NAMES.get(name, (name, False))
        try:
            self.t_crit = props("Tcrit", "T", 0, "P", 0, self.coolprop)
            self.p_crit = props("Pcrit", "T", 0, "P", 0, self.coolprop)
            self.t_triple = props("Ttriple", "T", 0, "P", 0, self.coolprop)
        except AerosimError as e:
            raise AerosimError(f"unknown CoolProp fluid for '{name}'") from e

    def density(self, T: float, P: float) -> float:
        return props("D", "T", T, "P", P, self.coolprop)

    def enthalpy(self, T: float, P: float) -> float:
        return props("H", "T", T, "P", P, self.coolprop)

    def cp(self, T: float, P: float) -> float:
        return props("C", "T", T, "P", P, self.coolprop)

    def viscosity(self, T: float, P: float) -> float:
        return props("V", "T", T, "P", P, self.coolprop)

    def conductivity(self, T: float, P: float) -> float:
        return props("L", "T", T, "P", P, self.coolprop)

    def saturation(self, T: float | None = None, P: float | None = None) -> Saturation:
        """Saturation state at a temperature or a pressure (give exactly one)."""
        if (T is None) == (P is None):
            raise ValueError("give exactly one of T or P")
        k, v = ("T", T) if T is not None else ("P", P)
        if T is not None and not (self.t_triple <= T < self.t_crit):
            raise AerosimError(f"{self.name}: T={T} K outside saturation range")
        return Saturation(
            temperature=props("T", k, v, "Q", 0, self.coolprop),
            pressure=props("P", k, v, "Q", 0, self.coolprop),
            rho_liquid=props("D", k, v, "Q", 0, self.coolprop),
            rho_vapor=props("D", k, v, "Q", 1, self.coolprop),
            h_liquid=props("H", k, v, "Q", 0, self.coolprop),
            h_vapor=props("H", k, v, "Q", 1, self.coolprop),
        )

    def pump_power(self, mdot: float, dp: float, T: float, P_in: float,
                   efficiency: float = 0.7) -> float:
        """Shaft power (W) to raise mdot by dp, incompressible approximation."""
        return mdot * dp / (self.density(T, P_in) * efficiency)


def blend_density(components: dict[str, float], T: float, P: float) -> float:
    """Density of a liquid blend by mass fraction, ideal volume mixing.

    Real ethanol/water mixes shrink ~2-4 % in volume; good enough for tank sizing.
    """
    total = sum(components.values())
    return 1.0 / sum((w / total) / Fluid(n).density(T, P) for n, w in components.items())
