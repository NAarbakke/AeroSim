"""Propellant names -> NASA CEA thermo-database species, with storage temperatures.

Configs can use a friendly alias ("LOX", "Ethanol"), a raw CEA species name
("C2H5OH(L)"), or a blend by mass fraction ({"Ethanol": 0.75, "Water": 0.25}).
Temperatures for liquids default to their normal boiling point / storage
temperature; override with {"name": ..., "T": ...} entries if needed.
"""

from __future__ import annotations

from dataclasses import dataclass

# alias -> (CEA species, default temperature K)
ALIASES: dict[str, tuple[str, float]] = {
    # oxidizers
    "LOX": ("O2(L)", 90.17),
    "O2": ("O2", 298.15),
    "N2O": ("N2O", 298.15),          # CEA has gaseous N2O only; fine for chamber enthalpy at 298 K
    "N2O4": ("N2O4(L)", 298.15),
    "NTO": ("N2O4(L)", 298.15),
    "H2O2": ("H2O2(L)", 298.15),
    "HTP": ("H2O2(L)", 298.15),      # blend with Water for <100% peroxide
    # fuels
    "LH2": ("H2(L)", 20.27),
    "H2": ("H2", 298.15),
    "LCH4": ("CH4(L)", 111.64),
    "Methane": ("CH4(L)", 111.64),
    "Ethanol": ("C2H5OH(L)", 298.15),
    "Methanol": ("CH3OH(L)", 298.15),
    "RP1": ("RP-1", 298.15),
    "RP-1": ("RP-1", 298.15),
    "Kerosene": ("RP-1", 298.15),
    "JetA": ("Jet-A(L)", 298.15),
    "Hydrazine": ("N2H4(L)", 298.15),
    "N2H4": ("N2H4(L)", 298.15),
    "MMH": ("CH6N2(L)", 298.15),
    "UDMH": ("C2H8N2(L),UDMH", 298.15),
    "Ammonia": ("NH3(L)", 239.82),
    "Water": ("H2O(L)", 298.15),
}


@dataclass(frozen=True)
class Component:
    species: str
    mass_fraction: float
    temperature: float


def resolve(spec) -> list[Component]:
    """Turn a propellant spec into CEA components.

    Accepted forms:
      "LOX" | "C2H5OH(L)"
      {"Ethanol": 0.75, "Water": 0.25}
      [{"name": "Ethanol", "fraction": 0.75, "T": 290}, {"name": "Water", "fraction": 0.25}]
    """
    if isinstance(spec, str):
        items = [{"name": spec, "fraction": 1.0}]
    elif isinstance(spec, dict):
        items = [{"name": k, "fraction": v} for k, v in spec.items()]
    else:
        items = list(spec)

    total = sum(float(i["fraction"]) for i in items)
    out = []
    for i in items:
        species, t_default = ALIASES.get(i["name"], (i["name"], 298.15))
        out.append(Component(species, float(i["fraction"]) / total, float(i.get("T", t_default))))
    return out


def key(spec) -> tuple:
    """Hashable identity of a propellant spec (for solver caching)."""
    return tuple((c.species, round(c.mass_fraction, 9), c.temperature) for c in resolve(spec))
