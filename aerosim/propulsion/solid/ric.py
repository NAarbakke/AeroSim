"""openMotor .ric motor files -> GrainMotor inputs (file format only; no openMotor code).

A .ric file is YAML: ``data: {config, grains, nozzle, propellant}`` plus a header
with two Python-specific tags (file-type enum, version tuple). They are read as
plain data with a safe loader; nothing in the file is executed.

Units in .ric are SI except tab molar mass (g/mol) and custom-grain outlines
(``dxfUnit``).
"""

from __future__ import annotations

from pathlib import Path

import yaml

from aerosim.propulsion.solid.grains import GRAIN_TYPES, Grain, MapGrain
from aerosim.propulsion.solid.propellant import BurnRateTab, TabbedPropellant


class RicLoader(yaml.SafeLoader):
    """SafeLoader that reads the .ric header's python/tuple and python/object/apply
    tags as plain sequences instead of constructing Python objects."""


RicLoader.add_constructor("tag:yaml.org,2002:python/tuple",
                          lambda loader, node: tuple(loader.construct_sequence(node)))
RicLoader.add_multi_constructor("tag:yaml.org,2002:python/object/apply:",
                                lambda loader, suffix, node: loader.construct_sequence(node))

_TYPES = {"BATES": "bates", "Finocyl": "finocyl", "Star Grain": "star", "X Core": "x_core",
          "C Grain": "c_grain", "D Grain": "d_grain", "Moon Burner": "moon_burner",
          "Rod and Tube": "rod_and_tube", "End Burner": "end_burner",
          "Custom Grain": "custom"}
_PROPS = {"diameter": "diameter", "length": "length", "inhibitedEnds": "inhibited_ends",
          "coreDiameter": "core_diameter", "finLength": "fin_length", "finWidth": "fin_width",
          "numFins": "num_fins", "numPoints": "num_points", "pointLength": "point_length",
          "pointWidth": "point_width", "slotLength": "slot_length", "slotWidth": "slot_width",
          "slotOffset": "slot_offset", "coreOffset": "core_offset",
          "rodDiameter": "rod_diameter"}
_DXF_UNIT = {"m": 1.0, "cm": 0.01, "mm": 0.001, "in": 0.0254, "ft": 0.3048}


def _grain(g: dict, map_dim: int) -> Grain:
    if g["type"] not in _TYPES:
        raise ValueError(f".ric grain type {g['type']!r} is not supported "
                         f"(supported: {sorted(_TYPES)})")
    cls = GRAIN_TYPES[_TYPES[g["type"]]]
    p = g["properties"]
    kw = {_PROPS[k]: v for k, v in p.items() if k in _PROPS}
    if "points" in p:
        s = _DXF_UNIT[p.get("dxfUnit", "m")]
        kw["polygons"] = [[(x * s, y * s) for x, y in poly] for poly in p["points"]]
    if issubclass(cls, MapGrain):
        kw["map_dim"] = int(map_dim)
    return cls(**kw)


def read_ric(path: str | Path) -> dict:
    """Return GrainMotor keyword arguments: grains, nozzle, propellant, limits,
    settings (ambient_pressure, timestep, burnout_web) and name."""
    path = Path(path)
    doc = yaml.load(path.read_text(), Loader=RicLoader)
    d = doc.get("data", doc)
    c, n, p = d["config"], d["nozzle"], d["propellant"]
    tabs = [BurnRateTab(a=t["a"], n=t["n"], gamma=t["k"], t_chamber=t["t"],
                        molar_mass=t["m"] / 1000.0, min_pressure=t["minPressure"],
                        max_pressure=t["maxPressure"]) for t in p["tabs"]]
    return {
        "grains": [_grain(g, c.get("mapDim", 501)) for g in d["grains"]],
        "nozzle": {"throat_diameter": n["throat"], "exit_diameter": n["exit"],
                   "efficiency": n["efficiency"], "divergence_deg": n["divAngle"],
                   "erosion_coeff": n.get("erosionCoeff", 0.0),
                   "slag_coeff": n.get("slagCoeff", 0.0)},
        "propellant": TabbedPropellant(p["name"], p["density"], tabs),
        "limits": {"max_pressure": c["maxPressure"], "max_mass_flux": c["maxMassFlux"],
                   "min_port_throat": c["minPortThroat"]},
        "settings": {"ambient_pressure": c["ambPressure"], "timestep": c["timestep"],
                     "burnout_web": c["burnoutWebThres"]},
        "name": path.parent.name if path.stem == "motor" else path.stem,
    }


def load_ric(path: str | Path, **overrides):
    """GrainMotor from a .ric file; keyword arguments override what the file says."""
    from aerosim.propulsion.solid.grain_motor import GrainMotor
    return GrainMotor(ric_file=path, **overrides)
