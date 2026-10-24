"""Mission runner: a YAML file picks one backend per layer and wires them together.

    propulsion  -> ThrustCurve | EngineDeck   (kind: launch_vehicle -> stages in the vehicle)
    atmosphere  -> AtmosphereModel
    flight      -> FlightResult

Swapping a backend (bates -> openmotor, isa -> forecast, point_mass -> rocketpy)
is a one-line change in the config, never a code change.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

import aerosim.atmosphere
import aerosim.flight
import aerosim.propulsion
import aerosim.thermo  # noqa: F401
from aerosim.core import FlightResult, registry
from aerosim.flight.vehicles import AircraftVehicle, LaunchVehicle, RocketVehicle


@dataclass
class MissionOutput:
    propulsion: Any
    atmosphere: Any
    flight: FlightResult


def _backend(section: dict, category: str):
    cfg = dict(section)
    name = cfg.pop("backend")
    return registry.create(category, name, **cfg)


def run_mission(config: dict | str | Path) -> MissionOutput:
    if not isinstance(config, dict):
        config = yaml.safe_load(Path(config).read_text())

    kind = config.get("kind", "rocket")
    if kind == "launch_vehicle":   # stages carry their own propulsion
        vehicle = LaunchVehicle(**config["vehicle"])
        atmosphere = _backend(config.get("atmosphere", {"backend": "isa"}), "atmosphere")
        sim = _backend(config.get("flight", {"backend": "orbital_ascent"}), "flight")
        return MissionOutput(vehicle, atmosphere, sim.run(vehicle, None, atmosphere))
    engine = _backend(config["propulsion"], "rocket_engine" if kind == "rocket" else "airbreathing")
    if kind == "rocket":
        propulsion = engine.thrust_curve()
        vehicle = RocketVehicle(**config["vehicle"])
    else:
        d = config["propulsion_deck"]
        propulsion = engine.deck(d["altitudes"], d["machs"], d["powers"])
        vehicle = AircraftVehicle(**config["vehicle"])

    atmosphere = _backend(config.get("atmosphere", {"backend": "isa"}), "atmosphere")
    sim = _backend(config.get("flight", {"backend": "point_mass"}), "flight")
    result = sim.run(vehicle, propulsion, atmosphere)
    return MissionOutput(propulsion, atmosphere, result)
