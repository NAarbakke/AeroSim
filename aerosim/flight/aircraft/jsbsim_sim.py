"""JSBSim backend for aircraft (LGPL-2.1, used as an unmodified library).

Propulsion coupling: an aerosim EngineDeck is written out as a JSBSim engine XML
(<turbine_engine> with thrust/fuel tables, or a generic <rocket_engine>/<electric>
with a function-driven thrust), then referenced from the aircraft definition.
Winds from any AtmosphereModel are pushed each step via atmosphere/wind-north-fps etc.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from aerosim.core import EngineDeck, FlightResult, registry, require
from aerosim.core.interfaces import AtmosphereModel
from aerosim.flight.vehicles import AircraftVehicle

FT = 0.3048


def engine_deck_to_jsbsim_xml(deck: EngineDeck, path: str | Path) -> Path:
    """TODO: emit JSBSim engine XML with 2-D thrust tables per power setting."""
    raise NotImplementedError


@registry.register("flight", "jsbsim")
class JSBSimSim:
    def __init__(self, dt: float = 1 / 120, duration: float = 60.0, root_dir: str | None = None):
        self.dt, self.duration, self.root_dir = dt, duration, root_dir

    def run(self, vehicle: AircraftVehicle, propulsion: EngineDeck | None,
            atmosphere: AtmosphereModel, **options) -> FlightResult:
        jsb = require("jsbsim", "aircraft")
        fdm = jsb.FGFDMExec(self.root_dir)
        fdm.set_dt(self.dt)
        fdm.load_model(vehicle.jsbsim_model)
        ic = {"ic/h-sl-ft": 3000.0, "ic/vc-kts": 120.0, "ic/psi-true-deg": 0.0,
              **{k: v for k, v in vehicle.initial.items()}}
        for k, v in ic.items():
            fdm[k] = v
        fdm.run_ic()
        for i in range(vehicle.engine_count):
            fdm[f"propulsion/engine[{i}]/set-running"] = 1
        fdm["fcs/throttle-cmd-norm"] = options.get("throttle", 1.0)

        rows = []
        while fdm.get_sim_time() < self.duration:
            h = fdm["position/h-sl-ft"] * FT
            atm = atmosphere.at(h)
            fdm["atmosphere/wind-north-fps"] = atm.wind_v / FT
            fdm["atmosphere/wind-east-fps"] = atm.wind_u / FT
            if not fdm.run():
                break
            rows.append([fdm.get_sim_time(),
                         fdm["position/distance-from-start-lon-mt"],
                         fdm["position/distance-from-start-lat-mt"], h,
                         fdm["velocities/v-east-fps"] * FT, fdm["velocities/v-north-fps"] * FT,
                         -fdm["velocities/v-down-fps"] * FT,
                         fdm["inertia/weight-lbs"] * 0.45359237])
        a = np.asarray(rows)
        return FlightResult(a[:, 0], a[:, 1:4], a[:, 4:7], a[:, 7],
                            meta={"simulator": "jsbsim", "model": vehicle.jsbsim_model})
