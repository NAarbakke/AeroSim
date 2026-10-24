"""openMotor backend (arbitrary grain geometry via Fast Marching Method).

LICENCE BOUNDARY: openMotor is GPL-3.0 and is NOT a dependency of aerosim.
This adapter imports it only at runtime if the user has installed it themselves
(clone github.com/reilleya/openMotor and put it on PYTHONPATH). See docs/licensing.md.

AeroSim's own ``grain_motor`` backend reads the same .ric files without openMotor;
this adapter remains as an independent cross-check.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from aerosim.core import ThrustCurve, registry, require
from aerosim.propulsion.solid.ric import RicLoader


@registry.register("rocket_engine", "openmotor")
class OpenMotorModel:
    def __init__(self, ric_file: str | Path, dry_mass: float = 0.0) -> None:
        self.ric_file = Path(ric_file)
        self.dry_mass = dry_mass

    def thrust_curve(self) -> ThrustCurve:
        motor_mod = require("motorlib.motor", "openmotor (install manually, GPL-3.0)")
        data = yaml.load(self.ric_file.read_text(), Loader=RicLoader)
        motor = motor_mod.Motor()
        motor.applyDict(data.get("data", data))
        sim = motor.runSimulation()
        # channel names checked against openMotor 0.6.2 (motorlib/simResult.py)
        ch = sim.channels
        t = np.asarray(ch["time"].getData(), float)
        F = np.asarray(ch["force"].getData(), float)
        mdot = np.asarray(ch["massFlow"].getData(), float)
        if mdot.ndim > 1:          # per-grain channels -> total
            mdot = mdot.sum(axis=1)
        m0 = float(np.asarray(ch["mass"].getData())[0].sum())
        return ThrustCurve(t, F, mdot, m0, self.dry_mass, self.ric_file.stem)
