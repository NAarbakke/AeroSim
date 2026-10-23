"""openMotor backend (arbitrary grain geometry via Fast Marching Method).

LICENCE BOUNDARY: openMotor is GPL-3.0 and is NOT a dependency of aerosim.
This adapter imports it only at runtime if the user has installed it themselves
(clone github.com/reilleya/openMotor and put it on PYTHONPATH). See docs/licensing.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from aerosim.core import ThrustCurve, registry, require


class _RicLoader(yaml.SafeLoader):
    """Safe loader for .ric files. openMotor writes two Python-specific tags in the
    header (file-type enum, version tuple); read them as plain data instead of using
    yaml.Loader, which would execute arbitrary constructors and import PyQt6."""


_RicLoader.add_constructor("tag:yaml.org,2002:python/tuple",
                           lambda loader, node: tuple(loader.construct_sequence(node)))
_RicLoader.add_multi_constructor("tag:yaml.org,2002:python/object/apply:",
                                 lambda loader, suffix, node: loader.construct_sequence(node))


@registry.register("rocket_engine", "openmotor")
class OpenMotorModel:
    def __init__(self, ric_file: str | Path, dry_mass: float = 0.0) -> None:
        self.ric_file = Path(ric_file)
        self.dry_mass = dry_mass

    def thrust_curve(self) -> ThrustCurve:
        motor_mod = require("motorlib.motor", "openmotor (install manually, GPL-3.0)")
        data = yaml.load(self.ric_file.read_text(), Loader=_RicLoader)
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
