"""OpenMDAO wrappers: expose aerosim models as components so engine design and
trajectory can be optimized together (pyCycle and Dymos are native OpenMDAO).

Examples of problems this layer is for:
- size a BATES grain + nozzle to hit a target apogee under a forecast wind
- choose liquid-engine Pc / MR / expansion ratio for max delta-v at fixed tank mass
- couple a pyCycle turbojet with a Dymos climb trajectory (min fuel to altitude)
"""

from __future__ import annotations

from aerosim.core import require


def solid_apogee_component():
    """Return an OpenMDAO ExplicitComponent: grain/nozzle geometry -> apogee.

    Uses finite differences around the built-in BatesMotor + PointMassRocketSim,
    which are fast enough for gradient-based optimization.
    """
    om = require("openmdao.api", "optim")
    from aerosim.atmosphere import StandardAtmosphere
    from aerosim.flight.rocket import PointMassRocketSim
    from aerosim.propulsion.solid import BatesMotor

    class SolidApogee(om.ExplicitComponent):
        def initialize(self):
            self.options.declare("vehicle")
            self.options.declare("motor_fixed", default={})
            self.options.declare("atmosphere", default=None)

        def setup(self):
            self.add_input("core_diameter", 0.02, units="m")
            self.add_input("grain_length", 0.10, units="m")
            self.add_input("throat_diameter", 0.012, units="m")
            self.add_output("apogee", 0.0, units="m")
            self.add_output("max_thrust", 0.0, units="N")
            self.declare_partials("*", "*", method="fd")

        def compute(self, inputs, outputs):
            motor = BatesMotor(grain_core_diameter=float(inputs["core_diameter"][0]),
                               grain_length=float(inputs["grain_length"][0]),
                               throat_diameter=float(inputs["throat_diameter"][0]),
                               **self.options["motor_fixed"])
            tc = motor.thrust_curve()
            atm = self.options["atmosphere"] or StandardAtmosphere()
            res = PointMassRocketSim().run(self.options["vehicle"], tc, atm)
            outputs["apogee"] = res.apogee
            outputs["max_thrust"] = tc.thrust.max()

    return SolidApogee
