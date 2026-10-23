"""Closed-loop examples: air brakes to a target apogee, and TVC on an unstable rocket.

Needs: pip install -e '.[control]'
"""

import math

import numpy as np

from aerosim.atmosphere import StandardAtmosphere
from aerosim.flight import RocketVehicle
from aerosim.flight.rocket import PlanarRocketSim, PointMassRocketSim
from aerosim.gnc import AirbrakeApogeeController, AttitudeHoldTVC, Sensor, design
from aerosim.propulsion.solid import BatesMotor

motor = BatesMotor(0.05, 0.02, 0.09, 4, 0.012, 6.0, dry_mass=0.6).thrust_curve()
atm = StandardAtmosphere(wind_speed=5)

# 1) air brakes ------------------------------------------------------------------------
rocket = RocketVehicle(dry_mass=3.0, diameter=0.066, rail_length=3.0, elevation_deg=88,
                       airbrake_cd_area=0.012)
free = PointMassRocketSim().run(rocket, motor, atm).apogee
braked = PointMassRocketSim(controller=AirbrakeApogeeController(target_apogee=2200)).run(rocket, motor, atm)
print(f"air brakes: open loop {free:.0f} m -> controlled {braked.apogee:.0f} m (target 2200)")

# 2) TVC: design with python-control, verify in the nonlinear planar sim -----------------
finless = RocketVehicle(dry_mass=3.0, diameter=0.066, rail_length=3.0, elevation_deg=90,
                        pitch_inertia=0.8, cg_to_nozzle=0.6, static_margin=-0.05)
op = design.TVCOperatingPoint(thrust=motor.total_impulse / motor.burn_time, lever_arm=0.6,
                              inertia=0.68, dynamic_pressure=0.5 * 1.2 * 150**2,
                              ref_area=finless.reference_area, cn_alpha=2.0, static_margin=-0.05)
gains = design.pd_by_pole_placement(op, wn=12, zeta=0.7)
loop = design.analyze_attitude_loop(op, gains["kp"], gains["kd"])
print(f"TVC design: kp={gains['kp']:.3f} kd={gains['kd']:.4f}  PM={loop['phase_margin_deg']:.0f} deg  "
      f"lower GM={loop['lower_gain_margin_db']:.0f} dB  overshoot={loop['overshoot_pct']:.0f} %")

gust = (0.3, 0.2, 8.0)   # 8 m/s side gust at t=0.3 s for 0.2 s
for name, ctrl in (("no TVC", None),
                   ("TVC", AttitudeHoldTVC(kp=gains["kp"], kd=gains["kd"], design_thrust=op.thrust))):
    r = PlanarRocketSim(controller=ctrl, gust=gust, sensor=Sensor({"q": 0.002})).run(finless, motor, atm)
    burn = r.time < motor.time[-1]
    dev = np.abs(np.degrees(r.meta["theta"][burn]) - 90).max()
    print(f"{name:>7}: max tilt during burn {dev:.2f} deg, max gimbal "
          f"{math.degrees(np.abs(r.meta['gimbal']).max()):.2f} deg")
