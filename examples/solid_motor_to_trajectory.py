"""Use the layers directly from Python instead of a YAML mission."""

from aerosim.atmosphere import StandardAtmosphere
from aerosim.core import write_eng
from aerosim.flight import RocketVehicle
from aerosim.flight.rocket import PointMassRocketSim
from aerosim.propulsion.solid import BatesMotor

motor = BatesMotor(grain_outer_diameter=0.05, grain_core_diameter=0.02, grain_length=0.09,
                   grain_count=4, throat_diameter=0.012, expansion_ratio=6.0, dry_mass=0.6)
curve = motor.thrust_curve()
print(f"Impulse {curve.total_impulse:.0f} N·s, burn {curve.burn_time:.2f} s, "
      f"Isp {curve.average_isp:.0f} s")
write_eng(curve, "bates_4grain.eng")          # open in OpenRocket too

rocket = RocketVehicle(dry_mass=3.0, diameter=0.066, rail_length=3.0, elevation_deg=85)
for wind in (0, 5, 10):
    r = PointMassRocketSim().run(rocket, curve, StandardAtmosphere(wind_speed=wind))
    print(f"wind {wind:>2} m/s -> apogee {r.apogee:6.0f} m, "
          f"drift {abs(r.position[-1, 0]):5.0f} m")
