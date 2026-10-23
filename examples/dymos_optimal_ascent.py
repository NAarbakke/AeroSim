"""Optimal throttle profile for maximum altitude (Goddard problem) with Dymos.

Needs: pip install -e '.[optim]'
"""

import numpy as np

from aerosim.optimization.trajectory import optimal_vertical_ascent, simulate_vertical_ascent

rocket = {"m0": 20.0, "propellant_mass": 8.0, "thrust_max": 2500.0, "isp": 230.0, "cd_area": 0.5 * 0.018}

full, _ = simulate_vertical_ascent(**rocket)
best = optimal_vertical_ascent(**rocket)
capped = optimal_vertical_ascent(**rocket, max_q=15e3)

print(f"full throttle      : {full:7.0f} m")
print(f"optimal throttle   : {best.apogee:7.0f} m   (max q {best.dynamic_pressure.max()/1e3:.1f} kPa)")
print(f"optimal, q<=15 kPa : {capped.apogee:7.0f} m")
burn = best.throttle > 0
print("throttle every ~5 s:", np.round(np.interp(np.arange(0, best.time[burn][-1], 5),
                                                  best.time[burn], best.throttle[burn]), 2))
