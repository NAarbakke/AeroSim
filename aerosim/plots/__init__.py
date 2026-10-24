"""AeroSim plotting (matplotlib, ``pip install -e '.[plot]'``).

Every function returns a matplotlib Figure; save with ``fig.savefig(path)``.

solid   thrust_curve, motor_history (GrainMotor.simulate()), grain_burnback
flight  flight_profile (FlightResult), ascent_overview, ascent_trajectory, ground_track
        (AscentResult from the orbital_ascent backend)
style   palette tokens, ``themed("dark")`` context, helpers for custom charts
"""

from aerosim.plots.flight import ascent_overview, ascent_trajectory, flight_profile, ground_track
from aerosim.plots.solid import grain_burnback, motor_history, thrust_curve
from aerosim.plots.style import series, themed, tokens

__all__ = ["ascent_overview", "ascent_trajectory", "flight_profile", "grain_burnback",
           "ground_track", "motor_history", "series", "themed", "thrust_curve", "tokens"]
