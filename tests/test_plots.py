"""Every plot renders (Agg backend) and returns a Figure."""

import numpy as np
import pytest

mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")

from aerosim import plots
from aerosim.atmosphere import StandardAtmosphere
from aerosim.flight import RocketVehicle
from aerosim.flight.rocket import PointMassRocketSim
from aerosim.propulsion.solid import BatesMotor, GrainMotor
from aerosim.propulsion.solid.grains import Bates


def test_solid_and_flight_plots(tmp_path):
    curve = BatesMotor(0.05, 0.02, 0.09, 4, 0.012, 6.0, name="4-grain").thrust_curve()
    gm = GrainMotor(grains=[{"type": "bates", "diameter": 0.05, "length": 0.09,
                             "core_diameter": 0.02}],
                    nozzle={"throat_diameter": 0.008, "exit_diameter": 0.02})
    flight = PointMassRocketSim().run(RocketVehicle(dry_mass=3, diameter=0.066), curve,
                                      StandardAtmosphere())
    figs = [plots.thrust_curve(curve), plots.thrust_curve([curve, gm.thrust_curve()]),
            plots.motor_history(gm.simulate()), plots.grain_burnback(Bates(0.05, 0.09,
                                                                           core_diameter=0.02)),
            plots.flight_profile(flight)]
    for i, f in enumerate(figs):
        f.savefig(tmp_path / f"{i}.png")
        assert (tmp_path / f"{i}.png").stat().st_size > 10_000


def test_series_order_is_fixed():
    assert plots.series(0) != plots.series(1)
    with pytest.raises(ValueError):
        plots.series(8)
    with plots.themed("dark") as t:
        assert t["surface"] == "#1a1a19"
        assert plots.series(0) == "#3987e5"
    assert np.all([plots.tokens()["surface"] == "#fcfcfb"])
