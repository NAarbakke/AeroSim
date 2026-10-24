"""Component drag build-up."""

import pytest

from aerosim.flight import aero
from aerosim.flight.vehicles import LaunchVehicle, Stage


def test_drag_curve_shape():
    c = aero.components(29.15, 2.5, 3.7)
    m, cd = c["mach"], c["total"]
    peak = m[cd.argmax()]
    assert 0.95 <= peak <= 1.2                       # transonic drag rise
    assert cd[0] < 0.25 and cd.max() < 1.0 and cd[-1] < cd.max()
    assert (c["wave"][m <= 0.8] == 0).all() and (c["friction"] > 0).all()
    # a blunter nose has more wave drag
    assert aero.cd0(2.0, 29.15, 2.5, 2.0)["wave"] > aero.cd0(2.0, 29.15, 2.5, 5.0)["wave"]


def test_vehicle_accepts_geometry():
    veh = LaunchVehicle(stages=[Stage("S1", 1000, 100, 300, 60)], payload_mass=10,
                        diameter=2.5, drag_curve={"length": 29.15, "nose_length": 3.7})
    assert veh.cd(0.0) == pytest.approx(aero.cd0(0.0, 29.15, 2.5, 3.7)["total"], abs=1e-4)
    assert veh.cd(1.05) > veh.cd(3.0) > veh.cd(0.5)
