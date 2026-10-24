import math

import numpy as np
import pytest

from aerosim.flight import orbits
from aerosim.flight.rocket.orbital_ascent import AscentProgram, OrbitalAscentSim
from aerosim.flight.vehicles import LaunchVehicle, Stage


def test_circular_orbit_elements():
    r = orbits.RE + 500e3
    v = math.sqrt(orbits.MU / r)
    o = orbits.Orbit.from_state(np.array([r, 0, 0]), np.array([0, v * math.cos(0.5), v * math.sin(0.5)]))
    assert o.perigee_alt == pytest.approx(500e3, abs=1)
    assert o.apogee_alt == pytest.approx(500e3, abs=1)
    assert o.inclination == pytest.approx(math.degrees(0.5))
    assert o.period == pytest.approx(2 * math.pi * math.sqrt(r**3 / orbits.MU))


def test_inertial_azimuth():
    # due-east launch gives inclination = latitude; polar = due north
    assert orbits.inertial_azimuth(28.5, 28.5) == pytest.approx(90.0)
    assert orbits.inertial_azimuth(62.8, 90.0) == pytest.approx(0.0, abs=1e-9)
    with pytest.raises(ValueError):
        orbits.inertial_azimuth(62.8, 30.0)


def test_altitude_on_ellipsoid():
    r0, _ = orbits.site_state(62.8, 40.3)
    assert orbits.altitude(r0) == pytest.approx(0.0, abs=100.0)


@pytest.fixture(scope="module")
def two_stage_flight():
    veh = LaunchVehicle(
        stages=[Stage("S1", 20_000, 2_000, 300, 120, isp_sl=270),
                Stage("S2", 4_000, 500, 330, 400, max_burns=2)],
        payload_mass=200, fairing_mass=100, diameter=1.5)
    prog = AscentProgram(target_inclination=51.6, kick_angle=12.0, vertical_time=6,
                         kick_time=10, upper_pitch=[], insertion_pitch=(12.0, -0.03),
                         transfer_apogee=300e3)
    return OrbitalAscentSim(45.9, 63.3, program=prog).run(veh)


def test_ascent_reaches_target_orbit(two_stage_flight):
    res = two_stage_flight
    assert res.meta["warnings"] == []
    assert res.orbits["transfer"].apogee_alt == pytest.approx(300e3, abs=1e3)
    assert res.final_orbit.perigee_alt == pytest.approx(300e3, abs=2e3)   # circularised
    assert res.final_orbit.inclination == pytest.approx(51.6, abs=0.5)
    assert "fairing_jettison" in res.events
    assert res.events["max_q"] < res.events["S1_burnout"]
    assert np.all(np.diff(res.time) > 0)
    # mass only ever decreases
    assert np.all(np.diff(res.mass) <= 1e-6)
