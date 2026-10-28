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


def test_optimize_program_improves_delivered_mass():
    from aerosim.flight.rocket.orbital_ascent import optimize_program

    veh = LaunchVehicle(
        stages=[Stage("S1", 20_000, 2_000, 300, 120, isp_sl=270),
                Stage("S2", 4_000, 500, 330, 400, max_burns=2)],
        payload_mass=200, fairing_mass=100, diameter=1.5)
    prog = AscentProgram(target_inclination=51.6, kick_angle=10.0, vertical_time=6, kick_time=10,
                         upper_pitch=[], transfer_apogee=300e3)
    sim = OrbitalAscentSim(45.9, 63.3, program=prog, rtol=1e-6)
    start = sim.run(veh).meta["propellant_left"]
    best, _veh, res, opt = optimize_program(
        sim, veh, {"kick_angle": (10.0, 6.0, 20.0)},
        constraints=[(lambda r, v: r.final_orbit.perigee_alt, 295e3, None, 2e3)], maxfev=14)
    assert res.meta["warnings"] == []
    assert res.meta["propellant_left"] >= start
    assert best.kick_angle == pytest.approx(opt.x_phys[0])


def test_kick_stage_coast_and_centred_apogee_burn():
    """Stage 1 coasts before separating; stage 2 alone reaches the transfer orbit and the kick
    stage only makes an apogee burn, started half its length before apogee."""
    from aerosim.flight.rocket.orbital_ascent import _time_to_apogee

    veh = LaunchVehicle(
        stages=[Stage("S1", 20_000, 2_000, 300, 120, isp_sl=270, coast=5.0),
                Stage("S2", 4_000, 400, 330, 250),
                Stage("Kick", 300, 100, 290, 120)],
        payload_mass=150, fairing_mass=100, diameter=1.5)
    prog = AscentProgram(target_inclination=51.6, kick_angle=12.0, vertical_time=6, kick_time=10,
                         upper_pitch=[[12.0, 4.0]], insertion_burn=False, centre_apogee_burn=True,
                         transfer_apogee=400e3)
    res = OrbitalAscentSim(45.9, 63.3, program=prog).run(veh)
    ev = res.events
    assert ev["S1_separation"] == pytest.approx(ev["S1_burnout"] + 5.0)
    assert "Kick_burn1_start" not in ev and ev["Kick_burn2_start"] > ev["S2_burnout"]
    # the burn straddles the transfer orbit's apogee
    i = int(np.searchsorted(res.time, ev["Kick_burn2_start"]))
    lead = _time_to_apogee(res.position[i], res.velocity[i])
    burn = ev["Kick_burn2_end"] - ev["Kick_burn2_start"]
    assert 0 < lead < burn
    o = res.final_orbit
    assert o.apogee_alt - o.perigee_alt < 0.5 * (res.orbits["transfer"].apogee_alt
                                                 - res.orbits["transfer"].perigee_alt)
