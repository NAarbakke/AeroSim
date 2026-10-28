"""Strela, Start-1, Long March 2C / CTS and Dnepr configs still fly their user's-manual missions
(calibrated or optimised by examples/slv_ascent.py; Rockot: test_rockot.py)."""

from pathlib import Path

import pytest

from aerosim.missions import run_mission

CFG = Path(__file__).parents[2] / "configs" / "orbital"


@pytest.fixture(scope="module")
def flights():
    return {}


def fly(flights, name):
    if name not in flights:
        flights[name] = run_mission(CFG / f"{name}.yaml").flight
    return flights[name]


# (event, guide speed m/s, guide altitude m): Strela UM fig. 3.4, Start-1 UH fig. 2-3,
# LM-2C UM table 3-2
WAYPOINTS = {
    "strela": [("Stage 1_separation", 3555, 70e3), ("Stage 2_burnout", 7727, 240e3)],
    "long_march_2c": [("Stage 1_burnout", 2036, 47.05e3), ("Stage 2_burnout", 6379, 146.9e3),
                      ("Stage 2 verniers_burnout", 7918, 181.1e3)],
    "start1": [("Stage 1_separation", 1283, 39.7e3), ("Stage 2_separation", 3347, 86.5e3),
               ("Stage 3_burnout", 5535, 167.9e3)],
}
# final orbit: altitude m, tolerance m, inclination deg
ORBITS = {
    "strela": (1000e3, 25e3, 90.0),
    "long_march_2c": (637.8e3, 40e3, 86.1),     # the CTS is a fixed-impulse solid
    "dnepr": (300e3, 10e3, 50.5),
}


@pytest.mark.parametrize("name", list(WAYPOINTS))
def test_waypoints_match_guide(flights, name):
    res = fly(flights, name)
    for ev, v, h in WAYPOINTS[name]:
        a = res.at(ev)
        assert a["speed"] == pytest.approx(v, rel=0.02), ev
        assert a["altitude"] == pytest.approx(h, abs=6e3), ev


@pytest.mark.parametrize("name", list(ORBITS))
def test_reaches_guide_orbit(flights, name):
    res = fly(flights, name)
    h, tol, inc = ORBITS[name]
    o = res.final_orbit
    assert o.perigee_alt == pytest.approx(h, abs=tol) and o.apogee_alt == pytest.approx(h, abs=tol)
    assert o.inclination == pytest.approx(inc, abs=0.5)
    assert res.meta["warnings"] == [] and res.meta["propellant_left"] >= 0


def test_long_march_peak_load_matches_manual(flights):
    """Not fitted: the stage 2 cutoff load follows from the fitted masses (UM s 6.3.3.1: 6.7 g)."""
    res = fly(flights, "long_march_2c")
    assert res.acceleration.max() / 9.80665 == pytest.approx(6.7, rel=0.05)


def test_dnepr_carries_guide_payload(flights):
    """Optimised, not calibrated: the guide's 3700 kg leaves stage 3 propellant over."""
    assert fly(flights, "dnepr").meta["propellant_left"] > 0


def test_start1_reaches_orbit(flights):
    """Stage 4 ends near the guide's 481 km: perigee on it, apogee raised by the prograde burn."""
    o = fly(flights, "start1").final_orbit
    assert o.perigee_alt == pytest.approx(481e3, abs=10e3) and o.apogee_alt < 570e3
    assert o.inclination == pytest.approx(97.3, abs=0.5)
