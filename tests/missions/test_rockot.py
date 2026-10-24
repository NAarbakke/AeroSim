"""Rockot / Breeze-KM config (calibrated to the User's Guide reference mission) still flies it."""

from pathlib import Path

import pytest

from aerosim.missions import run_mission

CFG = Path(__file__).parents[2] / "configs" / "examples" / "rocket" / "orbital" / "rockot.yaml"


@pytest.fixture(scope="module")
def rockot():
    return run_mission(CFG).flight


def test_reaches_guide_transfer_orbit(rockot):
    tr = rockot.orbits["transfer"]
    assert tr.perigee_alt == pytest.approx(160e3, abs=5e3)        # guide fig. 3-6
    assert tr.apogee_alt == pytest.approx(635.3e3, abs=1e3)
    assert rockot.final_orbit.inclination == pytest.approx(86.583, abs=0.5)
    assert rockot.meta["warnings"] == []


def test_stage_burnouts_match_guide(rockot):
    s1, s2 = rockot.at("Stage 1_burnout"), rockot.at("Stage 2_burnout")
    assert s1["speed"] == pytest.approx(3163, rel=0.02)
    assert s2["speed"] == pytest.approx(5634, rel=0.02)
    assert rockot.at("max_q")["dynamic_pressure"] == pytest.approx(61.29e3, rel=0.10)
