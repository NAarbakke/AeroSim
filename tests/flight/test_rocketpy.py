"""RocketPy 6-DOF vs built-in point-mass on the same vehicle, motor and drag."""

from pathlib import Path

import pytest
import yaml

pytest.importorskip("rocketpy")

from aerosim.missions import run_mission

CFG = Path(__file__).parents[2] / "configs" / "solid" / "knsb_isa.yaml"


def test_rocketpy_matches_point_mass_without_wind():
    cfg = yaml.safe_load(CFG.read_text())
    cfg["atmosphere"]["wind_speed"] = 0.0
    pm = run_mission(cfg).flight
    cfg["flight"] = {"backend": "rocketpy"}
    rp = run_mission(cfg).flight
    assert rp.meta["simulator"] == "rocketpy"
    assert rp.apogee == pytest.approx(pm.apogee, rel=0.01)
    assert rp.max_speed == pytest.approx(pm.max_speed, rel=0.01)
    assert rp.events["apogee"] == pytest.approx(pm.events["apogee"], rel=0.02)
    assert rp.position[-1, 0] == pytest.approx(pm.position[-1, 0], rel=0.05)   # ground range
