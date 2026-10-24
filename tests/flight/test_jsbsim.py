"""JSBSim F-16 airframe flown on an AeroSim engine deck with the built-in autopilot."""

from pathlib import Path

import pytest

pytest.importorskip("jsbsim")
pytest.importorskip("cantera")

from aerosim.missions.runner import run_mission

CFG = Path(__file__).parents[2] / "configs" / "aircraft" / "turbojet.yaml"


def test_f16_on_deck_holds_targets():
    out = run_mission(CFG)
    r, m = out.flight, out.flight.meta
    assert r.position[-1, 2] == pytest.approx(4000, abs=30)
    assert m["mach"][-1] == pytest.approx(0.8, abs=0.01)
    assert m["heading_deg"][-1] == pytest.approx(90, abs=2)
    assert m["fuel_used"] > 50                                   # the deck's fuel flow drains the tanks
    assert (m["thrust"] > 0).all()
