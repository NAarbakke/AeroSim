from pathlib import Path

import pytest

from aerosim.core import read_eng, write_eng
from aerosim.missions import run_mission

CFG = Path(__file__).parents[2] / "configs" / "examples" / "rocket" / "solid" / "knsb_isa.yaml"


def test_solid_mission_end_to_end():
    out = run_mission(CFG)
    r = out.flight
    assert 300 < r.apogee < 5000
    assert r.events["apogee"] < r.events["impact"]
    # wind from the west pushes the rocket east (it weathercocks upwind, but
    # with an east heading the downrange x must be positive at apogee)
    assert r.position[-1, 0] > 0


def test_eng_roundtrip(tmp_path):
    tc = run_mission(CFG).propulsion
    back = read_eng(write_eng(tc, tmp_path / "m.eng"))
    assert back.total_impulse == pytest.approx(tc.total_impulse, rel=0.02)
