"""Regenerative cooling via pyskyfire, and the expander cycle closed on it."""

import pytest

pytest.importorskip("pyskyfire")

from aerosim.propulsion.adapters.regen import regen_cooling
from aerosim.propulsion.liquid.cycle import CycleEngine
from aerosim.thermo.fluids import props


def test_lox_methane_chamber():
    r = regen_cooling("LOX", "LCH4", 3.4, 10e6, 100e3, 30)
    s = r.summary()
    assert r.t_coolant[0] == pytest.approx(111.64, abs=0.5)       # enters at storage T
    assert 200 < s["coolant_outlet_temperature"] < 600
    assert s["pressure_drop"] > 0
    assert 400 < s["max_wall_temperature"] < 1300                  # GRCop-42 limit ~1000 K class
    assert s["max_heat_flux"] > 1e7
    # heat load is the coolant's enthalpy rise
    h = [props("H", "T", r.t_coolant[i], "P", r.p_coolant[i], "Methane") for i in (0, -1)]
    assert s["heat_load"] == pytest.approx(r.mdot_coolant * (h[1] - h[0]), rel=1e-9)


def test_expander_closes_on_pyskyfire_jacket():
    e = CycleEngine("expander", "LOX", "LH2", 5.0, 3.275e6, 61, 73.4e3,
                    regen={"material": "ss347", "channel_height": 4e-3, "l_star": 0.9})
    o = e.solve()
    assert 100 < o["turbine_inlet_temperature"] < 400
    assert o["cooling_dp"] == pytest.approx(o["regen"]["pressure_drop"])
    assert o["isp_vac"] == pytest.approx(444.4, rel=0.03)          # RL10A-3-3A
