"""pyCycle turbojet / turbofan against published engine data (J79, CFM56-7B27)."""

import pytest

pytest.importorskip("pycycle")

from aerosim.core import AerosimError
from aerosim.propulsion.adapters.pycycle_engine import PyCycleEngine


def _build(engine):
    try:
        return engine.design_point()
    except (AerosimError, ValueError) as e:
        if "array element with a sequence" in str(e):
            pytest.skip("om-pycycle 4.4.0 is incompatible with numpy >= 2.5 (fixed on its master)")
        raise


def test_turbojet_j79():
    # J79-GE-17 (Wikipedia): 52.8 kN dry, OPR 13.5, 77 kg/s, TIT 1210 K, 0.84 lb/(lbf h)
    d = _build(PyCycleEngine("turbojet"))
    assert d["thrust"] == pytest.approx(52.8e3, rel=1e-3)
    assert d["air_flow"] == pytest.approx(77, rel=0.15)
    # no turbine cooling or bleed air in this cycle, so TSFC comes out ~11 % low
    assert d["tsfc_lb_lbf_h"] == pytest.approx(0.84, rel=0.15)
    od = PyCycleEngine("turbojet")
    od.design_point()
    far = od.operating_point(11e3, 0.9, 1.0)        # reached by continuation
    assert 0 < far["thrust"] < d["thrust"] and far["tsfc_lb_lbf_h"] > d["tsfc_lb_lbf_h"]


def test_turbofan_cfm56():
    # CFM56-7B27 (Wikipedia): 121 kN takeoff, BPR 5.1, OPR 32.7, takeoff TSFC 0.356-0.386,
    # sea-level air flow 307-355 kg/s (7B family)
    e = PyCycleEngine("turbofan")
    d = _build(e)
    assert d["opr"] == pytest.approx(32.7, rel=0.03)
    assert 0.356 <= d["tsfc_lb_lbf_h"] <= 0.386
    assert 307 <= d["air_flow"] <= 355
    cruise = e.operating_point(10.7e3, 0.78, 0.8)
    assert cruise["tsfc_lb_lbf_h"] > d["tsfc_lb_lbf_h"]
