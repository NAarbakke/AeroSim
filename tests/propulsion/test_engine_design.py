"""Engine design points from CEA and stages built from them."""

import math

import pytest

pytest.importorskip("cea")

from aerosim.core.units import G0
from aerosim.flight.vehicles import Stage
from aerosim.propulsion.liquid.design import engine_point

NTO_UDMH = {"oxidizer": "N2O4", "fuel": "UDMH"}


def test_isp_split_sets_exit_area():
    # RD-0233: the published vac / sea-level Isp gap fixes A_e/mdot, so eps follows
    e = engine_point(517.5e3, **NTO_UDMH, mixture_ratio=2.6, chamber_pressure=20.5e6,
                     isp_split=(310, 285), count=4)
    assert e["isp_vac"] - e["isp_sl"] == pytest.approx(25.0, abs=1e-6)
    assert 20 < e["expansion_ratio"] < 40
    assert e["isp_vac"] == pytest.approx(310, rel=0.03)            # predicted, not fitted
    assert e["thrust_vac"] == pytest.approx(4 * 517.5e3)
    assert e["mdot"] == pytest.approx(e["thrust_vac"] / (e["isp_vac"] * G0))
    # four 0.7 m nozzles fit the 2.5 m base
    assert math.sqrt(e["exit_area"] / 4 / math.pi) * 2 < 0.9


def test_exit_diameter_closes_geometry():
    e = engine_point(19.62e3, **NTO_UDMH, mixture_ratio=2.0, chamber_pressure=9.8e6,
                     exit_diameter=0.4)
    assert e["expansion_ratio"] * e["throat_area"] == pytest.approx(math.pi * 0.2**2, rel=2e-3)
    assert e["isp_ideal"] > e["isp_vac"] > 0.9 * e["isp_ideal"]


def test_losses_and_kinetics_order():
    kw = dict(thrust_vac=240e3, **NTO_UDMH, mixture_ratio=2.6, chamber_pressure=17.5e6,
              expansion_ratio=80)
    eq, mean, fr = (engine_point(**kw, kinetics=k)["isp_vac"] for k in ("equilibrium", "mean", "frozen"))
    assert eq > mean > fr
    ideal = engine_point(**kw, cstar_efficiency=1.0, cf_efficiency=1.0)
    assert ideal["isp_vac"] == pytest.approx(ideal["isp_ideal"])


def test_stage_from_engines_mixes_isp():
    st = Stage.from_engines("S2", [
        {"thrust_vac": 240e3, **NTO_UDMH, "mixture_ratio": 2.6, "chamber_pressure": 17.5e6,
         "expansion_ratio": 80},
        {"thrust_vac": 15.76e3, "isp_vac": 293}], dry_mass=1485, burn_time=183)
    main, vern = st.engines
    assert st.propellant_mass == pytest.approx((main["mdot"] + vern["mdot"]) * 183)
    assert vern["isp_vac"] < st.isp_vac < main["isp_vac"]
    assert st.thrust_vac == pytest.approx(255.76e3)
    up = Stage.from_engines("BK", [{"thrust_vac": 20e3, "isp_vac": 325.5}], dry_mass=1320,
                            propellant_mass=4975)
    assert up.burn_time == pytest.approx(4975 / (20e3 / (325.5 * G0)))
    assert up.isp_sl is None
