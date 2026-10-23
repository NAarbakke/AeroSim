"""Solid propellants via NASA CEA. Reference values: R. Nakka (KN), Sutton (APCP)."""

import pytest

pytest.importorskip("cea")

from aerosim.core import AerosimError
from aerosim.propulsion.solid import BatesMotor
from aerosim.thermo import solid_propellants as sp

GEO = {"grain_outer_diameter": 0.05, "grain_core_diameter": 0.02, "grain_length": 0.09,
           "grain_count": 4, "throat_diameter": 0.012, "expansion_ratio": 6.0}


def test_knsb_matches_nakka():
    p = sp.performance("KNSB", 6.895e6, 8.0)                  # 1000 psi
    assert p["t_chamber"] == pytest.approx(1600, abs=25)      # Nakka: 1600 K
    assert p["cstar"] == pytest.approx(908, rel=0.01)         # Nakka: 908 m/s
    assert sp.get("KNSB").theoretical_density == pytest.approx(1841, rel=0.01)  # 1.841 g/cc


def test_apcp_is_in_textbook_range():
    p = sp.performance("APCP", 6.895e6, 10.0)
    assert 3300 < p["t_chamber"] < 3600
    assert 255 < p["isp_ambient"] < 270                       # ~260-265 s SL class
    assert p["condensed_fraction"] > 0.2                      # liquid Al2O3 in the exhaust


def test_aluminium_raises_isp():
    metal = sp.performance("APCP", 7e6, 10.0)["isp_vac"]
    plain = sp.performance("APCP-NM", 7e6, 10.0)["isp_vac"]
    assert metal > plain + 10


def test_effective_gamma_reproduces_cea_cf():
    from aerosim.propulsion.nozzle import thrust_coefficient
    p = sp.performance("KNSB", 5e6, 6.0)
    assert thrust_coefficient(p["gamma_effective"], 6.0, 1.0, 0.0) == pytest.approx(p["cf_vac"], rel=1e-6)


def test_custom_recipe_and_unknown_ingredient():
    knsb_70 = sp.SolidPropellant("KNSB-70", {"KNO3": 70, "Sorbitol": 30}, 8e-3, 0.32)
    hot = sp.performance(knsb_70, 5e6, 6)["t_chamber"]
    assert hot > sp.performance("KNSB", 5e6, 6)["t_chamber"]  # closer to stoichiometric
    with pytest.raises(AerosimError):
        sp.performance(sp.SolidPropellant("bad", {"Unobtainium": 1}, 1e-3, 0.3), 5e6, 6)


def test_bates_with_cea_propellant():
    m = BatesMotor(**GEO, propellant="KNSB")
    tc = m.thrust_curve()
    assert 120 < tc.average_isp < 150                         # delivered, with efficiencies
    assert tc.chamber_pressure is not None and tc.chamber_pressure.max() > 1e6
    assert m.warnings == []
    apcp = BatesMotor(**GEO, propellant="APCP").thrust_curve()
    assert apcp.average_isp > tc.average_isp + 70


def test_burn_rate_range_warning():
    m = BatesMotor(**GEO, propellant="KNDX")
    m.thrust_curve()
    assert m.warnings and "burn-rate fit range" in m.warnings[0]
