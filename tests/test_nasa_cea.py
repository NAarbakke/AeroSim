"""NASA CEA backend. Skipped when the optional 'cea' package isn't installed."""

from pathlib import Path

import pytest

pytest.importorskip("cea")

from aerosim.core import AerosimError
from aerosim.missions import run_mission
from aerosim.thermo import NasaCEAModel, propellants


def test_rp1311_example8_lh2_lox():
    """NASA RP-1311 Example 8: LH2/LOX, O/F 5.55157, Pc 53.3172 bar, Ae/At 25 (IAC)."""
    m = NasaCEAModel()
    s = m.solve("LOX", "LH2", 5.55157, 53.3172e5, 25.0)
    assert s["cstar"] == pytest.approx(2332.3, rel=2e-3)
    assert s["isp_vac"] == pytest.approx(4348.0, rel=3e-3)       # m/s
    assert s["T"][0] == pytest.approx(3383.8, abs=5)
    assert s["ae_at"][2] == pytest.approx(25.0, rel=1e-3)


def test_chamber_contract_and_ambient_isp():
    p = NasaCEAModel().chamber("LOX", "RP1", 2.6, 70e5, 16.0)
    for k in ("isp_vac", "isp_sl", "cstar", "t_chamber", "gamma", "mw"):
        assert k in p
    assert 330 < p["isp_vac"] < 345
    assert p["isp_sl"] < p["isp_vac"]
    assert 3500 < p["t_chamber"] < 3800


def test_frozen_is_lower_than_equilibrium():
    eq = NasaCEAModel().chamber("LOX", "Ethanol", 1.3, 20e5, 4.0)["isp_vac"]
    fr = NasaCEAModel(frozen="chamber").chamber("LOX", "Ethanol", 1.3, 20e5, 4.0)["isp_vac"]
    assert fr < eq


def test_blend_dilution_lowers_temperature():
    # hold LOX per kg of *ethanol* fixed, so water only dilutes
    m = NasaCEAModel()
    pure = m.chamber("LOX", "Ethanol", 1.5, 20e5, 4.0)
    wet = m.chamber("LOX", {"Ethanol": 0.75, "Water": 0.25}, 1.5 * 0.75, 20e5, 4.0)
    assert wet["t_chamber"] < pure["t_chamber"] - 100


def test_nonphysical_case_raises():
    with pytest.raises(AerosimError):
        NasaCEAModel().chamber("LOX", "LH2", 0.05, 20e5, 4.0)   # far too fuel-rich to burn


def test_unknown_species_is_a_clear_error():
    with pytest.raises(AerosimError, match="could not build mixture"):
        NasaCEAModel().chamber("LOX", "NotAPropellant(L)", 2.0, 20e5, 4.0)


def test_alias_resolution():
    (c,) = propellants.resolve("LOX")
    assert c.species == "O2(L)" and c.temperature == pytest.approx(90.17)


def test_liquid_mission_end_to_end():
    cfg = Path(__file__).parents[1] / "configs" / "examples" / "liquid_rocket.yaml"
    out = run_mission(cfg)
    assert 200 < out.propulsion.average_isp < 280
    assert out.flight.apogee > 1000
