import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from aerosim.propulsion.solid import BatesMotor, GrainMotor, Propellant
from aerosim.propulsion.solid.grains import Bates, EndBurner, Finocyl, RodAndTube, make_grain
from aerosim.propulsion.solid.propellant import cstar_ideal

CASES = yaml.safe_load((Path(__file__).parent / "data" / "grain_motor_cases.yaml").read_text())


def test_map_grain_reproduces_exact_bates():
    pytest.importorskip("skfmm")
    exact = Bates(0.083, 0.15, core_diameter=0.028)
    mapped = Finocyl(0.083, 0.15, core_diameter=0.028, num_fins=0, map_dim=301)
    assert mapped.web_max == pytest.approx(exact.web_max, rel=0.01)
    for w in (0.0, 0.005, 0.015, 0.025):
        assert mapped.burn_area(w) == pytest.approx(exact.burn_area(w), rel=0.005)
        assert mapped.volume(w) == pytest.approx(exact.volume(w), rel=0.01)


def test_closed_form_grains():
    rt = RodAndTube(0.1, 0.2, inhibited_ends="Both", core_diameter=0.06, rod_diameter=0.03)
    solid = math.pi / 4 * (0.1**2 - 0.06**2 + 0.03**2) * 0.2
    assert rt.volume(0.0) == pytest.approx(solid)
    assert rt.web_max == pytest.approx(0.02)          # tube web > rod radius
    eb = EndBurner(0.05, 0.1)
    assert eb.burn_area(0.0) == eb.burn_area(0.09) == pytest.approx(math.pi / 4 * 0.05**2)
    assert eb.web_max == pytest.approx(0.1)


def test_cstar_ideal():
    # gamma 1.2, 3000 K, 25 g/mol -> textbook ~1540 m/s
    assert cstar_ideal(1.2, 3000.0, 0.025) == pytest.approx(1543, rel=0.005)


def test_grain_motor_matches_bates_motor():
    kw = {"dry_mass": 0.5, "name": "cmp"}
    b = BatesMotor(0.05, 0.02, 0.09, 4, 0.012, 6.0, Propellant(), nozzle_efficiency=0.95, **kw)
    g = GrainMotor(grains=[{"type": "bates", "diameter": 0.05, "length": 0.09,
                            "core_diameter": 0.02}] * 4,
                   nozzle={"throat_diameter": 0.012, "exit_diameter": 0.012 * math.sqrt(6.0),
                           "efficiency": 0.95, "divergence_deg": 0.0},
                   propellant=Propellant(), timestep=0.002, burnout_web=0.0, **kw)
    cb, cg = b.thrust_curve(), g.thrust_curve()
    assert cg.propellant_mass == pytest.approx(cb.propellant_mass, rel=1e-9)
    assert cg.total_impulse == pytest.approx(cb.total_impulse, rel=0.01)
    assert cg.burn_time == pytest.approx(cb.burn_time, rel=0.02)
    assert np.trapezoid(cg.mass_flow, cg.time) == pytest.approx(cg.propellant_mass, rel=1e-6)


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_against_openmotor_reference(case):
    """Geometry (mass, burn time) within a few %; Isp is 0-8 % above openMotor, whose
    nozzle model has extra losses (throat length, convergence) that are not modelled here."""
    grains = [make_grain(g) for g in case["grains"]]
    if any(not isinstance(g, (Bates, RodAndTube, EndBurner)) for g in grains):
        pytest.importorskip("skfmm")
        grains = [make_grain({**g, "map_dim": 301})
                  if g["type"] not in ("bates", "rod_and_tube", "end_burner") else g
                  for g in case["grains"]]
    m = GrainMotor(grains=grains, nozzle=case["nozzle"], propellant=case["propellant"],
                   ambient_pressure=case["ambient_pressure"], timestep=case["timestep"])
    r = m.simulate()
    t, F = r["time"], r["thrust"]
    impulse = np.trapezoid(F, t)
    on = np.nonzero(F > 0.001 * F.max())[0]
    burn = t[on[-1]] - t[on[0]]
    isp = impulse / (m.propellant_mass * 9.80665)
    ref = case["reference"]
    assert m.propellant_mass == pytest.approx(ref["propMass"], rel=0.015)
    assert burn == pytest.approx(ref["burnTime"], rel=0.06)
    assert -0.02 < isp / ref["isp"] - 1 < 0.08
