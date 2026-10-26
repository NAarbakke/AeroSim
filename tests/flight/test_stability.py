"""Pitch stability: Barrowman components, mass layout against the simulator, CG motion as tanks
drain, static margin sign, and the short-period roots."""

import math

import numpy as np
import pytest

from aerosim.flight.rocket.orbital_ascent import AscentProgram, OrbitalAscentSim
from aerosim.flight.stability import Layout, Part, Tank, barrowman, stability
from aerosim.flight.vehicles import LaunchVehicle, Stage


@pytest.fixture(scope="module")
def flight():
    veh = LaunchVehicle(stages=[Stage("S1", 20_000, 2_000, 300, 120, isp_sl=270),
                                Stage("S2", 4_000, 500, 330, 400, max_burns=2)],
                        payload_mass=200, fairing_mass=100, diameter=1.5)
    prog = AscentProgram(target_inclination=51.6, kick_angle=12.0, vertical_time=6,
                         transfer_apogee=300e3, coast_after=100)
    return veh, OrbitalAscentSim(45.0, 63.0, program=prog).run(veh)


def _layout(tail_flare=None):
    """1.5 m vehicle, 20 m long: fairing and payload, S2, S1 (x aft from the nose tip)."""
    shoulders = [(3.0, 3.5, 1.6, 1.5)] + ([tail_flare] if tail_flare else [])
    return Layout(
        diameter=1.5, nose_length=2.0, nose_diameter=1.6, shoulders=shoulders,
        parts=[Part("fairing", 100, 0.5, 3.0, "fairing_jettison"), Part("payload", 200, 1.0, 2.5),
               Part("S2 dry", 500, 3.5, 7.0), Part("S1 dry", 2000, 7.0, 20.0, "S1_burnout")],
        tanks=[Tank("S2 prop", "S2", 1.0, 3.6, 6.8), Tank("S1 ox", "S1", 0.7, 7.2, 14.0),
               Tank("S1 fuel", "S1", 0.3, 14.2, 19.0)],
        gimbal={"S1": 19.5, "S2": 6.9}, nozzle_exit={"S1": 20.0, "S2": 7.0},
        shoulders_until="fairing_jettison")


def test_barrowman_components():
    lay = _layout()
    (cn_nose, x_nose), (cn_sh, x_sh) = barrowman(lay)
    assert cn_nose == pytest.approx(2 * (1.6 / 1.5) ** 2) and x_nose == pytest.approx(0.466 * 2.0)
    assert cn_sh < 0 and 3.0 < x_sh < 3.5                 # boattail: negative lift between its ends


def test_layout_mass_matches_the_simulator_and_cg_moves(flight):
    veh, res = flight
    s = stability(res, veh, _layout(), t_end=400)
    assert s["layout_mass_error"] < 1e-9
    # at liftoff, the CG is the mass-weighted centroid of parts and full tanks
    items = [(100, 1.75), (200, 1.75), (500, 5.25), (2000, 13.5), (4000, 5.2), (14000, 10.6), (6000, 16.6)]
    assert s["x_cg"][0] == pytest.approx(sum(m * x for m, x in items) / sum(m for m, _ in items), abs=0.15)
    burn = s["t"] < res.events["S1_burnout"]
    assert s["x_cg"][burn].min() < s["x_cg"][0] - 1.0     # S1 tanks drain: CG moves forward
    assert np.all(s["margin"][np.isfinite(s["margin"])] < 0)   # no fins: unstable


def test_tail_flare_makes_it_stable_and_roots_satisfy_the_pitch_equation(flight):
    veh, res = flight
    s = stability(res, veh, _layout(tail_flare=(18.0, 20.0, 1.5, 3.6)), t_end=120)
    ok = np.isfinite(s["margin"])
    assert ok.sum() > 10 and np.all(s["margin"][ok] > 0)      # CP pulled behind the CG
    assert np.all(np.isfinite(s["omega"][ok])) and np.all(s["zeta"][ok] > 0)
    # unstable layout: s = ln2 / t_double solves I s^2 - M_q s - M_alpha = 0
    u = stability(res, veh, _layout(), t_end=120)
    i = int(np.nanargmin(u["t_double"]))
    sr = math.log(2) / u["t_double"][i]
    assert u["inertia"][i] * sr**2 - u["m_q"][i] * sr - u["m_alpha"][i] == pytest.approx(0, abs=1e-6 * u["m_alpha"][i])
