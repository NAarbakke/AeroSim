"""6-DOF ascent: consistency with the 3-DOF model, wind response, autopilot margins, RocketPy
wind adapter and the close-up page with attitude data."""

import json
import re

import numpy as np
import pytest

from aerosim.flight.rocket.ascent6dof import (
    RocketPyWind,
    SixDofAscentSim,
    WindProfile,
    autopilot_margins,
)
from aerosim.flight.rocket.orbital_ascent import AscentProgram, OrbitalAscentSim
from aerosim.flight.stability import Layout, Part, Tank
from aerosim.flight.vehicles import LaunchVehicle, Stage

SITE = (45.0, 63.0)


def _vehicle():
    return LaunchVehicle(stages=[Stage("S1", 20_000, 2_000, 300, 120, isp_sl=270),
                                 Stage("S2", 4_000, 500, 330, 400, max_burns=2)],
                         payload_mass=200, fairing_mass=100, diameter=1.5)


def _program():
    return AscentProgram(target_inclination=51.6, kick_angle=12.0, vertical_time=6,
                         transfer_apogee=300e3, coast_after=0.0)


LAYOUT = Layout(
    diameter=1.5, nose_length=2.0, nose_diameter=1.6, shoulders=[(3.0, 3.5, 1.6, 1.5)],
    parts=[Part("fairing", 100, 0.5, 3.0, "fairing_jettison"), Part("payload", 200, 1.0, 2.5),
           Part("S2 dry", 500, 3.5, 7.0), Part("S1 dry", 2000, 7.0, 20.0, "S1_burnout")],
    tanks=[Tank("S2 prop", "S2", 1.0, 3.6, 6.8), Tank("S1 ox", "S1", 0.7, 7.2, 14.0),
           Tank("S1 fuel", "S1", 0.3, 14.2, 19.0)],
    gimbal={"S1": 19.5, "S2": 6.9}, nozzle_exit={"S1": 20.0, "S2": 7.0},
    shoulders_until="fairing_jettison")


@pytest.fixture(scope="module")
def windy():
    veh = _vehicle()
    wind = WindProfile(peak=30.0, peak_altitude=9e3)
    return veh, SixDofAscentSim(*SITE, program=_program(), layout=LAYOUT, wind=wind).run(veh)


def test_without_normal_force_it_flies_the_3dof_trajectory():
    veh = _vehicle()
    ref = OrbitalAscentSim(*SITE, program=_program()).run(veh)
    sim = SixDofAscentSim(*SITE, program=_program(), layout=LAYOUT)
    sim._aero = {k: [(c * 1e-9, x) for c, x in v] for k, v in sim._aero.items()}
    res = sim.run(veh)
    t = ref.events["S1_burnout"]
    fpa = lambda r: np.interp(t, r.time, r.flight_path_angle)
    assert fpa(res) == pytest.approx(fpa(ref), abs=0.3)
    assert res.final_orbit.apogee_alt == pytest.approx(ref.final_orbit.apogee_alt, rel=0.02)
    assert np.allclose(np.linalg.norm(res.attitude, axis=1), 1.0)


def test_wind_gives_angle_of_attack_held_by_the_gimbal(windy):
    _veh, res = windy
    # (the small test vehicle has no performance margin for the wind: it may fall short of apogee)
    band = (res.altitude > 7e3) & (res.altitude < 11e3) & (res.time < res.events["S1_burnout"])
    assert np.nanmax(res.alpha_total[band]) > 1.0                 # 30 m/s crosswind at ~400 m/s
    assert np.nanmax(res.gimbal_total[band]) > 0.2
    assert np.nanmax(res.gimbal_total) <= 7.01                      # actuator limit
    k = int(np.nanargmax(np.where(band, res.q_alpha, -1)))            # q from the air-relative speed
    assert res.q_alpha[k] == pytest.approx(res.dynamic_pressure[k] * res.alpha_total[k], rel=0.15)
    # the wind blows from the west: the air-relative velocity has a westward part
    assert np.all(np.isfinite(res.wind_speed)) and res.wind_speed.max() > 25


def test_autopilot_margins(windy):
    pytest.importorskip("control")
    veh, res = windy
    m = autopilot_margins(res, veh, LAYOUT, t_end=res.events["S2_burn1_end"])
    assert m["stable"].all() and m["t"].size > 20
    q_band = m["t"] < res.events["S1_burnout"]
    assert np.all(m["pm"][q_band] > 25) and np.all(m["gm_upper"] > 6)
    assert np.nanmax(m["gm_lower"][q_band]) < -6            # aerodynamic margin exists in the air


def test_rocketpy_wind_adapter():
    rocketpy = pytest.importorskip("rocketpy")
    env = rocketpy.Environment(latitude=SITE[0], longitude=SITE[1])
    env.set_atmospheric_model(type="custom_atmosphere", wind_u=[(0, 0), (10e3, 20.0)], wind_v=0)
    from aerosim.flight import orbits
    r, _ = orbits.site_state(*SITE)
    w = RocketPyWind(env).vector(r, 10e3)
    e, n, _ = orbits.enu(r)
    assert np.dot(w, e) == pytest.approx(20.0) and abs(np.dot(w, n)) < 1e-9
    assert WindProfile(surface=0, peak=20, peak_altitude=10e3).vector(r, 10e3) @ e == pytest.approx(20.0)


def test_closeup_page_with_attitude(windy):
    from aerosim.plots.ascent_closeup import ascent_closeup_html
    veh, res = windy
    page = ascent_closeup_html(res, veh, title="6dof", t_end=300, layout=LAYOUT)
    data = json.loads(re.search(r'id="data">(.*?)</script>', page, re.DOTALL).group(1))
    assert data["six"] and len(data["traj"]["t"]) == len(data["f"]["t"]) == len(data["f"]["bx"])
    b = np.array([data["f"]["bx"], data["f"]["by"], data["f"]["bz"]], float)
    assert np.allclose(np.linalg.norm(b, axis=0), 1.0, atol=1e-3)
    assert data["traj"]["u"][0] == pytest.approx(0.0, abs=0.01)   # starts at the site
    assert "pm" in data["f"] and 'id="traj3d"' in page
