import math

import numpy as np
import pytest

from aerosim.atmosphere import StandardAtmosphere
from aerosim.flight import RocketVehicle
from aerosim.flight.rocket import PlanarRocketSim, PointMassRocketSim
from aerosim.gnc import PID, Actuator, AirbrakeApogeeController, AttitudeHoldTVC, Sensor
from aerosim.propulsion.solid import BatesMotor

TC = BatesMotor(0.05, 0.02, 0.09, 4, 0.012, 6.0, dry_mass=0.6).thrust_curve()
ATM = StandardAtmosphere(wind_speed=5)


# -- blocks ------------------------------------------------------------------------------
def test_pid_antiwindup_releases_quickly():
    pid = PID(1.0, 10.0, 0.0, dt=0.01, out_min=-1, out_max=1)
    for _ in range(500):
        pid(5.0)                      # saturated hard for 5 s
    assert pid._i < 1.0               # integrator did not wind up
    assert pid(-0.5) < 1.0            # leaves saturation immediately on sign change


def test_actuator_rate_and_position_limits():
    a = Actuator(tau=0.0, rate_limit=1.0, min=0, max=0.5)
    assert a.step(10.0, 0.1) == pytest.approx(0.1)
    for _ in range(20):
        a.step(10.0, 0.1)
    assert a.position == 0.5


def test_sensor_is_seeded():
    s1, s2 = Sensor({"theta": 0.01}, seed=3), Sensor({"theta": 0.01}, seed=3)
    assert s1.measure({"theta": 1.0}) == s2.measure({"theta": 1.0})


# -- air brakes --------------------------------------------------------------------------
def test_apogee_prediction_matches_vacuum_limit():
    assert AirbrakeApogeeController.predict_apogee(100, 50, 2, 1.2, 0.0) == pytest.approx(
        100 + 50**2 / (2 * 9.80665))


@pytest.mark.parametrize("target", [2500.0, 2200.0])
def test_airbrakes_hit_target_apogee(target):
    veh = RocketVehicle(dry_mass=3.0, diameter=0.066, rail_length=3.0, elevation_deg=88,
                        airbrake_cd_area=0.012)
    open_loop = PointMassRocketSim().run(veh, TC, ATM).apogee
    closed = PointMassRocketSim(controller=AirbrakeApogeeController(target)).run(veh, TC, ATM)
    assert open_loop > target + 300
    assert closed.apogee == pytest.approx(target, rel=0.01)
    assert closed.meta["airbrake"][closed.time < TC.time[-1]].max() == 0.0   # none before burnout


def test_closed_loop_without_brakes_matches_open_loop():
    veh = RocketVehicle(dry_mass=3.0, diameter=0.066, rail_length=3.0, elevation_deg=88)
    ol = PointMassRocketSim().run(veh, TC, ATM).apogee
    cl = PointMassRocketSim(controller=AirbrakeApogeeController(1e9)).run(veh, TC, ATM).apogee
    assert cl == pytest.approx(ol, rel=2e-3)


# -- planar rigid body + TVC -------------------------------------------------------------
STABLE = RocketVehicle(dry_mass=3.0, diameter=0.066, rail_length=3.0, elevation_deg=88,
                       pitch_inertia=0.8, cg_to_nozzle=0.6, static_margin=0.1)
UNSTABLE = RocketVehicle(dry_mass=3.0, diameter=0.066, rail_length=3.0, elevation_deg=90,
                         pitch_inertia=0.8, cg_to_nozzle=0.6, static_margin=-0.05)


def test_planar_agrees_with_point_mass_for_stable_rocket():
    planar = PlanarRocketSim().run(STABLE, TC, ATM).apogee
    pm = PointMassRocketSim().run(STABLE, TC, ATM).apogee
    assert planar == pytest.approx(pm, rel=0.03)


def test_stable_rocket_weathercocks_into_wind():
    # heading north (downrange = +x); theta < 90 deg means nose tilted north
    burn = TC.time[-1]
    head = PlanarRocketSim().run(STABLE, TC, StandardAtmosphere(wind_speed=8, wind_from_deg=0))
    tail = PlanarRocketSim().run(STABLE, TC, StandardAtmosphere(wind_speed=8, wind_from_deg=180))
    th_head = math.degrees(np.interp(burn, head.time, head.meta["theta"]))
    th_tail = math.degrees(np.interp(burn, tail.time, tail.meta["theta"]))
    assert th_head < 87.0            # wind from north: nose turns north, into the wind
    assert th_tail > 88.0            # wind from south: nose turns south (back past rail angle)


def _tvc_gains():
    pytest.importorskip("control")
    from aerosim.gnc import design
    op = design.TVCOperatingPoint(
        thrust=TC.total_impulse / TC.burn_time, lever_arm=0.6, inertia=0.68,
        dynamic_pressure=0.5 * 1.2 * 150**2, ref_area=UNSTABLE.reference_area,
        cn_alpha=2.0, static_margin=-0.05)
    return op, design.pd_by_pole_placement(op, wn=12, zeta=0.7)


def test_tvc_design_has_margins():
    from aerosim.gnc import design
    op, g = _tvc_gains()
    assert op.m_alpha > 0                                    # aerodynamically unstable plant
    a = design.analyze_attitude_loop(op, g["kp"], g["kd"])
    assert a["stable"] and a["phase_margin_deg"] > 40
    assert a["lower_gain_margin_db"] < -6                    # tolerates 2x gain loss
    lqr = design.lqr_attitude(op)
    assert np.all(np.real(lqr["closed_loop_poles"]) < 0)


def test_tvc_holds_unstable_rocket_through_gust():
    op, g = _tvc_gains()
    gust = (0.3, 0.2, 8.0)
    ol = PlanarRocketSim(gust=gust).run(UNSTABLE, TC, ATM)
    ctrl = AttitudeHoldTVC(kp=g["kp"], kd=g["kd"], design_thrust=op.thrust)
    cl = PlanarRocketSim(controller=ctrl, gust=gust,
                         sensor=Sensor({"q": 0.002}, seed=1)).run(UNSTABLE, TC, ATM)
    burn_ol, burn_cl = ol.time < TC.time[-1], cl.time < TC.time[-1]
    dev_ol = np.abs(np.degrees(ol.meta["theta"][burn_ol]) - 90).max()
    dev_cl = np.abs(np.degrees(cl.meta["theta"][burn_cl]) - 90).max()
    assert dev_cl < 0.5 < dev_ol
    assert np.abs(cl.meta["gimbal"]).max() <= math.radians(UNSTABLE.gimbal_limit_deg) + 1e-12
