"""Liquid engine cycles (power balance) against published engine data."""

import pytest

pytest.importorskip("cea")
pytest.importorskip("CoolProp")

from aerosim.core import registry
from aerosim.propulsion.liquid.cycle import CycleEngine


def test_gas_generator_f1():
    # F-1 (Wikipedia): Pc 70 bar, MR 2.27, eps 16, 7.77 MN vac, Isp 304 s vac / 263 s SL,
    # turbopump 41 MW, turbine inlet 1,500 F (1089 K)
    e = CycleEngine("gas_generator", "LOX", "RP1", 2.27, 7.0e6, 16, 7.77e6,
                    turbine_inlet_temperature=1089, turbine_pressure_ratio=16)
    o = e.solve()
    assert o["turbine_power"] == pytest.approx(41e6, rel=0.2)
    assert o["isp_vac"] == pytest.approx(304, rel=0.04)
    assert o["isp_sl"] == pytest.approx(263, rel=0.05)
    assert 0 < o["turbine_flow_fraction"] < 0.05
    assert o["drive_gas"]["temperature"] == pytest.approx(1089, abs=1)
    # power balance closes: turbine power = pump power
    assert o["turbine_power"] == pytest.approx(o["pump_power_ox"] + o["pump_power_fuel"], rel=1e-6)


def test_staged_combustion_rd180_and_rs25():
    # RD-180 (Wikipedia): 26.7 MPa, MR 2.72, eps 36.87, 4.15 MN vac, Isp 338 vac / 311 SL
    rd = CycleEngine("staged_combustion", "LOX", "RP1", 2.72, 26.7e6, 36.87, 4.15e6,
                     rich="oxidizer", turbine_inlet_temperature=772, pump_efficiency=0.75,
                     turbine_efficiency=0.8, injector_dp=0.15).solve()
    assert rd["isp_vac"] == pytest.approx(338, rel=0.01)
    assert rd["isp_sl"] == pytest.approx(311, rel=0.01)
    assert 1.5 < rd["turbine_pressure_ratio"] < 2.5
    assert rd["drive_gas"]["mixture_ratio"] > 20                  # oxidizer-rich preburner
    # RS-25 (Wikipedia): 20.64 MPa, MR 6.03; HPFTP discharge 45 MPa, HPFTP+HPOTP 70 MW
    rs = CycleEngine("staged_combustion", "LOX", "LH2", 6.03, 20.64e6, 69, 2.09e6,
                     turbine_inlet_temperature=1000, pump_efficiency=0.75, turbine_efficiency=0.75,
                     tank_pressure_ox=2.9e6, tank_pressure_fuel=1.9e6).solve()
    assert rs["fuel_discharge_pressure"] == pytest.approx(45e6, rel=0.15)
    assert rs["turbine_power"] == pytest.approx(70.4e6, rel=0.3)
    assert rs["drive_gas"]["mixture_ratio"] < 2                    # fuel-rich preburner


def test_expander_pressure_fed_and_electric():
    ex = CycleEngine("expander", "LOX", "LH2", 5.0, 3.275e6, 61, 73.4e3,
                     coolant_outlet_temperature=200).solve()
    assert ex["turbine_pressure_ratio"] > 1.05
    assert ex["fuel_discharge_pressure"] > ex["ox_discharge_pressure"]
    pf = CycleEngine("pressure_fed", "N2O4", "MMH", 1.65, 1.0e6, 40, 4e3).solve()
    assert pf["tank_pressure_fuel"] == pytest.approx(1.0e6 * (1 + 0.20 + 0.15 + 0.05))
    assert pf["tank_pressure_ox"] == pytest.approx(1.0e6 * (1 + 0.20 + 0.05))
    el = CycleEngine("electric_pump", "LOX", "RP1", 2.6, 1.2e6, 14, 25.8e3, burn_time=150).solve()
    assert el["electric_power"] > el["pump_power"] > 0
    assert el["battery_mass"] > 0


def test_cycle_engine_thrust_curve_via_registry():
    e = registry.create("rocket_engine", "liquid_cycle", cycle="gas_generator", oxidizer="LOX",
                        fuel="RP1", mixture_ratio=2.3, chamber_pressure=7e6, expansion_ratio=16,
                        thrust_vac=1e6, burn_time=60, turbine_inlet_temperature=1000)
    tc = e.thrust_curve()
    assert tc.burn_time == pytest.approx(60)
    assert tc.average_isp == pytest.approx(e.solve()["isp_vac"], rel=0.05)
