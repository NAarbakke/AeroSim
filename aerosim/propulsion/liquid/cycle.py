"""Liquid engine cycles: turbopump power balance -> delivered Isp, flows, pump and turbine duty.

    CycleEngine(cycle="gas_generator", oxidizer="LOX", fuel="RP1", mixture_ratio=2.27,
                chamber_pressure=7.0e6, expansion_ratio=16, thrust_vac=7.77e6).solve()

Cycles (0-D steady power balance, one turbopump shaft per cycle):

- ``pressure_fed``      tanks at injector + cooling pressure; reports the tank pressures
- ``electric_pump``     pumps driven by motors from batteries; reports power, battery and
                        motor mass (Rocket Lab Rutherford style)
- ``gas_generator``     open cycle: a fuel-rich (or ox-rich) gas generator at the
                        turbine-inlet temperature drives the turbine, whose exhaust is
                        dumped through its own nozzle; the turbine flow is solved for
- ``staged_combustion`` closed: all of one propellant plus enough of the other to reach the
                        turbine-inlet temperature burns in a preburner, drives the turbine and
                        enters the main chamber; the turbine pressure ratio is solved for
                        (``rich="fuel"`` SSME / RS-25, ``rich="oxidizer"`` RD-170 / RD-180)
- ``expander``          closed: the fuel, heated in the cooling jacket, drives the turbine
                        (RL10); turbine inlet temperature from pyskyfire's regen solution
                        (``regen={...}``) or given as ``coolant_outlet_temperature``

Physics: pump work is the isentropic enthalpy rise of the real fluid (CoolProp) over the pump
efficiency, or dp/rho for propellants without an equation of state (NTO, UDMH, MMH, N2H4,
H2O2). Turbine drive gas comes from NASA CEA at the generator / preburner mixture ratio
that gives the turbine-inlet temperature (equilibrium; for fuel-rich hydrocarbons this
over-predicts temperature at a given mixture ratio, so set the temperature, not the ratio),
expanded through the turbine pressure ratio at the turbine efficiency (ideal-gas with CEA
gamma and molar mass). The main chamber's delivered Isp is ``design.engine_point`` (CEA with
textbook c* and nozzle efficiencies). Pressure budget: injector drop and cooling-jacket
drop as fractions of chamber pressure, or the jacket drop from pyskyfire when ``regen`` is
given.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq

from aerosim.core import AerosimError, ThrustCurve, registry
from aerosim.core.units import G0
from aerosim.propulsion.liquid.design import engine_point
from aerosim.thermo.combustion import propellants

R_U = 8314.462618
CYCLES = ("pressure_fed", "electric_pump", "gas_generator", "staged_combustion", "expander")
# liquid densities (kg/m^3, ~293 K) for propellants CoolProp has no equation of state for
DENSITY = {"N2O4": 1440.0, "NTO": 1440.0, "UDMH": 791.0, "MMH": 875.0, "N2H4": 1004.0,
           "Hydrazine": 1004.0, "H2O2": 1440.0, "HTP": 1440.0, "JetA": 800.0}


def _name(spec) -> str:
    return spec if isinstance(spec, str) else next(iter(spec))


def _first_root(f, xs):
    """Bracket of the first rise through zero of f along xs; stops where f cannot be evaluated
    (e.g. pump pressures beyond the fluid's equation of state)."""
    prev = None
    for x in xs:
        try:
            y = f(x)
        except AerosimError:
            break
        if prev is not None and prev[1] < 0 <= y:
            return prev[0], x
        prev = (x, y)
    return None, None


class Pump:
    """Pump on one propellant: specific work and outlet temperature between two pressures."""

    def __init__(self, propellant, inlet_pressure: float, efficiency: float) -> None:
        self.name, self.p_in, self.eta = _name(propellant), float(inlet_pressure), efficiency
        self.t_in = propellants.resolve(propellant)[0].temperature
        self.rho = DENSITY.get(self.name)
        if self.rho is None:
            from aerosim.thermo.fluids import coolprop_name, props
            self._fluid, self._props = coolprop_name(self.name), props
            self.h_in = props("H", "T", self.t_in, "P", self.p_in, self._fluid)
            self.s_in = props("S", "T", self.t_in, "P", self.p_in, self._fluid)
            self.rho = props("D", "T", self.t_in, "P", self.p_in, self._fluid)

    def work(self, p_out: float) -> float:
        """J/kg of shaft work to raise the propellant to ``p_out``."""
        if p_out <= self.p_in:
            return 0.0
        if not hasattr(self, "_fluid"):
            return (p_out - self.p_in) / (self.rho * self.eta)
        h_s = self._props("H", "P", p_out, "S", self.s_in, self._fluid)
        return (h_s - self.h_in) / self.eta

    def outlet_temperature(self, p_out: float) -> float:
        if not hasattr(self, "_fluid"):
            return self.t_in
        return self._props("T", "P", p_out, "H", self.h_in + self.work(p_out), self._fluid)


@dataclass
class CycleEngine:
    cycle: str
    oxidizer: object
    fuel: object
    mixture_ratio: float
    chamber_pressure: float          # Pa
    expansion_ratio: float
    thrust_vac: float                # N, whole engine (main chamber + turbine exhaust)
    burn_time: float = 100.0         # s, for thrust_curve and battery sizing
    dry_mass: float = 0.0
    tank_pressure_ox: float = 0.35e6
    tank_pressure_fuel: float = 0.35e6
    injector_dp: float = 0.20        # main injector drop / chamber pressure
    preburner_injector_dp: float = 0.15
    cooling_dp: float = 0.15         # jacket drop / chamber pressure (replaced by ``regen``)
    line_dp: float = 0.05            # feed line + valve drop / chamber pressure (pressure-fed)
    pump_efficiency: float = 0.70
    turbine_efficiency: float = 0.60
    turbine_inlet_temperature: float = 900.0     # K, gas generator / preburner
    rich: str = "fuel"               # gas generator / preburner side
    turbine_pressure_ratio: float = 15.0         # gas generator turbines only
    gg_exhaust_pressure_ratio: float = 10.0      # turbine exhaust nozzle, gas generator only
    coolant_outlet_temperature: float | None = None   # K, expander without pyskyfire
    regen: dict | None = None        # kwargs for adapters.regen.regen_cooling (pyskyfire)
    motor_efficiency: float = 0.90   # electric pump: motor x inverter
    battery_specific_energy: float = 200.0       # Wh/kg
    battery_specific_power: float = 5.0e3        # W/kg
    motor_specific_power: float = 5.0e3          # W/kg
    cstar_efficiency: float = 0.98
    cf_efficiency: float = 0.98
    name: str = "liquid engine"
    _out: dict = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.cycle not in CYCLES:
            raise ValueError(f"cycle must be one of {CYCLES}")
        if self.rich not in ("fuel", "oxidizer"):
            raise ValueError("rich must be 'fuel' or 'oxidizer'")

    # -- pieces ------------------------------------------------------------------
    def _drive_gas(self, pressure: float) -> dict:
        """Mixture ratio and gas properties of the generator / preburner at the turbine-inlet T."""
        from aerosim.propulsion.liquid.design import _cea
        cea = _cea(None)
        T = lambda mr: cea.solve(self.oxidizer, self.fuel, mr, pressure, 2.0)
        mr0 = self.mixture_ratio
        lo, hi = (0.03 * mr0, 0.5 * mr0) if self.rich == "fuel" else (3.0 * mr0, 40.0 * mr0)
        grid = np.geomspace(lo, hi, 16)
        ts = []
        for mr in grid:
            try:
                ts.append(float(T(mr)["T"][0]))
            except AerosimError:
                ts.append(np.nan)
        ts = np.asarray(ts) - self.turbine_inlet_temperature
        ok = np.isfinite(ts)
        g, t = grid[ok], ts[ok]
        for i in range(len(g) - 1):
            if t[i] * t[i + 1] <= 0:
                mr = brentq(lambda m: float(T(m)["T"][0]) - self.turbine_inlet_temperature,
                            g[i], g[i + 1], xtol=1e-4)
                s = T(mr)
                gam, mw = float(s["gamma"][0]), float(s["MW"][0])
                return {"mixture_ratio": mr, "temperature": float(s["T"][0]), "gamma": gam,
                        "mw": mw, "cp": gam / (gam - 1) * R_U / mw}
        raise AerosimError(f"no {self.rich}-rich mixture ratio gives "
                           f"{self.turbine_inlet_temperature:.0f} K in [{lo:.3g}, {hi:.3g}]")

    @staticmethod
    def _gas_work(gas: dict, pr: float, eta: float) -> tuple[float, float]:
        """(J/kg turbine work, outlet temperature) for an ideal-gas expansion through ``pr``."""
        k = (gas["gamma"] - 1) / gas["gamma"]
        w = eta * gas["cp"] * gas["temperature"] * (1 - pr ** -k)
        return w, gas["temperature"] - w / gas["cp"]

    def _regen(self, mdot_fuel: float, inlet_t: float, inlet_p: float):
        from aerosim.propulsion.adapters.regen import regen_cooling
        kw = {"l_star": 1.0, **(self.regen or {})}
        return regen_cooling(self.oxidizer, self.fuel, self.mixture_ratio, self.chamber_pressure,
                             self._chamber["thrust_vac_main"], self.expansion_ratio, ambient_pressure=0.0,
                             coolant_inlet_temperature=inlet_t, coolant_inlet_pressure=inlet_p,
                             mdot_coolant=mdot_fuel, **kw)

    # -- solve -------------------------------------------------------------------
    def solve(self) -> dict:
        """Close the cycle. Returns flows (kg/s), powers (W), pressures (Pa), Isp (s)."""
        if self._out is not None:
            return self._out
        pc, mr = self.chamber_pressure, self.mixture_ratio
        ch = engine_point(1.0, self.oxidizer, self.fuel, mr, pc, expansion_ratio=self.expansion_ratio,
                          cstar_efficiency=self.cstar_efficiency, cf_efficiency=self.cf_efficiency)
        v_main = ch["isp_vac"] * G0
        p_inj = pc * (1 + self.injector_dp)
        ox_pump = Pump(self.oxidizer, self.tank_pressure_ox, self.pump_efficiency)
        fu_pump = Pump(self.fuel, self.tank_pressure_fuel, self.pump_efficiency)
        out = {"cycle": self.cycle, "isp_vac_chamber": ch["isp_vac"], "isp_sl_chamber": ch["isp_sl"],
               "cstar": ch["cstar"], "t_chamber": ch["t_chamber"]}
        fo, ff = mr / (1 + mr), 1 / (1 + mr)
        cool_dp = self.cooling_dp * pc
        regen = None

        if self.cycle in ("pressure_fed", "electric_pump"):
            mdot = self.thrust_vac / v_main
            p_ox, p_fu = p_inj + self.line_dp * pc, p_inj + cool_dp + self.line_dp * pc
            self._chamber = {"thrust_vac_main": self.thrust_vac}
            if self.regen is not None:
                regen = self._regen(mdot * ff, fu_pump.t_in, p_fu)
                p_fu = p_inj + regen.pressure_drop + self.line_dp * pc
            out |= {"mdot": mdot, "mdot_main": mdot, "mdot_turbine": 0.0,
                    "isp_vac": ch["isp_vac"], "isp_sl": ch["isp_sl"],
                    "ox_discharge_pressure": p_ox, "fuel_discharge_pressure": p_fu}
            if self.cycle == "pressure_fed":
                out |= {"tank_pressure_ox": p_ox, "tank_pressure_fuel": p_fu}
            else:
                pw = mdot * (fo * ox_pump.work(p_ox) + ff * fu_pump.work(p_fu))
                elec = pw / self.motor_efficiency
                batt = max(elec * self.burn_time / (self.battery_specific_energy * 3600.0),
                           elec / self.battery_specific_power)
                out |= {"pump_power": pw, "electric_power": elec, "battery_mass": batt,
                        "motor_mass": pw / self.motor_specific_power}

        elif self.cycle == "gas_generator":
            gg = self._drive_gas(p_inj)
            w_t, t_out = self._gas_work(gg, self.turbine_pressure_ratio, self.turbine_efficiency)
            p_ox, p_fu = p_inj * (1 + self.preburner_injector_dp), p_inj * (1 + self.preburner_injector_dp) + cool_dp
            w_ox, w_fu = ox_pump.work(p_ox), fu_pump.work(p_fu)
            mg = gg["mixture_ratio"]
            # turbine power = pump power for main + generator flows:
            #   m_t w_t = m_main (fo w_ox + ff w_fu) + m_t (mg w_ox + w_fu) / (1 + mg)
            k = (fo * w_ox + ff * w_fu) / (w_t - (mg * w_ox + w_fu) / (1 + mg))
            if k <= 0:
                raise AerosimError("gas-generator turbine cannot drive the pumps "
                                   "(raise pressure ratio, temperature or efficiencies)")
            kx = (gg["gamma"] - 1) / gg["gamma"]
            v_gg = math.sqrt(2 * gg["cp"] * t_out * (1 - self.gg_exhaust_pressure_ratio ** -kx))
            m_main = self.thrust_vac / (v_main + k * v_gg)
            m_t = k * m_main
            mdot = m_main + m_t
            f_sl_main = m_main * ch["isp_sl"] * G0
            out |= {"mdot": mdot, "mdot_main": m_main, "mdot_turbine": m_t,
                    "turbine_flow_fraction": m_t / mdot, "drive_gas": gg,
                    "turbine_pressure_ratio": self.turbine_pressure_ratio,
                    "turbine_inlet_pressure": p_inj, "turbine_outlet_temperature": t_out,
                    "turbine_power": m_t * w_t, "exhaust_isp": v_gg / G0,
                    "pump_power_ox": (m_main * fo + m_t * mg / (1 + mg)) * w_ox,
                    "pump_power_fuel": (m_main * ff + m_t / (1 + mg)) * w_fu,
                    "ox_discharge_pressure": p_ox, "fuel_discharge_pressure": p_fu,
                    "isp_vac": self.thrust_vac / (mdot * G0),
                    "isp_sl": (f_sl_main + m_t * v_gg) / (mdot * G0)}

        elif self.cycle == "staged_combustion":
            mdot = self.thrust_vac / v_main
            m_ox, m_fu = mdot * fo, mdot * ff
            p_hot = p_inj                                   # turbine exhaust into the main injector
            pb = self._drive_gas(p_hot * 3.0)               # composition barely depends on pressure
            mg = pb["mixture_ratio"]
            m_t = m_fu * (1 + mg) if self.rich == "fuel" else m_ox * (1 + 1 / mg)
            m_pb_other = m_t - (m_fu if self.rich == "fuel" else m_ox)

            def balance(pr):
                p_pb = p_hot * pr
                p_rich = p_pb * (1 + self.preburner_injector_dp)
                w_t = self._gas_work(pb, pr, self.turbine_efficiency)[0]
                if self.rich == "fuel":
                    fuel = m_fu * fu_pump.work(p_rich + cool_dp)
                    ox = (m_ox - m_pb_other) * ox_pump.work(p_inj) + m_pb_other * ox_pump.work(p_rich)
                else:
                    ox = m_ox * ox_pump.work(p_rich)
                    fuel = ((m_fu - m_pb_other) * fu_pump.work(p_inj + cool_dp)
                            + m_pb_other * fu_pump.work(max(p_rich, p_inj + cool_dp)))
                return m_t * w_t - ox - fuel, ox, fuel

            a, b = _first_root(lambda p: balance(p)[0], np.geomspace(1.02, 12.0, 60))
            if a is None:
                raise AerosimError("staged-combustion cycle does not close for turbine pressure "
                                   "ratios up to 12 (raise turbine temperature or efficiencies)")
            pr = brentq(lambda p: balance(p)[0], a, b, xtol=1e-6)
            p_pb = p_hot * pr
            _, pw_ox, pw_fu = balance(pr)
            pump = pw_ox + pw_fu
            p_rich = p_pb * (1 + self.preburner_injector_dp)
            out |= {"mdot": mdot, "mdot_main": mdot, "mdot_turbine": m_t,
                    "turbine_flow_fraction": m_t / mdot, "drive_gas": pb,
                    "turbine_pressure_ratio": pr, "preburner_pressure": p_pb,
                    "turbine_inlet_pressure": p_pb, "turbine_power": pump,
                    "pump_power_ox": pw_ox, "pump_power_fuel": pw_fu,
                    "turbine_outlet_temperature": self._gas_work(pb, pr, self.turbine_efficiency)[1],
                    "ox_discharge_pressure": p_rich if self.rich == "oxidizer" else p_inj,
                    "fuel_discharge_pressure": p_rich + cool_dp if self.rich == "fuel" else p_inj + cool_dp,
                    "isp_vac": ch["isp_vac"], "isp_sl": ch["isp_sl"]}

        else:  # expander
            mdot = self.thrust_vac / v_main
            m_ox, m_fu = mdot * fo, mdot * ff
            self._chamber = {"thrust_vac_main": self.thrust_vac}
            p_guess = 2.5 * pc
            if self.regen is not None:
                regen = self._regen(m_fu, fu_pump.outlet_temperature(p_guess), p_guess)
                t_in, cool_dp = regen.coolant_outlet_temperature, regen.pressure_drop
            elif self.coolant_outlet_temperature is not None:
                t_in = self.coolant_outlet_temperature
            else:
                raise AerosimError("expander cycle needs regen={...} or coolant_outlet_temperature")
            from aerosim.thermo.fluids import coolprop_name, props
            fl = coolprop_name(_name(self.fuel))

            def balance(p_dis):
                p_t_in = p_dis - cool_dp
                h_in, s_in = props("H", "T", t_in, "P", p_t_in, fl), props("S", "T", t_in, "P", p_t_in, fl)
                w_t = self.turbine_efficiency * (h_in - props("H", "P", p_inj, "S", s_in, fl))
                pump = m_fu * fu_pump.work(p_dis) + m_ox * ox_pump.work(p_inj)
                return m_fu * w_t - pump, pump, w_t

            a, b = _first_root(lambda p: balance(p)[0],
                               np.linspace(p_inj + cool_dp + 0.02 * pc, 8 * pc, 80))
            if a is None:
                raise AerosimError(f"expander cycle does not close: jacket outlet {t_in:.0f} K gives "
                                   "too little turbine power at this chamber pressure")
            p_dis = brentq(lambda p: balance(p)[0], a, b, xtol=1.0)
            pump = balance(p_dis)[1]
            out |= {"mdot": mdot, "mdot_main": mdot, "mdot_turbine": m_fu, "turbine_flow_fraction": ff,
                    "turbine_inlet_temperature": t_in, "turbine_inlet_pressure": p_dis - cool_dp,
                    "turbine_pressure_ratio": (p_dis - cool_dp) / p_inj, "turbine_power": pump,
                    "fuel_discharge_pressure": p_dis, "ox_discharge_pressure": p_inj,
                    "pump_power_fuel": m_fu * fu_pump.work(p_dis), "pump_power_ox": m_ox * ox_pump.work(p_inj),
                    "isp_vac": ch["isp_vac"], "isp_sl": ch["isp_sl"]}

        out |= {"thrust_vac": self.thrust_vac, "exit_area": ch["exit_area"] * out["mdot_main"] * v_main,
                "cooling_dp": cool_dp}
        if regen is not None:
            out["regen"] = regen.summary()
            self.regen_result = regen
        self._out = out
        return out

    def thrust_curve(self, samples: int = 50) -> ThrustCurve:
        o = self.solve()
        t = np.linspace(0.0, self.burn_time, samples)
        F = np.full(samples, self.thrust_vac)
        m = np.full(samples, o["mdot"])
        F[-1] = m[-1] = 0.0
        return ThrustCurve(t, F, m, float(np.trapezoid(m, t)), self.dry_mass, self.name,
                           exit_area=o["exit_area"], vacuum=True)


@registry.register("rocket_engine", "liquid_cycle")
def _factory(**kw) -> CycleEngine:
    return CycleEngine(**kw)
