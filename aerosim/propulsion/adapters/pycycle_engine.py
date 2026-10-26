"""pyCycle backend (NASA, Apache-2.0, built on OpenMDAO): gas-turbine design and off-design.

    eng = PyCycleEngine("turbofan", design={"thrust": 121e3, "t4": 1640, "fan_pr": 1.65,
                                             "bpr": 5.1, "hpc_pr": 19.8})
    eng.operating_point(altitude=0, mach=0.0, power=1.0)   # thrust, fuel_flow, tsfc, ...
    eng.deck(altitudes, machs, powers)                      # -> EngineDeck

Architectures (element models and maps from pyCycle; cycles follow its example_cycles):

- ``turbojet``  inlet, compressor (AXI5 map), combustor, turbine (LPT2269 map),
                convergent-divergent nozzle, one shaft
- ``turbofan``  two-spool separate-flow: fan (FanMap) on the low spool with the LPT; HPC
                (HPCMap) on the high spool with the HPT; core and bypass convergent nozzles

Design point: inlet mass flow is solved for the design thrust, fuel-air ratio for the
turbine-inlet temperature ``t4`` and turbine pressure ratios for zero net shaft power.
Off design: the nozzle throat areas are frozen at their design values; fuel-air ratio is
solved for the turbine-inlet temperature and mass flow, bypass ratio and spool speeds for
nozzle flow and shaft-power matching on the component maps. Throttle ``power`` in [0, 1]
maps linearly to turbine-inlet temperature between ``t4_idle`` and ``t4``. Real engines cap
T4 at altitude well below takeoff; full design T4 at cruise can drive the spools off their
maps (no convergence), so decks at altitude should use power below ~0.9.

Thermodynamics: pyCycle's tabulated air / Jet-A products by default (``THERMO``; set it to
CEA equilibrium for janaf species). Needs om-pycycle newer than 4.4.0 on numpy >= 2.5 (the
fix is on pyCycle's master). Off-design points are reached by adaptive continuation from the
last converged point: altitude, Mach and T4 move towards the target in steps; a step that
fails (NaN residuals or no convergence) restores the last converged solution and is halved.
Newton on the full cycle is sensitive to its start, so a direct jump that converges on one
machine can diverge on another (different floating-point kernels); the walk makes the result
independent of that. If the walk stalls, it restarts from the design point. SI in and out.
"""

from __future__ import annotations

import numpy as np

from aerosim.core import AerosimError, EngineDeck, registry, require

LBF_H = 0.45359237 / 3600 / 4.4482216152605      # lb/(lbf h) -> kg/(N s)
# pyCycle thermodynamics: tabulated air / Jet-A products (fast) or chemical equilibrium (CEA)
THERMO = {"method": "TABULAR", "data": lambda pyc: pyc.AIR_JETA_TAB_SPEC, "fuel": "FAR"}


def _turbojet_cls(om, pyc):
    class Turbojet(pyc.Cycle):
        def setup(self):
            self.options["thermo_method"] = THERMO["method"]
            self.options["thermo_data"] = THERMO["data"](pyc)
            design = self.options["design"]
            self.add_subsystem("fc", pyc.FlightConditions())
            self.add_subsystem("inlet", pyc.Inlet())
            self.add_subsystem("comp", pyc.Compressor(map_data=pyc.AXI5, map_extrap=True),
                               promotes_inputs=["Nmech"])
            self.add_subsystem("burner", pyc.Combustor(fuel_type=THERMO["fuel"]))
            self.add_subsystem("turb", pyc.Turbine(map_data=pyc.LPT2269, map_extrap=True),
                               promotes_inputs=["Nmech"])
            self.add_subsystem("nozz", pyc.Nozzle(nozzType="CD", lossCoef="Cv"))
            self.add_subsystem("shaft", pyc.Shaft(num_ports=2), promotes_inputs=["Nmech"])
            self.add_subsystem("perf", pyc.Performance(num_nozzles=1, num_burners=1))
            self.pyc_connect_flow("fc.Fl_O", "inlet.Fl_I", connect_w=False)
            self.pyc_connect_flow("inlet.Fl_O", "comp.Fl_I")
            self.pyc_connect_flow("comp.Fl_O", "burner.Fl_I")
            self.pyc_connect_flow("burner.Fl_O", "turb.Fl_I")
            self.pyc_connect_flow("turb.Fl_O", "nozz.Fl_I")
            self.connect("comp.trq", "shaft.trq_0")
            self.connect("turb.trq", "shaft.trq_1")
            self.connect("fc.Fl_O:stat:P", "nozz.Ps_exhaust")
            self.connect("inlet.Fl_O:tot:P", "perf.Pt2")
            self.connect("comp.Fl_O:tot:P", "perf.Pt3")
            self.connect("burner.Wfuel", "perf.Wfuel_0")
            self.connect("inlet.F_ram", "perf.ram_drag")
            self.connect("nozz.Fg", "perf.Fg_0")
            bal = self.add_subsystem("balance", om.BalanceComp())
            bal.add_balance("FAR", eq_units="degR", lower=1e-4, val=0.017, rhs_name="T4_target")
            self.connect("balance.FAR", "burner.Fl_I:FAR")
            self.connect("burner.Fl_O:tot:T", "balance.lhs:FAR")
            if design:
                bal.add_balance("W", units="lbm/s", eq_units="lbf", rhs_name="Fn_target")
                self.connect("balance.W", "inlet.Fl_I:stat:W")
                self.connect("perf.Fn", "balance.lhs:W")
                bal.add_balance("turb_PR", val=1.5, lower=1.001, upper=8, eq_units="hp", rhs_val=0.0)
                self.connect("balance.turb_PR", "turb.PR")
                self.connect("shaft.pwr_net", "balance.lhs:turb_PR")
            else:
                bal.add_balance("Nmech", val=1.5, units="rpm", lower=500.0, eq_units="hp", rhs_val=0.0)
                self.connect("balance.Nmech", "Nmech")
                self.connect("shaft.pwr_net", "balance.lhs:Nmech")
                bal.add_balance("W", val=168.0, units="lbm/s", eq_units="inch**2")
                self.connect("balance.W", "inlet.Fl_I:stat:W")
                self.connect("nozz.Throat:stat:area", "balance.lhs:W")
            _newton(self, om)
            super().setup()
    return Turbojet


def _turbofan_cls(om, pyc):
    class Turbofan(pyc.Cycle):
        def setup(self):
            self.options["thermo_method"] = THERMO["method"]
            self.options["thermo_data"] = THERMO["data"](pyc)
            design = self.options["design"]
            self.add_subsystem("fc", pyc.FlightConditions())
            self.add_subsystem("inlet", pyc.Inlet())
            self.add_subsystem("fan", pyc.Compressor(map_data=pyc.FanMap, map_extrap=True),
                               promotes_inputs=[("Nmech", "LP_Nmech")])
            self.add_subsystem("splitter", pyc.Splitter())
            self.add_subsystem("hpc", pyc.Compressor(map_data=pyc.HPCMap, map_extrap=True),
                               promotes_inputs=[("Nmech", "HP_Nmech")])
            self.add_subsystem("burner", pyc.Combustor(fuel_type=THERMO["fuel"]))
            self.add_subsystem("hpt", pyc.Turbine(map_data=pyc.HPTMap, map_extrap=True),
                               promotes_inputs=[("Nmech", "HP_Nmech")])
            self.add_subsystem("lpt", pyc.Turbine(map_data=pyc.LPTMap, map_extrap=True),
                               promotes_inputs=[("Nmech", "LP_Nmech")])
            self.add_subsystem("core_nozz", pyc.Nozzle(nozzType="CV", lossCoef="Cv"))
            self.add_subsystem("byp_duct", pyc.Duct())
            self.add_subsystem("byp_nozz", pyc.Nozzle(nozzType="CV", lossCoef="Cv"))
            self.add_subsystem("lp_shaft", pyc.Shaft(num_ports=2), promotes_inputs=[("Nmech", "LP_Nmech")])
            self.add_subsystem("hp_shaft", pyc.Shaft(num_ports=2), promotes_inputs=[("Nmech", "HP_Nmech")])
            self.add_subsystem("perf", pyc.Performance(num_nozzles=2, num_burners=1))
            self.connect("inlet.Fl_O:tot:P", "perf.Pt2")
            self.connect("hpc.Fl_O:tot:P", "perf.Pt3")
            self.connect("burner.Wfuel", "perf.Wfuel_0")
            self.connect("inlet.F_ram", "perf.ram_drag")
            self.connect("core_nozz.Fg", "perf.Fg_0")
            self.connect("byp_nozz.Fg", "perf.Fg_1")
            self.connect("fan.trq", "lp_shaft.trq_0")
            self.connect("lpt.trq", "lp_shaft.trq_1")
            self.connect("hpc.trq", "hp_shaft.trq_0")
            self.connect("hpt.trq", "hp_shaft.trq_1")
            self.connect("fc.Fl_O:stat:P", "core_nozz.Ps_exhaust")
            self.connect("fc.Fl_O:stat:P", "byp_nozz.Ps_exhaust")
            bal = self.add_subsystem("balance", om.BalanceComp())
            bal.add_balance("FAR", eq_units="degR", lower=1e-4, val=0.025, rhs_name="T4_target")
            self.connect("balance.FAR", "burner.Fl_I:FAR")
            self.connect("burner.Fl_O:tot:T", "balance.lhs:FAR")
            if design:
                bal.add_balance("W", units="lbm/s", eq_units="lbf", rhs_name="Fn_target")
                self.connect("balance.W", "fc.W")
                self.connect("perf.Fn", "balance.lhs:W")
                for sp, tb in (("lp", "lpt"), ("hp", "hpt")):
                    bal.add_balance(f"{tb}_PR", val=2.0, lower=1.001, upper=10, eq_units="hp",
                                    use_mult=True, mult_val=-1)
                    self.connect(f"balance.{tb}_PR", f"{tb}.PR")
                    self.connect(f"{sp}_shaft.pwr_in_real", f"balance.lhs:{tb}_PR")
                    self.connect(f"{sp}_shaft.pwr_out_real", f"balance.rhs:{tb}_PR")
            else:
                bal.add_balance("W", units="lbm/s", lower=1.0, upper=3000.0, eq_units="inch**2")
                self.connect("balance.W", "fc.W")
                self.connect("core_nozz.Throat:stat:area", "balance.lhs:W")
                bal.add_balance("BPR", lower=0.5, upper=20.0, eq_units="inch**2")
                self.connect("balance.BPR", "splitter.BPR")
                self.connect("byp_nozz.Throat:stat:area", "balance.lhs:BPR")
                for sp in ("lp", "hp"):
                    bal.add_balance(f"{sp}_Nmech", val=1.5, units="rpm", lower=200.0, eq_units="hp",
                                    use_mult=True, mult_val=-1)
                    self.connect(f"balance.{sp}_Nmech", f"{sp.upper()}_Nmech")
                    self.connect(f"{sp}_shaft.pwr_in_real", f"balance.lhs:{sp}_Nmech")
                    self.connect(f"{sp}_shaft.pwr_out_real", f"balance.rhs:{sp}_Nmech")
            self.pyc_connect_flow("fc.Fl_O", "inlet.Fl_I")
            self.pyc_connect_flow("inlet.Fl_O", "fan.Fl_I")
            self.pyc_connect_flow("fan.Fl_O", "splitter.Fl_I")
            self.pyc_connect_flow("splitter.Fl_O1", "hpc.Fl_I")
            self.pyc_connect_flow("hpc.Fl_O", "burner.Fl_I")
            self.pyc_connect_flow("burner.Fl_O", "hpt.Fl_I")
            self.pyc_connect_flow("hpt.Fl_O", "lpt.Fl_I")
            self.pyc_connect_flow("lpt.Fl_O", "core_nozz.Fl_I")
            self.pyc_connect_flow("splitter.Fl_O2", "byp_duct.Fl_I")
            self.pyc_connect_flow("byp_duct.Fl_O", "byp_nozz.Fl_I")
            _newton(self, om, maxiter=50, armijo=True)
            super().setup()
    return Turbofan


def _newton(group, om, maxiter=20, armijo=False):
    n = group.nonlinear_solver = om.NewtonSolver()
    n.options.update({"atol": 1e-6, "rtol": 1e-8, "iprint": -1, "maxiter": maxiter,
                      "solve_subsystems": True, "max_sub_solves": 200,
                      "reraise_child_analysiserror": False, "err_on_non_converge": True})
    if armijo:
        ls = n.linesearch = om.ArmijoGoldsteinLS()
        ls.options.update({"maxiter": 3, "rho": 0.75})
    group.linear_solver = om.DirectSolver()


TURBOJET_DEFAULTS = {"altitude": 0.0, "mach": 1e-6, "thrust": 52.8e3, "t4": 1210.0,
                     "comp_pr": 13.5, "comp_eff": 0.83, "turb_eff": 0.86, "burner_dpqp": 0.03,
                     "nozzle_cv": 0.99, "nmech": 8070.0}
TURBOFAN_DEFAULTS = {"altitude": 0.0, "mach": 1e-6, "thrust": 121e3, "t4": 1640.0, "bpr": 5.1,
                     "fan_pr": 1.65, "fan_eff": 0.89, "hpc_pr": 19.8, "hpc_eff": 0.86,
                     "hpt_eff": 0.89, "lpt_eff": 0.90, "burner_dpqp": 0.05, "bypass_dpqp": 0.015,
                     "core_cv": 0.993, "bypass_cv": 0.994, "lp_nmech": 4666.0, "hp_nmech": 14700.0}


# (station number, name, pyCycle flow port) in flow order
STATIONS = {
    "turbojet": [("0", "Free stream", "fc.Fl_O"), ("2", "Compressor face", "inlet.Fl_O"),
                 ("3", "Compressor exit", "comp.Fl_O"), ("4", "Turbine inlet", "burner.Fl_O"),
                 ("5", "Turbine exit", "turb.Fl_O"), ("9", "Nozzle exit", "nozz.Fl_O")],
    "turbofan": [("0", "Free stream", "fc.Fl_O"), ("2", "Fan face", "inlet.Fl_O"),
                 ("21", "Fan exit, core", "splitter.Fl_O1"), ("13", "Fan exit, bypass", "splitter.Fl_O2"),
                 ("3", "HPC exit", "hpc.Fl_O"), ("4", "HPT inlet", "burner.Fl_O"),
                 ("45", "LPT inlet", "hpt.Fl_O"), ("5", "LPT exit", "lpt.Fl_O"),
                 ("9", "Core nozzle exit", "core_nozz.Fl_O"), ("19", "Bypass nozzle exit", "byp_nozz.Fl_O")],
}


@registry.register("airbreathing", "pycycle")
class PyCycleEngine:
    """Gas-turbine design point + off-design points on pyCycle's component maps (SI units)."""

    def __init__(self, architecture: str = "turbojet", design: dict | None = None,
                 t4_idle: float | None = None, name: str | None = None) -> None:
        if architecture not in ("turbojet", "turbofan"):
            raise ValueError("architecture must be 'turbojet' or 'turbofan'")
        self.architecture = architecture
        base = TURBOJET_DEFAULTS if architecture == "turbojet" else TURBOFAN_DEFAULTS
        unknown = set(design or {}) - set(base)
        if unknown:
            raise ValueError(f"unknown design keys {sorted(unknown)}; expected {sorted(base)}")
        self.design = {**base, **(design or {})}
        self.t4_idle = t4_idle or 0.6 * self.design["t4"]
        self.name = name or f"pyCycle {architecture}"
        self._prob = None

    # -- model -------------------------------------------------------------------
    def build(self):
        om = require("openmdao.api", "optim")
        pyc = require("pycycle.api", "airbreathing")
        cls = (_turbojet_cls if self.architecture == "turbojet" else _turbofan_cls)(om, pyc)
        d, fan = self.design, self.architecture == "turbofan"

        class MP(pyc.MPCycle):
            def setup(mp):
                mp.pyc_add_pnt("DESIGN", cls())
                mp.pyc_add_pnt("OD", cls(design=False))
                if fan:
                    for k, v in {"inlet.MN": 0.751, "fan.MN": 0.4578, "splitter.MN1": 0.3104,
                                 "splitter.MN2": 0.4518, "hpc.MN": 0.2442, "burner.MN": 0.1025,
                                 "hpt.MN": 0.3650, "lpt.MN": 0.4127, "byp_duct.MN": 0.4589}.items():
                        mp.set_input_defaults(f"DESIGN.{k}", v)
                    mp.set_input_defaults("DESIGN.splitter.BPR", d["bpr"])
                    mp.set_input_defaults("DESIGN.LP_Nmech", d["lp_nmech"], units="rpm")
                    mp.set_input_defaults("DESIGN.HP_Nmech", d["hp_nmech"], units="rpm")
                    mp.pyc_add_cycle_param("burner.dPqP", d["burner_dpqp"])
                    mp.pyc_add_cycle_param("byp_duct.dPqP", d["bypass_dpqp"])
                    mp.pyc_add_cycle_param("core_nozz.Cv", d["core_cv"])
                    mp.pyc_add_cycle_param("byp_nozz.Cv", d["bypass_cv"])
                    mp.pyc_use_default_des_od_conns()
                    mp.pyc_connect_des_od("core_nozz.Throat:stat:area", "balance.rhs:W")
                    mp.pyc_connect_des_od("byp_nozz.Throat:stat:area", "balance.rhs:BPR")
                else:
                    mp.set_input_defaults("DESIGN.Nmech", d["nmech"], units="rpm")
                    for k, v in {"inlet.MN": 0.60, "comp.MN": 0.02, "burner.MN": 0.02, "turb.MN": 0.4}.items():
                        mp.set_input_defaults(f"DESIGN.{k}", v)
                    mp.pyc_add_cycle_param("burner.dPqP", d["burner_dpqp"])
                    mp.pyc_add_cycle_param("nozz.Cv", d["nozzle_cv"])
                    mp.pyc_use_default_des_od_conns()
                    mp.pyc_connect_des_od("nozz.Throat:stat:area", "balance.rhs:W")
                super(MP, mp).setup()

        prob = om.Problem(reports=False)
        prob.model = MP()
        prob.setup(check=False)
        prob.set_solver_print(level=-1)
        sv = prob.set_val
        sv("DESIGN.fc.alt", d["altitude"], units="m")
        sv("DESIGN.fc.MN", max(d["mach"], 1e-6))
        sv("DESIGN.balance.Fn_target", d["thrust"], units="N")
        sv("DESIGN.balance.T4_target", d["t4"], units="K")
        if fan:
            for k in ("fan_pr", "fan_eff", "hpc_pr", "hpc_eff", "hpt_eff", "lpt_eff"):
                comp, q = k.split("_")
                sv(f"DESIGN.{comp}.{q.upper() if q == 'pr' else q}", d[k])
            guess = {"DESIGN.balance.FAR": 0.025, "DESIGN.balance.W": d["thrust"] / 4.448 / 35,
                     "DESIGN.balance.lpt_PR": 4.0, "DESIGN.balance.hpt_PR": 3.0,
                     "OD.balance.FAR": 0.025, "OD.balance.W": d["thrust"] / 4.448 / 35,
                     "OD.balance.BPR": d["bpr"], "OD.balance.lp_Nmech": d["lp_nmech"],
                     "OD.balance.hp_Nmech": d["hp_nmech"], "OD.hpt.PR": 3.0, "OD.lpt.PR": 4.0,
                     "OD.fan.map.RlineMap": 2.0, "OD.hpc.map.RlineMap": 2.0}
        else:
            sv("DESIGN.comp.PR", d["comp_pr"])
            sv("DESIGN.comp.eff", d["comp_eff"])
            sv("DESIGN.turb.eff", d["turb_eff"])
            guess = {"DESIGN.balance.FAR": 0.0175, "DESIGN.balance.W": d["thrust"] / 4.448 / 70,
                     "DESIGN.balance.turb_PR": 4.46, "OD.balance.W": d["thrust"] / 4.448 / 70,
                     "OD.balance.FAR": 0.0168, "OD.balance.Nmech": d["nmech"], "OD.turb.PR": 4.67}
        for k, v in guess.items():
            prob[k] = v
        self._prob, self._guess = prob, guess
        self._set_od(d["altitude"], max(d["mach"], 1e-6), d["t4"])
        self._run()
        self._cond = np.array([d["altitude"], max(d["mach"], 1e-6), d["t4"]])
        return prob

    def _set_od(self, altitude, mach, t4):
        p = self._prob
        p.set_val("OD.fc.alt", altitude, units="m")
        p.set_val("OD.fc.MN", max(mach, 1e-6))
        p.set_val("OD.balance.T4_target", t4, units="K")

    # the full solution vector (design and off-design point): saved after each converged
    # solve so a failed Newton step can go back to it
    def _snapshot(self):
        return self._prob.model._outputs.asarray().copy()

    def _restore(self, state):
        self._prob.model._outputs.asarray()[:] = state

    def _walk(self, target, first_step: float = 0.25, min_step: float = 1 / 64):
        """Adaptive continuation from the last converged (altitude, Mach, T4) to ``target``."""
        start, target = self._cond.copy(), np.asarray(target, float)
        good, k, step = self._snapshot(), 0.0, first_step
        while k < 1.0:
            step = min(step, 1.0 - k)
            cond = start + (k + step) * (target - start)
            self._set_od(*cond)
            try:
                self._run()
            except AerosimError:
                self._restore(good)
                self._set_od(*self._cond)
                step /= 2
                if step < min_step:
                    raise
                continue
            k += step
            good, self._cond = self._snapshot(), cond
            step *= 2

    def _run(self):
        from openmdao.api import AnalysisError
        try:
            self._prob.run_model()
        except (AnalysisError, RuntimeError) as e:    # RuntimeError: singular Jacobian off the maps
            raise AerosimError(f"pyCycle did not converge: {str(e)[:200]}") from e

    def as_subsystem(self):
        """The pyCycle problem (design + off-design point) for coupling in OpenMDAO."""
        return self._prob or self.build()

    # -- results -----------------------------------------------------------------
    def _point(self, pt: str) -> dict[str, float]:
        p = self._prob
        g = lambda k, u=None: float(p.get_val(f"{pt}.{k}", units=u)[0])
        fan = self.architecture == "turbofan"
        fn, wf = g("perf.Fn", "N"), g("perf.Wfuel", "kg/s")
        out = {"thrust": fn, "fuel_flow": wf, "tsfc": wf / fn if fn > 0 else float("nan"),
               "tsfc_lb_lbf_h": wf / fn / LBF_H if fn > 0 else float("nan"),
               "air_flow": g("inlet.Fl_O:stat:W", "kg/s"), "opr": g("perf.OPR"),
               "t4": g("burner.Fl_O:tot:T", "K"), "far": g("balance.FAR"),
               "gross_thrust": g("perf.Fg", "N"), "ram_drag": g("inlet.F_ram", "N")}
        if fan:
            out |= {"bpr": g("splitter.BPR"), "fan_pr": g("fan.PR"), "hpc_pr": g("hpc.PR"),
                    "lp_rpm": g("LP_Nmech", "rpm"), "hp_rpm": g("HP_Nmech", "rpm")}
        else:
            out |= {"comp_pr": g("comp.PR"), "rpm": g("Nmech", "rpm")}
        return out

    def stations(self, point: str = "DESIGN") -> list[dict]:
        """Flow-station states along the engine: total T (K) and p (Pa), static T, Mach, area (m^2)."""
        if self._prob is None:
            self.build()
        p, out = self._prob, []
        for num, name, path in STATIONS[self.architecture]:
            g = lambda q, u=None, path=path: float(p.get_val(f"{point}.{path}:{q}", units=u)[0])
            st = {"station": num, "name": name, "tt": g("tot:T", "K"), "pt": g("tot:P", "Pa"),
                  "ts": g("stat:T", "K"), "mach": g("stat:MN")}
            try:
                st["area"] = g("stat:area", "m**2")
            except (KeyError, ValueError):
                st["area"] = float("nan")
            out.append(st)
        return out

    def design_point(self) -> dict[str, float]:
        if self._prob is None:
            self.build()
        return self._point("DESIGN")

    def operating_point(self, altitude: float, mach: float, power: float) -> dict[str, float]:
        """Off-design point; ``power`` 0..1 sets T4 between ``t4_idle`` and design ``t4``."""
        if self._prob is None:
            self.build()
        t4 = self.t4_idle + float(np.clip(power, 0, 1.2)) * (self.design["t4"] - self.t4_idle)
        target = (altitude, max(mach, 1e-6), t4)
        try:
            self._walk(target)
        except AerosimError:
            self.build()                    # from the design point
            self._walk(target)
        return self._point("OD")

    def deck(self, altitudes, machs, powers) -> EngineDeck:
        """Thrust / fuel-flow table. Points are solved in a serpentine order (each from the
        previous converged state); points that do not converge are filled from neighbours."""
        alts, mns, pws = (np.asarray(a, float) for a in (altitudes, machs, powers))
        F = np.full((len(alts), len(mns), len(pws)), np.nan)
        W = F.copy()
        for i, h in enumerate(alts):
            js = range(len(mns)) if i % 2 == 0 else reversed(range(len(mns)))
            for j in js:
                for k in sorted(range(len(pws)), key=lambda k: -pws[k]):
                    try:
                        o = self.operating_point(h, mns[j], pws[k])
                        F[i, j, k], W[i, j, k] = o["thrust"], o["fuel_flow"]
                    except AerosimError:
                        self.build()                # reset to the design solution
        for a in (F, W):
            bad = ~np.isfinite(a)
            if bad.all():
                raise AerosimError("no off-design point converged")
            if bad.any():
                from scipy.interpolate import NearestNDInterpolator
                pts = np.argwhere(~bad)
                a[bad] = NearestNDInterpolator(pts, a[~bad])(np.argwhere(bad))
        return EngineDeck(alts, mns, pws, F, W, name=self.name)
