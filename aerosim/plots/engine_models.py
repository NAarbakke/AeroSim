"""3D engine figures: geometry and flow fields of AeroSim engine models, for ``site.engines3d_html``.

Each builder turns a solved AeroSim model into an engine description: bodies (surfaces of
revolution, planar walls, grains, tanks, rotor rings) plus the fields to colour them by and the
cards to show. What is real and what is schematic:

- liquid engines   chamber + Rao bell from pyskyfire, sized by the cycle solution (throat and
                   exit area); gas temperature, pressure and Mach along the nozzle from the
                   area ratio with CEA's throat gamma (isentropic, frozen); hot-wall
                   temperature where a pyskyfire cooling solution exists
- solid motor      the grains' real cross-sections: port outlines from the regression map at
                   each web, stepped through the burn with the simulated web history
- hybrid motor     grain port radius and tank liquid level from the simulation; tank shape
                   (length = 4 diameters) is assumed
- ramjet/scramjet  station values from the cycle; flowpath proportions are schematic, scaled
                   to the capture area
- gas turbines     station values from pyCycle; diameters from pyCycle's station flow areas,
                   lengths and blade rows schematic
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq


def _r(a, d=4):
    return np.round(np.asarray(a, float), d).tolist()


def _mach_from_area(ar: float, gamma: float, supersonic: bool) -> float:
    if ar <= 1.0 + 1e-9:
        return 1.0
    g = gamma

    def f(m):
        return (1 / m) * ((2 / (g + 1)) * (1 + (g - 1) / 2 * m * m)) ** ((g + 1) / (2 * (g - 1))) - ar
    return brentq(f, 1.0 + 1e-9, 50.0) if supersonic else brentq(f, 1e-6, 1.0 - 1e-9)


def nozzle_fields(x, r, t_chamber: float, p_chamber: float, gamma: float) -> dict:
    """Isentropic (frozen) gas Mach, temperature and pressure along a chamber contour."""
    x, r = np.asarray(x, float), np.asarray(r, float)
    it = int(np.argmin(r))
    rt = r[it]
    mach = np.array([_mach_from_area((ri / rt) ** 2, gamma, i > it) for i, ri in enumerate(r)])
    T = t_chamber / (1 + (gamma - 1) / 2 * mach**2)
    p = p_chamber * (T / t_chamber) ** (gamma / (gamma - 1))
    return {"T": T, "p": p / 1e6, "M": mach}


GAS_FIELDS = {"T": {"label": "Gas temperature", "unit": "K", "d": 0},
              "p": {"label": "Gas pressure", "unit": "MPa", "d": 2},
              "M": {"label": "Mach number", "unit": "", "d": 2}}


def liquid_engine(name: str, engine, solution: dict, l_star: float = 1.0, chambers: int = 1,
                  regen=None, contour=None) -> dict:
    """``engine``: a solved ``CycleEngine`` (or anything with oxidizer, fuel, mixture_ratio,
    chamber_pressure, expansion_ratio); ``solution``: its ``solve()`` output (needs exit_area).
    ``regen``: a pyskyfire ``RegenResult`` adds hot-wall temperature. ``contour``: (x, r) to use
    instead of a new pyskyfire contour."""
    from aerosim.propulsion.adapters.regen import chamber_contour
    from aerosim.propulsion.liquid.design import _cea
    eps = engine.expansion_ratio
    if contour is None:
        at = solution["exit_area"] / eps / chambers
        x, r, _ = chamber_contour(math.sqrt(at / math.pi), eps, l_star)
    else:
        x, r = (np.asarray(a, float) for a in contour)
    o = np.argsort(x)
    x, r = x[o], r[o]
    keep = np.unique(np.r_[np.linspace(0, x.size - 1, min(x.size, 160)).astype(int), np.argmin(r)])
    x, r = x[keep], r[keep]                 # ~160 points, throat kept
    ch = _cea(None).chamber(engine.oxidizer, engine.fuel, engine.mixture_ratio, engine.chamber_pressure, eps)
    f = nozzle_fields(x, r, ch["t_chamber"], engine.chamber_pressure, ch["gamma_throat"])
    fields = dict(GAS_FIELDS)
    if regen is not None:
        ro = np.argsort(regen.x)
        f["Tw"] = np.interp(x, regen.x[ro], regen.t_hot_wall[ro])
        fields = {"Tw": {"label": "Hot-wall temperature", "unit": "K", "d": 0}, **fields}
    rt, re_ = float(r.min()), float(r[-1])
    wall = {"type": "lathe", "name": "chamber and nozzle", "x": _r(x), "r": _r(r),
            "f": {k: _r(v, 3) for k, v in f.items()}}
    plate = {"type": "lathe", "name": "injector", "x": _r([x[0], x[0]]), "r": _r([0.0, r[0]]), "color": "ink2"}
    cards = {"vacuum thrust": f"{solution['thrust_vac'] / 1e3:,.0f} kN" + (f" ({chambers} chambers)" if chambers > 1 else ""),
             "chamber pressure": f"{engine.chamber_pressure / 1e6:.1f} MPa",
             "mixture ratio": f"{engine.mixture_ratio:.2f}", "vacuum Isp": f"{solution['isp_vac']:.0f} s",
             "throat diameter": f"{2 * rt * 100:.1f} cm", "exit diameter": f"{2 * re_ * 100:.0f} cm",
             "expansion ratio": f"{eps:g}", "chamber temperature": f"{ch['t_chamber']:.0f} K"}
    if "turbine_power" in solution:
        cards["turbopump power"] = f"{solution['turbine_power'] / 1e6:.2f} MW"
    return {"name": name, "group": "Liquid rocket engines", "fields": fields,
            "field": "Tw" if regen is not None else "T", "bodies": [wall, plate], "cards": cards,
            "scale": "to scale (one chamber)" if chambers > 1 else "to scale"}


def _resample(loop: np.ndarray, n: int = 160) -> list:
    loop = np.asarray(loop, float)
    seg = np.r_[0, np.cumsum(np.hypot(*np.diff(np.vstack([loop, loop[:1]]), axis=0).T))]
    s = np.linspace(0, seg[-1], n, endpoint=False)
    closed = np.vstack([loop, loop[:1]])
    return _r(np.c_[np.interp(s, seg, closed[:, 0]), np.interp(s, seg, closed[:, 1])], 5)


def solid_motor(name: str, motor, levels: int = 28, max_frames: int = 240) -> dict:
    """GrainMotor: grains stacked from the forward end, regressing through the simulated burn."""
    sim = motor.simulate()
    t, webs = sim["time"], np.asarray(sim["webs"])
    idx = np.unique(np.r_[np.linspace(0, t.size - 1, min(t.size, max_frames)).astype(int), t.size - 1])
    bodies, x0, gap = [], 0.0, 0.002
    R = max(g.diameter for g in motor.grains) / 2
    for i, g in enumerate(motor.grains):
        lv = np.linspace(0.0, g.web_max, levels)
        holes, islands, lens, starts = [], [], [], []
        for w in lv:
            o = g.port_outline(float(w)) if w < g.web_max else {"holes": [], "islands": []}
            holes.append([_resample(c) for c in o["holes"]])
            islands.append([_resample(c) for c in o["islands"]])
            inh = g.inhibited_ends.lower()
            fwd = 0.0 if inh in ("top", "both") else float(w)     # an uninhibited face recedes by w
            aft = 0.0 if inh in ("bottom", "both") else float(w)
            lens.append(round(max(g.length - fwd - aft, 0.0), 5))
            starts.append(round(x0 + fwd, 5))
        lvl = [int(np.argmin(np.abs(lv - w))) for w in webs[idx, i]]
        bodies.append({"type": "grain", "name": f"grain {i + 1} ({type(g).__name__})", "R": g.diameter / 2,
                       "x0": starts, "len": lens, "holes": holes, "islands": islands, "frame": lvl, "color": "s3"})
        x0 += g.length + gap
    nz = motor.nozzle
    rt, re_ = nz.throat_diameter / 2, nz.exit_diameter / 2
    xn = x0 + 0.004
    conv = (R - rt) / math.tan(math.radians(45))
    div = (re_ - rt) / math.tan(math.radians(nz.divergence_deg if hasattr(nz, "divergence_deg") else 12))
    bodies += [{"type": "lathe", "name": "case", "x": _r([-0.004, xn]), "r": _r([R + 0.002, R + 0.002]),
                "color": "ink2", "opacity": 0.25},
               {"type": "lathe", "name": "nozzle", "x": _r([xn, xn + conv, xn + conv + div]),
                "r": _r([R, rt, re_]), "color": "ink2"}]
    kn = sim["kn"]
    return {"name": name, "group": "Solid and hybrid motors", "fields": {}, "field": None, "bodies": bodies,
            "cards": {"propellant": motor.model.name if hasattr(motor.model, "name") else "",
                      "propellant mass": f"{motor.propellant_mass:.3f} kg",
                      "total impulse": f"{np.trapezoid(sim['thrust'], t):.0f} N·s",
                      "burn time": f"{t[-1]:.2f} s", "peak pressure": f"{sim['pressure'].max() / 1e6:.2f} MPa",
                      "initial Kn": f"{kn[0]:.0f}"},
            "time": {"t": _r(t[idx], 3), "cards": {"chamber pressure, MPa": _r(sim["pressure"][idx] / 1e6, 3),
                                                   "thrust, N": _r(sim["thrust"][idx], 1),
                                                   "Kn": _r(kn[idx], 0)}},
            "scale": "to scale"}


def hybrid_motor(name: str, motor, history, max_frames: int = 240) -> dict:
    h = history
    idx = np.unique(np.r_[np.linspace(0, h.time.size - 1, min(h.time.size, max_frames)).astype(int), h.time.size - 1])
    L, R = motor.grain_length, motor.outer_radius
    tank_v = motor.tank["volume"] if motor.tank else 0.0
    rt_ = (tank_v / (8 * math.pi)) ** (1 / 3) if tank_v else R      # tank 4 diameters long (assumed)
    lt = 8 * rt_
    x_t0, x_g0 = -lt - 0.12, 0.0
    rth = math.sqrt(motor.throat_area / math.pi)
    re_ = rth * math.sqrt(motor.expansion_ratio)
    xn = L + 0.06
    bodies = [{"type": "tank", "name": "N2O tank", "x0": x_t0, "len": lt, "R": rt_,
               "fill": _r(np.nan_to_num(h.liquid_fraction[idx], nan=0.0) if h.liquid_fraction is not None
                          else np.zeros(idx.size), 4)},
              {"type": "lathe", "name": "injector", "x": _r([-0.04, -0.04]), "r": _r([0, R]), "color": "ink2"},
              {"type": "tube", "name": "fuel grain", "x0": x_g0, "len": L, "R": R,
               "r_in": _r(h.port_radius[idx], 5), "color": "s3"},
              {"type": "lathe", "name": "case", "x": _r([-0.05, xn]), "r": _r([R + 0.004, R + 0.004]),
               "color": "ink2", "opacity": 0.25},
              {"type": "lathe", "name": "nozzle", "x": _r([xn, xn + (R - rth), xn + (R - rth) + (re_ - rth) / 0.21]),
               "r": _r([R, rth, re_]), "color": "ink2"}]
    return {"name": name, "group": "Solid and hybrid motors", "fields": {}, "field": None, "bodies": bodies,
            "cards": {"propellants": f"{motor.oxidizer} / {motor.fuel}", "N2O used": f"{h.oxidizer_used:.2f} kg",
                      "fuel burned": f"{h.fuel_used:.2f} kg", "burn time": f"{h.time[-1]:.1f} s",
                      "average Isp": f"{motor.average_isp:.0f} s", "end": h.end_reason},
            "time": {"t": _r(h.time[idx], 3), "cards": {"thrust, kN": _r(h.thrust[idx] / 1e3, 3),
                                                        "chamber, MPa": _r(h.chamber_pressure[idx] / 1e6, 3),
                                                        "tank, MPa": _r(h.tank_pressure[idx] / 1e6, 3),
                                                        "O/F": _r(h.of_ratio[idx], 2),
                                                        "port radius, mm": _r(h.port_radius[idx] * 1e3, 2)}},
            "scale": "to scale; tank proportions assumed"}


FLOW_FIELDS = {"T": {"label": "Static temperature", "unit": "K", "d": 0},
               "p": {"label": "Pressure", "unit": "kPa", "d": 1}}


def ramjet(name: str, point: dict, capture_area: float) -> dict:
    """Axisymmetric ramjet, schematic proportions scaled to the capture radius."""
    Rc = math.sqrt(capture_area / math.pi)
    # stations along x (in capture radii): 0 lip, 2.5 diffuser exit, 6 burner exit, 7 throat, 8.5 exit
    xs = np.array([-1.6, 0.0, 2.5, 6.0, 7.0, 8.5]) * Rc
    T = np.array([point["t0"], point["t0"], point["tt0"], point["t4"], point["t4"] * 0.86, point["t9"]])
    p = np.array([point["p0"], point["p0"], point["pt2"], point["pt4"], point["pt4"] * 0.53, point["p0"]]) / 1e3
    cx = np.array([0.0, 0.3, 2.5, 6.0, 7.0, 8.5]) * Rc
    cr = np.array([1.0, 1.05, 1.25, 1.25, 0.8, 1.15]) * Rc
    bx = np.array([-1.6, 0.6, 3.0]) * Rc
    br = np.array([0.0, 0.55, 0.0]) * Rc
    fx = lambda x: {"T": _r(np.interp(x, xs, T), 1), "p": _r(np.interp(x, xs, p), 2)}
    xc = np.linspace(cx[0], cx[-1], 60)
    xb = np.linspace(bx[0], bx[-1], 40)
    bodies = [{"type": "lathe", "name": "cowl", "x": _r(xc), "r": _r(np.interp(xc, cx, cr)), "f": fx(xc)},
              {"type": "lathe", "name": "centre body", "x": _r(xb), "r": _r(np.interp(xb, bx, br)), "f": fx(xb)}]
    return {"name": name, "group": "Air-breathing engines", "fields": FLOW_FIELDS, "field": "T", "bodies": bodies,
            "cards": {"flight": f"Mach {point['mach']:.1f}, {point['altitude'] / 1e3:.1f} km",
                      "thrust": f"{point['thrust'] / 1e3:.2f} kN", "Isp": f"{point['isp']:.0f} s",
                      "air flow": f"{point['air_flow']:.2f} kg/s", "burner exit": f"{point['t4']:.0f} K",
                      "inlet recovery": f"{point['pt2'] / point['pt0']:.3f}"},
            "stations": [{"station": "0", "name": "Free stream", "T": point["t0"], "p": point["p0"]},
                         {"station": "2", "name": "Burner entry (total)", "T": point["tt0"], "p": point["pt2"]},
                         {"station": "4", "name": "Burner exit", "T": point["t4"], "p": point["pt4"]},
                         {"station": "9", "name": "Nozzle exit", "T": point["t9"], "p": point["p0"]}],
            "scale": "schematic flowpath, scaled to the capture area"}


def scramjet(name: str, point: dict, capture_area: float) -> dict:
    """Two-dimensional scramjet (ramp inlet, isolator, combustor, nozzle), schematic."""
    w = math.sqrt(2 * capture_area)          # width = 2 x capture height
    H = capture_area / w
    xs = np.array([-5.0, 0.0, 2.0, 7.0, 12.0]) * H
    T = np.array([point["t0"], point["t3"], point["t3"], point["t4"], point["t9"]])
    p = np.array([point["p0"], point["p3"], point["p3"], point["p3"], point["p0"]]) / 1e3
    x = np.linspace(xs[0], xs[-1], 80)
    y0 = np.interp(x, np.array([-5, 0, 7, 12]) * H, np.array([-1.2, 0.0, -0.3, -1.6]) * H)
    y1 = np.where(x >= 0, np.interp(x, np.array([0, 2, 7, 9, 12]) * H, np.array([0.25, 0.25, 0.4, 0.4, 0.4]) * H),
                  np.nan)
    f = {"T": _r(np.interp(x, xs, T), 1), "p": _r(np.interp(x, xs, p), 2)}
    bodies = [{"type": "planar", "name": "flowpath", "x": _r(x), "y0": _r(y0), "y1": _r(y1), "w": round(w, 4), "f": f}]
    return {"name": name, "group": "Air-breathing engines", "fields": FLOW_FIELDS, "field": "T", "bodies": bodies,
            "cards": {"flight": f"Mach {point['mach']:.0f}, {point['altitude'] / 1e3:.0f} km",
                      "thrust": f"{point['thrust'] / 1e3:.2f} kN", "Isp": f"{point['isp']:.0f} s",
                      "burner entry": f"{point['t3']:.0f} K, {point['v3']:.0f} m/s",
                      "burner exit": f"{point['t4']:.0f} K", "air flow": f"{point['air_flow']:.2f} kg/s"},
            "stations": [{"station": "0", "name": "Free stream", "T": point["t0"], "p": point["p0"]},
                         {"station": "3", "name": "Burner entry", "T": point["t3"], "p": point["p3"]},
                         {"station": "4", "name": "Burner exit", "T": point["t4"], "p": point["p3"]},
                         {"station": "9", "name": "Nozzle exit", "T": point["t9"], "p": point["p0"]}],
            "scale": "schematic flowpath, scaled to the capture area"}


GT_FIELDS = {"tt": {"label": "Total temperature", "unit": "K", "d": 0},
             "pt": {"label": "Total pressure", "unit": "kPa", "d": 0}}


def gas_turbine(name: str, engine, design: dict) -> dict:
    """pyCycle turbojet / turbofan cutaway: station states coloured along the flow path."""
    st = {s["station"]: s for s in engine.stations()}
    fan = engine.architecture == "turbofan"
    R = math.sqrt(st["2"]["area"] / math.pi)
    bodies = []

    def core_field(xpos: dict, x):
        k = sorted(xpos, key=xpos.get)
        xx = [xpos[s] for s in k]
        return {"tt": _r(np.interp(x, xx, [st[s]["tt"] for s in k]), 1),
                "pt": _r(np.interp(x, xx, [st[s]["pt"] / 1e3 for s in k]), 1)}

    def ring(x, r0, r1, n=0, color="ink2"):
        bodies.append({"type": "ring", "x": round(x, 4), "r0": round(r0, 4), "r1": round(r1, 4), "blades": n, "color": color})

    if not fan:
        D = 2 * R
        pos = {"2": 0.8 * D, "3": 2.2 * D, "4": 3.0 * D, "5": 3.45 * D, "9": 4.4 * D}
        cx = np.array([0, 0.8, 2.2, 2.4, 3.0, 3.45, 3.9, 4.4]) * D
        cr = np.array([1.08, 1.0, 1.0, 1.15, 1.15, 1.05, 0.85, math.sqrt(st["9"]["area"] / math.pi) / R]) * R
        hx = np.array([0.55, 0.8, 2.2, 2.4, 3.0, 3.45, 3.8]) * D
        hr = np.array([0.0, 0.45, 0.72, 0.6, 0.6, 0.62, 0.0]) * R
        x = np.linspace(0, 4.4 * D, 90)
        xh = np.linspace(hx[0], hx[-1], 60)
        bodies += [{"type": "lathe", "name": "casing", "x": _r(x), "r": _r(np.interp(x, cx, cr)), "f": core_field(pos, x)},
                   {"type": "lathe", "name": "hub", "x": _r(xh), "r": _r(np.interp(xh, hx, hr)), "f": core_field(pos, xh)}]
        for k in range(8):                                             # compressor stages
            xk = (0.85 + k * 0.17) * D
            ring(xk, np.interp(xk, hx, hr), 0.97 * np.interp(xk, cx, cr), 24)
        for xk in (3.1 * D, 3.3 * D):                                  # turbine stages
            ring(xk, np.interp(xk, hx, hr), 0.97 * np.interp(xk, cx, cr), 30, "s1")
        cards = {"thrust (sea-level static)": f"{design['thrust'] / 1e3:.1f} kN",
                 "pressure ratio": f"{design['comp_pr']:.1f}", "turbine inlet": f"{design['t4']:.0f} K",
                 "air flow": f"{design['air_flow']:.1f} kg/s", "TSFC": f"{design['tsfc_lb_lbf_h']:.3f} lb/(lbf·h)",
                 "inlet diameter": f"{2 * R:.2f} m"}
    else:
        D = 2 * R
        core_R = math.sqrt(st["21"]["area"] / math.pi + (0.35 * R) ** 2)
        pos = {"2": 0.35 * D, "21": 0.55 * D, "3": 1.25 * D, "4": 1.6 * D, "45": 1.75 * D, "5": 2.2 * D, "9": 2.55 * D}
        x = np.linspace(0.5 * D, 2.55 * D, 90)
        cx = np.array([0.5, 0.6, 1.25, 1.3, 1.6, 1.75, 2.2, 2.55]) * D
        cr = np.array([core_R / R, 0.62, 0.42, 0.5, 0.5, 0.5, 0.55, 0.45]) * R
        hx = np.array([0.0, 0.35, 0.6, 1.25, 1.3, 1.6, 1.75, 2.2, 2.75]) * D
        hr = np.array([0.0, 0.32, 0.33, 0.33, 0.28, 0.28, 0.3, 0.3, 0.0]) * R
        xh = np.linspace(0, 2.75 * D, 90)
        nx = np.array([0.0, 0.35, 1.4, 1.75]) * D
        nr = np.array([1.06, 1.0, 1.0, math.sqrt(st["19"]["area"] / math.pi + (0.62 * R) ** 2) / R]) * R
        xn = np.linspace(0, 1.75 * D, 60)
        byp = {"tt": _r(np.interp(xn, [0.35 * D, 0.45 * D], [st["2"]["tt"], st["13"]["tt"]]), 1),
               "pt": _r(np.interp(xn, [0.35 * D, 0.45 * D], [st["2"]["pt"] / 1e3, st["13"]["pt"] / 1e3]), 1)}
        bodies += [{"type": "lathe", "name": "nacelle", "x": _r(xn), "r": _r(np.interp(xn, nx, nr)), "f": byp},
                   {"type": "lathe", "name": "core casing", "x": _r(x), "r": _r(np.interp(x, cx, cr)), "f": core_field(pos, x)},
                   {"type": "lathe", "name": "hub and plug", "x": _r(xh), "r": _r(np.interp(xh, hx, hr)), "f": core_field(pos, xh)}]
        ring(0.4 * D, 0.32 * R, 0.98 * R, 24)                           # fan
        for k in range(9):                                             # HPC
            xk = (0.65 + k * 0.067) * D
            ring(xk, np.interp(xk, hx, hr), 0.97 * np.interp(xk, cx, cr), 30)
        for k, xk in enumerate((1.65, 1.82, 1.95, 2.05, 2.15)):        # HPT + LPT
            ring(xk * D, np.interp(xk * D, hx, hr), 0.97 * np.interp(xk * D, cx, cr), 36, "s1")
        cards = {"thrust (sea-level static)": f"{design['thrust'] / 1e3:.0f} kN",
                 "bypass ratio": f"{design['bpr']:.1f}", "overall pressure ratio": f"{design['opr']:.1f}",
                 "turbine inlet": f"{design['t4']:.0f} K", "air flow": f"{design['air_flow']:.0f} kg/s",
                 "TSFC": f"{design['tsfc_lb_lbf_h']:.3f} lb/(lbf·h)", "fan diameter": f"{2 * R:.2f} m"}
    return {"name": name, "group": "Air-breathing engines", "fields": GT_FIELDS, "field": "tt", "bodies": bodies,
            "cards": cards, "stations": [{"station": s["station"], "name": s["name"], "T": s["tt"], "p": s["pt"]}
                                         for s in st.values()],
            "scale": "diameters from pyCycle flow areas; lengths and blade rows schematic"}
