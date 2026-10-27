"""Pitch stability of a launch vehicle along its ascent: centre of gravity as tanks drain and
stages drop, centre of pressure, static margin, and the short-period pitch dynamics.

    st = stability(result, vehicle, ROCKOT_LAYOUT)        # arrays per output sample
    st["margin"], st["t_double"], st["gimbal_per_aoa"] ...

Mass: a ``Layout`` places every dry part (spread uniformly over [x0, x1], x measured aft from
the nose tip) and every propellant tank. Parts leave at their event (``until``: a key of
``result.events``). A stage's tanks hold ``share`` of its propellant; what is left of it at time
t follows from the stage's constant mass flow over the phases in which it burned. Thrust settles
the liquid at the bottom of each tank, so the remaining propellant fills [x_bottom - f L,
x_bottom] (f = fraction left) and its centroid moves aft as the tank empties. Pitch inertia
about the CG treats parts and liquid as uniform rods (no slosh) plus the cylinder's radial term.

Aerodynamics: Barrowman's slender-body method (Barrowman 1967), linear in angle of attack:
an ogive nose gives C_Nα = 2 (d_n/d)^2 at 0.466 of its length; a change of diameter (shoulder
or boattail) gives 2 [(d_aft/d)^2 - (d_fore/d)^2] at its own centre of pressure. A cylindrical
body adds no linear lift, so a launcher without fins carries its lift near the nose. Body
crossflow lift, which grows with α^2 and pulls the CP aft at larger angles, is left out.

Static margin = (x_cp - x_cg) / d, in calibers: positive is stable (CP behind the CG).

Short period, rigid body, pitch about the CG (Barrowman / Mandell, *Topics in Advanced Model
Rocketry*, ch. 5), I θ'' = M_q θ' + M_α α with
    M_α = q A C_Nα (x_cg - x_cp)                     restoring if negative (N m / rad)
    M_q = -q A Σ C_Nα,i (x_i - x_cg)^2 / V           aerodynamic pitch damping
          - ṁ (x_exit - x_cg)^2                       jet damping of the exhaust
Unstable (M_α > 0): disturbances grow as e^{s t}, s = [M_q + sqrt(M_q^2 + 4 I M_α)] / 2I; the
time to double, ln 2 / s, is what the steering must beat. Stable: natural frequency
sqrt(-M_α / I) and damping ratio -M_q / (2 sqrt(-M_α I)).
Control: the engine deflection that holds one degree of α, from the moment balance
δ/α = q A C_Nα (x_cg - x_cp) / (F (x_gimbal - x_cg)).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Part:
    name: str
    mass: float              # kg
    x0: float                # m from the nose tip
    x1: float
    until: str | None = None  # event at which it leaves (separation, jettison)


@dataclass
class Tank:
    name: str
    stage: str               # stage whose propellant it holds
    share: float             # fraction of that stage's propellant
    x_top: float
    x_bottom: float


@dataclass
class Layout:
    diameter: float                      # reference diameter, m
    nose_length: float                   # tangent ogive
    nose_diameter: float
    shoulders: list[tuple[float, float, float, float]]   # (x_fore, x_aft, d_fore, d_aft)
    parts: list[Part]
    tanks: list[Tank]
    gimbal: dict[str, float]             # stage -> x of the steering thrust point
    nozzle_exit: dict[str, float]        # stage -> x of the nozzle exit plane
    shoulders_until: str | None = None   # event that removes the shoulders (fairing jettison)
    notes: dict = field(default_factory=dict)


def barrowman(layout: Layout, with_shoulders: bool = True) -> list[tuple[float, float]]:
    """[(C_Nα per rad, x_cp m)] of the nose and each shoulder, referred to ``layout.diameter``."""
    d = layout.diameter
    comps = [(2.0 * (layout.nose_diameter / d) ** 2, 0.466 * layout.nose_length)]
    for xf, xa, df, da in (layout.shoulders if with_shoulders else []):
        if abs(df - da) < 1e-9:
            continue
        r = df / da
        comps.append((2.0 * ((da / d) ** 2 - (df / d) ** 2),
                      xf + (xa - xf) / 3.0 * (1.0 + (1.0 - r) / (1.0 - r * r))))
    return comps


def mass_properties(layout: Layout, left: dict[str, float], full: dict[str, float],
                    present) -> tuple[float, float, float]:
    """(mass, x_cg, pitch inertia about the CG) with ``left[stage]`` kg of propellant out of
    ``full[stage]`` and the parts for which ``present(part)`` is true."""
    items = []                                                 # (mass, centroid, length)
    for p in layout.parts:
        if present(p):
            items.append((p.mass, 0.5 * (p.x0 + p.x1), p.x1 - p.x0))
    for tk in layout.tanks:
        mt = max(left.get(tk.stage, 0.0), 0.0) * tk.share
        if mt <= 0:
            continue
        L = tk.x_bottom - tk.x_top
        f = mt / (full[tk.stage] * tk.share)
        items.append((mt, tk.x_bottom - 0.5 * f * L, f * L))
    m = sum(a for a, _, _ in items)
    xcg = sum(a * x for a, x, _ in items) / m
    r2 = (layout.diameter / 2) ** 2
    inertia = sum(a * (ln * ln / 12 + (x - xcg) ** 2 + r2 / 4) for a, x, ln in items)
    return m, xcg, inertia


def _stage_of_phase(vehicle, name: str):
    for st in vehicle.stages:
        if name == st.name or name.startswith(st.name + " burn"):
            return st
    return None


def stability(res, vehicle, layout: Layout, t_end: float | None = None,
              min_q: float = 200.0) -> dict[str, np.ndarray]:
    """Mass properties, CP, static margin and pitch dynamics at each output sample.
    Aerodynamic quantities are NaN where q < ``min_q`` Pa (no meaningful aerodynamics)."""
    from aerosim.atmosphere.standard import isa
    t_all = res.time
    keep = t_all <= (t_end if t_end is not None else t_all[-1]) + 1e-9
    t = t_all[keep]
    n = t.size
    stages = {s.name: s for s in vehicle.stages}
    burned = {k: np.zeros(n) for k in stages}
    ph = res.phase[keep]
    st0 = _stage_of_phase(vehicle, res.phases[ph[0]])        # output starts after liftoff
    if st0 is not None:
        burned[st0.name][0] = st0.mass_flow * (t[0] - res.events.get("liftoff", 0.0))
    for i in range(1, n):
        st = _stage_of_phase(vehicle, res.phases[ph[i]])
        for b in burned.values():
            b[i] = b[i - 1]
        if st is not None:
            burned[st.name][i] += st.mass_flow * (t[i] - t[i - 1])
    ev = res.events
    mres = res.mass[keep]
    full = {k: s.propellant_mass for k, s in stages.items()}

    def present(event, mass, i):
        """Before the event; at the event sample itself, whichever side the simulator recorded
        (the mass falls by ``mass`` right after a sample taken before the drop)."""
        te = ev.get(event, np.inf) if event else np.inf
        if t[i] < te - 1e-9:
            return True
        if t[i] > te + 1e-9 or i + 1 >= n:
            return False
        return mres[i] - mres[i + 1] > 0.5 * mass

    out = {k: np.full(n, np.nan) for k in ("mass", "x_cg", "inertia", "x_cp", "cna", "margin",
                                           "m_alpha", "m_q", "t_double", "omega", "zeta",
                                           "gimbal_per_aoa", "q")}
    for i in range(n):
        left = {k: full[k] - burned[k][i] for k in stages}
        m, xcg, inertia = mass_properties(layout, left, full,
                                          lambda p, i=i: present(p.until, p.mass, i))
        out["mass"][i], out["x_cg"][i], out["inertia"][i] = m, xcg, inertia
        h = res.altitude[keep][i]
        v = res.speed[keep][i]
        q = res.dynamic_pressure[keep][i]
        out["q"][i] = q
        if q < min_q or v <= 0:
            continue
        shoulders = layout.shoulders_until is None or t[i] < ev.get(layout.shoulders_until, np.inf)
        comps = barrowman(layout, shoulders)
        cna = sum(c for c, _ in comps)
        xcp = sum(c * x for c, x in comps) / cna
        A = math.pi / 4 * layout.diameter**2
        m_alpha = q * A * cna * (xcg - xcp)
        st = _stage_of_phase(vehicle, res.phases[ph[i]])
        m_q = -q * A * sum(c * (x - xcg) ** 2 for c, x in comps) / v
        thrust = 0.0
        if st is not None:
            m_q -= st.mass_flow * (layout.nozzle_exit[st.name] - xcg) ** 2
            thrust = st.thrust(isa(max(h, 0.0))[0] if h < 200e3 else 0.0)
        out["x_cp"][i], out["cna"][i] = xcp, cna
        out["margin"][i] = (xcp - xcg) / layout.diameter
        out["m_alpha"][i], out["m_q"][i] = m_alpha, m_q
        if m_alpha > 0:
            s = (m_q + math.sqrt(m_q * m_q + 4 * inertia * m_alpha)) / (2 * inertia)
            out["t_double"][i] = math.log(2) / s
        else:
            out["omega"][i] = math.sqrt(-m_alpha / inertia)
            out["zeta"][i] = -m_q / (2 * math.sqrt(-m_alpha * inertia))
        if st is not None and thrust > 0:
            arm = layout.gimbal[st.name] - xcg
            out["gimbal_per_aoa"][i] = q * A * cna * (xcg - xcp) / (thrust * arm)
    out["t"] = t
    out["layout_mass_error"] = float(np.max(np.abs(out["mass"] - mres) / mres))
    return out


# Rockot / Breeze-KM, x aft from the nose tip (29.15 m overall). Lengths and diameters follow
# the User's Guide (fairing 2.62 x 6.7 m, 2.5 m stages); stage split, tank positions (sized from
# N2O4 / UDMH densities at a 2.6 mixture ratio), the share of dry mass in the engines and the
# steering points are estimates. Masses are those of configs/orbital/rockot_first_principles.yaml.
ROCKOT_LAYOUT = Layout(
    diameter=2.5, nose_length=3.7, nose_diameter=2.62,
    shoulders=[(6.7, 7.85, 2.62, 2.5)], shoulders_until="fairing_jettison",
    parts=[
        Part("Fairing", 700, 1.2, 6.7, "fairing_jettison"),
        Part("Payload", 1570, 2.2, 4.0),
        Part("Adapter", 50, 4.0, 4.2),
        Part("Breeze-KM dry", 1320, 4.2, 6.85),
        Part("Stage 2 structure and transition", 0.6 * 1485, 6.7, 11.75, "Stage 2_burnout"),
        Part("Stage 2 engines", 0.4 * 1485, 10.9, 11.75, "Stage 2_burnout"),
        Part("Stage 1 structure and interstage", 0.65 * 5695, 11.75, 28.05, "Stage 1_burnout"),
        Part("Stage 1 engines", 0.35 * 5695, 26.9, 29.15, "Stage 1_burnout"),
    ],
    tanks=[
        Tank("Breeze-KM propellant", "Breeze-KM", 1.0, 4.5, 6.6),
        Tank("Stage 2 oxidiser", "Stage 2", 2.6 / 3.6, 7.95, 9.45),
        Tank("Stage 2 fuel", "Stage 2", 1.0 / 3.6, 9.6, 10.7),
        Tank("Stage 1 oxidiser", "Stage 1", 2.6 / 3.6, 12.4, 20.7),
        Tank("Stage 1 fuel", "Stage 1", 1.0 / 3.6, 20.95, 26.85),
    ],
    gimbal={"Stage 1": 27.6, "Stage 2": 11.4, "Breeze-KM": 6.5},
    nozzle_exit={"Stage 1": 29.15, "Stage 2": 12.2, "Breeze-KM": 7.0},
    notes={"control": "Stage 1 steers by gimballing its engines and stage 2 by its four vernier "
                      "engines (assumed thrust points); no fins."},
)
