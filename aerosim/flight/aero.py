"""Zero-lift drag of a slender launch vehicle from its geometry (component build-up).

    Cd(M) = skin friction + nose wave drag + base drag,   referenced to the body area

- skin friction: turbulent flat plate, Cf = 0.455 / (log10 Re)^2.58, compressibility factor
  (1 + 0.144 M^2)^-0.65, body form factor 1 + 60/(l/d)^3 + 0.0025 l/d below M 1
  (Raymer, *Aircraft Design*, 12.5); wetted area of a cone-cylinder
- nose wave drag (M > 1): (1.59 + 1.83/M^2) * atan(0.5 / (l_n/d))^1.69 (Fleeman, *Tactical
  Missile Design*, 2.4), ramped smoothly from zero at M 0.8 to its M 1.05 value
- base drag: 0.12 + 0.13 M^2 below M 1, 0.25/M above (Fleeman), times the fraction of the
  base not filled by the nozzles and plume (``base_exposed``)

Reynolds number enters only logarithmically; one representative value is used for the
whole ascent (default 1e8, a 30 m vehicle near max q).
"""

from __future__ import annotations

import math

import numpy as np

MACHS = (0.0, 0.3, 0.6, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.4, 1.6, 2.0, 2.5, 3.0, 4.0,
         5.0, 7.0, 10.0)


def _wave(m: float, fineness: float) -> float:
    return (1.59 + 1.83 / m**2) * math.atan(0.5 / fineness) ** 1.69


def cd0(mach: float, length: float, diameter: float, nose_length: float,
        base_exposed: float = 0.5, reynolds: float = 1e8) -> dict[str, float]:
    """Zero-lift drag components at one Mach number."""
    m = max(float(mach), 0.0)
    ld, fn = length / diameter, nose_length / diameter
    s_ref = math.pi / 4 * diameter**2
    r = diameter / 2
    s_wet = math.pi * diameter * (length - nose_length) + math.pi * r * math.hypot(r, nose_length)
    cf = 0.455 / math.log10(reynolds) ** 2.58 * (1 + 0.144 * m * m) ** -0.65
    ff = 1 + 60 / ld**3 + 0.0025 * ld if m < 1 else 1.0
    friction = cf * ff * s_wet / s_ref
    if m <= 0.8:
        wave = 0.0
    elif m < 1.05:
        x = (m - 0.8) / 0.25
        wave = _wave(1.05, fn) * x * x * (3 - 2 * x)          # smoothstep through transonic
    else:
        wave = _wave(m, fn)
    base = (0.12 + 0.13 * m * m if m < 1 else 0.25 / m) * base_exposed
    return {"friction": friction, "wave": wave, "base": base, "total": friction + wave + base}


def drag_curve(length: float, diameter: float, nose_length: float, base_exposed: float = 0.5,
               reynolds: float = 1e8, machs=MACHS) -> list[tuple[float, float]]:
    """[(mach, cd), ...] for ``LaunchVehicle.drag_curve``."""
    return [(float(m), round(cd0(m, length, diameter, nose_length, base_exposed, reynolds)["total"], 4))
            for m in machs]


def components(length, diameter, nose_length, base_exposed=0.5, reynolds=1e8, machs=MACHS):
    """Arrays of each component over ``machs`` (for plotting the build-up)."""
    rows = [cd0(m, length, diameter, nose_length, base_exposed, reynolds) for m in machs]
    return {"mach": np.asarray(machs, float),
            **{k: np.array([r[k] for r in rows]) for k in ("friction", "wave", "base", "total")}}
