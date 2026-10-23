"""Isentropic nozzle relations shared by solid and liquid models."""

from __future__ import annotations

import math

from scipy.optimize import brentq


def area_ratio(mach: float, gamma: float) -> float:
    g = gamma
    return (1 / mach) * ((2 / (g + 1)) * (1 + (g - 1) / 2 * mach**2)) ** ((g + 1) / (2 * (g - 1)))


def exit_mach(expansion_ratio: float, gamma: float) -> float:
    """Supersonic exit Mach for a given Ae/At."""
    if expansion_ratio <= 1.0:
        return 1.0
    return brentq(lambda m: area_ratio(m, gamma) - expansion_ratio, 1.0 + 1e-9, 50.0)


def pressure_ratio(mach: float, gamma: float) -> float:
    """p / p0 for isentropic flow."""
    return (1 + (gamma - 1) / 2 * mach**2) ** (-gamma / (gamma - 1))


def thrust_coefficient(gamma: float, expansion_ratio: float, pc: float, pa: float) -> float:
    """Ideal Cf including pressure thrust (no separation model)."""
    g = gamma
    pe_pc = pressure_ratio(exit_mach(expansion_ratio, g), g)
    momentum = math.sqrt(
        2 * g**2 / (g - 1) * (2 / (g + 1)) ** ((g + 1) / (g - 1)) * (1 - pe_pc ** ((g - 1) / g))
    )
    return momentum + (pe_pc - pa / pc) * expansion_ratio
