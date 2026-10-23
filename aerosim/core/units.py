"""Unit conversions. Internally aerosim is pure SI; convert only at the edges
(file readers, adapters for libraries that use other units, e.g. JSBSim)."""

G0 = 9.80665            # m/s^2
R_UNIVERSAL = 8.314462618  # J/(mol K)

LBF_TO_N = 4.4482216152605
PSI_TO_PA = 6894.757293168
FT_TO_M = 0.3048
LBM_TO_KG = 0.45359237
BAR_TO_PA = 1.0e5


def rankine_to_kelvin(r: float) -> float:
    return r * 5.0 / 9.0


def kelvin_to_rankine(k: float) -> float:
    return k * 9.0 / 5.0
