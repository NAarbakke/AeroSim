"""Legacy RocketCEA backend (GPL-3.0). Kept only for cross-checking against
NASA CEA; install with ``pip install 'aerosim[rocketcea]'``. Uses RocketCEA's
own propellant names (e.g. "LOX", "Ethanol"), not aerosim aliases."""

from __future__ import annotations

from aerosim.core import registry, require
from aerosim.core.units import PSI_TO_PA, rankine_to_kelvin


@registry.register("combustion", "rocketcea")
class RocketCEAModel:
    def __init__(self, frozen: bool = False) -> None:
        self._mod = require("rocketcea.cea_obj", "rocketcea")
        self.frozen = frozen
        self._cache: dict[tuple[str, str], object] = {}

    def _obj(self, ox: str, fuel: str):
        key = (ox, fuel)
        if key not in self._cache:
            self._cache[key] = self._mod.CEA_Obj(oxName=ox, fuelName=fuel)
        return self._cache[key]

    def chamber(self, oxidizer, fuel, mixture_ratio, chamber_pressure, expansion_ratio):
        c = self._obj(oxidizer, fuel)
        pc_psia = chamber_pressure / PSI_TO_PA
        frozen = 1 if self.frozen else 0
        isp_vac = c.get_Isp(Pc=pc_psia, MR=mixture_ratio, eps=expansion_ratio, frozen=frozen)
        isp_sl, _mode = c.estimate_Ambient_Isp(
            Pc=pc_psia, MR=mixture_ratio, eps=expansion_ratio, Pamb=14.696, frozen=frozen
        )
        mw, gamma = c.get_Chamber_MolWt_gamma(Pc=pc_psia, MR=mixture_ratio, eps=expansion_ratio)
        return {
            "isp_vac": float(isp_vac),
            "isp_sl": float(isp_sl),
            "cstar": float(c.get_Cstar(Pc=pc_psia, MR=mixture_ratio)) * 0.3048,  # ft/s -> m/s
            "t_chamber": rankine_to_kelvin(float(c.get_Tcomb(Pc=pc_psia, MR=mixture_ratio))),
            "gamma": float(gamma),
            "mw": float(mw),
        }
