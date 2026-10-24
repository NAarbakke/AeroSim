"""Regenerative cooling of a thrust chamber via pyskyfire (MIT, github.com/ask-hovik/pyskyfire).

    regen_cooling(oxidizer="LOX", fuel="LCH4", mixture_ratio=3.4, chamber_pressure=10e6,
                  thrust=100e3, expansion_ratio=30)  ->  RegenResult

pyskyfire builds the chamber (CEA hot gas, Rao contour sized from thrust and L*), lays
coolant channels over it and solves the coupled hot-gas / wall / coolant problem along the
circuit (Bartz-type hot side, real-fluid coolant from CoolProp). This wrapper takes AeroSim
propellant aliases and returns the numbers a cycle analysis needs: coolant outlet
temperature and pressure drop, heat load, and peak wall temperature and heat flux.

Propellants: the hot gas uses the CEA species of ``aerosim.thermo.combustion.propellants``
at their storage temperatures; the coolant (the fuel by default) uses its CoolProp fluid
(RP-1 -> n-dodecane surrogate).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from aerosim.core import AerosimError, require
from aerosim.thermo.combustion import propellants
from aerosim.thermo.fluids import coolprop_name

MATERIALS = {"copper": "copper", "grcop42": "GRCop42", "grcop84": "GRCop84_AM",
             "ss304": "StainlessSteel304", "ss347": "StainlessSteel347",
             "inconel718": "Inconel718", "inconel625": "Inconel625"}


@dataclass
class RegenResult:
    x: np.ndarray                   # m, axial station (throat at 0)
    radius: np.ndarray              # m, chamber contour radius at x
    t_hot_wall: np.ndarray          # K
    t_cold_wall: np.ndarray         # K, coolant-side wall
    t_coolant: np.ndarray           # K, bulk coolant static
    p_coolant: np.ndarray           # Pa, coolant static
    heat_flux: np.ndarray           # W/m^2, hot side
    coolant_velocity: np.ndarray    # m/s
    mdot_coolant: float             # kg/s
    coolant: str                    # CoolProp fluid
    chamber: dict = field(default_factory=dict)   # pyskyfire's sized chamber (r_t, A_t, Isp...)

    @property
    def coolant_outlet_temperature(self) -> float:
        return float(self.t_coolant[-1])

    @property
    def pressure_drop(self) -> float:
        return float(self.p_coolant[0] - self.p_coolant[-1])

    @property
    def max_wall_temperature(self) -> float:
        return float(self.t_hot_wall.max())

    @property
    def max_heat_flux(self) -> float:
        return float(self.heat_flux.max())

    @property
    def heat_load(self) -> float:
        """W, heat picked up by the coolant (energy balance on its enthalpy rise)."""
        from aerosim.thermo.fluids import props
        h = [props("H", "T", self.t_coolant[i], "P", self.p_coolant[i], self.coolant) for i in (0, -1)]
        return float(self.mdot_coolant * (h[1] - h[0]))

    def summary(self) -> dict[str, float]:
        return {"coolant_outlet_temperature": self.coolant_outlet_temperature,
                "pressure_drop": self.pressure_drop, "heat_load": self.heat_load,
                "max_wall_temperature": self.max_wall_temperature,
                "max_heat_flux": self.max_heat_flux, "mdot_coolant": self.mdot_coolant}


def _psf_fluid(psf, spec, kind: str):
    comps = propellants.resolve(spec)
    return (psf.thermo.Fluid(type=kind, propellants=[c.species for c in comps],
                             fractions=[c.mass_fraction for c in comps]),
            comps[0].temperature)


def regen_cooling(oxidizer, fuel, mixture_ratio: float, chamber_pressure: float, thrust: float,
                  expansion_ratio: float, ambient_pressure: float = 101_325.0, l_star: float = 1.0, chamber_aspect_ratio: float = 1.5,
                  coolant: str = "fuel", coolant_fraction: float = 1.0,
                  mdot_coolant: float | None = None,
                  coolant_inlet_temperature: float | None = None,
                  coolant_inlet_pressure: float | None = None,
                  n_channels: int | None = None, channel_height: float = 2.5e-3,
                  rib_fraction: float = 0.3, wall_thickness: float = 0.8e-3,
                  material: str = "grcop42", roughness: float = 6e-6, helix_angle: float = 0.0,
                  cooled_span: tuple[float, float] = (1.0, -1.0), nodes: int = 100) -> RegenResult:
    """Steady regen-cooling solution for a chamber sized by pyskyfire from thrust and L*.

    thrust is the thrust at ``ambient_pressure`` that pyskyfire sizes the throat for (pass
    ambient_pressure=0 for a vacuum rating);
    coolant flows from the nozzle end of ``cooled_span`` (1 = exit) to the injector (-1).
    coolant_fraction: share of the coolant propellant's flow sent through the channels;
    mdot_coolant (kg/s) overrides it with an absolute coolant flow.
    chamber_aspect_ratio: chamber length / diameter handed to pyskyfire's contour generator.
    n_channels: default = one per ~4 mm of throat circumference (at least 20).
    """
    psf = require("pyskyfire", "liquid")
    fu, t_fu = _psf_fluid(psf, fuel, "fuel")
    ox, t_ox = _psf_fluid(psf, oxidizer, "oxidizer")
    try:
        at = psf.thermo.Aerothermodynamics.from_F_eps_Lstar(
            fu=fu, ox=ox, MR=float(mixture_ratio), p_c=float(chamber_pressure), F=float(thrust),
            eps=float(expansion_ratio), L_star=float(l_star), T_fu_in=t_fu, T_ox_in=t_ox,
            p_amb=float(ambient_pressure))
    except Exception as e:  # CEA / pyskyfire raise plain exceptions on bad inputs
        raise AerosimError(f"pyskyfire could not size the chamber: {e}") from e
    cool_spec = fuel if coolant == "fuel" else oxidizer
    cool_name = coolprop_name(cool_spec if isinstance(cool_spec, str) else next(iter(cool_spec)))
    cool = psf.thermo.FluidTransport(psf.thermo.Fluid(type=coolant, propellants=[cool_name],
                                                      fractions=[1.0]))
    mdot = mdot_coolant or (at.mdot_fu if coolant == "fuel" else at.mdot_ox) * coolant_fraction
    t_in = coolant_inlet_temperature or (t_fu if coolant == "fuel" else t_ox)
    p_in = coolant_inlet_pressure or 1.5 * chamber_pressure

    # chamber volume from L*; pyskyfire picks the chamber radius for the aspect ratio. Not
    # every ratio has a contour for a given volume and throat, so fall back to nearby ones.
    for ar in dict.fromkeys([float(chamber_aspect_ratio), 1.5, 1.2, 1.0, 0.8, 2.0, 0.6, 2.5, 3.0]):
        try:
            xs, rs = psf.regen.contour.generate_contour(
                V_c=at.V_c, AR_c=ar, r_t=at.r_t, area_ratio=float(expansion_ratio),
                nozzle="rao", R_1f=1.5, R_2f=2.0, R_3f=0.382)
            break
        except ValueError:
            continue
    else:
        raise AerosimError(f"no chamber contour for L*={l_star} at any aspect ratio 0.6-3")
    contour = psf.regen.Contour(xs, rs, name="chamber")
    n = n_channels or max(20, int(2 * math.pi * at.r_t / 4e-3))
    mat = getattr(psf.common.solids, MATERIALS.get(material.lower(), material))
    circuit = psf.regen.CoolingCircuit(
        name="regen", contour=contour, coolant_transport=cool,
        cross_section=psf.regen.CrossSectionSquared(
            sidewall_half_thickness=lambda pitch_mid: rib_fraction * pitch_mid / 2),
        span=list(cooled_span),
        placement=psf.regen.SurfacePlacement(n_channel_positions=n,
                                             helix_angle=lambda: math.radians(helix_angle)),
        walls=[psf.regen.Wall(material=mat, thickness=wall_thickness)],
        roughness=roughness, channel_height=lambda: channel_height)
    tc = psf.regen.ThrustChamber(contour=contour, combustion_transport=at,
                                 cooling_circuits=[circuit], n_nodes=150)
    bc = psf.regen.BoundaryConditions(T_coolant_in=t_in, p_coolant_in=p_in, mdot_coolant=mdot)
    try:
        d = psf.regen.coupled_steady_heating_analysis(tc, nodes=nodes, circuit_index=0,
                                                      boundary_conditions=bc, output=False)
    except Exception as e:
        raise AerosimError(f"pyskyfire regen solve failed: {e}") from e
    x = np.asarray(d.x, float)
    T = np.asarray(d.T, float)              # columns: coolant, coolant-side wall, ..., hot wall
    opt = {k: v for k, v in at.optimum.items() if isinstance(v, (int, float))}
    return RegenResult(x=x, radius=np.interp(x, xs, rs), t_hot_wall=T[:, -1], t_cold_wall=T[:, 1],
                       t_coolant=np.asarray(d.T_static, float), p_coolant=np.asarray(d.p_static, float),
                       heat_flux=np.asarray(d.dQ_dA, float), coolant_velocity=np.asarray(d.velocity, float),
                       mdot_coolant=float(mdot), coolant=cool_name,
                       chamber=opt | {"n_channels": n, "aspect_ratio": ar})
