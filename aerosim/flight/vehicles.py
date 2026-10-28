"""Vehicle descriptions, independent of which simulator flies them."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RocketVehicle:
    dry_mass: float                 # kg, airframe without motor
    diameter: float                 # m, reference
    drag_coefficient: float = 0.45  # constant Cd, or replace with drag_curve
    drag_curve: list[tuple[float, float]] | None = None   # [(mach, cd), ...]
    rail_length: float = 5.0        # m
    elevation_deg: float = 85.0     # launch rail elevation
    heading_deg: float = 0.0        # launch azimuth, clockwise from north
    # actuated drag (air brakes): Cd*A added at full deployment, m^2
    airbrake_cd_area: float = 0.0
    airbrake_tau: float = 0.1       # s, actuator time constant
    airbrake_rate: float = 2.0      # fraction/s, deploy rate limit
    # rigid-body pitch-plane data (planar simulator, TVC)
    pitch_inertia: float | None = None   # kg m^2 at full propellant (scaled with mass)
    cg_to_nozzle: float = 0.5       # m, TVC lever arm
    static_margin: float = 0.1      # m, CP aft of CG (>0 = aerodynamically stable)
    cn_alpha: float = 2.0           # 1/rad, normal-force slope
    pitch_damping: float = 2.0      # -Cmq, dimensionless pitch damping
    gimbal_limit_deg: float = 5.0
    gimbal_tau: float = 0.02        # s
    gimbal_rate_deg: float = 150.0  # deg/s
    # higher-fidelity fields used by the RocketPy backend (positions from the nose tip, m)
    length: float | None = None
    cg_position: float | None = None      # airframe CG without motor
    nose: dict = field(default_factory=dict)
    fins: dict = field(default_factory=dict)
    parachutes: list[dict] = field(default_factory=list)

    @property
    def reference_area(self) -> float:
        return 3.141592653589793 / 4 * self.diameter**2

    def cd(self, mach: float) -> float:
        if not self.drag_curve:
            return self.drag_coefficient
        import numpy as np
        m, c = zip(*self.drag_curve)
        return float(np.interp(mach, m, c))


@dataclass
class AircraftVehicle:
    jsbsim_model: str               # aircraft folder name in the JSBSim root, e.g. "c172x"
    engine_count: int = 1
    initial: dict = field(default_factory=dict)   # altitude, speed, heading, lat, lon


@dataclass
class Stage:
    """One propulsive stage. Thrust at altitude: F = F_vac - p_amb * A_exit, with
    A_exit implied by the sea-level Isp (``isp_sl``; None = vacuum-only engine).
    Mass flow is constant at propellant_mass / burn_time (full-thrust equivalent).
    ``coast``: s of unpowered flight after cutoff before the stage separates (lower stages)."""

    name: str
    propellant_mass: float          # kg, usable
    dry_mass: float                 # kg, dropped at separation
    isp_vac: float                  # s
    burn_time: float                # s, at full thrust
    isp_sl: float | None = None     # s
    max_burns: int = 1              # restartable upper stages > 1
    coast: float = 0.0              # s, cutoff to separation

    @property
    def mass_flow(self) -> float:
        return self.propellant_mass / self.burn_time

    @property
    def thrust_vac(self) -> float:
        return self.mass_flow * self.isp_vac * 9.80665

    @property
    def exit_area(self) -> float:
        if self.isp_sl is None:
            return 0.0
        return self.mass_flow * (self.isp_vac - self.isp_sl) * 9.80665 / 101_325.0

    def thrust(self, ambient_pressure: float) -> float:
        return max(self.thrust_vac - ambient_pressure * self.exit_area, 0.0)

    @classmethod
    def from_engines(cls, name: str, engines: list[dict], dry_mass: float,
                     burn_time: float | None = None, propellant_mass: float | None = None,
                     max_burns: int = 1, coast: float = 0.0, **_) -> Stage:
        """Stage from engine design points (``aerosim.propulsion.liquid.design.engine_point``
        dicts, or the keyword dicts that produce them). Isp is the mass-flow-weighted mix;
        give ``burn_time`` to size the propellant (mdot x t) or ``propellant_mass`` to get the
        burn time. Each engine dict is kept in ``engines`` for reporting."""
        from aerosim.propulsion.liquid.design import engine_point
        pts = [e if "mdot" in e else engine_point(**e) for e in engines]
        F = sum(e["thrust_vac"] for e in pts)
        mdot = sum(e["mdot"] for e in pts)
        ae = sum(e["exit_area"] for e in pts)
        if propellant_mass is None:
            if burn_time is None:
                raise ValueError(f"{name}: give burn_time or propellant_mass")
            propellant_mass = mdot * burn_time
        else:
            burn_time = propellant_mass / mdot
        isp = F / (mdot * 9.80665)
        st = cls(name, propellant_mass, dry_mass, isp, burn_time,
                 isp - 101_325.0 * ae / (mdot * 9.80665) if ae > 0 else None, max_burns, coast)
        st.engines = pts
        return st


@dataclass
class LaunchVehicle:
    """Multi-stage orbital launcher, stages bottom to top (the last one carries the payload)."""

    stages: list[Stage]
    payload_mass: float             # kg
    fairing_mass: float = 0.0       # kg, jettisoned during flight
    adapter_mass: float = 0.0       # kg, stays with the last stage
    diameter: float = 2.5           # m, reference for drag
    # [(mach, cd), ...], or geometry {length, nose_length, base_exposed?, reynolds?} for the
    # component build-up in aerosim.flight.aero
    drag_curve: list[tuple[float, float]] | dict = field(default_factory=lambda: [
        (0.0, 0.30), (0.8, 0.33), (1.0, 0.48), (1.2, 0.52), (1.5, 0.45),
        (2.0, 0.38), (3.0, 0.30), (5.0, 0.25), (10.0, 0.22)])   # generic slender LV
    name: str = "launch vehicle"

    def __post_init__(self) -> None:
        self.stages = [s if isinstance(s, Stage) else
                       Stage.from_engines(**s) if "engines" in s else Stage(**s)
                       for s in self.stages]
        if isinstance(self.drag_curve, dict):
            from aerosim.flight.aero import drag_curve
            self.drag_curve = drag_curve(diameter=self.diameter, **self.drag_curve)

    @property
    def liftoff_mass(self) -> float:
        return (sum(s.propellant_mass + s.dry_mass for s in self.stages) + self.payload_mass
                + self.fairing_mass + self.adapter_mass)

    @property
    def reference_area(self) -> float:
        return 3.141592653589793 / 4 * self.diameter**2

    def cd(self, mach: float) -> float:
        import numpy as np
        m, c = zip(*self.drag_curve)
        return float(np.interp(mach, m, c))
