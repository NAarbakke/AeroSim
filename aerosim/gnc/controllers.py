"""Ready-made flight controllers built from gnc.blocks."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from aerosim.gnc.blocks import PID

G0 = 9.80665


@dataclass
class AirbrakeApogeeController:
    """Deploy drag brakes after burnout to hit a target apogee.

    Each step predicts the coast apogee in closed form for the current drag,
        h_ap = h + m / (2k) * ln(1 + k v_z^2 / (m g)),   k = 0.5 * rho * Cd*A_total,
    and runs a PI loop on (predicted - target). The command is the brake fraction 0..1.
    State keys used: altitude, vz, mass, density, burnout, cd_area (current Cd*A, m^2),
    brake_cd_area (Cd*A added at full deployment, m^2).
    """

    target_apogee: float
    kp: float = 2.0e-3            # brake fraction per metre of predicted overshoot
    ki: float = 5.0e-4
    dt: float = 0.02
    pid: PID = field(init=False)

    def __post_init__(self) -> None:
        self.pid = PID(self.kp, self.ki, 0.0, self.dt, out_min=0.0, out_max=1.0)

    def reset(self) -> None:
        self.pid.reset()

    @staticmethod
    def predict_apogee(h: float, vz: float, mass: float, rho: float, cd_area: float) -> float:
        if vz <= 0:
            return h
        k = 0.5 * rho * cd_area
        if k <= 0:
            return h + vz**2 / (2 * G0)
        return h + mass / (2 * k) * math.log1p(k * vz**2 / (mass * G0))

    def update(self, t: float, s: dict) -> dict:
        if not s["burnout"] or s["vz"] <= 0:
            return {"airbrake": 0.0}
        # predict with brakes retracted: error > 0 means we would overshoot
        pred = self.predict_apogee(s["altitude"], s["vz"], s["mass"], s["density"], s["cd_area"])
        return {"airbrake": self.pid(pred - self.target_apogee), "predicted_apogee": pred}


@dataclass
class AttitudeHoldTVC:
    """Thrust-vector-control attitude hold for the planar rocket.

    PID on pitch-angle error with the measured pitch rate as the derivative,
    gains scheduled by thrust so loop bandwidth stays constant as thrust varies:
    control authority is T*l_t/I, so gains scale with T_design/T.
    State keys: theta, q, thrust. Command: gimbal (rad).
    """

    kp: float
    kd: float
    ki: float = 0.0
    theta_ref: float = math.pi / 2         # rad from horizontal (vertical)
    gimbal_limit: float = math.radians(5)
    design_thrust: float | None = None     # N; None = no scheduling
    dt: float = 0.005
    pid: PID = field(init=False)

    def __post_init__(self) -> None:
        self.pid = PID(self.kp, self.ki, self.kd, self.dt,
                       out_min=-self.gimbal_limit, out_max=self.gimbal_limit)

    def reset(self) -> None:
        self.pid.reset()

    def update(self, t: float, s: dict) -> dict:
        thrust = s.get("thrust", 0.0)
        if thrust <= 1.0:
            return {"gimbal": 0.0}
        scale = 1.0 if not self.design_thrust else min(self.design_thrust / thrust, 5.0)
        err = self.theta_ref - s["theta"]
        return {"gimbal": self.pid(err, -s["q"], scale=scale)}
