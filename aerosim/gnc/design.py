"""Control design and analysis with python-control.

Linear models of the vehicle around an operating point, controller synthesis
(PD/PID by pole placement, LQR), and loop analysis (margins, closed-loop poles,
step response). The resulting gains plug straight into gnc.controllers and are
then verified in the nonlinear flight simulators.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from aerosim.core import require


def _ct():
    return require("control", "control")


@dataclass(frozen=True)
class TVCOperatingPoint:
    """Pitch-plane operating point of a thrust-vectored rocket."""

    thrust: float               # N
    lever_arm: float            # m, CG to gimbal pivot
    inertia: float              # kg m^2, pitch
    dynamic_pressure: float = 0.0   # Pa
    ref_area: float = 0.0       # m^2
    cn_alpha: float = 0.0       # 1/rad
    static_margin: float = 0.0  # m, CP aft of CG (>0 aerodynamically stable)
    actuator_tau: float = 0.02  # s

    @property
    def k_delta(self) -> float:
        """Pitch acceleration per radian of gimbal, T*l_t/I."""
        return self.thrust * self.lever_arm / self.inertia

    @property
    def m_alpha(self) -> float:
        """Aerodynamic pitch stiffness (1/s^2): <0 stable, >0 unstable."""
        return -self.dynamic_pressure * self.ref_area * self.cn_alpha * self.static_margin / self.inertia


def tvc_plant(op: TVCOperatingPoint, include_actuator: bool = True):
    """State space of theta'' = M_alpha*theta + K_delta*delta, delta' = (delta_cmd - delta)/tau.

    States [theta, q(, delta)], input delta_cmd, outputs [theta, q].
    Flight-path angle treated as slow (short-period approximation, alpha ~ theta).
    """
    ct = _ct()
    if include_actuator:
        A = [[0, 1, 0], [op.m_alpha, 0, op.k_delta], [0, 0, -1 / op.actuator_tau]]
        B = [[0], [0], [1 / op.actuator_tau]]
        C = [[1, 0, 0], [0, 1, 0]]
    else:
        A, B, C = [[0, 1], [op.m_alpha, 0]], [[0], [op.k_delta]], [[1, 0], [0, 1]]
    return ct.ss(A, B, C, 0, inputs="delta_cmd", outputs=["theta", "q"])


def pd_by_pole_placement(op: TVCOperatingPoint, wn: float, zeta: float = 0.7,
                         ki_ratio: float = 0.0) -> dict[str, float]:
    """PD(+I) gains putting the rigid-body poles at wn, zeta (actuator neglected).

    theta'' = K_delta*(kp*e - kd*q) + M_alpha*theta
      => kp = (wn^2 + M_alpha)/K_delta, kd = 2*zeta*wn/K_delta.
    ``ki_ratio`` sets ki = ki_ratio * kp * wn / 10 (slow integrator for trim).
    """
    kp = (wn**2 + op.m_alpha) / op.k_delta
    kd = 2 * zeta * wn / op.k_delta
    return {"kp": kp, "kd": kd, "ki": ki_ratio * kp * wn / 10}


def lqr_attitude(op: TVCOperatingPoint, q_theta: float = 1.0, q_rate: float = 0.1,
                 r_delta: float = 1.0) -> dict:
    """LQR on [theta, q] (actuator neglected). Returns gains as kp/kd and the raw K."""
    ct = _ct()
    sys = tvc_plant(op, include_actuator=False)
    K, _S, E = ct.lqr(sys.A, sys.B, np.diag([q_theta, q_rate]), [[r_delta]])
    K = np.asarray(K).ravel()
    return {"kp": float(K[0]), "kd": float(K[1]), "K": K, "closed_loop_poles": E}


def analyze_attitude_loop(op: TVCOperatingPoint, kp: float, kd: float, ki: float = 0.0) -> dict:
    """Loop analysis with the actuator lag included.

    Controller u = kp*e + ki*int(e) - kd*q with e = theta_ref - theta. The loop is
    broken at the actuator input for margins.
    """
    ct = _ct()
    s = ct.tf("s")
    tau = op.actuator_tau
    # delta_cmd -> theta: K_delta / ((s^2 - M_alpha)(tau s + 1)), built directly (well conditioned)
    P_theta = op.k_delta / ((s**2 - op.m_alpha) * (tau * s + 1))
    # with q = s*theta, the loop transfer is L = (kp + ki/s + kd*s) * P_theta
    C = kp + kd * s + (ki / s if ki else 0)
    L = ct.minreal(C * P_theta, verbose=False)
    gms, pms, _sms, wpcs, wgcs, _ = ct.stability_margins(L, returnall=True)
    gms, wpcs = np.atleast_1d(gms), np.atleast_1d(wpcs)
    pms, wgcs = np.atleast_1d(pms), np.atleast_1d(wgcs)
    finite = [(g, w) for g, w in zip(gms, wpcs) if np.isfinite(g) and 0 < g < 1e8]
    lower = [g for g, _ in finite if g < 1]     # gain may drop by this factor (unstable plants)
    upper = [g for g, _ in finite if g > 1]     # gain may rise by this factor
    T = ct.feedback(L, 1)
    poles = ct.poles(T)
    info = ct.step_info(T) if np.all(np.real(poles) < 0) else {}
    i_pm = int(np.argmin(pms)) if len(pms) else None
    return {
        "gain_margin_db": 20 * math.log10(min(upper)) if upper else math.inf,
        "lower_gain_margin_db": 20 * math.log10(max(lower)) if lower else None,
        "phase_margin_deg": float(pms[i_pm]) if i_pm is not None else math.inf,
        "crossover_rad_s": float(wgcs[i_pm]) if i_pm is not None else float("nan"),
        "closed_loop_poles": poles,
        "stable": bool(np.all(np.real(poles) < 0)),
        "rise_time": info.get("RiseTime"),
        "overshoot_pct": info.get("Overshoot"),
        "settling_time": info.get("SettlingTime"),
        "loop": L, "closed_loop": T,
    }


def discretize(sys, dt: float, method: str = "zoh"):
    """Discrete-time equivalent for flight-software implementation."""
    return _ct().sample_system(sys, dt, method=method)
