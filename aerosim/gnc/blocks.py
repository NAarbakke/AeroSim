"""Discrete-time GNC building blocks: controllers, actuators, sensors.

All blocks are stepped by the flight simulator at the controller rate with a
zero-order hold in between, like flight software on a real vehicle.

Controller contract::

    controller.update(t, state) -> commands
        state:    dict of measured quantities (see flight sims for keys)
        commands: dict, e.g. {"gimbal": rad} or {"airbrake": 0..1}
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Controller(Protocol):
    dt: float  # s, update period

    def reset(self) -> None: ...

    def update(self, t: float, state: dict) -> dict: ...


@dataclass
class PID:
    """PID with output limits, conditional-integration anti-windup and a
    first-order filtered derivative. Derivative may be supplied directly
    (e.g. measured pitch rate) to avoid differentiating noise."""

    kp: float
    ki: float = 0.0
    kd: float = 0.0
    dt: float = 0.01
    out_min: float = -math.inf
    out_max: float = math.inf
    d_filter_tau: float = 0.0          # s, 0 = unfiltered
    _i: float = field(default=0.0, init=False, repr=False)
    _e_prev: float | None = field(default=None, init=False, repr=False)
    _d: float = field(default=0.0, init=False, repr=False)

    def reset(self) -> None:
        self._i, self._e_prev, self._d = 0.0, None, 0.0

    def __call__(self, error: float, error_rate: float | None = None, scale: float = 1.0) -> float:
        """One step. ``scale`` multiplies all gains (for gain scheduling)."""
        if error_rate is None:
            raw = 0.0 if self._e_prev is None else (error - self._e_prev) / self.dt
            a = self.dt / (self.d_filter_tau + self.dt)
            self._d += a * (raw - self._d)
            error_rate = self._d
        self._e_prev = error
        u_unsat = scale * (self.kp * error + self.ki * self._i + self.kd * error_rate)
        u = min(max(u_unsat, self.out_min), self.out_max)
        # integrate only if not pushing further into saturation
        if u == u_unsat or (u_unsat > u) != (error > 0):
            self._i += error * self.dt
        return u


@dataclass
class Actuator:
    """First-order lag with rate and position limits (servo, gimbal, air brake)."""

    tau: float = 0.02                  # s, time constant
    rate_limit: float = math.inf       # units/s
    min: float = -math.inf
    max: float = math.inf
    position: float = 0.0

    def reset(self, position: float = 0.0) -> None:
        self.position = position

    def step(self, command: float, dt: float) -> float:
        cmd = min(max(command, self.min), self.max)
        rate = (cmd - self.position) / self.tau if self.tau > 0 else (cmd - self.position) / dt
        rate = min(max(rate, -self.rate_limit), self.rate_limit)
        self.position = min(max(self.position + rate * dt, self.min), self.max)
        return self.position


@dataclass
class Sensor:
    """Adds bias and white noise to selected state keys (seeded for repeatability)."""

    noise_std: dict[str, float] = field(default_factory=dict)
    bias: dict[str, float] = field(default_factory=dict)
    seed: int = 0

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def measure(self, state: dict) -> dict:
        out = dict(state)
        for k in set(self.noise_std) | set(self.bias):
            if k in out:
                out[k] = out[k] + self.bias.get(k, 0.0) + self._rng.normal(0.0, self.noise_std.get(k, 0.0))
        return out
