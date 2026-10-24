"""Feed-network backend (interface only).

Planned: ``fullflow``, a network solver for tanks, lines, pumps and injectors
(github.com/saakethramoju/FullFlow). Engine cycles (turbopump power balance) are built in:
``propulsion/liquid/cycle.py``; regenerative cooling is ``adapters/regen.py`` (pyskyfire).

Contract: a FeedSystem resolves the operating point the chamber actually sees,
i.e. returns {"chamber_pressure", "mixture_ratio", "mdot_ox", "mdot_fuel"} as a
function of time (tank blowdown) or throttle command. LiquidEngine then consumes it.
"""

from __future__ import annotations

from typing import Protocol


class FeedSystem(Protocol):
    def operating_point(self, t: float, throttle: float = 1.0) -> dict[str, float]: ...


class FullFlowFeed:
    def __init__(self, network_file: str) -> None:
        self.network_file = network_file

    def operating_point(self, t: float, throttle: float = 1.0) -> dict[str, float]:
        raise NotImplementedError("TODO: build FullFlow Network and solve at time t")
