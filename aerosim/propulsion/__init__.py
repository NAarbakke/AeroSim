"""Propulsion layer: solid, hybrid, liquid and airbreathing engines -> ThrustCurve | EngineDeck.

Built-in models live in solid/, hybrid/, liquid/, airbreathing/. Wrappers around external
libraries that are not finished yet live in adapters/.
"""

from aerosim.propulsion import adapters, airbreathing, hybrid, liquid, solid  # noqa: F401  (registers backends)
