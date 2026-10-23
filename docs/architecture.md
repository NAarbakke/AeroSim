# Architecture

## Data flow

```
       thermo (CEA, Cantera, CoolProp, solid propellant library)
                    │ chamber performance / gas & fluid props
                    ▼
 ┌──────────── propulsion ─────────────┐
 │ solid   ─┐                          │
 │ liquid  ─┼─► ThrustCurve            │      atmosphere (ISA, GFS, HRRR, IFS, ERA5)
 │ airbreathing ─► EngineDeck          │                │ AtmosphereModel.at(z)
 │   └ turbomachinery ─► ComponentMap  │                │
 └──────────────┬──────────────────────┘                │
                ▼                                       ▼
          flight (point_mass, planar, RocketPy, JSBSim)  ◄──────┘
                │   ▲
         state  ▼   │ commands (gimbal, air brake, throttle)
          gnc (controllers, actuators, sensors; design via python-control)
                │
                ▼
           FlightResult ──► data_io / plots / Monte Carlo
                ▲
  optimization (OpenMDAO, Dymos, pyCycle) wraps any of the above
```

## Rules

1. **Layers talk only through `core.types`.** A flight sim never sees a RocketPy
   motor or an openMotor grain; it sees a `ThrustCurve`. This is what lets you
   swap any backend with a one-line config change.
2. **Dependencies point inward.** `core` imports nothing from aerosim. `thermo`,
   `atmosphere` import only `core`. `propulsion` may use `thermo`. `flight` may use
   `atmosphere`. Only `optimization` and `missions` may import everything.
3. **Heavy libraries are optional and lazy.** Import them inside functions via
   `core.require(module, extra)`; never at module top level.
4. **SI everywhere inside.** Convert at the adapter boundary (RocketCEA is
   English units, JSBSim is feet/lbs).
5. **Every external backend has a built-in counterpart** (BATES vs openMotor,
   turbojet vs pyCycle, point-mass vs RocketPy). They cross-check each other and keep
   tests runnable without the heavy stack.

## Adding a backend

```python
from aerosim.core import ThrustCurve, registry, require

@registry.register("rocket_engine", "hybrid")
class HybridMotor:
    def __init__(self, **params): ...
    def thrust_curve(self) -> ThrustCurve: ...
```

Import it in the layer's `__init__.py`, add a config example and a test. Done.

## Who does which chemistry

| Question | Tool | Where |
|---|---|---|
| Liquid propellant density, vapour pressure, tank blowdown, coolant props, pump work | **CoolProp** (real-fluid equations of state) | `thermo/fluids.py`, `propulsion/liquid/tanks.py` |
| Rocket chamber: flame temperature, products, c*, Isp through the nozzle | **NASA CEA** (equilibrium; shifting or frozen expansion) | `thermo/nasa_cea.py` |
| Gas-turbine flow path: cp(T), γ, enthalpy/entropy of air and burned gas along compressor → burner → turbine → nozzle | **Cantera** (mixture thermo) | `thermo/gas.py` |
| How *fast* chemistry happens: ignition delay, gas generators, residence time, non-equilibrium | **Cantera** (finite-rate kinetics) | `thermo/kinetics.py` |

CEA and Cantera overlap on equilibrium; `test_cantera_and_cea_agree_on_chamber`
holds them within 0.3 %. Data flow for a liquid engine:
tank (CoolProp: how much propellant, at what state) → chamber (CEA: what it
burns to) → `ThrustCurve`. For a jet engine: ISA air → Cantera gas through the
cycle → `EngineDeck`.

Solid propellants are CEA problems too: the whole grain is one premixed reactant
(e.g. KNO3 + sorbitol, AP + Al + HTPB). Ingredients missing from CEA's database
(sugars, HTPB) are custom reactants in `thermo/solid_propellants.py`. The BATES
model tabulates c*(Pc) and an effective γ (chosen so the ideal nozzle reproduces
CEA's Cf, including condensed K2CO3 / Al2O3) and solves the chamber mass balance
at every web step.

## Control: three different jobs

| Job | Where | Tool |
|---|---|---|
| **Design** a controller on a linear model: gains, margins, LQR, step response | `gnc/design.py` | python-control |
| **Fly** it: discrete controller + actuator + sensor inside the nonlinear sim | `gnc/blocks.py`, `gnc/controllers.py`, flight sims (`controller=`) | built-in |
| **Plan** the best command history offline (throttle schedule, max-q) | `optimization/trajectory.py` | Dymos (OpenMDAO) |

Typical loop: Dymos finds the optimal throttle/pitch schedule → python-control
designs the tracking/attitude loop around it → the flight sim verifies the
closed loop with actuator limits, sensor noise, wind and gusts.

Flight sims call `controller.update(t, state)` every `controller.dt` and hold the
command (zero-order hold) through an `Actuator`. Point-mass supports `airbrake`;
planar supports `gimbal` (TVC). Gains for TVC scale with thrust (control
authority is T·l/I).

## Turbomachinery → cycle → flight

`turbomachinery.py` turns blade geometry into a `ComponentMap` (TurboFlow
meanline). `pycycle_engine.py` consumes maps in its Compressor/Turbine elements
and sweeps altitude × Mach × power into an `EngineDeck`. The flight layer
(JSBSim) only ever sees the deck.

## Roadmap

1. Finish RocketPy `Rocket` construction; validate point-mass vs RocketPy.
2. pyCycle turbojet from its `simple_turbojet` example; validate vs built-in turbojet.
3. NASA CEA done: add transport properties (cea transport=True) for regen-cooling heat flux.
4. Monte Carlo module (dispersions on wind, thrust, Cd) in `missions/`.
5. ERA5 reanalysis adapter (`cdsapi`) for post-flight reconstruction.
6. Hybrid motors on top of `SelfPressurisingTank` (N2O) + regression-rate fuel grain;
   FullFlow feed-system coupling for liquid blowdown.
7. Two-phase injector model (Dyer / HEM) for N2O, replacing SPI.
8. Kerosene/ethanol kinetics mechanisms; 1-D finite-rate nozzle with Cantera.
9. Controllers in mission YAML; throttle command for liquid engines (Dymos schedule
   tracking in closed loop); full 6-DOF TVC/fin control via RocketPy controllers.
10. Piecewise burn-rate laws (Nakka KNSB segments), erosive burning, ignition transient.
11. Dymos: gravity-turn ascent with pitch control; IPOPT via pyoptsparse when available.
