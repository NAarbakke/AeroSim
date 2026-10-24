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
          flight (point_mass, planar, orbital_ascent, RocketPy, JSBSim)  ◄──────┘
                │   ▲
         state  ▼   │ commands (gimbal, air brake, throttle)
          gnc (controllers, actuators, sensors; design via python-control)
                │
                ▼
           FlightResult ──► core.eng / aerosim.plots / Monte Carlo
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
4. **SI everywhere inside.** Convert at the adapter boundary (JSBSim is feet/lbs).
5. **Every external backend has a built-in counterpart** (BATES vs openMotor,
   turbojet vs pyCycle, point-mass vs RocketPy). They cross-check each other and keep
   tests runnable without the heavy stack.
6. **Unfinished wrappers live in `adapters/`.** `propulsion/adapters/` and
   `flight/adapters/` hold wrappers around external libraries that are interface-only
   or partly built. Once one is finished and tested, move it next to its built-in
   counterpart (e.g. `flight/rocket/`). Registry names do not change, so configs keep working.

## Package layout

```
aerosim/
  core/          types, interfaces, registry, units, errors, eng (RASP files)
  thermo/        combustion/ (NASA CEA)  gas/ (Cantera)  fluids.py (CoolProp)
  propulsion/    nozzle.py  solid/  liquid/  airbreathing/  adapters/
  atmosphere/    standard, profile, weather
  flight/        vehicles.py  rocket/  adapters/
  gnc/           blocks, controllers, design
  optimization/  mdao, trajectory
  missions/      runner (YAML -> backends -> FlightResult)
tests/           same folder names as aerosim/
configs/         solid, liquid, orbital, aircraft (example YAML missions)
```

## Adding a backend

```python
from aerosim.core import ThrustCurve, registry, require

@registry.register("rocket_engine", "hybrid")
class HybridMotor:
    def __init__(self, **params): ...
    def thrust_curve(self) -> ThrustCurve: ...
```

Import it in the layer's `__init__.py` (or `adapters/__init__.py` if it wraps an
unfinished external library), add an example config under `configs/` and a
test in the matching `tests/<layer>/` folder. Done.

## Who does which chemistry

| Question | Tool | Where |
|---|---|---|
| Liquid propellant density, vapour pressure, tank blowdown, coolant props, pump work | **CoolProp** (real-fluid equations of state) | `thermo/fluids.py`, `propulsion/liquid/tanks.py` |
| Rocket chamber: flame temperature, products, c*, Isp through the nozzle | **NASA CEA** (equilibrium; shifting or frozen expansion) | `thermo/combustion/nasa_cea.py` |
| Gas-turbine flow path: cp(T), γ, enthalpy/entropy of air and burned gas along compressor → burner → turbine → nozzle | **Cantera** (mixture thermo) | `thermo/gas/mixture.py` |
| How *fast* chemistry happens: ignition delay, gas generators, residence time, non-equilibrium | **Cantera** (finite-rate kinetics) | `thermo/gas/kinetics.py` |

CEA and Cantera overlap on equilibrium; `test_cantera_and_cea_agree_on_chamber`
holds them within 0.3 %. Data flow for a liquid engine:
tank (CoolProp: how much propellant, at what state) → chamber (CEA: what it
burns to) → `ThrustCurve`. For a jet engine: ISA air → Cantera gas through the
cycle → `EngineDeck`.

Solid propellants are CEA problems too: the whole grain is one premixed reactant
(e.g. KNO3 + sorbitol, AP + Al + HTPB). Ingredients missing from CEA's database
(sugars, HTPB) are custom reactants in `thermo/combustion/solid_propellants.py`. The BATES
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

`propulsion/adapters/turbomachinery.py` turns blade geometry into a `ComponentMap` (TurboFlow
meanline). `propulsion/adapters/pycycle_engine.py` consumes maps in its Compressor/Turbine elements
and sweeps altitude × Mach × power into an `EngineDeck`. The flight layer
(JSBSim, `flight/adapters/jsbsim_sim.py`) only ever sees the deck.

## Solid motors: two models, one propellant interface

`BatesMotor` (web-stepped, cylindrical grains only) is the fast model for
optimisation loops. `GrainMotor` time-steps any stack of grains: closed-form
geometry for BATES, rod-and-tube and end burners; every other core shape is drawn
on a grid and regressed with a fast-marching distance transform (scikit-fmm),
perimeter from marching squares (scikit-image). Both use `solid/propellant.py`
(manual, NASA CEA library, or openMotor-style tabbed burn-rate laws) and the
shared `solve_pressure` mass balance. `ric.py` reads openMotor files (format only);
the openMotor adapter stays as an independent cross-check. GrainMotor is a clean
implementation, so AeroSim's licence is unaffected by openMotor's GPL.

## Orbital ascent

`flight/rocket/orbital_ascent.py` flies a `LaunchVehicle` (stages bottom to top)
over a rotating WGS-84 Earth with J2 (`flight/orbits.py`): vertical rise, pitch
kick along the rotation-corrected launch azimuth, gravity turn, programmed pitch
with yaw nulling of out-of-plane velocity, closed-loop climb-rate insertion burn,
coast to apogee and a circularisation burn. `solve_program` fits programme and
vehicle parameters to orbit or waypoint targets by least squares; the Rockot
example (`examples/rockot_ascent.py`) uses it to calibrate unpublished stage
masses to the User's Guide reference trajectory.

That calibration is grey-box parameter estimation: it reproduces the reference,
but the fitted masses absorb whatever the model gets wrong. The first-principles
route predicts instead and keeps the reference for checking only:

- engines: `propulsion/liquid/design.py` runs NASA CEA at the published chamber
  pressure and mixture ratio, averages shifting-equilibrium and frozen expansion,
  and applies textbook c* and Cf efficiencies (0.98 each, not tuned). The nozzle
  comes from the expansion ratio, the exit diameter, or a published vacuum /
  sea-level Isp pair (their difference fixes A_e/mdot). `Stage.from_engines`
  mixes engines into a stage; propellant = mass flow x burn time.
- drag: `flight/aero.py` builds Cd(M) from length, diameter and nose length.
- steering: `optimize_program` chooses the programme that maximises delivered
  mass (last-stage propellant left in the final orbit) under constraints
  (bounded Nelder-Mead with penalties), as a guidance designer would.

`examples/rockot_first_principles.py` applies this to Rockot and then compares
with the guide's waypoints.

## Plotting

`aerosim.plots` returns matplotlib figures in one validated, colour-vision-safe
palette (light and dark), one y-axis per panel, small multiples instead of twin
axes, text in ink colours, legends whenever there are two or more series.
`plots/replay.py` writes an animated 3D replay of one or more ascents as a
standalone HTML page that works offline (three.js and flight data embedded): globe with
coastlines (Natural Earth, public domain), live telemetry, charts and time
controls.

## Roadmap

1. RocketPy: fins / drag from geometry, parachute descent validation.
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
10. Erosive burning and ignition transient in GrainMotor; nozzle throat-length and
    convergence losses (GrainMotor Isp runs 0-8 % above openMotor without them).
11. Dymos: gravity-turn ascent with pitch control; IPOPT via pyoptsparse when available.
12. Orbital ascent: drop-zone impact points, payload-vs-orbit sweeps, SSO plane change.
