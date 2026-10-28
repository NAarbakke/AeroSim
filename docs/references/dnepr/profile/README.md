# Dnepr-1 profile

Extracted from the *Dnepr Space Launch System User's Guide*, Issue 2, November 2001,
`../Dnepr-User-Guide.pdf`. `vehicle.yaml` holds the specs with a source tag on every value,
`performance.csv` holds the payload curves, and this file covers the flight sequence, how the
numbers were read, and what is missing.

## Vehicle

Three-stage, all-storable (N2O4/UDMH) SS-18 conversion launched from a silo at Baikonur. Stages 1
and 2 are unmodified SS-18 stages. Stage 3 is the SS-18 post-boost stage, with only its control
system changed. A Space Head Module (fairing, intermediate section, adapter, GDS or EPM) sits on top.

| Stage | Vacuum thrust | Propellant | Stack mass at ignition |
|---|---|---|---|
| 1 | 4523 kN (461.2 tf), 4 chambers | 147 900 kg | 208 900 kg |
| 2 | 760 kN (77.5 tf), main + 4-chamber steering | 36 740 kg | 47 380 kg |
| 3 | 18.6 / 7.8 kN (1.9 / 0.8 tf), two modes | 1 910 kg | 6 266 kg |

**Reading Table 4.1-1.** The guide labels the masses "Liftoff mass (with the spacecraft mass of
2000 kg)", given per stage. They only make sense as the stack mass at each stage's ignition, so
208.9 t is the liftoff mass. That reading gives:

- Stage 1 dry mass 13.6 t (61.0 t at burnout minus the 47.4 t upper stack).
- Stage 2 dry mass plus fairing 4.4 t.
- Stage 3 dry mass plus SHM 2.4 t, with the 2000 kg SC.

These masses are tagged `[interp]` and `[der]` in the YAML.

## Flight sequence (Figures 4.1-4 and 4.1-5, no times given)

1. Mortar ejection from the canister. Stage 1 ignites at H ≈ 20 m.
2. Stage 1 shutdown and separation; stage 2 ignition.
3. Fairing separation during the stage 2 burn. The fairing still shields RF until about 275 s
   (§9.9), which bounds fairing jettison at ≲ 275 s.
4. Stage 2 shutdown and separation; stage 3 ignites at full power.
5. Stage 3 makes a programmed 180° turn, then switches to the throttled-back mode. It is a
   "drag" (pulling) stage: the SHM is attached to its upper end and its nozzles face forward.
6. GDS (or EPM cover) jettison.
7. The SC separates as stage 3 backs away under low thrust; there are no spring pushers.
   Multi-SC missions add a separation for each tier.

The drop zones are fixed by SS-18 heritage, so only 50.5°, 64.5°, 87.3° and 98° are offered. For
the southward 87.3° and 98° launches, a mobile telemetry station near Salalah, Oman covers SC
separation.

## Trajectory constraints usable for fitting

- Peak axial load is 7.5 g at the end of stage 1 and 7.8 g at the end of stage 2 (Table 9.2-2,
  including gravity). Stage 3 loads are −0.3 to −0.5 g, negative because the stage pulls.
- The 35 s duration of liftoff, transonic and max-q vibration (Table 9.3-3) and of the acoustic
  load (Table 9.5-1) roughly brackets the dense-atmosphere phase. The 831 s "rest" duration is
  the powered flight from there to separation.
- Fairing pressure (Figure 9.7-1) decays to ~0 by about 100 s.
- The stage 3 plume table (Table 9.8-1) gives an undisturbed gas velocity of 3670–3764 m/s at the
  SC. That suggests a stage 3 exhaust velocity in the high-3000s m/s, or Isp ≈ 375 s. This is an
  upper bound: a plume expanding into vacuum speeds up beyond the nozzle exit velocity.

## Model

`configs/orbital/dnepr.yaml`, with steering optimised by `python examples/slv_ascent.py dnepr`.

- Thrust, propellant and the stack masses are the guide's. Isp is outside data (318/293, 340
  and 309 s), and burn times follow from it. There is no trajectory to calibrate against, so the
  steering maximises the stage 3 propellant left on the headline mission: 3700 kg to 300 km
  circular at 50.5°.
- **The model carries up to 5050 kg there, 37 % more than the guide.** The outside Isp values,
  the reading of the mass table, or both are optimistic. The optimum also has its 30° pitch kick
  at the bound, a steep early turn.
- Loads are a partial check: peak acceleration is 7.2 g at stage 2 cutoff, against
  7.8 ± 0.5 g in the guide.
- To make this a calibration, fit one Isp scale to the 300 km payload and check the model against
  the rest of Figure 4.1-3.

## Gaps for engine/trajectory modelling

- **No Isp for any stage.** Thrust and propellant are given, so Isp fixes the burn times.
  Public SS-18 sources give the engines as RD-264, RD-0255 and RD-869 (tagged `[ext]`); take
  Isp from those sources and check them against the payload curve.
- No event times, altitudes or velocities; the mission profile figures are drawings only.
- No aerodynamic data, and no fairing mass or jettison time.
- Site coordinates are rounded to 46 N, 63 E in the guide. Sites 106/109 are about
  45.95 N, 63.50 E `[ext]`.
