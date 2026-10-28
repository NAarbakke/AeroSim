# Long March 2C profile

Extracted from the *LM-2C User's Manual* (CALT, Issue 1999),
`../16906331-China-s-Long-March-2C-Users-Manual.pdf`. The files are:

- `vehicle.yaml`: stage specs, fairing and adapters, launch sites and loads, with a source tag
  on every value.
- `trajectory.csv`: the tabulated reference trajectory.
- `performance.csv`: the payload curves.

Of the four new references, this manual is the most complete for modelling. It gives thrust,
Isp and propellant mass for each engine, plus a state vector at every event.

## Engines

| | Thrust | Isp | Mass flow [der] | Propellant | Burn |
|---|---|---|---|---|---|
| Stage 1, 4 × DaFY6-2 | 2961.6 kN SL | 260.7 s SL | 1158.5 kg/s | 162 706 kg | 0–120.27 s |
| Stage 2 main, DaFY20-1 | 741.4 kN vac | 298.0 s | 253.7 kg/s | 54 667 kg (shared) | 121.77–305.77 s |
| Stage 2 verniers, 4 × DaFY21-1 | 4 × 11.8 kN vac | 289.0 s | 16.65 kg/s | | 121.77–566.23 s |
| CTS solid motor | 9.75 kN mean | 285.9 s | 3.48 kg/s | 121.7 kg | 35 s |

The manual gives Isp in N·s/kg, divided here by g0 = 9.80665 to get seconds.

**Stage 2 closes.** At the derived flows, the two-stage timeline uses 54.08 t, 99 % of the
54.67 t load. The verniers run for 260 s after main engine cutoff and handle the final
velocity trim and the re-orientation before SC separation.

**Stage 1 does not close.** At 1158.5 kg/s, the full 162.7 t would burn for 140 s, but
shutdown is at 120.27 s, which uses only 139.3 t. Possible explanations:

- The tabulated propellant includes unusable propellant or a reserve.
- The thrust or Isp is quoted for a different operating point.
- The 233 t liftoff mass in §2.2 is wrong; Table 1-1 gives 213 t for the same vehicle, and
  the 20 t gap is close to the 23 t mismatch.

Fit stage 1 against the trajectory table rather than trusting both mass and flow.

## Reference trajectory (`trajectory.csv`)

Table 3-2 is a full LM-2C/CTS ascent from **TSLC** (the liftoff row is 38.66 N, 111.61 E at
1452 m). It flies south, close to the Iridium-style 630 km, ~86° missions, and the injection
accuracy table (3-5) also refers to h = 630 km. The sequence:

1. Pitch-over at 10 s.
2. Stage 1 cutoff at 120.3 s: 47.1 km, 2036 m/s, γ = 22.8°.
3. Hot staging; fairing jettison at 231.7 s: 117.6 km, 3698 m/s.
4. Stage 2 main cutoff at 301.2 s: 146.9 km, 6379 m/s.
5. The verniers burn on to 613.3 s, reaching 181 km and 7918 m/s, level (local γ ≈ −0.2°,
   see below). This puts the stack at the perigee of a ~181 × 638 km transfer orbit.
6. After a half-orbit coast, the CTS ignites at 2888 s at 637.8 km apogee, then trims velocity
   and separates the SC at 3013 s (637.6 km, 7521 m/s). The CTS deorbits at 3213 s.

**The angle column is not the local flight-path angle.** The table's "ballistic inclination"
is measured from the launch point's horizontal plane. Above the local horizontal it is larger
by the central angle flown, range / 6378.14 km in radians: 22.8° → 23.3° at stage 1 cutoff,
−2.5° → 3.6° at main engine cutoff and −25.6° → −0.2° at vernier shutdown.
`examples/slv_ascent.py` converts it before comparing.

**The CTS timing does not follow from the coast.** From the vernier-shutdown state, a Kepler
coast reaches apogee about 2780 s later, at ~3390 s, but the table has the CTS igniting at
637.8 km at 2888 s. The model flies the physics and reaches apogee near 3330 s.

Figure 3-2 also plots axial acceleration, velocity, altitude and Mach against time:

- Axial acceleration peaks at ~4.6 g at stage 1 cutoff and ~6.7 g at stage 2 main cutoff.
- Mach peaks at ~12.5 near 600 s.

For the two-stage LM-2C, Table 3-1 gives event times only, with SC separation at 569.2 s.

## Payload (`performance.csv`)

- **Stated values:** 3366 kg to 200 km / 63° (two-stage), and 1456 kg to 900 km SSO (CTS).
- **Inconsistent CTS figure:** 500 km / 50° is given as 3000 kg in §3.2.3 but 2800 kg in
  Table 2-3. Figure 3-4a gives ~2750 kg at 500 km, so 3000 kg belongs to 300 km.
- **Digitized values:** the rest are read from Figures 3-3a/b and 3-4a, ±40 kg. The figure
  pages also list XSLC curves (3-3c, 3-4b) and perigee-velocity curves (3-3d) that are not
  digitized here.

## Model

`configs/orbital/long_march_2c.yaml` (LM-2C/CTS), calibrated by
`python examples/slv_ascent.py long_march_2c`.

- Engine flows and Isp are the manual's. Stage 1 propellant is the 139.3 t that its flow burns in
  120.27 s. Stage 2 flies as two stages: main engine plus verniers to 301.2 s, then verniers alone
  to 613.3 s. The fit sets the stage 1 and stage 2 dry masses, the stage 1 vacuum Isp (289.1 s)
  and the steering.
- Model against Table 3-2 (local flight-path angle): stage 1 cutoff 47.2 km / 2040 m/s / 23.3°
  (47.1 / 2036 / 23.3), main engine cutoff 146.6 km / 6371 m/s / 3.4° (146.9 / 6379 / 3.6),
  vernier shutdown 181.1 km / 7908 m/s / −0.2° (181.1 / 7918 / −0.2). Transfer orbit
  178 × 638 km, i = 86.2°.
- **The fitted lift-off mass is 212.1 t**, which sides with Table 1-1 (213 t) over §2.2 (233 t).
  Stage 1's 11.2 t dry mass then includes the ~23 t of tabulated propellant that the timeline
  never burns, or that propellant is not loaded.
- Not fitted, but it matches: the peak load is 6.6 g at main engine cutoff, against the manual's
  6.7 g.
- The fairing is dropped by the manual's 1135 W/m² criterion at 226 s, against 231.7 s in the
  table.

## Gaps

- No stage dry masses. Stage 2 dry mass follows from the trajectory once stage 1 is fitted.
  The liftoff mass is quoted twice, as 233 t and 213 t.
- No sea-level/vacuum split for stage 1: only ground values are given.
- No aerodynamic data. The fairing geometry (hemisphere, 15° cone, 3.35 m cylinder) is
  enough for a Cd estimate.
- The stage 2 total thrust in the text (798.1 kN) does not equal main plus verniers
  (788.6 kN).
