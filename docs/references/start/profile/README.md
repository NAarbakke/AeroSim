# Start-1 profile

Extracted from the *START-1 Space Launch System User's Handbook, Volume I*, Issue 1, May 2002,
`../START-1_user_handbook_vol1.pdf`. The files are:

- `vehicle.yaml`: specs, burn timeline, loads and flight record.
- `trajectory.csv`: two tabulated SSO ascents.
- `performance.csv`: the payload table.

Volume II, which covers ground operations, is not in the folder.

## Vehicle

An all-solid, four-stage launcher built from SS-25 Topol elements, cold-launched from a
canister. It weighs 47 t at liftoff and is 1.8 m in diameter and 22.7 m long. A small solid
post-boost propulsion system (PBPS) trims the orbit after stage 4 and stays attached to it.
Between stage 3 and stage 4, the vehicle coasts unpowered under a nitrogen cold-gas attitude
control system (GRACS).

Burn sequence (Figure 2-4, which matches the event times in Figure 2-3):

| Phase | Duration | Window (481 km case) |
|---|---|---|
| Stage 1 burn | 63 s | ~3–66 s |
| Coast before separation | 21 s | 66–87 s |
| Stage 2 burn | 60 s | 87–148 s |
| Stage 3 burn | 63 s | 148–211 s |
| GRACS coast | 150–500 s | 211–488 s |
| Stage 4 burn | 53 s | 488–538 s (~541 s with tail-off) |
| Delay | 3 s | |
| PBPS burn | 200 s | 538–763 s |
| SC separation | 375 s after stage 4 burnout | 912 s |

This is a classic "loft, coast to apogee, circularise" ascent. Stages 1–3 put the vehicle on a
suborbital arc with apoapsis at the target altitude. Stage 4 lifts the speed from 5012 to
7704 m/s at γ ≈ 4.7° → 0°. Stage 3 separation (211 s, 5535 m/s) is the peak speed before the
coast: the vehicle slows to 5012 m/s as it climbs from 168 km to 478 km.

## Trajectory constraints usable for fitting

- `trajectory.csv` gives two cases flown on the same boost-phase program: 481 km with the PBPS
  and 620.8 km without it. Each row has time, velocity, altitude, range, flight-path angle and
  ground latitude/longitude. The two cases are identical through stage 1 (87 s). From stage 2 on,
  the no-PBPS case flies a lofted program (θ = 22.8° vs 18.1° at 148 s) to reach its higher
  apogee.
- Stage 4 Δv ≈ 2690 m/s (481 km case) and 2850 m/s (no-PBPS case), straight from the tables.
- Peak axial loads are 5.15, 6.5, 6.5 and 10.0 g for stages 1–4 with a 240 kg SC (Table 4-1).
  Figure 4-1 gives the stage 4 peak against SC mass: ~13 g at 50 kg, 10 g at 250 kg, 8.4 g at
  400 kg. With the 53 s burn, that curve pins stage 4 thrust and mass.
- The fairing is jettisoned above 150 km at ≤ 5500 m/s, at 295 s in both cases.
- Acoustic loads peak at 138 dB during the 30 s transonic phase.

## Model

`configs/orbital/start1.yaml`, calibrated by `python examples/slv_ascent.py start1`. This is the
weakest of the models, because the handbook gives no masses, thrust or Isp.

- Dry masses are estimates. Stage 1 and 2 Isp and the stage 1-3 propellant masses are fitted
  to the Figure 2-3 waypoints and the 47 t lift-off mass. Stage 4 propellant (1600 kg) is sized
  from the ~2.7 km/s it adds between ignition and burnout. The PBPS is not flown; its mass is in
  stage 4.
- Model against handbook: every staging point within 2 km, 30 m/s and 0.5°, and fairing release
  at 295 s.
- **Lift-off comes out at 51.4 t, not 47 t**, with stage 1 and 2 Isp at the top of their
  plausible ranges (285 s and 300 s). The dry-mass estimates are too heavy.
- Stage 4 burns prograde, centred on apogee. The real stage 4 ignites earlier (488 s, still
  climbing at 4.7°) and steers to hold altitude, and the PBPS trims afterwards. So the model
  ignites 55 s late and ends on 479 × 552 km rather than 481 km circular.

## Gaps

- **No stage masses, propellant loads, thrust or Isp.** Only the liftoff mass (47 t), the burn
  times and the g-peaks are given. A model has to fit motor sizes to the trajectory table,
  or start from public Topol stage data `[ext]` and calibrate it.
- No launch-site coordinates. Svobodny is about 51.8 N, 128.4 E `[ext]`; the first tabulated
  ground point is 52.2 N, 128.0 E at 87 s.
- The orbit radii printed in Figures 2-3 and 2-5 (6838 and 6978 km) equal the stated altitude plus
  6357 km, the polar radius. Altitudes in these figures are measured from that radius, not from
  the 6378 km equatorial radius.
- Figure 2-5 mislabels the 585 s point as "4th Stage Motor Burnout"; it is stage 4 ignition.
- Payload in Table 2-1 is SC mass, presumably excluding the adapter; the handbook does not say.
