# Strela profile

Extracted from the *Strela Launch Vehicle User Manual*, Second Edition, 1999 (NPO
Mashinostroyenia), `../180618736-Strela.pdf`. The files are:

- `vehicle.yaml`: specs, loads and fairing heating.
- `trajectory.csv`: the reference injection.
- `performance.csv`: the payload curves.

Most of the useful content is in figures (Figs 2.2, 3.1, 3.4, 7.1), so many values here are
read off drawings.

## Vehicle

The RS-18 (UR-100NUTTH) ICBM used unchanged. It is two liquid N2O4/UDMH stages plus the
missile's mechanisms and instruments section (MIS). The MIS carries guidance and acts as a
single-burn kick stage at apogee, and the space head section with the fairing sits on top. The
vehicle weighs ~105 t at liftoff and is 2.5 m in diameter.

**Shared hardware with Rockot.** Rockot uses the same two stages under a Breeze-KM. The Rockot
User's Guide, already modelled in `configs/orbital/rockot.yaml`, gives the engine data this
manual lacks: Isp 310/285 s (vac/SL) for stage 1 and 320 s vac for stage 2, plus burn times of
121 s and 183 s. Strela's Figure 3.4 has stage 2 running from 126.1 s to 309.1 s, also exactly
183 s, and Figure 7.1 shows stage 1 cutoff near 123 s. The Rockot stage values can therefore be
reused directly, with Strela's MIS in place of Breeze-KM.

## Reference mission (`trajectory.csv`, Fig. 3.4)

700 kg to a 1000 km circular orbit at 90° from Svobodny:

| t (s) | Event | V (m/s) | H (km) | θ (°) | Range (km) |
|---|---|---|---|---|---|
| 51 | Max-q, 6706 kg/m² (65.8 kPa) | 566 | 48.5 | 49 | 5 |
| 126.1 | Stage 1/2 separation | 3555 | 70 | 20 | 120 |
| 164.25 | Fairing jettison | 3992 | 114 | 15 | 254 |
| 309.1 | Stage 2/MIS separation, coast begins | 7727 | 240 | 7 | 1056 |
| 1575 | Apogee, MIS ignition | 6894 | 1000 | 0 | 10 174 |
| 1760 | SC separation | 7371 | 1000 | 0 | 11 229 |

This is direct ascent to an elliptical transfer orbit with its apogee at the target altitude,
then a single MIS circularisation burn: 185 s, Δv ≈ 477 m/s.

**Max-q row is suspect.** At 48.5 km, standard-atmosphere density at 566 m/s gives only
0.16 kPa, not the stated 65.8 kPa. A climb to 48.5 km within 51 s and 5 km of range is also
impossible. 65.8 kPa at 566 m/s needs ρ = 0.41 kg/m³, which ISA places at **10.0 km**. The
altitude is most likely ~10 km and the "48.5" a misprint; treat the row's altitude as unknown.

Stage 1 drops at 1300 km downrange and stage 2 at 1600 km.

## Constraints usable for fitting

- Peak axial loads are 8.7 g (stage 1), 6.7 g (stage 2) and 0.4 g (MIS) (Table 7.2).
- Figure 7.1 (axial load against time for 100 kg and 1600 kg SC) shows the stage 1 g ramp up to
  ~123 s and the stage 2 ramp from ~2 g to ~7 g by ~300 s.
- Fairing jettison is at 110–150 km, chosen against fairing heating (Tables 7.9 and 7.10). The
  SHS-2 cone section reaches 1120 W/m² on a 250 km mission.
- SHS-1 adds 105 kg of payload over the SHS-2 curves.

## Model

`configs/orbital/strela.yaml`, calibrated by `python examples/slv_ascent.py strela`.

- Stages 1 and 2 use Rockot's Isp (310/285 s, 320 s) and dry masses. Their propellant and the
  steering are fitted to the Figure 3.4 waypoints. The MIS is sized from the 477 m/s burn with an
  assumed 290 s Isp and 1000 kg dry mass.
- Stage 2 flies the stack onto a −555 × 1000 km transfer orbit by itself. The MIS burn is centred
  on apogee, as the 185 s burn in Figure 3.4 is.
- Model against manual: stage 1 separation 73.1 km / 3570 m/s / 21.0° (70 / 3555 / 20), stage 2
  separation 240.4 km / 7770 m/s / 6.8° (240 / 7727 / 7). Final orbit 995 × 1016 km, i = 89.7°,
  with 33 kg of MIS propellant left.
- **Lift-off comes out at 96.2 t, not ~105 t.** The simulator's mass flow is propellant / burn
  time, so more stage 1 propellant would also mean more thrust, which the trajectory does not
  allow. The Rockot calibration has the same kind of gap (117 t fitted against its guide's 107 t).
- The model reaches apogee ~95 s later than the manual's MIS ignition: its stage 2 burns out
  43 m/s faster, on a slightly larger transfer orbit.

## Gaps

- **No stage masses, thrust or Isp** in the manual. Use the Rockot stage data and keep the
  liftoff mass at ≤ 105 t.
- No MIS propellant, thrust or mass. The 185 s burn and 477 m/s Δv for a 700 kg SC constrain
  them.
- No site coordinates.
- The manual's ranges (560–2000 kg, 200–2000 km) do not quite match Figure 3.1, which starts at
  250 km with ~1440 kg for SHS-2. The 2000 kg figure is probably for 200 km on the best
  inclination with SHS-1, but the manual does not show this.
