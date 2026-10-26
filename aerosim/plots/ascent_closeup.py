"""Close-up replay of a launch through the atmosphere: the rocket up close, the sky and Earth as
they look from its altitude, the atmospheric layers it climbs through, and how the air shapes the
flight (drag, dynamic pressure, Mach, heating, thrust lost to back pressure, Δv losses).

    data = ascent_physics(result, vehicle)            # forces and losses along the trajectory
    page = ascent_closeup_html(result, vehicle, title=..., nav=...)

Everything is derived from the simulator's own models at each output sample: drag
D = ½ρv²·C_D(M)·A with the vehicle's C_D curve, thrust F = F_vac − p_a·A_e for the stage that is
burning, weight m·g, heat flux ½ρv³ (the fairing-jettison criterion), and the classic losses
∫D/m dt (drag), ∫g·sin γ dt (gravity), ∫(F_vac − F)/m dt (back pressure). Temperature in the
atmosphere column is the US76 kinetic temperature up to 1000 km (``kinetic_temperature``).

The 3D scene is centred on the rocket: sky and Earth are one shader that traces each view ray
against the curved Earth and a Rayleigh + Mie atmosphere (single scattering), so the sky darkens
to space and the horizon curves as the rocket climbs. The Rockot / Breeze-KM model is built
from its published dimensions (29.15 m long, 2.5 m stages, 2.62 m fairing); details such as
colours, nozzle lengths and the stage split are approximate.
"""

from __future__ import annotations

import html
import math

import numpy as np

from aerosim.atmosphere.standard import isa, kinetic_temperature
from aerosim.flight import orbits
from aerosim.plots.replay import _importmap
from aerosim.plots.site import _r, _thin
from aerosim.plots.web import CHART_JS, CSS, data_tag, theme_tag, themed

# (name, base m, top m, colour, what happens there); boundaries per US76 temperature structure
LAYERS = [
    ("Troposphere", 0, 11_000, "#6fa8dc",
     ("Weather, and three quarters of the air's mass. Temperature falls 6.5 °C per km. The rocket "
     "is still slow, but the air is dense: drag and dynamic pressure build fastest here.")),
    ("Tropopause", 11_000, 20_000, "#8fb6d9",
     ("A lid of constant −56.5 °C. Dynamic pressure peaks around here (max-Q): speed has grown "
     "faster than the air has thinned. The structure feels its highest aerodynamic load.")),
    ("Stratosphere", 20_000, 47_000, "#9fc5a8",
     ("Ozone absorbs ultraviolet light, so it warms with height. The air thins a hundredfold: "
     "drag falls away even as speed climbs past Mach 5, and the engines gain thrust.")),
    ("Stratopause", 47_000, 51_000, "#b9d39b",
     "The warm top of the stratosphere, about −2.5 °C. Less than 0.1 % of the air is above."),
    ("Mesosphere", 51_000, 85_000, "#c9a96e",
     ("The coldest layer, below −86 °C at the top; meteors burn up here. The first stage burns out "
     "in it, and stage 2 starts in near-vacuum.")),
    ("Mesopause", 85_000, 91_000, "#a08cc0",
     "The coldest place in the atmosphere, about −87 °C."),
    ("Thermosphere", 91_000, 600_000, "#c08ca8",
     ("Hot by temperature (500–1000 °C) but so thin it carries almost no heat. The Kármán line, "
     "the conventional edge of space, is at 100 km; auroras glow here; the ISS orbits at about "
     "420 km. The fairing goes once the heating it shields against has died away.")),
    ("Exosphere", 600_000, 1_000_000, "#7d7a9c",
     "Atoms on ballistic paths, merging into space. The 650 km target orbit is at its base."),
]
MARKERS = [("Kármán line", 100_000)]


def _stage_for_phase(vehicle, name: str):
    for st in vehicle.stages:
        if name == st.name or name.startswith(st.name + " burn"):
            return st
    return None


def ascent_physics(res, vehicle, t_end: float | None = None) -> dict:
    """Forces, losses and atmosphere along an ``AscentResult`` (arrays per output sample)."""
    t = res.time
    keep = t <= (t_end if t_end is not None else t[-1]) + 1e-9
    t = t[keep]
    h, v, m = res.altitude[keep], res.speed[keep], res.mass[keep]
    fpa = res.flight_path_angle[keep]
    ph = res.phase[keep]
    r = res.position[keep]
    n = t.size
    p = np.zeros(n)
    rho = np.zeros(n)
    a_snd = np.full(n, np.inf)
    T = np.zeros(n)
    for i, hh in enumerate(h):
        pa, ta, ra = isa(max(hh, 0.0))
        p[i], rho[i], T[i] = pa, ra, kinetic_temperature(max(hh, 0.0))
        a_snd[i] = math.sqrt(1.4 * 287.05287 * ta)
    air = h < 200e3
    p[~air] = rho[~air] = 0.0
    mach = np.where(air, v / a_snd, np.nan)
    q = 0.5 * rho * v**2
    cd = np.array([vehicle.cd(M) if np.isfinite(M) else 0.0 for M in mach])
    drag = q * cd * vehicle.reference_area
    g = np.array([np.linalg.norm(orbits.gravity(x)) for x in r])
    thrust, fvac = np.zeros(n), np.zeros(n)
    for i in range(n):
        st = _stage_for_phase(vehicle, res.phases[ph[i]])
        if st is not None:
            thrust[i], fvac[i] = st.thrust(p[i]), st.thrust_vac
    rates = {"drag": drag / m, "gravity": g * np.sin(np.radians(fpa)) * (thrust > 0),
             "pressure": (fvac - thrust) / m}
    losses = {k: np.cumsum(np.r_[0.0, 0.5 * (r_[1:] + r_[:-1]) * np.diff(t)]) for k, r_ in rates.items()}
    p0 = isa(0.0)[0]
    return {"t": t, "h": h, "v": v, "mach": mach, "q": q, "cd": cd, "drag": drag, "thrust": thrust,
            "thrust_vac": fvac, "weight": m * g, "mass": m, "fpa": fpa, "rho": rho, "p": p, "T": T,
            "heat": 0.5 * rho * v**3, "air_above": p / p0, "acc": res.acceleration[keep] / 9.80665,
            "downrange": res.downrange[keep], "phase": ph, "losses": losses}


def _crossings(t, h, levels):
    out = []
    for name, z in levels:
        k = np.nonzero((h[:-1] < z) & (h[1:] >= z))[0]
        if k.size:
            i = k[0]
            out.append((name, float(t[i] + (z - h[i]) / (h[i + 1] - h[i]) * (t[i + 1] - t[i]))))
    return out


def mach1_time(t, mach) -> float | None:
    """When the Mach number first reaches 1 (interpolated between samples)."""
    m = np.nan_to_num(np.asarray(mach, float))
    k = np.nonzero((m[:-1] < 1) & (m[1:] >= 1))[0]
    if not k.size:
        return None
    i = k[0]
    return float(t[i] + (1 - m[i]) / (m[i + 1] - m[i]) * (t[i + 1] - t[i]))


def _events(res, d) -> list[tuple[str, float]]:
    t, h = d["t"], d["h"]
    ev = [("Liftoff", 0.0)]
    t1 = mach1_time(t, d["mach"])
    if t1 is not None:
        ev.append(("Mach 1", t1))
    names = {"max_q": "Max-Q", "fairing_jettison": "Fairing jettison"}
    for key, val in res.events.items():
        if val > t[-1] or key == "liftoff":
            continue
        label = names.get(key) or key.replace("_", " ").replace("burnout", "burnout and separation") \
            .replace("burn1 start", "ignition").replace("burn1 end", "cut-off (transfer orbit)") \
            .replace("burn2 start", "second burn").replace("burn2 end", "cut-off (final orbit)")
        ev.append((label, float(val)))
    ev += [(f"Into the {n.lower()}", tt) for n, tt in _crossings(t, h, [(L[0], L[1]) for L in LAYERS[1:]])]
    ev += _crossings(t, h, MARKERS)
    seen, out = set(), []
    for e in sorted(ev, key=lambda e: e[1]):
        if e[0] not in seen:
            seen.add(e[0])
            out.append(e)
    return out


def _temperature_profile() -> list[list[float]]:
    zs = np.r_[np.arange(0, 120e3, 1e3), np.arange(120e3, 1000e3 + 1, 10e3)]
    return [[float(z), round(kinetic_temperature(z), 1), float(f"{isa(z)[2]:.3e}")] for z in zs]


_CGCP = ('<p class="cgcp"><i style="background:#ffd23f"></i>centre of gravity'
         '<i style="background:#ff3b3b"></i>centre of pressure</p>')

_CSS = r"""
html,body{height:100%}body{display:flex;flex-direction:column}
#app{flex:1;min-height:0;display:grid;grid-template-columns:minmax(0,1fr) 420px;
grid-template-rows:minmax(0,1fr) auto}
#scene{position:relative;min-height:0;overflow:hidden;background:#000}#scene canvas{display:block}
#hud{position:absolute;left:20px;top:16px;pointer-events:none;max-width:62%;color:#fff;
text-shadow:0 1px 4px #000}
#hud h1{margin:0;font-size:20px;font-weight:600}#hud p{margin:2px 0 0;font-size:13px;opacity:.85}
#hud .cgcp i{display:inline-block;width:9px;height:9px;border-radius:50%;margin:0 5px 0 12px}
#hud .cgcp i:first-child{margin-left:0}
#layer{position:absolute;left:20px;top:104px;max-width:430px;pointer-events:none;color:#fff;
background:rgba(10,14,22,.55);border-left:4px solid #6fa8dc;border-radius:6px;padding:8px 12px;
font-size:12.5px;line-height:1.45;text-shadow:0 1px 2px #000}
#layer b{display:block;font-size:15px;margin-bottom:2px}
#toast{position:absolute;left:50%;top:22%;transform:translate(-50%,0);color:#fff;font-size:22px;
font-weight:600;text-shadow:0 2px 8px #000;opacity:0;transition:opacity .4s;pointer-events:none;
white-space:nowrap}
#clock{position:absolute;left:20px;bottom:30px;color:#fff;font:600 28px/1 Consolas,"SF Mono",monospace;
pointer-events:none;text-shadow:0 1px 4px #000;font-variant-numeric:tabular-nums}
#clock small{display:block;font:500 12px/1.6 "Segoe UI",system-ui,sans-serif;opacity:.85}
#gauges{position:absolute;right:16px;top:16px;display:grid;grid-template-columns:repeat(2,auto);
gap:2px 14px;color:#fff;text-shadow:0 1px 3px #000;font-variant-numeric:tabular-nums;
pointer-events:none;text-align:right}
#gauges div{font:600 18px/1.1 Consolas,"SF Mono",monospace}
#gauges span{display:block;font:500 10.5px/1.3 "Segoe UI",system-ui,sans-serif;opacity:.8}
#side{grid-row:1/3;grid-column:2;overflow-y:auto;background:var(--surface);
border-left:1px solid var(--grid);padding:16px 18px 20px}
canvas.chart.t{height:150px}#column{width:100%;height:440px;display:block}
#bar{grid-column:1;display:flex;align-items:center;gap:12px;padding:10px 20px;
background:var(--surface);border-top:1px solid var(--grid);flex-wrap:wrap}
#play{min-width:76px;font-weight:600}#scrub{flex:1;min-width:160px;accent-color:var(--accent)}
.note{font-size:12px;color:var(--ink2);margin:6px 0}
@media (max-width:900px){#app{grid-template-columns:1fr;grid-template-rows:62vh auto auto}
#side{grid-row:3;grid-column:1;border-left:0;border-top:1px solid var(--grid)}#bar{grid-row:2}
#gauges{display:none}#layer{max-width:80%}}
"""

_JS = r"""
import * as THREE from "three";
import {OrbitControls} from "three/addons/controls/OrbitControls.js";
const D = JSON.parse(document.getElementById("data").textContent), TK = TKC;
const F = D.f, N = F.t.length, tEnd = F.t[N - 1], rad = Math.PI / 180;
const RE = 6371.0;                                   // km, Earth radius for the sky shader

// ---------- renderer, camera ----------
const host = document.getElementById("scene");
const renderer = new THREE.WebGLRenderer({antialias: true, logarithmicDepthBuffer: true});
renderer.setPixelRatio(Math.min(devicePixelRatio, 2)); host.prepend(renderer.domElement);
renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.0;
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, 1, 0.05, 2e6);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; controls.minDistance = 8; controls.maxDistance = 6000;

// ---------- sky + Earth: one background shader (ray vs curved Earth, single scattering) ----------
// units: km; the rocket sits at altitude uAlt above the Earth, local vertical +y
const SUN = new THREE.Vector3(-0.45, 0.62, 0.64).normalize();
const skyMat = new THREE.ShaderMaterial({
  side: THREE.BackSide, depthWrite: false, depthTest: false,
  uniforms: {uAlt: {value: 0}, uCam: {value: new THREE.Vector3()}, uSun: {value: SUN}, uRot: {value: 0}},
  vertexShader: `varying vec3 vDir; void main(){ vDir = normalize((modelMatrix * vec4(position, 0.0)).xyz);
    vec4 p = projectionMatrix * viewMatrix * vec4((modelMatrix * vec4(position, 1.0)).xyz, 1.0); gl_Position = p.xyww; }`,
  fragmentShader: `
    precision highp float; varying vec3 vDir;
    uniform float uAlt, uRot; uniform vec3 uCam, uSun;
    const float RE = 6371.0, RA = 6471.0, HR = 8.0, HM = 1.2;
    const vec3 BR = vec3(5.5e-3, 13.0e-3, 22.4e-3); const float BM = 21e-3;
    vec2 rs(vec3 o, vec3 d, float r){ float b = dot(o,d), c = dot(o,o) - r*r, h = b*b - c;
      if (h < 0.0) return vec2(1e9, -1e9); h = sqrt(h); return vec2(-b - h, -b + h); }
    float hash(vec3 p){ p = fract(p * 0.3183099 + 0.1); p *= 17.0; return fract(p.x * p.y * p.z * (p.x + p.y + p.z)); }
    float noise(vec3 x){ vec3 i = floor(x), f = fract(x); f = f*f*(3.0-2.0*f);
      return mix(mix(mix(hash(i), hash(i+vec3(1,0,0)), f.x), mix(hash(i+vec3(0,1,0)), hash(i+vec3(1,1,0)), f.x), f.y),
                 mix(mix(hash(i+vec3(0,0,1)), hash(i+vec3(1,0,1)), f.x), mix(hash(i+vec3(0,1,1)), hash(i+vec3(1,1,1)), f.x), f.y), f.z); }
    float fbm(vec3 p){ float a = 0.5, s = 0.0; for (int i = 0; i < 6; i++){ s += a * noise(p); p *= 2.03; a *= 0.5; } return s; }
    void main(){
      vec3 d = normalize(vDir), o = uCam + vec3(0.0, RE + uAlt, 0.0);
      vec2 ta = rs(o, d, RA), tg = rs(o, d, RE);
      bool ground = tg.x > 0.0;
      float t0 = max(ta.x, 0.0), t1 = ground ? tg.x : ta.y;
      vec3 col = vec3(0.0); vec3 tau = vec3(0.0);
      if (t1 > t0) {                        // in-scattered sunlight along the ray (12 steps, 4 to the sun)
        float ds = (t1 - t0) / 12.0; float odR = 0.0, odM = 0.0; vec3 sR = vec3(0.0), sM = vec3(0.0);
        for (int i = 0; i < 12; i++) {
          vec3 p = o + d * (t0 + ds * (float(i) + 0.5)); float hh = length(p) - RE;
          float dR = exp(-hh / HR) * ds, dM = exp(-hh / HM) * ds; odR += dR; odM += dM;
          vec2 tl = rs(p, uSun, RA); float dl = tl.y / 4.0, lR = 0.0, lM = 0.0; vec2 te = rs(p, uSun, RE); bool lit = te.x > 1e8 || te.y < 0.0;   // sun ray misses the Earth
          for (int j = 0; j < 4; j++) { float h2 = length(p + uSun * dl * (float(j) + 0.5)) - RE;
            lR += exp(-h2 / HR) * dl; lM += exp(-h2 / HM) * dl; }
          vec3 att = exp(-(BR * (odR + lR) + BM * 1.1 * (odM + lM)));
          if (lit) { sR += att * dR; sM += att * dM; }
        }
        float mu = dot(d, uSun), pR = 0.0597 * (1.0 + mu*mu), g = 0.76;
        float pM = 0.1194 * ((1.0-g*g) * (1.0+mu*mu)) / ((2.0+g*g) * pow(1.0 + g*g - 2.0*g*mu, 1.5));
        col = 40.0 * (sR * BR * pR + sM * BM * pM);
        tau = BR * odR + BM * 1.1 * odM;
      }
      if (ground) {                         // the surface: boreal forest, lakes, cloud, lit by the sun
        vec3 p = normalize(o + d * tg.x);
        float c = cos(uRot), s = sin(uRot); vec3 q = vec3(c*p.x - s*p.y, s*p.x + c*p.y, p.z) * 900.0;
        float n = fbm(q), lake = smoothstep(0.62, 0.66, fbm(q * 2.1 + 7.0)), cl = smoothstep(0.6, 0.85, fbm(q * 0.35 + 3.0));
        vec3 alb = mix(vec3(0.05, 0.09, 0.045), vec3(0.11, 0.12, 0.07), n);
        alb = mix(alb, vec3(0.02, 0.04, 0.07), lake); alb = mix(alb, vec3(0.75), cl * 0.6);
        float lam = max(dot(p, uSun), 0.0);
        col += alb * lam * 1.6 * exp(-tau);
      } else {                              // the sun disc in thin air
        col += vec3(1.0, 0.95, 0.85) * smoothstep(0.9996, 0.99985, dot(d, uSun)) * 20.0 * exp(-tau);
      }
      gl_FragColor = vec4(col, 1.0);
      #include <tonemapping_fragment>
      #include <colorspace_fragment>
    }`
});
const sky = new THREE.Mesh(new THREE.SphereGeometry(1, 64, 32), skyMat); sky.frustumCulled = false; sky.renderOrder = -10;
scene.add(sky);
// stars: fade in as the sky above goes dark
const starGeo = new THREE.BufferGeometry(), sp = [];
for (let i = 0; i < 2500; i++) { const u = Math.random() * 2 - 1, a = Math.random() * 6.283, r = Math.sqrt(1 - u * u);
  sp.push(r * Math.cos(a) * 9e5, u * 9e5, r * Math.sin(a) * 9e5); }
starGeo.setAttribute("position", new THREE.Float32BufferAttribute(sp, 3));
const stars = new THREE.Points(starGeo, new THREE.PointsMaterial({color: 0xffffff, size: 1.6, sizeAttenuation: false, transparent: true, opacity: 0, depthWrite: false}));
scene.add(stars);

// ---------- lights ----------
const sunL = new THREE.DirectionalLight(0xffffff, 2.2); sunL.position.copy(SUN).multiplyScalar(100); scene.add(sunL);
const hemi = new THREE.HemisphereLight(0x9cc4ff, 0x2b3322, 0.9); scene.add(hemi);

// ---------- Rockot / Breeze-KM, true scale (m), body axis +y, origin at the nozzle exit plane ----------
const M = D.model;
const paint = (c, extra = {}) => new THREE.MeshStandardMaterial({color: c, roughness: 0.55, metalness: 0.15, ...extra});
const matBody = paint(0x6c7560), matDark = paint(0x3f463a), matFair = paint(0xeef0f2, {roughness: 0.4}),
  matBand = paint(0x2b4a86), matNoz = paint(0x2a2622, {metalness: 0.7, roughness: 0.35}),
  matBreeze = paint(0xb9bec4, {metalness: 0.5}), matGold = paint(0xc8a24a, {metalness: 0.8, roughness: 0.3});
const cyl = (r0, r1, h, mat, y, seg = 48) => { const m = new THREE.Mesh(new THREE.CylinderGeometry(r1, r0, h, seg, 1, false), mat); m.position.y = y + h / 2; return m; };
function bell(re, rt, len, mat) {          // nozzle bell, throat at the top
  const pts = []; for (let k = 0; k <= 16; k++) { const f = k / 16; pts.push(new THREE.Vector2(rt + (re - rt) * Math.pow(f, 0.6), len * (1 - f))); }
  return new THREE.Mesh(new THREE.LatheGeometry(pts, 32), mat);
}
function groupAt(y) { const g = new THREE.Group(); g.position.y = y; return g; }
const R = M.d / 2, RF = M.fairing_d / 2;
const parts = {};
// stage 1: four nozzles under an aft skirt
parts.s1 = groupAt(0);
parts.s1.add(cyl(R * 1.03, R, 1.2, matDark, M.nozzle1), cyl(R, R, M.s1 - 1.2, matBody, M.nozzle1 + 1.2));
for (let k = 0; k < 4; k++) { const a = k * Math.PI / 2 + Math.PI / 4, b = bell(M.nozzle1_re, M.nozzle1_re * 0.42, M.nozzle1, matNoz);
  b.position.set(Math.cos(a) * R * 0.48, 0, Math.sin(a) * R * 0.48); parts.s1.add(b); }
const duct = new THREE.Mesh(new THREE.BoxGeometry(0.18, M.s1 - 1.6, 0.22), matDark);   // cable duct
duct.position.set(R + 0.06, M.nozzle1 + M.s1 / 2 + 0.2, 0); parts.s1.add(duct);
// interstage (goes with stage 1) and stage 2 with its nozzle inside the interstage
const yS2 = M.nozzle1 + M.s1;
parts.s1.add(cyl(R, R, M.inter, matDark, yS2));
parts.s2 = groupAt(yS2 + M.inter);
parts.s2.add(cyl(R, R, M.s2, matBody, 0), cyl(R, RF, M.trans, matDark, M.s2));   // transition stays with stage 2
const n2 = bell(M.nozzle2_re, M.nozzle2_re * 0.35, M.inter * 0.95, matNoz); n2.position.y = -M.inter * 0.95; parts.s2.add(n2);
// upper composite: transition, Breeze-KM and payload under the fairing
parts.up = groupAt(yS2 + M.inter + M.s2 + M.trans);
parts.breeze = new THREE.Group();
parts.breeze.add(cyl(RF * 0.92, RF * 0.92, M.breeze, matBreeze, 0.1));
const n3 = bell(M.nozzle3_re, 0.12, 0.9, matNoz); n3.position.y = 0.1 - 0.9; parts.breeze.add(n3);
const payload = new THREE.Mesh(new THREE.BoxGeometry(1.5, 2.0, 1.5), matGold); payload.position.y = M.breeze + 1.3;
const panel = new THREE.Mesh(new THREE.BoxGeometry(0.05, 1.7, 1.2), paint(0x1b2a55, {metalness: 0.3}));
panel.position.set(0.8, payload.position.y, 0); parts.breeze.add(payload, panel);
parts.up.add(parts.breeze);
// fairing: two halves, cylinder + tangent ogive
const ogive = []; const Ln = M.nose, rho_ = (RF * RF + Ln * Ln) / (2 * RF);
for (let k = 0; k <= 24; k++) { const yy = Ln * k / 24; ogive.push(new THREE.Vector2(Math.max(Math.sqrt(rho_ * rho_ - yy * yy) + RF - rho_, 0.001), yy)); }
const fprof = [new THREE.Vector2(RF, 0), new THREE.Vector2(RF, M.fairing - Ln), ...ogive.map(v => new THREE.Vector2(v.x, v.y + M.fairing - Ln))];
parts.fair = [];
for (const s of [0, 1]) {
  const half = new THREE.Group();
  const g = new THREE.LatheGeometry(fprof, 40, s * Math.PI, Math.PI);
  half.add(new THREE.Mesh(g, matFair));
  const band = new THREE.Mesh(new THREE.CylinderGeometry(RF * 1.002, RF * 1.002, 0.35, 40, 1, true, s * Math.PI, Math.PI), matBand);
  band.position.y = 0.9; half.add(band);
  parts.up.add(half); parts.fair.push(half);
}
const L_TOT = yS2 + M.inter + M.s2 + M.trans + M.fairing;
const stack = new THREE.Group(); stack.add(parts.s1, parts.s2, parts.up);
const rocket = new THREE.Group(); rocket.add(stack); scene.add(rocket);       // pivot: centre of the flying stack
const tip = new THREE.Mesh(new THREE.SphereGeometry(0.35, 16, 12), new THREE.MeshBasicMaterial({color: 0xff7a3a, transparent: true, opacity: 0, depthWrite: false, blending: THREE.AdditiveBlending})); tip.position.y = L_TOT + 0.1; stack.add(tip);
let centre = L_TOT / 2;
// centre of gravity (yellow) and of pressure (red), just outside the skin so they stay visible
const mark = c => new THREE.Mesh(new THREE.SphereGeometry(0.32, 20, 14), new THREE.MeshBasicMaterial({color: c, depthTest: false, transparent: true}));
const cgMark = mark(0xffd23f), cpMark = mark(0xff3b3b); cgMark.renderOrder = cpMark.renderOrder = 5;
stack.add(cgMark, cpMark);

// plume: bright core + an outer envelope that balloons as the ambient pressure falls
const plumeMat = (c, o) => new THREE.MeshBasicMaterial({color: c, transparent: true, opacity: o, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide});
function plume(r, len) { const g = new THREE.Group();
  const core = new THREE.Mesh(new THREE.CylinderGeometry(r * 0.95, r * 0.15, len, 24, 1, true), plumeMat(0xff9a3c, 0.55)); core.position.y = -len / 2;
  const env = new THREE.Mesh(new THREE.CylinderGeometry(r, r * 2.0, len * 1.8, 24, 1, true), plumeMat(0xd8502a, 0.14));
  env.geometry.translate(0, -len * 0.9, 0);                 // top at the nozzle exit, so scaling widens downstream
  g.add(core, env); g.core = core; g.env = env; return g; }
const pl1 = plume(R * 0.9, 18); pl1.position.y = 0; parts.s1.add(pl1);
const pl2 = plume(M.nozzle2_re, 10); pl2.position.y = -M.inter * 0.95; parts.s2.add(pl2);
const pl3 = plume(M.nozzle3_re, 5); pl3.position.y = 0.1 - 0.9; parts.breeze.add(pl3);

// Mach cone (half-angle asin 1/M from the nose) and vapour cone near Mach 1
const shock = new THREE.Mesh(new THREE.ConeGeometry(1, 1, 48, 1, true), new THREE.MeshBasicMaterial({color: 0xcfe6ff, transparent: true, opacity: 0, depthWrite: false, side: THREE.DoubleSide}));
scene.add(shock);
const vapour = new THREE.Mesh(new THREE.CylinderGeometry(R * 1.05, R * 2.6, 6, 40, 1, true), new THREE.MeshBasicMaterial({color: 0xffffff, transparent: true, opacity: 0, depthWrite: false, side: THREE.DoubleSide}));
scene.add(vapour);

// air: streaks that pass the rocket at the true relative wind, fewer and fainter as the air thins
const NS = 900, streakPos = new Float32Array(NS * 6), seeds = [];
for (let i = 0; i < NS; i++) seeds.push([Math.random() * 2 - 1, Math.random() * 2 - 1, Math.random() * 2 - 1]);
const streakGeo = new THREE.BufferGeometry(); streakGeo.setAttribute("position", new THREE.BufferAttribute(streakPos, 3));
const streaks = new THREE.LineSegments(streakGeo, new THREE.LineBasicMaterial({color: 0xffffff, transparent: true, opacity: 0.0, depthWrite: false}));
scene.add(streaks);
let windPhase = 0;

// launch pad and tower (left behind on the ground)
const pad = new THREE.Group();
const padBase = new THREE.Mesh(new THREE.CylinderGeometry(14, 16, 2, 32), paint(0x55585c)); padBase.position.y = -1; pad.add(padBase);
const tower = new THREE.Mesh(new THREE.BoxGeometry(3, 34, 3), paint(0x8a3b2c)); tower.position.set(-6, 17, 0); pad.add(tower);
scene.add(pad);

// layer boundaries: a wide translucent sheet at each base altitude, seen as the rocket passes it
const sheets = D.layers.slice(1).map(L => { const m = new THREE.Mesh(new THREE.CircleGeometry(900, 64),
  new THREE.MeshBasicMaterial({color: L.color, transparent: true, opacity: 0, side: THREE.DoubleSide, depthWrite: false}));
  m.rotation.x = -Math.PI / 2; m.userData.alt = L.base; scene.add(m); return m; });
D.markers.forEach(Mk => { const m = new THREE.Mesh(new THREE.RingGeometry(300, 900, 64), new THREE.MeshBasicMaterial({color: 0xffffff, transparent: true, opacity: 0, side: THREE.DoubleSide, depthWrite: false}));
  m.rotation.x = -Math.PI / 2; m.userData.alt = Mk.alt; scene.add(m); sheets.push(m); });

// ---------- sampling ----------
function find(a, x) { let lo = 0, hi = a.length - 1; if (x <= a[0]) return 0; if (x >= a[hi]) return hi - 1;
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (a[mid] <= x) lo = mid; else hi = mid; } return lo; }
function sample(t) {
  const i = find(F.t, t), j = Math.min(i + 1, N - 1), f = j > i ? Math.min(Math.max((t - F.t[i]) / (F.t[j] - F.t[i]), 0), 1) : 0;
  const s = {i}; for (const k in F) { const a = F[k]; if (!Array.isArray(a)) continue;
    const x = a[i], y = a[j]; s[k] = (x == null || y == null) ? (f < 0.5 ? x : y) : x + (y - x) * f; }
  return s;
}
const EV = D.events;
const tSep1 = D.sep.s1, tSep2 = D.sep.s2, tFair = D.sep.fairing;

// ---------- side panel ----------
const gauges = document.getElementById("gauges");
const G = [["h", "altitude, km", v => (v / 1e3).toFixed(1)], ["v", "speed, m/s", v => v.toFixed(0)],
  ["mach", "Mach", v => v == null ? "–" : v.toFixed(2)], ["q", "dyn. pressure, kPa", v => (v / 1e3).toFixed(1)],
  ["thrust", "thrust, kN", v => (v / 1e3).toFixed(0)], ["drag", "drag, kN", v => (v / 1e3).toFixed(1)],
  ["acc", "acceleration, g", v => v.toFixed(2)], ["air_above", "air above, %", v => v >= 0.01 ? (v * 100).toFixed(1) : (v * 100).toPrecision(2)]];
if (D.stability) G.push(["margin", "static margin, cal", v => v == null ? "–" : v.toFixed(2)],
                        ["t_double", "time to double, s", v => v == null ? "–" : v.toFixed(2)]);
gauges.innerHTML = G.map(([k, l]) => `<div data-k="${k}">–<span>${l}</span></div>`).join("");
const layerBox = document.getElementById("layer"), toast = document.getElementById("toast");
let lastLayer = -1, toastT = 0;
function layerAt(h) { let k = 0; D.layers.forEach((L, i) => { if (h >= L.base) k = i; }); return k; }

// atmosphere column: layers, US76 temperature, the rocket's altitude
const col = document.getElementById("column");
const Hs = h => Math.log10(1 + h / 2000) / Math.log10(1 + D.hmax / 2000);
function column(h) {
  const [c, w, hh] = setup(col), top = 10, bot = hh - 26, Y = z => bot - Hs(z) * (bot - top), X0 = 64, X1 = w - 12;
  let lastY = 1e9;
  D.layers.forEach(L => { const y0 = Y(L.base), y1 = Y(Math.min(L.top, D.hmax));
    c.fillStyle = L.color; c.globalAlpha = 0.18; c.fillRect(X0, y1, X1 - X0, y0 - y1); c.globalAlpha = 1;
    c.strokeStyle = TK.grid; c.beginPath(); c.moveTo(X0, y0); c.lineTo(X1, y0); c.stroke();
    if (lastY - y0 > 12) { c.fillStyle = TK.ink2; c.textAlign = "right"; c.fillText((L.base / 1e3).toFixed(0) + " km", X0 - 6, y0 + 4); lastY = y0; }
    if (y0 - y1 > 13) { c.fillStyle = TK.ink; c.textAlign = "left"; c.fillText(L.name, X0 + 6, (y0 + y1) / 2 + 4); } });
  D.markers.forEach(Mk => { const y = Y(Mk.alt); c.setLineDash([4, 3]); c.strokeStyle = TK.ink2; c.beginPath(); c.moveTo(X0, y); c.lineTo(X1, y); c.stroke(); c.setLineDash([]);
    c.fillStyle = TK.ink2; c.textAlign = "right"; c.fillText(Mk.name, X1 - 4, y - 3); });
  // temperature: 150 K at X0+40 .. 1100 K at X1
  const TX = T => X0 + 40 + (T - 150) / (1100 - 150) * (X1 - X0 - 40);
  c.strokeStyle = TK.series[1]; c.lineWidth = 2; c.beginPath();
  D.profile.forEach(([z, T], k) => { if (z > D.hmax) return; k ? c.lineTo(TX(T), Y(z)) : c.moveTo(TX(T), Y(z)); }); c.stroke(); c.lineWidth = 1;
  c.fillStyle = TK.ink2; c.textAlign = "center";
  for (const T of [200, 500, 1000]) c.fillText(T + " K", TX(T), hh - 8);
  // rocket
  const y = Y(Math.max(h, 0)); c.fillStyle = TK.series[0]; c.beginPath(); c.moveTo(X0 + 4, y); c.lineTo(X0 + 18, y - 7); c.lineTo(X0 + 18, y + 7); c.closePath(); c.fill();
  c.strokeStyle = TK.series[0]; c.beginPath(); c.moveTo(X0 + 18, y); c.lineTo(X1, y); c.stroke();
}
document.getElementById("lg-col").innerHTML = `<span><i style="background:${TK.series[1]}"></i>temperature (US76)</span><span><i style="background:${TK.series[0]}"></i>Rockot</span>`;

// time charts with several series
const CH = [
  {id: "c-h", series: [["h", "altitude", 1e-3]], unit: "km"},
  {id: "c-v", series: [["v", "speed", 1]], unit: "m/s"},
  {id: "c-q", series: [["q", "dynamic pressure", 1e-3]], unit: "kPa"},
  {id: "c-f", series: [["thrust", "thrust", 1e-3], ["drag", "drag", 1e-3], ["weight", "weight", 1e-3]], unit: "kN"},
  {id: "c-l", series: [["loss_gravity", "gravity", 1], ["loss_drag", "drag", 1], ["loss_pressure", "back pressure", 1]], unit: "m/s"},
];
if (D.stability) CH.push(
  {id: "c-x", series: [["x_cg", "centre of gravity", 1], ["x_cp", "centre of pressure", 1]], unit: "m"},
  {id: "c-m", series: [["margin", "static margin", 1]], unit: "cal"},
  {id: "c-d", series: [["t_double", "time to double", 1]], unit: "s"},
  {id: "c-g", series: [["gimbal_per_aoa", "gimbal per degree of AoA", 1]], unit: "deg/deg"});
else document.getElementById("stab").remove();
CH.forEach(ch => legend(document.getElementById("lg-" + ch.id), ch.series.map(([k, n], i) => ({name: n, color: TK.series[i]}))));
const tView = () => +document.getElementById("span").value;
function chart(ch, t) {
  const cv = document.getElementById(ch.id), [c, w, h] = setup(cv), t1 = Math.min(tEnd, tView());
  let ymin = 0, ymax = -1e99;
  for (const [k, , s] of ch.series) for (let i = 0; i < N && F.t[i] <= t1; i++) { const v = F[k][i];
    if (v != null) { ymax = Math.max(ymax, v * s); ymin = Math.min(ymin, v * s); } }
  if (ymax < 0) ymax = 0;
  const [y0, y1, yst] = axisRange(ymin, Math.max(ymax, 1e-9), 4, true), [, x1, xst] = axisRange(0, t1, 4, true);
  const L = 44, B = 18, T = 6, W = w - L - 10, H = h - B - T, X = x => L + x / x1 * W, Y = y => T + H - (y - y0) / (y1 - y0) * H;
  c.strokeStyle = TK.grid; c.fillStyle = TK.ink2;
  for (let y = y0; y <= y1 + yst / 2; y += yst) { c.beginPath(); c.moveTo(L, Y(y)); c.lineTo(L + W, Y(y)); c.stroke(); c.textAlign = "right"; c.fillText(fmtN(y, yst), L - 6, Y(y) + 4); }
  for (let x = 0; x <= x1 + xst / 2; x += xst) { c.textAlign = x + xst > x1 + xst / 2 ? "right" : "center"; c.fillText(fmtN(x, xst) + (x + xst > x1 + xst / 2 ? " s" : ""), X(x), h - 3); }
  ch.series.forEach(([k, , s], n) => { const a = F[k];
    for (const [full, alpha] of [[true, 0.25], [false, 1]]) { c.strokeStyle = TK.series[n]; c.globalAlpha = alpha; c.lineWidth = 2; c.beginPath(); let pen = false;
      for (let i = 0; i < N && F.t[i] <= t1; i++) { if (!full && F.t[i] > t) break; const v = a[i]; if (v == null) { pen = false; continue; }
        pen ? c.lineTo(X(F.t[i]), Y(v * s)) : c.moveTo(X(F.t[i]), Y(v * s)); pen = true; } c.stroke(); }
    c.globalAlpha = 1; });
  c.strokeStyle = TK.axis; c.beginPath(); c.moveTo(X(Math.min(t, t1)), T); c.lineTo(X(Math.min(t, t1)), T + H); c.stroke();
}

// events list
const evList = document.getElementById("events");
const evItems = EV.map(([n, te]) => { const li = document.createElement("li"); li.innerHTML = `${n}<b>${te.toFixed(0)} s</b>`;
  li.onclick = () => { t = te - 2; scrub.value = t / tEnd * 1000; }; evList.append(li); return [te, li]; });

// ---------- cameras ----------
const camSel = document.getElementById("cam");
function setCam() {
  const m = camSel.value, c = new THREE.Vector3(0, 0, 0);
  controls.target.copy(c);
  const off = {close: [26, 10, 30], side: [0, 6, 70], below: [10, -40, 18], wide: [260, 90, 380], nose: [5, 8, 6]}[m];
  camera.position.copy(c).add(new THREE.Vector3(...off));
  if (m === "nose") controls.target.set(0, L_TOT / 2, 0);
  controls.autoRotate = m === "close"; controls.autoRotateSpeed = 0.35;
}
camSel.onchange = setCam;

// ---------- time ----------
let t = 0, playing = false, last = performance.now();
const play = document.getElementById("play"), rate = document.getElementById("rate"),
  scrub = document.getElementById("scrub"), clock = document.getElementById("clock");
for (const r of D.rates) rate.insertAdjacentHTML("beforeend", `<option value="${r}"${r === D.rate ? " selected" : ""}>×${r}</option>`);
play.onclick = () => { if (t >= tEnd) t = 0; playing = !playing; play.textContent = playing ? "Pause" : "Play"; };
scrub.oninput = () => { t = scrub.value / 1000 * tEnd; };
addEventListener("keydown", e => { if (e.code === "Space" && e.target === document.body) { e.preventDefault(); play.click(); } });
function resize() { const r = host.getBoundingClientRect(); renderer.setSize(r.width, r.height, false);
  renderer.domElement.style.width = r.width + "px"; renderer.domElement.style.height = r.height + "px";
  camera.aspect = r.width / r.height; camera.updateProjectionMatrix(); }
addEventListener("resize", resize); resize(); setCam();
const fmtT = s => `${String(Math.floor(s / 60)).padStart(2, "0")}:${(s % 60).toFixed(1).padStart(4, "0")}`;
const tmp = new THREE.Vector3();

function sepOffset(part, s, dt, dirSide) {
  // separated part, in the rocket frame: falls back along the body axis as the upper stage
  // accelerates away (1/2 a dt^2, a from the flight data), drifts sideways and tumbles slowly
  const back = 0.5 * Math.max(s.acc, 0.5) * 9.81 * dt * dt + 2 * dt;
  part.position.x = dirSide * 1.2 * dt; part.position.z = 0.4 * dt * dirSide;
  part.rotation.z = dirSide * 0.02 * dt; part.rotation.x = 0.01 * dt;
  return back;
}

function frame(now) {
  const dtw = Math.min((now - last) / 1000, 0.25); last = now;
  if (playing) { t = Math.min(t + dtw * rate.value, tEnd); if (t >= tEnd) { playing = false; play.textContent = "Play"; } scrub.value = t / tEnd * 1000; }
  const s = sample(t), h = s.h, hk = h / 1e3;
  // attitude: body axis along the flight path (vertical rise, then the gravity turn)
  const pitch = Math.max(Math.min(s.fpa, 90), -10) * rad;
  rocket.rotation.set(0, 0, -(Math.PI / 2 - pitch));
  const want = t < tSep1 ? L_TOT / 2 : t < tSep2 ? (yS2 + M.inter + L_TOT) / 2 : yS2 + M.inter + M.s2 + M.trans + (t < tFair ? M.fairing : M.breeze + 3) / 2;
  centre += (want - centre) * Math.min(1, dtw * 1.5); stack.position.y = -centre;
  // staging
  const y1 = parts.s1.userData.y0 ?? (parts.s1.userData.y0 = parts.s1.position.y);
  if (t > tSep1) { const dt = t - tSep1, back = sepOffset(parts.s1, s, dt, 1); parts.s1.position.y = y1 - back; parts.s1.visible = back < 4000; }
  else { parts.s1.position.set(0, y1, 0); parts.s1.rotation.set(0, 0, 0); parts.s1.visible = true; }
  const y2 = parts.s2.userData.y0 ?? (parts.s2.userData.y0 = parts.s2.position.y);
  if (t > tSep2) { const dt = t - tSep2, back = sepOffset(parts.s2, s, dt, -1); parts.s2.position.y = y2 - back; parts.s2.visible = back < 4000; }
  else { parts.s2.position.set(0, y2, 0); parts.s2.rotation.set(0, 0, 0); parts.s2.visible = true; }
  parts.fair.forEach((f, k) => { const sd = k ? -1 : 1;
    if (t > tFair) { const dt = t - tFair; f.position.set(sd * (1.5 * dt + 0.05 * dt * dt), -0.6 * dt * dt, 0); f.rotation.set(0, 0, -sd * Math.min(0.25 * dt, 2.5)); f.visible = dt < 120; }
    else { f.position.set(0, 0, 0); f.rotation.set(0, 0, 0); f.visible = true; } });
  // engines: which plume burns, and how wide (ambient pressure)
  const burning = s.thrust > 1, stageNow = t < tSep1 ? 1 : t < tSep2 ? 2 : 3;
  const pr = Math.max(s.p, 1) / 101325, grow = Math.min(Math.pow(1 / pr, 0.15), 4);   // under-expanded plume spreads
  for (const [pl, k] of [[pl1, 1], [pl2, 2], [pl3, 3]]) { const on = burning && stageNow === k; pl.visible = on;
    if (on) { const flick = 0.92 + 0.08 * Math.sin(now * 0.05 + k); pl.env.scale.set(grow, 1 + 0.4 * (grow - 1), grow); pl.core.scale.set(1, flick, 1);
      pl.env.material.opacity = 0.14 / grow; } }
  // the air: streaks (density), Mach cone, vapour cone, nose heating
  const dens = Math.pow(Math.max(s.rho, 1e-9) / 1.225, 0.33), vrel = s.v;
  streaks.material.opacity = Math.min(0.5 * dens, 0.5) * Math.min(1, s.v / 250);   // seen once it moves fast
  windPhase += dtw * rate.value * vrel;
  const dir = new THREE.Vector3(Math.cos(pitch), Math.sin(pitch), 0), len = Math.min(0.02 * vrel + 0.5, 9), box = 60;
  const mid = tmp.set(0, 0, 0), e1x = -dir.y, e1y = dir.x;   // e1 across the flow, e2 = z
  for (let i = 0; i < NS; i++) { const [a, b, cc] = seeds[i];
    const along = ((a * box - windPhase) % (2 * box) + 3 * box) % (2 * box) - box;   // air moves past, wraps around
    const px = mid.x + dir.x * along + e1x * b * 25, py = mid.y + dir.y * along + e1y * b * 25, pz = cc * 25;
    streakPos.set([px, py, pz, px + dir.x * len, py + dir.y * len, pz], 6 * i); }
  streakGeo.attributes.position.needsUpdate = true;
  const nose = new THREE.Vector3(0, L_TOT - centre, 0).applyEuler(rocket.rotation);
  const qn = Math.min(s.q / 30000, 1), mach = s.mach ?? 0;
  if (mach > 1.05 && qn > 0.01) { const mu = Math.asin(1 / mach), Lc = 45, rc = Lc * Math.tan(mu);
    shock.visible = true; shock.scale.set(rc, Lc, rc); shock.position.copy(nose).addScaledVector(dir, -Lc / 2);
    shock.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir); shock.material.opacity = 0.09 * qn; } else shock.visible = false;
  const vap = Math.max(0, 1 - Math.abs(mach - 1) / 0.12) * Math.min(1, s.rho / 0.5);
  vapour.visible = vap > 0.01; vapour.material.opacity = 0.45 * vap;
  vapour.position.copy(new THREE.Vector3(0, yS2 + M.inter + M.s2 + M.trans + 0.5 - centre, 0).applyEuler(rocket.rotation)); vapour.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
  if (D.stability) {
    cgMark.visible = s.x_cg != null; if (s.x_cg != null) cgMark.position.set(0, L_TOT - s.x_cg, R + 0.4);
    cpMark.visible = s.x_cp != null; if (s.x_cp != null) cpMark.position.set(0, L_TOT - s.x_cp, R + 0.4);
  } else { cgMark.visible = cpMark.visible = false; }
  const heat = Math.min(s.heat / D.heatmax, 1);
  tip.visible = heat > 0.02 && t < tFair; tip.material.opacity = 0.9 * Math.pow(heat, 1.5);
  // world around: pad, layer sheets, sky
  pad.position.set(-s.downrange, -h - centre, 0); pad.visible = h < 6000;
  sheets.forEach(m => { const dz = m.userData.alt - h; m.position.set(0, dz, 0); const a = Math.max(0, 1 - Math.abs(dz) / 1500); m.material.opacity = 0.32 * a; m.visible = a > 0; });
  skyMat.uniforms.uAlt.value = hk; skyMat.uniforms.uCam.value.copy(camera.position).multiplyScalar(1e-3); skyMat.uniforms.uRot.value = s.downrange / 6371e3;
  sky.position.copy(camera.position); sky.scale.setScalar(1e5);
  stars.position.copy(camera.position); stars.material.opacity = Math.min(Math.max((hk - 25) / 45, 0), 1);
  hemi.intensity = 0.25 + 0.75 * Math.min(1, Math.pow(Math.max(s.rho, 1e-9) / 1.225, 0.25));
  // panel
  for (const [k, , f] of G) gauges.querySelector(`[data-k="${k}"]`).firstChild.nodeValue = (k === "mach" && (s.mach == null || hk > 100)) ? "–" : f(s[k]);
  const li = layerAt(h);
  if (li !== lastLayer) { const L = D.layers[li]; layerBox.innerHTML = `<b>${L.name} · ${(L.base / 1e3).toFixed(0)}–${(L.top / 1e3).toFixed(0)} km</b>${L.text}`;
    layerBox.style.borderColor = L.color; if (lastLayer >= 0 && playing) { toast.textContent = li > lastLayer ? `Into the ${L.name.toLowerCase()}` : L.name; toastT = now; } lastLayer = li; }
  toast.style.opacity = now - toastT < 2200 ? 1 : 0;
  evItems.forEach(([te, li2]) => li2.classList.toggle("done", te <= t));
  clock.firstChild.nodeValue = `T+${fmtT(t)}`; clock.lastChild.textContent = `×${rate.value}`;
  controls.update(); renderer.render(scene, camera);
  column(h); CH.forEach(ch => chart(ch, t));
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
"""


def ascent_closeup_html(res, vehicle, *, title: str, subtitle: str = "", nav: str = "",
                        t_end: float | None = None, model: dict | None = None, layout=None,
                        theme: str = "dark", rates=(1, 2, 5, 10, 20, 50), rate: float = 5) -> str:
    """The close-up page for one ascent. ``model``: Rockot-like dimensions in m (see ROCKOT);
    ``layout``: an ``aerosim.flight.stability.Layout`` for the CG / CP and stability charts."""
    d = ascent_physics(res, vehicle, t_end)

    idx = _thin(d["t"].size, 2500)
    keys = {"t": 2, "h": 1, "v": 2, "mach": 3, "q": 1, "drag": 0, "thrust": 0, "weight": 0, "fpa": 3,
            "rho": None, "p": 2, "heat": 0, "air_above": None, "acc": 3, "downrange": 1}
    f = {}
    for k, nd in keys.items():
        a = np.asarray(d[k], float)[idx]
        f[k] = [None if not np.isfinite(x) else (float(f"{x:.4g}") if nd is None else round(float(x), nd)) for x in a]
    for k, a in d["losses"].items():
        f[f"loss_{k}"] = _r(np.asarray(a)[idx], 1)
    stab = None
    if layout is not None:
        from aerosim.flight.stability import stability
        stab = stability(res, vehicle, layout, t_end)
        for k, nd in (("x_cg", 2), ("x_cp", 2), ("margin", 2), ("t_double", 3), ("gimbal_per_aoa", 3)):
            a = np.asarray(stab[k], float)[idx]
            f[k] = [round(float(x), nd) if np.isfinite(x) else None for x in a]
    ev = _events(res, d)
    names = {e[0]: e[1] for e in ev}
    stages = [s.name for s in vehicle.stages]
    sep = {"s1": names.get(f"{stages[0]} burnout and separation", 1e9),
           "s2": names.get(f"{stages[1]} burnout and separation", 1e9) if len(stages) > 2 else 1e9,
           "fairing": names.get("Fairing jettison", 1e9)}
    data = {"f": f, "events": ev, "sep": sep, "model": model or ROCKOT,
            "layers": [{"name": n, "base": b, "top": tp, "color": c, "text": tx} for n, b, tp, c, tx in LAYERS],
            "markers": [{"name": n, "alt": a} for n, a in MARKERS], "profile": _temperature_profile(),
            "hmax": float(max(700e3, d["h"].max() * 1.1)), "heatmax": float(max(d["heat"].max(), 1.0)),
            "rates": list(rates), "rate": rate, "stability": stab is not None}
    loss = {k: float(v[-1]) for k, v in d["losses"].items()}
    k_mq = int(np.argmax(d["q"]))
    stab_note = ""
    if stab is not None:
        mg, td = stab["margin"], stab["t_double"]
        k_t = int(np.argmin(np.where(np.isfinite(td), td, np.inf)))
        s1 = stab["t"] <= res.events.get(f"{vehicle.stages[0].name}_burnout", np.inf)
        stab_note = (f"Static margin {np.nanmin(mg):.1f} to {np.nanmax(mg):.1f} calibers: the CP is ahead of "
                     f"the CG throughout, so the airframe alone is unstable. The fastest divergence is at "
                     f"{stab['t'][k_t]:.0f} s, doubling every {td[k_t]:.2f} s. The CG moves "
                     f"{stab['x_cg'][s1].max() - stab['x_cg'][s1].min():.1f} m over the stage 1 burn.")
    summary = (f"Max-Q {d['q'][k_mq] / 1e3:.0f} kPa at {d['h'][k_mq] / 1e3:.1f} km, {d['t'][k_mq]:.0f} s. "
               f"Losses to the end of the replay: gravity {loss['gravity']:.0f} m/s, drag "
               f"{loss['drag']:.0f} m/s, back pressure {loss['pressure']:.0f} m/s.")
    body = f"""<div id="app">
 <div id="scene"><div id="hud"><h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p>{_CGCP if layout is not None else ""}</div>
  <div id="layer"></div><div id="toast"></div><div id="gauges"></div>
  <div id="clock">T+0<small>&nbsp;</small></div></div>
 <div id="side">
  <h3>Atmosphere</h3><div class="legend" id="lg-col"></div><canvas id="column" aria-label="Atmospheric layers, temperature and the rocket's altitude"></canvas>
  <p class="note">Altitude on a log scale, so the thin lower layers stay visible. Temperature: US Standard Atmosphere 1976.</p>
  <h3>Altitude</h3><div class="legend" id="lg-c-h"></div><canvas class="chart t" id="c-h"></canvas>
  <h3>Speed relative to the air</h3><div class="legend" id="lg-c-v"></div><canvas class="chart t" id="c-v"></canvas>
  <h3>Dynamic pressure ½ρv²</h3><div class="legend" id="lg-c-q"></div><canvas class="chart t" id="c-q"></canvas>
  <p class="note">Speed rises while density falls: their product peaks at max-Q, the largest aerodynamic load.</p>
  <h3>Forces</h3><div class="legend" id="lg-c-f"></div><canvas class="chart t" id="c-f"></canvas>
  <p class="note">Thrust grows as the air thins (less back pressure on the nozzles); drag is ½ρv²·C<sub>D</sub>·A.</p>
  <h3>Where the Δv goes</h3><div class="legend" id="lg-c-l"></div><canvas class="chart t" id="c-l"></canvas>
  <p class="note">{html.escape(summary)}</p>
  <div id="stab">
  <h3>Pitch stability</h3>
  <p class="note">Rockot has no fins: its lift acts near the nose (Barrowman slender-body method), well
  ahead of the centre of gravity, so the bare airframe is unstable and the engines must steer it
  straight. The CG moves as the tanks drain from the top and as stages and the fairing drop.</p>
  <div class="legend" id="lg-c-x"></div><canvas class="chart t" id="c-x"></canvas>
  <p class="note">Positions measured aft from the nose tip.</p>
  <h3>Static margin</h3><div class="legend" id="lg-c-m"></div><canvas class="chart t" id="c-m"></canvas>
  <p class="note">(x<sub>CP</sub> &minus; x<sub>CG</sub>) / diameter, in calibers; negative is unstable. Shown where q &gt; 200 Pa.</p>
  <h3>Pitch divergence: time to double</h3><div class="legend" id="lg-c-d"></div><canvas class="chart t" id="c-d"></canvas>
  <p class="note">How fast an uncorrected pitch disturbance grows: aerodynamic stiffness against pitch inertia,
  with aerodynamic and exhaust-jet damping. Shortest near max-Q, where the steering must be fastest.</p>
  <h3>Steering needed</h3><div class="legend" id="lg-c-g"></div><canvas class="chart t" id="c-g"></canvas>
  <p class="note">Engine deflection that holds 1&deg; of angle of attack (moment balance). {html.escape(stab_note)}</p>
  </div>
  <h3>Events</h3><ul id="events" class="ev"></ul>
  <p class="note">Rockot / Breeze-KM model drawn from published dimensions (29.15 m, 2.5 m stages,
  2.62 m fairing); colours and details approximate. Flight: the first-principles model on the
  Rockot page. Earth and sky: a single-scattering atmosphere shader; the ground is procedural.</p>
 </div>
 <div id="bar"><button id="play" aria-label="Play or pause">Play</button>
  <select id="rate" aria-label="Time scale"></select>
  <input id="scrub" type="range" min="0" max="1000" value="0" aria-label="Flight time">
  <select id="cam" aria-label="Camera"><option value="close">Close (orbiting)</option><option value="side">Side</option>
   <option value="below">From below</option><option value="nose">Nose</option><option value="wide">Wide</option></select>
  <select id="span" aria-label="Chart time span"><option value="320">Charts: first 320 s</option><option value="{d['t'][-1]:.0f}">Charts: whole replay</option></select>
 </div>
</div>
{theme_tag(theme)}{data_tag(data)}<script type="importmap">{_importmap()}</script>
<script type="module">{CHART_JS}{_JS}</script>"""
    css = _CSS + "#events{list-style:none;margin:0;padding:0;font-size:12.5px}#events li{display:flex;gap:8px;padding:3px 0;color:var(--muted);border-bottom:1px solid var(--grid);cursor:pointer}#events li.done{color:var(--ink)}#events li b{font-weight:500;margin-left:auto;color:var(--ink2)}"
    return themed(f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title><style>{CSS}{css}</style></head><body>
{nav}
{body}
</body></html>""", theme)


# Rockot / Breeze-KM dimensions, m. Overall 29.15 m, 2.5 m core, 2.62 m x 6.7 m fairing (User's
# Guide); the stage split, nozzle sizes (from the engines' exit areas) and the transition are
# estimates that keep the overall length.
ROCKOT = {"d": 2.5, "fairing_d": 2.62, "nozzle1": 1.1, "s1": 15.8, "inter": 0.5, "s2": 3.9,
          "trans": 1.15, "fairing": 6.7, "nose": 3.7, "breeze": 2.65,
          "nozzle1_re": 0.36, "nozzle2_re": 0.42, "nozzle3_re": 0.22}
