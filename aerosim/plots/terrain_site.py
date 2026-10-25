"""Low-level flight over real terrain as offline HTML pages (three.js), in the style of the
results site (``site.py``).

    terrain_replay_html(grid, models, title=..., labels=..., attribution=...)   # one region
    terrain_hub_html(regions, ...)                                              # index page

``grid`` is a TerrainGrid (already at display resolution); its heights go into the page
as int16 in steps of 0.25 m with a bit-packed water mask, and the browser builds the mesh:
vertex colours from a height ramp blended to rock on steep slopes, water where the model is
flat. ``models``: [{"name", "flight": FlightResult from ``terrain.flight.fly_route``,
"track": Track, "profile": Profile}]; positions are in the grid frame (x east, y north of
the grid centre, m), headings from grid north.
"""

from __future__ import annotations

import base64
import html
import io

import numpy as np

from aerosim.plots.site import _r, _thin
from aerosim.plots.style import THEMES
from aerosim.plots.web import CHART_JS, CSS, data_tag, importmap_tag, nav, theme_tag, themed

RAMPS = {   # (height m, colour): lowland -> rock -> snow
    "norway": [(0, "#4f6b3c"), (150, "#64804a"), (450, "#8c8f63"), (750, "#9a8b72"),
               (1000, "#a3a09a"), (1250, "#d9dbdc"), (1500, "#f4f6f7")],
    "arctic": [(0, "#5e6152"), (120, "#77776a"), (350, "#8d8b82"), (600, "#b3b5b3"),
               (850, "#dfe3e6"), (1200, "#f7f9fa")],
}
ROCK, WATER = "#6f6a64", "#2c5672"
QUANT = 0.25


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


AIRCRAFT_CREDIT = ('F-16 model: "F-16 Fighter Jet" by iedalton (Sketchfab), CC BY 4.0, '
                   "scaled to 15.06 m")


def _aircraft() -> dict:
    """The F-16 GLB (``data/f16``, see its ATTRIBUTION.md) for the page, with the nozzle and fin
    tip read from its vertices (for the exhaust glow and the route-coloured fin light)."""
    import json
    import struct
    from importlib import resources
    glb = resources.files("aerosim.plots").joinpath("data/f16/f16.glb").read_bytes()
    n = struct.unpack_from("<I", glb, 12)[0]
    doc = json.loads(glb[20:20 + n])
    binary = glb[20 + n + 8:]
    pts = []
    for p in doc["meshes"][0]["primitives"]:
        a = doc["accessors"][p["attributes"]["POSITION"]]
        v = doc["bufferViews"][a["bufferView"]]
        pts.append(np.frombuffer(binary, "<f4", a["count"] * 3, v["byteOffset"]).reshape(-1, 3))
    P = np.vstack(pts)
    tail = P[P[:, 0] < P[:, 0].min() + 0.3]
    fin = P[np.argmax(P[:, 1])]
    return {"glb": base64.b64encode(glb).decode("ascii"),
            "nozzle": [round(float(P[:, 0].min()), 2), round(float(tail[:, 1].mean()), 2)],
            "fin": [round(float(fin[0]), 2), round(float(fin[1]), 2), 0.0]}


def hillshade_png(grid, ramp: str, width: int = 520, water=None) -> str:
    """Shaded-relief map of the grid as a PNG data URI (north up)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource, LinearSegmentedColormap
    k = max(1, grid.x.size // width)
    g = grid.downsample(k)
    z = g.z.astype(float)
    hs, cs = zip(*RAMPS[ramp])
    top = hs[-1]
    cmap = LinearSegmentedColormap.from_list("r", [(h / top, c) for h, c in zip(hs, cs)])
    ls = LightSource(azdeg=315, altdeg=40)
    rgb = ls.shade(np.clip(z, 0, top), cmap=cmap, vert_exag=2.0, dx=g.spacing, dy=g.spacing,
                   blend_mode="soft", vmin=0, vmax=top)
    w = g.water if water is None else water[::k, ::k]
    rgb[w] = np.r_[[int(WATER[i:i + 2], 16) / 255 for i in (1, 3, 5)], 1.0]
    buf = io.BytesIO()
    plt.imsave(buf, rgb[::-1], format="png")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _pack_flight(m: dict, color: str, max_points: int = 5000, prof_points: int = 1500) -> dict:
    f, tr, pr = m["flight"], m["track"], m["profile"]
    me = f.meta
    idx = _thin(f.time.size, max_points)
    p = f.position[idx]
    ti = _thin(tr.x.size, 1500)
    pi = _thin(pr.s.size, prof_points)
    fi = _thin(f.time.size, prof_points)
    return {
        "name": m["name"], "color": color, "t": _r(f.time[idx], 2),
        "x": _r(p[:, 0], 1), "y": _r(p[:, 1], 1), "z": _r(p[:, 2], 1),
        "att": {"pitch": _r(me["pitch_deg"][idx], 1), "roll": _r(me["roll_deg"][idx], 1),
                "heading": _r(me["heading_deg"][idx], 1)},
        "ch": {"agl": _r(me["agl"][idx], 1), "alt": _r(p[:, 2], 1), "mach": _r(me["mach"][idx], 3),
               "nz": _r(me["nz"][idx], 2), "roll": _r(me["roll_deg"][idx], 1),
               "dist": _r(me["s"][idx] / 1e3, 2), "gnd": _r(me["terrain"][idx], 1),
               "kt": _r(me["airspeed"][idx] / 0.514444, 0), "xt": _r(me["cross_track"][idx], 1)},
        "track": {"x": _r(tr.x[ti], 1), "y": _r(tr.y[ti], 1)},
        "prof": {"s": _r(pr.s[pi] / 1e3, 3), "terrain": _r(pr.terrain[pi], 1),
                 "floor": _r(pr.floor[pi], 1), "ref": _r(pr.altitude[pi], 1),
                 "fs": _r(me["s"][fi] / 1e3, 3), "fz": _r(f.position[fi, 2], 1)},
        "length": tr.length / 1e3,
    }


def flight_stats(m: dict) -> dict[str, str]:
    f, me = m["flight"], m["flight"].meta
    err = f.position[:, 2] - me["h_ref"]
    return {"route, km": f"{me['route_length'] / 1e3:.1f}", "time, s": f"{f.time[-1]:.0f}",
            "lowest above ground, m": f"{me['min_agl']:.1f}",
            "mean above ground, m": f"{np.mean(me['agl']):.0f}",
            "altitude error (rms), m": f"{np.sqrt(np.mean(err ** 2)):.1f}",
            "cross-track (max), m": f"{np.abs(me['cross_track']).max():.0f}",
            "load factor, g": f"{me['nz'].min():.1f} … {me['nz'].max():.1f}".replace("-0.0", "0.0"),
            "bank (max), °": f"{np.abs(me['roll_deg']).max():.0f}",
            "fuel used, kg": f"{me['fuel_used']:.0f}",
            "outcome": me["status"]}


def _subnav(links, here: str) -> str:
    """A row of links between sibling pages (the regions), the current one in bold."""
    if not links:
        return ""
    items = " · ".join(f"<b>{html.escape(n)}</b>" if h == here else f'<a href="{h}">{html.escape(n)}</a>'
                       for n, h in links)
    return f'<p class="note" style="margin-top:0">{items}</p>'


_CSS = r"""
html,body{height:100%}body{display:flex;flex-direction:column}
#app{flex:1;min-height:0;display:grid;grid-template-columns:minmax(0,1fr) 410px;
grid-template-rows:minmax(0,1fr) auto}
#scene{position:relative;min-height:0;overflow:hidden}#scene canvas{display:block}
#hud{position:absolute;left:20px;top:16px;pointer-events:none;max-width:70%;
text-shadow:0 1px 3px var(--page)}
#hud h1{margin:0;font-size:20px;font-weight:600;letter-spacing:.2px}
#hud p{margin:2px 0 0;color:var(--ink2);font-size:13px}
#clock{position:absolute;left:20px;bottom:30px;font:600 28px/1 Consolas,"SF Mono",monospace;
pointer-events:none;font-variant-numeric:tabular-nums;text-shadow:0 1px 3px var(--page)}
#clock small{display:block;font:500 12px/1.6 "Segoe UI",system-ui,sans-serif;color:var(--ink2)}
#attr{position:absolute;left:20px;right:20px;bottom:8px;font-size:10.5px;color:var(--ink2);
pointer-events:none;text-shadow:0 1px 2px var(--page);line-height:1.3}
#labels{position:absolute;inset:0;pointer-events:none;overflow:hidden}
#labels div{position:absolute;transform:translate(-50%,-100%);font-size:12px;font-weight:600;
color:#fff;text-shadow:0 0 3px #000,0 0 6px #000;white-space:nowrap}
#labels div.sea{font-style:italic;font-weight:500;color:#cfe3f2}
#labels div.pk::after{content:"";display:block;width:6px;height:6px;margin:2px auto 0;
border-radius:50%;background:#fff;box-shadow:0 0 3px #000}
#side{grid-row:1/3;grid-column:2;overflow-y:auto;background:var(--surface);
border-left:1px solid var(--grid);padding:16px 18px 20px}
canvas.chart.t{height:150px}#mini{width:100%;display:block;border-radius:6px;cursor:crosshair}
#bar{grid-column:1;display:flex;align-items:center;gap:12px;padding:10px 20px;
background:var(--surface);border-top:1px solid var(--grid);flex-wrap:wrap}
#play{min-width:76px;font-weight:600}#scrub{flex:1;min-width:160px;accent-color:var(--accent)}
.note{font-size:12px;color:var(--ink2);margin:6px 0}
@media (max-width:900px){#app{grid-template-columns:1fr;grid-template-rows:62vh auto auto}
#side{grid-row:3;grid-column:1;border-left:0;border-top:1px solid var(--grid)}#bar{grid-row:2}
#clock{font-size:20px}}
"""

_JS = r"""
import * as THREE from "three";
import {OrbitControls} from "three/addons/controls/OrbitControls.js";
import {GLTFLoader} from "three/addons/loaders/GLTFLoader.js";
const D = JSON.parse(document.getElementById("data").textContent), TK = TKC;
const M = D.models, tEnd = Math.max(...M.map(m => m.t[m.t.length - 1]));
const rad = Math.PI / 180, b64 = s => Uint8Array.from(atob(s), c => c.charCodeAt(0));

// ---------- terrain mesh (scene: x = east m, y = up m, z = -north m) ----------
const T = D.terrain, NX = T.nx, NY = T.ny, NV = NX * NY;
const HQ = new Int16Array(b64(T.h).buffer), WB = b64(T.water);
const water = k => (WB[k >> 3] >> (k & 7)) & 1;
const tpos = new Float32Array(NV * 3), tcol = new Float32Array(NV * 3);
for (let j = 0, k = 0; j < NY; j++) for (let i = 0; i < NX; i++, k++) {
  tpos[3 * k] = T.x0 + i * T.dx; tpos[3 * k + 1] = HQ[k] * T.q; tpos[3 * k + 2] = -(T.y0 + j * T.dx); }
const idx = new Uint32Array((NX - 1) * (NY - 1) * 6);
for (let j = 0, n = 0; j < NY - 1; j++) for (let i = 0; i < NX - 1; i++) {
  const a = j * NX + i, b = a + 1, c = a + NX, d = c + 1;
  idx[n++] = a; idx[n++] = b; idx[n++] = c; idx[n++] = b; idx[n++] = d; idx[n++] = c; }
const tgeo = new THREE.BufferGeometry();
tgeo.setAttribute("position", new THREE.BufferAttribute(tpos, 3)); tgeo.setIndex(new THREE.BufferAttribute(idx, 1));
tgeo.computeVertexNormals();
const ramp = T.ramp.map(([h, c]) => [h, new THREE.Color(c)]), rock = new THREE.Color(T.rock), wcol = new THREE.Color(T.waterColor);
const nrm = tgeo.attributes.normal.array, cc = new THREE.Color();
for (let k = 0; k < NV; k++) {
  const h = HQ[k] * T.q;
  if (water(k)) cc.copy(wcol);
  else {
    let r = ramp.length - 1; for (let q = 1; q < ramp.length; q++) if (h < ramp[q][0]) { r = q; break; }
    const [h0, c0] = ramp[r - 1] || ramp[0], [h1, c1] = ramp[r];
    cc.copy(c0).lerp(c1, Math.min(Math.max((h - h0) / Math.max(h1 - h0, 1), 0), 1));
    const ny = nrm[3 * k + 1], steep = Math.min(Math.max((0.86 - ny) / 0.22, 0), 1);
    if (h < ramp[ramp.length - 2][0]) cc.lerp(rock, steep * 0.8);
  }
  tcol[3 * k] = cc.r; tcol[3 * k + 1] = cc.g; tcol[3 * k + 2] = cc.b;
}
tgeo.setAttribute("color", new THREE.BufferAttribute(tcol, 3));
const world = new THREE.Group();          // terrain and lines: scale.y = height exaggeration
world.add(new THREE.Mesh(tgeo, new THREE.MeshLambertMaterial({vertexColors: true})));

const host = document.getElementById("scene");
const renderer = new THREE.WebGLRenderer({antialias: true, logarithmicDepthBuffer: true});
renderer.setPixelRatio(Math.min(devicePixelRatio, 2)); host.prepend(renderer.domElement);
const scene = new THREE.Scene(), sky = new THREE.Color(TK.dark ? "#1b2633" : "#cfdbe6");
scene.background = sky;
const span = Math.hypot(T.dx * NX, T.dx * NY);
scene.fog = new THREE.Fog(sky, span * 0.35, span * 1.4);
scene.add(world, new THREE.HemisphereLight(0xdfe9f3, 0x3a3328, 0.9));
const sun = new THREE.DirectionalLight(0xffffff, 1.6); sun.position.set(-1, 0.75, -1); scene.add(sun);
const camera = new THREE.PerspectiveCamera(55, 1, 2, span * 3);
const controls = new OrbitControls(camera, renderer.domElement); controls.enableDamping = true;
controls.maxPolarAngle = Math.PI * 0.495;

// ---------- aircraft: the glTF F-16 (true size, nose +x, origin near the CG) ----------
let F16 = null;
if (D.aircraft) try { F16 = (await new GLTFLoader().parseAsync(b64(D.aircraft.glb).buffer, "")).scene; }
  catch (e) { console.warn("F-16 glTF failed to load, using the simple shape", e); }
function f16(color) {
  if (!F16) return simpleF16(color);
  const g = new THREE.Group(); g.add(F16.clone(true));
  const glow = new THREE.Mesh(new THREE.ConeGeometry(0.42, 2.4, 12, 1, true), new THREE.MeshBasicMaterial({color: 0xffb347, transparent: true, opacity: 0.8}));
  glow.rotation.z = Math.PI / 2; glow.position.set(D.aircraft.nozzle[0] - 1.2, D.aircraft.nozzle[1], 0); g.add(glow); g.glow = glow;
  const tag = new THREE.Mesh(new THREE.SphereGeometry(0.25, 12, 8), new THREE.MeshBasicMaterial({color}));   // route colour:
  tag.position.fromArray(D.aircraft.fin); g.add(tag);                                                       // a light on the fin tip
  return g;
}
// fallback: an F-16-like shape from primitives
function simpleF16(color) {
  const g = new THREE.Group(), mat = new THREE.MeshPhongMaterial({color: 0x8d949b, shininess: 40, side: THREE.DoubleSide});
  const acc = new THREE.MeshPhongMaterial({color, shininess: 40, side: THREE.DoubleSide});
  const fus = new THREE.Mesh(new THREE.CylinderGeometry(0.55, 0.7, 11, 16), mat); fus.rotation.z = Math.PI / 2; fus.position.x = 0.5; g.add(fus);
  const nose = new THREE.Mesh(new THREE.ConeGeometry(0.55, 4, 16), mat); nose.rotation.z = -Math.PI / 2; nose.position.x = 8; g.add(nose);
  const can = new THREE.Mesh(new THREE.SphereGeometry(0.5, 12, 8, 0, Math.PI * 2, 0, Math.PI / 2), new THREE.MeshPhongMaterial({color: 0x3b4d5e, shininess: 90}));
  can.scale.set(2.6, 0.9, 1); can.position.set(4.2, 0.45, 0); g.add(can);
  const plan = (pts, m, y = 0) => { const s = new THREE.Shape(); s.moveTo(pts[0][0], pts[0][1]);
    for (const p of pts.slice(1)) s.lineTo(p[0], p[1]); s.closePath();
    const me = new THREE.Mesh(new THREE.ShapeGeometry(s), m); me.rotation.x = Math.PI / 2; me.position.y = y; return me; };
  g.add(plan([[3.6, 0.6], [-1.6, 4.98], [-3.0, 4.98], [-3.4, 0.6], [-3.4, -0.6], [-3.0, -4.98], [-1.6, -4.98], [3.6, -0.6]], mat, -0.1));
  g.add(plan([[5.5, 0.55], [2.5, 1.0], [2.5, -1.0], [5.5, -0.55]], mat, -0.1));                       // strakes
  g.add(plan([[-3.8, 0.6], [-5.6, 2.7], [-6.4, 2.7], [-6.2, 0.6], [-6.2, -0.6], [-6.4, -2.7], [-5.6, -2.7], [-3.8, -0.6]], mat, -0.2));
  const fs = new THREE.Shape(); fs.moveTo(-2.6, 0.5); fs.lineTo(-5.6, 4.0); fs.lineTo(-6.6, 4.0); fs.lineTo(-6.3, 0.5); fs.closePath();
  g.add(new THREE.Mesh(new THREE.ShapeGeometry(fs), acc));                                             // fin, model colour
  const intake = new THREE.Mesh(new THREE.BoxGeometry(2.4, 0.8, 1.0), mat); intake.position.set(2.2, -0.75, 0); g.add(intake);
  const glow = new THREE.Mesh(new THREE.ConeGeometry(0.42, 2.4, 12, 1, true), new THREE.MeshBasicMaterial({color: 0xffb347, transparent: true, opacity: 0.8}));
  glow.rotation.z = Math.PI / 2; glow.position.x = -6.3; g.add(glow); g.glow = glow;
  return g;
}

// ---------- flights ----------
let exag = 1;
for (const m of M) {
  const n = m.t.length; m.p3 = new Float32Array(n * 3); m.g3 = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) { m.p3.set([m.x[i], m.z[i], -m.y[i]], 3 * i); m.g3.set([m.x[i], m.ch.gnd[i] + 4, -m.y[i]], 3 * i); }
  const line = (arr, opa, w) => { const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.BufferAttribute(arr, 3));
    return new THREE.Line(g, new THREE.LineBasicMaterial({color: m.color, transparent: true, opacity: opa})); };
  m.ghost = line(m.p3, 0.35); m.line = line(m.p3, 1); m.gtrack = line(m.g3, 0.55);
  world.add(m.ghost, m.line, m.gtrack);
  m.craft = f16(m.color); scene.add(m.craft);
  m.drop = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]),
    new THREE.LineDashedMaterial({color: 0xffffff, dashSize: 8, gapSize: 6, transparent: true, opacity: 0.7}));
  m.shadow = new THREE.Mesh(new THREE.RingGeometry(6, 10, 24), new THREE.MeshBasicMaterial({color: 0x000000, transparent: true, opacity: 0.45, side: THREE.DoubleSide}));
  m.shadow.rotation.x = -Math.PI / 2; scene.add(m.drop, m.shadow);
}
function find(a, x) { let lo = 0, hi = a.length - 1; if (x <= a[0]) return 0; if (x >= a[hi]) return hi - 1;
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (a[mid] <= x) lo = mid; else hi = mid; } return lo; }
function sample(m, t) {
  const i = find(m.t, t), j = Math.min(i + 1, m.t.length - 1);
  const f = j > i ? Math.min(Math.max((t - m.t[i]) / (m.t[j] - m.t[i]), 0), 1) : 0, L = a => a[i] + (a[j] - a[i]) * f;
  const s = {i, x: L(m.x), y: L(m.y), z: L(m.z), ch: {}, done: t >= m.t[m.t.length - 1]};
  for (const k in m.ch) s.ch[k] = L(m.ch[k]);
  let dh = m.att.heading[j] - m.att.heading[i]; dh -= 360 * Math.round(dh / 360);
  s.att = {pitch: L(m.att.pitch), roll: L(m.att.roll), heading: m.att.heading[i] + dh * f};
  return s;
}
function pose(m, s) {
  const g = m.craft; g.position.set(s.x, s.z * exag, -s.y);
  g.rotation.set(s.att.roll * rad, (90 - s.att.heading) * rad, s.att.pitch * rad, "YZX");
  const gz = s.ch.gnd * exag; m.shadow.position.set(s.x, gz + 1.5, -s.y);
  const a = m.drop.geometry.attributes.position; a.setXYZ(0, s.x, s.z * exag, -s.y); a.setXYZ(1, s.x, gz, -s.y);
  a.needsUpdate = true; m.drop.computeLineDistances();
}

// ---------- labels ----------
const lab = document.getElementById("labels");
for (const L of D.labels) { const d = document.createElement("div"); d.textContent = L.name; d.className = L.kind; lab.append(d); L.el = d; }
const tmp = new THREE.Vector3();
function labels() {
  const r = host.getBoundingClientRect();
  for (const L of D.labels) { tmp.set(L.x, (L.z + (L.kind === "pk" ? 0 : 60)) * exag, -L.y);
    const dist = tmp.distanceTo(camera.position); tmp.project(camera);
    const vis = tmp.z < 1 && Math.abs(tmp.x) < 1.1 && Math.abs(tmp.y) < 1.1 && dist < span * 0.9;
    L.el.style.display = vis ? "block" : "none";
    if (vis) { L.el.style.left = ((tmp.x + 1) / 2 * r.width) + "px"; L.el.style.top = ((1 - tmp.y) / 2 * r.height) + "px"; } }
}

// ---------- cameras ----------
const camSel = document.getElementById("cam"), who = document.getElementById("who");
M.forEach((m, k) => who.insertAdjacentHTML("beforeend", `<option value="${k}">${m.name}</option>`));
let mode = camSel.value, act = 0;
const qCock = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), -Math.PI / 2);
function overview() {
  const c = new THREE.Vector3(0, 0, 0), d = span * 0.62;
  camera.position.set(c.x - d * 0.25, d * 0.62 * Math.max(exag, 1) ** 0.3, c.z + d * 0.75); controls.target.copy(c);
}
function setMode() {
  mode = camSel.value; fixedYaw = null; controls.enabled = mode === "orbit" || mode === "overview";
  if (mode === "overview") overview();
  if (mode === "orbit") { const p = M[act].craft.position; controls.target.copy(p);
    camera.position.copy(p).add(new THREE.Vector3(-600, 350, 600)); }
  camera.up.set(0, 1, 0);
}
camSel.onchange = setMode; who.onchange = () => { act = +who.value; setMode(); };
const chase = {pos: null}, lvl = new THREE.Quaternion();
const fol = {roll: document.getElementById("f-roll"), pitch: document.getElementById("f-pitch"),
  yaw: document.getElementById("f-yaw")};
let fixedYaw = null;                     // compass direction held while yaw follow is off
for (const b of Object.values(fol)) b.onchange = () => { fixedYaw = null; };
// camera attitude for the close-behind and cockpit views: the jet's roll / pitch / yaw where
// "Follow" is ticked, level / fixed otherwise (the cockpit always turns with the jet)
function camQuat(g, cockpit) {
  const a = M[act].s.att, yawOn = fol.yaw.checked || cockpit;
  if (fol.roll.checked && fol.pitch.checked && yawOn) return g.quaternion;
  if (!yawOn && fixedYaw === null) fixedYaw = a.heading;
  const yaw = yawOn ? a.heading : fixedYaw;
  return lvl.setFromEuler(new THREE.Euler(fol.roll.checked ? a.roll * rad : 0, (90 - yaw) * rad,
    fol.pitch.checked ? a.pitch * rad : 0, "YZX"));
}
function cameraFrame(dt) {
  const g = M[act].craft, p = g.position;
  if (mode === "chase") {
    const hd = M[act].s.att.heading * rad, fw = new THREE.Vector3(Math.sin(hd), 0, -Math.cos(hd));
    const want = p.clone().addScaledVector(fw, -90).add(new THREE.Vector3(0, 22, 0));
    chase.pos = chase.pos && chase.pos.distanceTo(want) < 400 ? chase.pos.lerp(want, 1 - Math.exp(-dt * 5)) : want.clone();
    camera.position.copy(chase.pos); camera.up.set(0, 1, 0); camera.lookAt(p.clone().addScaledVector(fw, 120));
  } else if (mode === "tail") {          // close behind and above the tail, see camQuat
    const q = camQuat(g, false), want = p.clone().add(new THREE.Vector3(-30, 6, 0).applyQuaternion(q));
    chase.pos = chase.pos && chase.pos.distanceTo(want) < 200 ? chase.pos.lerp(want, 1 - Math.exp(-dt * 12)) : want.clone();
    camera.position.copy(chase.pos); camera.up.set(0, 1, 0).applyQuaternion(q);
    camera.lookAt(p.clone().add(new THREE.Vector3(80, 0, 0).applyQuaternion(q)));
  } else if (mode === "cockpit") {
    const q = camQuat(g, true), off = new THREE.Vector3(5.2, 1.1, 0).applyQuaternion(g.quaternion);
    camera.position.copy(p).add(off); camera.quaternion.copy(q).multiply(qCock);
  } else {
    chase.pos = null;
    if (mode === "orbit") { const d = p.clone().sub(controls.target); controls.target.add(d); camera.position.add(d); }
    controls.update();
  }
  const dist = camera.position.distanceTo(p), k = mode === "overview" ? Math.max(1, dist / 450) : (mode === "orbit" ? Math.max(1, dist / 300) : 1);
  for (const m of M) { m.craft.scale.setScalar(k); m.craft.visible = !(mode === "cockpit" && m === M[act]); }
}

// ---------- exaggeration ----------
const ex = document.getElementById("exag");
ex.onchange = () => { exag = ex.checked ? D.exag : 1; world.scale.y = exag; if (mode === "overview") overview(); };

// ---------- side panel ----------
const cards = document.getElementById("cards");
for (const m of M) { const c = document.createElement("div"); c.className = "card";
  c.innerHTML = `<h2><span class="sw" style="background:${m.color}"></span>${m.name}<span class="ph"></span></h2>
   <div class="tm">${D.cards.map(k => `<div data-k="${k.k}">–<span>${k.label}</span></div>`).join("")}</div>`;
  cards.append(c); m.card = c; }
legend(document.getElementById("lg-agl"), M); legend(document.getElementById("lg-nz"), M);
document.getElementById("lg-prof").innerHTML = `<span><i style="background:${TK.muted}"></i>terrain under the route</span>
 <span><i style="background:${TK.ink2};opacity:.7"></i>corridor + clearance</span><span><i style="background:${TK.ink}"></i>reference</span>
 <span><i class="pl"></i>flown</span>`;
function timeChart(cv, key, t, extra) {
  const [c, w, h] = setup(cv);
  let ymin = 1e99, ymax = -1e99; for (const m of M) for (const v of m.ch[key]) { ymin = Math.min(ymin, v); ymax = Math.max(ymax, v); }
  if (extra != null) { ymin = Math.min(ymin, extra); ymax = Math.max(ymax, extra); }
  const [y0, y1, yst] = axisRange(ymin, ymax, 4, true), [, x1, xst] = axisRange(0, tEnd, 4, true);
  const L = 42, B = 18, Tp = 6, W = w - L - 10, Hh = h - B - Tp, X = x => L + x / x1 * W, Y = y => Tp + Hh - (y - y0) / (y1 - y0) * Hh;
  c.strokeStyle = TK.grid; c.lineWidth = 1; c.fillStyle = TK.ink2;
  for (let y = y0; y <= y1 + yst / 2; y += yst) { c.beginPath(); c.moveTo(L, Y(y)); c.lineTo(L + W, Y(y)); c.stroke();
    c.textAlign = "right"; c.fillText(fmtN(y, yst), L - 6, Y(y) + 4); }
  for (let x = 0; x <= x1 + xst / 2; x += xst) { c.textAlign = x + xst > x1 + xst / 2 ? "right" : "center";
    c.fillText(fmtN(x, xst) + (x + xst > x1 + xst / 2 ? " s" : ""), X(x), h - 3); }
  if (extra != null) { c.strokeStyle = TK.ink2; c.setLineDash([5, 4]); c.beginPath(); c.moveTo(L, Y(extra)); c.lineTo(L + W, Y(extra)); c.stroke(); c.setLineDash([]); }
  for (const m of M) {
    for (const [full, alpha] of [[true, 0.25], [false, 1]]) {
      c.strokeStyle = m.color; c.globalAlpha = alpha; c.lineWidth = 2; c.lineJoin = "round"; c.beginPath();
      for (let i = 0; i < m.t.length; i++) { if (!full && m.t[i] > t) break;
        i ? c.lineTo(X(m.t[i]), Y(m.ch[key][i])) : c.moveTo(X(m.t[i]), Y(m.ch[key][i])); }
      c.stroke(); }
    c.globalAlpha = 1; c.beginPath(); c.arc(X(Math.min(t, m.t[m.t.length - 1])), Y(m.s.ch[key]), 5, 0, 7);
    c.fillStyle = m.color; c.fill(); c.lineWidth = 2; c.strokeStyle = TK.surface; c.stroke();
  }
}
// profile along the active route, flown part and current position on top
const pcv = document.getElementById("prof");
function profile(t) {
  const m = M[act], P = m.prof, [c, w, h] = setup(pcv);
  let ymax = 0; for (const v of P.ref) ymax = Math.max(ymax, v); for (const v of P.fz) ymax = Math.max(ymax, v);
  const [y0, y1, yst] = axisRange(0, ymax, 4, true), [, x1, xst] = axisRange(0, m.length, 4, true);
  const L = 42, B = 18, Tp = 6, W = w - L - 10, Hh = h - B - Tp, X = x => L + x / x1 * W, Y = y => Tp + Hh - (y - y0) / (y1 - y0) * Hh;
  c.strokeStyle = TK.grid; c.lineWidth = 1; c.fillStyle = TK.ink2;
  for (let y = y0; y <= y1 + yst / 2; y += yst) { c.beginPath(); c.moveTo(L, Y(y)); c.lineTo(L + W, Y(y)); c.stroke();
    c.textAlign = "right"; c.fillText(fmtN(y, yst), L - 6, Y(y) + 4); }
  for (let x = 0; x <= x1 + xst / 2; x += xst) { c.textAlign = x + xst > x1 + xst / 2 ? "right" : "center";
    c.fillText(fmtN(x, xst) + (x + xst > x1 + xst / 2 ? " km" : ""), X(x), h - 3); }
  c.fillStyle = TK.muted; c.globalAlpha = 0.45; c.beginPath(); c.moveTo(X(0), Y(0));
  for (let i = 0; i < P.s.length; i++) c.lineTo(X(P.s[i]), Y(Math.max(P.terrain[i], 0)));
  c.lineTo(X(P.s[P.s.length - 1]), Y(0)); c.closePath(); c.fill(); c.globalAlpha = 1;
  const ln = (xs, ys, col, dash, wd = 1.5, upto = 1e99) => { c.strokeStyle = col; c.lineWidth = wd; c.setLineDash(dash || []); c.beginPath();
    for (let i = 0; i < xs.length && xs[i] <= upto; i++) i ? c.lineTo(X(xs[i]), Y(ys[i])) : c.moveTo(X(xs[i]), Y(ys[i])); c.stroke(); c.setLineDash([]); };
  c.globalAlpha = 0.7; ln(P.s, P.floor, TK.ink2, [4, 3], 1); c.globalAlpha = 1;
  ln(P.s, P.ref, TK.ink, null, 1.5);
  const s = m.s.ch.dist; c.globalAlpha = 0.3; ln(P.fs, P.fz, m.color, null, 2); c.globalAlpha = 1; ln(P.fs, P.fz, m.color, null, 2, s);
  c.beginPath(); c.arc(X(s), Y(m.s.ch.alt), 5, 0, 7); c.fillStyle = m.color; c.fill(); c.lineWidth = 2; c.strokeStyle = TK.surface; c.stroke();
  document.querySelector("#lg-prof .pl").style.background = m.color;
}
// minimap: shaded relief, routes, aircraft
const mini = document.getElementById("mini"), mimg = new Image(); mimg.src = D.minimap;
function minimap() {
  if (!mimg.complete) return;
  const r = mini.getBoundingClientRect(); mini.style.height = (r.width * D.mapAspect) + "px";
  const [c, w, h] = setup(mini); c.drawImage(mimg, 0, 0, w, h);
  const X = x => (x - D.mapBox[0]) / (D.mapBox[1] - D.mapBox[0]) * w, Y = y => h - (y - D.mapBox[2]) / (D.mapBox[3] - D.mapBox[2]) * h;
  for (const m of M) { c.strokeStyle = m.color; c.lineWidth = 2; c.beginPath();
    m.track.x.forEach((x, i) => i ? c.lineTo(X(x), Y(m.track.y[i])) : c.moveTo(X(x), Y(m.track.y[i]))); c.stroke(); }
  for (const m of M) { const s = m.s, hd = s.att.heading * rad;
    c.beginPath(); c.arc(X(s.x), Y(s.y), m === M[act] ? 6 : 4.5, 0, 7); c.fillStyle = m.color; c.fill();
    c.lineWidth = 2; c.strokeStyle = "#fff"; c.stroke();
    c.beginPath(); c.moveTo(X(s.x), Y(s.y)); c.lineTo(X(s.x) + 14 * Math.sin(hd), Y(s.y) - 14 * Math.cos(hd)); c.stroke(); }
}
mini.onclick = e => { const r = mini.getBoundingClientRect(), m = M[act];
  const x = D.mapBox[0] + e.offsetX / r.width * (D.mapBox[1] - D.mapBox[0]), y = D.mapBox[3] - e.offsetY / r.height * (D.mapBox[3] - D.mapBox[2]);
  let best = 0, bd = 1e99; for (let i = 0; i < m.t.length; i++) { const d = (m.x[i] - x) ** 2 + (m.y[i] - y) ** 2; if (d < bd) { bd = d; best = i; } }
  t = m.t[best]; scrub.value = t / tEnd * 1000; };

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
addEventListener("resize", resize); resize();
const fmtT = s => `${String(Math.floor(s / 60)).padStart(2, "0")}:${(s % 60).toFixed(1).padStart(4, "0")}`;
for (const m of M) { m.s = sample(m, 0); pose(m, m.s); }
setMode();
function frame(now) {
  const dt = Math.min((now - last) / 1000, 0.25); last = now;
  if (playing) { t = Math.min(t + dt * rate.value, tEnd); if (t >= tEnd) { playing = false; play.textContent = "Play"; }
    scrub.value = t / tEnd * 1000; }
  for (const m of M) {
    const s = sample(m, t); m.s = s; pose(m, s);
    m.line.geometry.setDrawRange(0, s.i + 1); m.gtrack.geometry.setDrawRange(0, s.i + 1);
    m.craft.glow.scale.set(1, 0.5 + s.ch.nz * 0.15, 1);
    for (const k of D.cards) m.card.querySelector(`[data-k="${k.k}"]`).firstChild.nodeValue = s.ch[k.k].toFixed(k.d);
    m.card.querySelector(".ph").textContent = s.done ? "arrived" : `${(100 * s.ch.dist / m.length).toFixed(0)} %`;
  }
  cameraFrame(dt);
  clock.firstChild.nodeValue = `T+${fmtT(t)}`; clock.lastChild.textContent = `×${rate.value}  ·  ${camSel.options[camSel.selectedIndex].text}`;
  renderer.render(scene, camera); labels();
  timeChart(document.getElementById("agl"), "agl", t, D.clearance);
  timeChart(document.getElementById("nz"), "nz", t, null);
  profile(t); minimap();
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
"""


def terrain_replay_html(grid, models: list[dict], *, title: str, subtitle: str = "",
                        labels: list[dict] = (), attribution: str = "", ramp: str = "norway",
                        clearance: float = 100.0, exag: float = 2.0, rates=(1, 2, 5, 10, 20),
                        rate: float = 5, pages: list | None = None, current: str = "",
                        notes: str = "", subnav: list[tuple[str, str]] = (), here: str = "",
                        theme: str = "dark") -> str:
    """One region: terrain mesh (``grid`` at display resolution), flights, charts, minimap.

    labels: [{"name", "x", "y", "z" (m), "kind": "pk" (point) | "sea" | ""}].
    notes: HTML for the side panel (method, data)."""
    T = THEMES[theme]
    w = grid.water
    bits = np.packbits(w.ravel(), bitorder="little")
    zq = np.clip(np.round(grid.z / QUANT), -32768, 32767).astype("<i2")
    data = {
        "terrain": {"nx": int(grid.x.size), "ny": int(grid.y.size), "x0": float(grid.x[0]),
                    "y0": float(grid.y[0]), "dx": grid.spacing, "q": QUANT, "h": _b64(zq),
                    "water": _b64(bits), "ramp": RAMPS[ramp], "rock": ROCK, "waterColor": WATER},
        "models": [_pack_flight(m, T["series"][i]) for i, m in enumerate(models)],
        "labels": [dict(lb) for lb in labels],
        "cards": [{"k": "agl", "label": "above ground, m", "d": 0}, {"k": "alt", "label": "altitude, m", "d": 0},
                  {"k": "kt", "label": "airspeed, kt", "d": 0}, {"k": "nz", "label": "load factor, g", "d": 1},
                  {"k": "roll", "label": "bank, °", "d": 0}, {"k": "dist", "label": "along route, km", "d": 1}],
        "rates": list(rates), "rate": rate, "clearance": clearance, "exag": exag,
        "minimap": hillshade_png(grid, ramp, 420, w),
        "mapBox": [float(grid.x[0]), float(grid.x[-1]), float(grid.y[0]), float(grid.y[-1])],
        "mapAspect": float((grid.y[-1] - grid.y[0]) / (grid.x[-1] - grid.x[0])),
        "aircraft": _aircraft(),
    }
    attribution = f"{attribution}  ·  {AIRCRAFT_CREDIT}" if attribution else AIRCRAFT_CREDIT
    stats = [flight_stats(m) for m in models]
    rows = "".join(f"<tr><td>{html.escape(k)}</td>" + "".join(f"<td>{html.escape(s[k])}</td>" for s in stats)
                   + "</tr>" for k in stats[0])
    head = "".join(f'<th><span class="sw" style="display:inline-block;background:{T["series"][i]}"></span> '
                   f'{html.escape(m["name"])}</th>' for i, m in enumerate(models))
    page = f"""<div id="app">
 <div id="scene"><div id="labels"></div><div id="hud"><h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p></div>
  <div id="clock">T+0<small>&nbsp;</small></div><div id="attr">{html.escape(attribution)}</div></div>
 <div id="side">{_subnav(subnav, here)}<div id="cards"></div>
  <h3>Map</h3><canvas id="mini" aria-label="Shaded-relief map with both routes; click to jump"></canvas>
  <p class="note">Click the map to jump to that point of the selected flight.</p>
  <h3>Profile along the route</h3><div class="legend" id="lg-prof"></div><div class="cw"><canvas class="chart t" id="prof"></canvas></div>
  <h3>Height above ground (m)</h3><div class="legend" id="lg-agl"></div><canvas class="chart t" id="agl"></canvas>
  <p class="note">Dashed: set clearance, {clearance:.0f} m.</p>
  <h3>Load factor (g)</h3><div class="legend" id="lg-nz"></div><canvas class="chart t" id="nz"></canvas>
  <h3>Flight summary</h3><table><tr><th></th>{head}</tr>{rows}</table>
  {notes}
  <h3>Aircraft model</h3><p class="note">"F-16 Fighter Jet" by
  <a href="https://sketchfab.com/iedalton">iedalton</a>, from
  <a href="https://sketchfab.com/3d-models/f-16-fighter-jet-d84491f443384ee488593cc6f0f0839e">Sketchfab</a>,
  licensed <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>. Scaled uniformly to the real
  15.06 m length; the model is stylised, so its wing spans 13.7 m (real: 9.96 m). The flight
  dynamics are JSBSim's F-16, not this shape.</p>
 </div>
 <div id="bar"><button id="play" aria-label="Play or pause">Play</button>
  <select id="rate" aria-label="Time scale"></select>
  <input id="scrub" type="range" min="0" max="1000" value="0" aria-label="Flight time">
  <select id="cam" aria-label="Camera"><option value="tail">Close behind</option><option value="chase">Chase</option><option value="cockpit">Cockpit</option>
   <option value="orbit">Orbit (follow)</option><option value="overview">Overview</option></select>
  <select id="who" aria-label="Aircraft to follow"></select>
  <span class="muted" style="font-size:13px">Camera follows:</span>
  <label><input type="checkbox" id="f-roll"> roll</label><label><input type="checkbox" id="f-pitch"> pitch</label>
  <label><input type="checkbox" id="f-yaw"> yaw</label>
  <label><input type="checkbox" id="exag"> Heights ×{exag:g}</label></div>
</div>
{theme_tag(theme)}{data_tag(data)}{importmap_tag(gltf=True)}
<script type="module">{CHART_JS}{_JS}</script>"""
    nav_html = nav(pages, current) if pages else ""
    return themed(f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title><style>{CSS}{_CSS}</style></head><body>
{nav_html}
{page}
</body></html>""", theme)


_HUB_CSS = r"""
main{max-width:1180px;margin:0 auto;padding:24px 20px 40px}
h1{font-size:24px;margin:8px 0 4px}h2{font-size:17px;margin:28px 0 8px}
p{max-width:820px;color:var(--ink2)}p b{color:var(--ink)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:18px;margin-top:16px}
a.region{display:block;text-decoration:none;border:1px solid var(--grid);border-radius:10px;overflow:hidden;
background:var(--surface)}a.region:hover{border-color:var(--axis)}
a.region img{display:block;width:100%;height:auto}a.region div{padding:12px 14px}
a.region h3{margin:0 0 4px;font-size:15px}a.region p{margin:0;font-size:13px}
table.v{max-width:820px}ol{max-width:820px;color:var(--ink2);padding-left:20px}ol li{margin:4px 0}
ol b{color:var(--ink)}
"""


def terrain_hub_html(regions: list[dict], validation: list[dict], *, pages: list, method: str,
                     credits: str = "", current: str = "index.html", theme: str = "dark") -> str:
    """Index: a card per region (shaded-relief thumbnail, summary) and the DEM validation.

    regions: [{"href", "title", "text", "img" (data URI)}]; validation: rows of
    {"label", "value"} for the table."""
    cards = "".join(f"""<a class="region" href="{r['href']}"><img src="{r['img']}" alt="Shaded relief of {html.escape(r['title'])}">
<div><h3>{html.escape(r['title'])}</h3><p>{r['text']}</p></div></a>""" for r in regions)
    vrows = "".join(f"<tr><td>{html.escape(v['label'])}</td><td>{html.escape(v['value'])}</td></tr>" for v in validation)
    body = f"""<main><h1>F-16 low-level flights over real terrain</h1>
<p>The JSBSim F-16 airframe, on an AeroSim turbojet deck, flies terrain-following routes at 100 m above
the ground through a Norwegian fjord and across Novaya Zemlya. Terrain heights come from Kartverket's
national laser-scanned terrain model and the Copernicus 30 m elevation model. Each route is planned
on those heights, and JSBSim is given the real ground height under the aircraft every step.</p>
<div class="grid">{cards}</div>
<h2>How it works</h2>{method}
<h2>Checking the terrain data</h2>
<p>Over Lysefjorden the two sources are independent measurements of the same ground: Kartverket's
model from airborne laser scanning, Copernicus's from TanDEM-X radar. Their difference is the check
on both the data and the resampling here. Copernicus measures the surface, tree tops included, so it sits
slightly higher where there is forest; above the tree line the two agree to a few metres.</p>
<table class="v">{vrows}</table>
{"<h2>Data and credits</h2>" + credits if credits else ""}
</main>"""
    return themed(f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Terrain flights</title><style>{CSS}{_HUB_CSS}</style></head><body>
{nav(pages, current)}
{body}
</body></html>""", theme)
