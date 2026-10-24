"""Animated 3D replay of orbital ascents as a standalone HTML page (three.js, no Python deps).

``write_replay(results, path, site=(lat, lon))`` writes one self-contained page:

- globe      Earth-fixed 3D view (drag to orbit, scroll to zoom): coastlines, graticule, the
             launch site, and each trajectory drawn as it flies, with staging events marked
- telemetry  per model: flight phase, altitude, speed, dynamic pressure, acceleration, mass,
             downrange
- charts     altitude and speed against time, the axis widening as the replay advances
- controls   play / pause (space), time scale (x10 to x1000; coasts run 10x faster), scrub
             bar, follow-the-vehicle camera, altitude exaggeration

Several results (e.g. calibrated vs first-principles models) fly together in the fixed
palette order of ``aerosim.plots.style``. Everything is embedded, so the page works offline
from a double-clicked file: the flight data (a few hundred kB per model) and three.js r160
(MIT, ``data/three/``, ~0.9 MB as data: URLs in the import map).
"""

from __future__ import annotations

import base64
import html
import json
from importlib import resources

import numpy as np

from aerosim.plots.style import THEMES

EVENT_LABELS = {"max_q": "Max q", "Stage 1_burnout": "Stage 1 separation",
                "fairing_jettison": "Fairing jettison", "Stage 2_burnout": "Stage 2 separation"}


def _event_label(k: str) -> str:
    if k in EVENT_LABELS:
        return EVENT_LABELS[k]
    for a, b in (("_burn1_end", " burn 1 cutoff"), ("_burn2_end", " burn 2 cutoff"),
                 ("_burnout", " burnout")):
        k = k.replace(a, b)
    return k.replace("_", " ")


def _pack(res, max_points: int = 4000) -> dict:
    """Arrays the page needs, rounded to what the eye can see."""
    n = res.time.size
    idx = np.unique(np.r_[np.linspace(0, n - 1, min(n, max_points)).astype(int), n - 1])
    r = lambda a, d: np.round(np.asarray(a, float)[idx], d).tolist()
    ev = {k: float(v) for k, v in res.events.items() if k != "liftoff" and not k.endswith("_start")
          and v <= res.time[-1]}
    ev = dict(sorted(ev.items(), key=lambda kv: kv[1]))
    return {"t": r(res.time, 1), "lat": r(res.latitude, 4), "lon": r(res.longitude, 4),
            "alt": r(res.altitude / 1e3, 3), "v": r(res.speed / 1e3, 4),
            "q": r(res.dynamic_pressure / 1e3, 2), "g": r(res.acceleration / 9.80665, 2),
            "m": r(res.mass / 1e3, 3), "dr": r(res.downrange / 1e3, 1),
            "ph": np.asarray(res.phase)[idx].astype(int).tolist(),
            "phases": list(res.phases), "events": [[_event_label(k), t] for k, t in ev.items()]}


def _importmap() -> str:
    """three.js and OrbitControls inlined as data: URLs, so the page needs no network."""
    def url(name):
        b = resources.files("aerosim.plots").joinpath(f"data/three/{name}").read_bytes()
        return "data:text/javascript;base64," + base64.b64encode(b).decode("ascii")
    imports = {"three": url("three.module.min.js"),
               "three/addons/controls/OrbitControls.js": url("OrbitControls.js")}
    return json.dumps({"imports": imports}, separators=(",", ":"))


def replay_html(results, site: tuple[float, float], title: str = "Ascent replay",
                subtitle: str = "", theme: str = "dark") -> str:
    """The replay page as an HTML string (see module docstring)."""
    if not isinstance(results, dict):
        results = {"Model": results}
    T = THEMES[theme]
    coast = json.loads(resources.files("aerosim.plots").joinpath("data/coastline_110m.json")
                       .read_text(encoding="utf-8"))["lines"]
    data = {"models": [{"name": k, "color": T["series"][i], **_pack(r)}
                       for i, (k, r) in enumerate(results.items())],
            "site": list(site), "coast": coast, "theme": T, "dark": theme == "dark"}
    page = _PAGE.replace("__TITLE__", html.escape(title)).replace("__SUB__", html.escape(subtitle))
    page = page.replace("__SERIES0__", T["series"][0])
    for k, v in T.items():
        if isinstance(v, str):
            page = page.replace(f"__{k.upper()}__", v)
    page = page.replace("__DATA__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
    return page.replace("__IMPORTMAP__", _importmap())


def write_replay(results, path, site: tuple[float, float], title: str = "Ascent replay",
                 subtitle: str = "", theme: str = "dark") -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(replay_html(results, site, title, subtitle, theme))


_PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--surface:__SURFACE__;--page:__PAGE__;--ink:__INK__;--ink2:__INK2__;--muted:__MUTED__;
--grid:__GRID__;--axis:__AXIS__}
*{box-sizing:border-box}html,body{margin:0;height:100%;background:var(--page);color:var(--ink);
font:14px/1.4 "Segoe UI",system-ui,-apple-system,Helvetica,Arial,sans-serif}
#app{display:grid;grid-template-columns:minmax(0,1fr) 400px;grid-template-rows:minmax(0,1fr) auto;
height:100vh}
#globe{position:relative;min-height:0;background:var(--page)}#globe canvas{display:block}
#hud{position:absolute;left:20px;top:16px;pointer-events:none}
#hud h1{margin:0;font-size:20px;font-weight:600;letter-spacing:.2px}
#hud p{margin:2px 0 0;color:var(--ink2);font-size:13px}
#clock{position:absolute;left:20px;bottom:14px;font:600 30px/1 Consolas,"SF Mono",monospace;
pointer-events:none;font-variant-numeric:tabular-nums}
#clock small{display:block;font:500 12px/1.6 "Segoe UI",system-ui,sans-serif;color:var(--ink2)}
#side{grid-row:1/3;grid-column:2;overflow-y:auto;background:var(--surface);
border-left:1px solid var(--grid);padding:16px 18px 20px}
.card{border:1px solid var(--grid);border-radius:8px;padding:10px 12px;margin-bottom:10px}
.card h2{margin:0 0 6px;font-size:13px;font-weight:600;display:flex;align-items:center;gap:8px}
.sw{width:12px;height:12px;border-radius:3px;flex:none}
.card .ph{margin-left:auto;font-weight:500;color:var(--ink2)}
.tm{display:grid;grid-template-columns:repeat(3,1fr);gap:4px 10px}
.tm div{font-variant-numeric:tabular-nums;font-size:15px;font-weight:600}
.tm span{display:block;font-size:11px;font-weight:400;color:var(--ink2)}
h3{margin:16px 0 4px;font-size:13px;font-weight:600}
canvas.chart{width:100%;height:150px;display:block}
.legend{display:flex;gap:14px;font-size:12px;color:var(--ink2);margin-bottom:2px}
.legend i{display:inline-block;width:14px;height:3px;border-radius:2px;margin-right:5px;
vertical-align:middle}
#events{list-style:none;margin:0;padding:0;font-size:12.5px}
#events li{display:flex;gap:8px;padding:3px 0;color:var(--muted);border-bottom:1px solid var(--grid)}
#events li.done{color:var(--ink)}#events li b{font-weight:500;margin-left:auto;
font-variant-numeric:tabular-nums;color:var(--ink2)}
#bar{grid-column:1;display:flex;align-items:center;gap:12px;padding:10px 20px;
background:var(--surface);border-top:1px solid var(--grid);flex-wrap:wrap}
button,select{font:inherit;color:var(--ink);background:var(--page);border:1px solid var(--axis);
border-radius:6px;padding:5px 12px;cursor:pointer}
button:focus-visible,select:focus-visible,input:focus-visible{outline:2px solid __SERIES0__;
outline-offset:2px}
#play{min-width:76px;font-weight:600}
#scrub{flex:1;min-width:160px;accent-color:__SERIES0__}
label{display:flex;align-items:center;gap:6px;color:var(--ink2);font-size:13px;cursor:pointer}
@media (max-width:900px){#app{grid-template-columns:1fr;grid-template-rows:60vh auto auto;height:auto}
#side{grid-row:3;grid-column:1;border-left:0;border-top:1px solid var(--grid)}#bar{grid-row:2}
#clock{font-size:22px}}
</style></head><body>
<div id="app">
 <div id="globe"><div id="hud"><h1>__TITLE__</h1><p>__SUB__</p></div>
  <div id="clock">T+00:00<small>&nbsp;</small></div></div>
 <div id="side">
  <div id="cards"></div>
  <h3>Altitude (km)</h3><div class="legend" id="lg1"></div><canvas class="chart" id="c1"></canvas>
  <h3>Speed, Earth-relative (km/s)</h3><div class="legend" id="lg2"></div>
  <canvas class="chart" id="c2"></canvas>
  <h3>Events</h3><ul id="events"></ul>
 </div>
 <div id="bar">
  <button id="play" aria-label="Play or pause">Play</button>
  <select id="rate" aria-label="Time scale">
   <option value="10">×10</option><option value="30" selected>×30</option>
   <option value="100">×100</option><option value="300">×300</option><option value="1000">×1000</option>
  </select>
  <input id="scrub" type="range" min="0" max="1000" value="0" aria-label="Flight time">
  <label><input type="checkbox" id="follow"> Follow vehicle</label>
  <label><input type="checkbox" id="exag"> Altitude ×5</label>
 </div>
</div>
<script type="application/json" id="data">__DATA__</script>
<script type="importmap">__IMPORTMAP__</script>
<script type="module">
import * as THREE from "three";
import {OrbitControls} from "three/addons/controls/OrbitControls.js";
const D = JSON.parse(document.getElementById("data").textContent), TK = D.theme;
const R = 6.371, KM = 1e-3;                       // scene unit = 1000 km
const M = D.models, tEnd = Math.max(...M.map(m => m.t[m.t.length - 1]));
const rad = Math.PI / 180;
const xyz = (lat, lon, r) => new THREE.Vector3(r * Math.cos(lat * rad) * Math.cos(lon * rad),
  r * Math.sin(lat * rad), -r * Math.cos(lat * rad) * Math.sin(lon * rad));

// ---------- scene ----------
const host = document.getElementById("globe");
const renderer = new THREE.WebGLRenderer({antialias: true});
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
host.prepend(renderer.domElement);
const scene = new THREE.Scene(); scene.background = new THREE.Color(TK.page);
const camera = new THREE.PerspectiveCamera(40, 1, 0.01, 200);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; controls.minDistance = R * 1.02; controls.maxDistance = 60;
scene.add(new THREE.AmbientLight(0xffffff, D.dark ? 0.55 : 0.8));
const sun = new THREE.DirectionalLight(0xffffff, D.dark ? 1.1 : 0.7); camera.add(sun);
sun.position.set(-2, 3, 4); scene.add(camera);
scene.add(new THREE.Mesh(new THREE.SphereGeometry(R, 128, 96),
  new THREE.MeshPhongMaterial({color: D.dark ? 0x17324d : 0xcfe0ee, shininess: 8, specular: 0x223344})));
scene.add(new THREE.Mesh(new THREE.SphereGeometry(R * 1.018, 96, 64),
  new THREE.MeshBasicMaterial({color: D.dark ? 0x3d7fd6 : 0x7fb0e8, transparent: true,
    opacity: D.dark ? 0.10 : 0.12, side: THREE.BackSide})));
function lineSegs(paths, r, color, opacity) {
  const p = [];
  for (const path of paths) for (let i = 1; i < path.length; i++) {
    const a = xyz(path[i - 1][1], path[i - 1][0], r), b = xyz(path[i][1], path[i][0], r);
    p.push(a.x, a.y, a.z, b.x, b.y, b.z);
  }
  const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.Float32BufferAttribute(p, 3));
  return new THREE.LineSegments(g, new THREE.LineBasicMaterial({color, transparent: true, opacity}));
}
scene.add(lineSegs(D.coast, R * 1.0015, D.dark ? 0x8fb3d6 : 0x52514e, D.dark ? 0.75 : 0.7));
const grat = [];
for (let la = -60; la <= 60; la += 30) { const l = []; for (let lo = -180; lo <= 180; lo += 3) l.push([lo, la]); grat.push(l); }
for (let lo = -180; lo < 180; lo += 30) { const l = []; for (let la = -90; la <= 90; la += 3) l.push([lo, la]); grat.push(l); }
scene.add(lineSegs(grat, R * 1.001, D.dark ? 0x9fb4c8 : 0x52514e, 0.12));
const disc = (() => { const c = document.createElement("canvas"); c.width = c.height = 64;
  const x = c.getContext("2d"); x.fillStyle = "#fff"; x.beginPath(); x.arc(32, 32, 30, 0, 7); x.fill();
  return new THREE.CanvasTexture(c); })();
const dot = (color, size) => new THREE.Points(new THREE.BufferGeometry(),
  new THREE.PointsMaterial({color, size, sizeAttenuation: false, map: disc, transparent: true,
    alphaTest: 0.5}));
const site = dot(TK.ink, 7);
site.geometry.setAttribute("position", new THREE.Float32BufferAttribute(xyz(D.site[0], D.site[1], R * 1.002).toArray(), 3));
scene.add(site);

// ---------- trajectories ----------
let exag = 1;
for (const m of M) {
  const n = m.t.length;
  m.pos = new Float32Array(n * 3);
  m.ghost = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({color: m.color, transparent: true, opacity: 0.35}));
  m.line = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({color: m.color}));
  m.ghost.geometry.setAttribute("position", new THREE.BufferAttribute(m.pos, 3));
  m.line.geometry.setAttribute("position", new THREE.BufferAttribute(m.pos, 3));
  m.craft = dot(m.color, 11); m.craft.geometry.setAttribute("position", new THREE.Float32BufferAttribute([0, 0, 0], 3));
  m.halo = dot(TK.surface, 17); m.halo.geometry.setAttribute("position", m.craft.geometry.attributes.position);
  m.evPts = dot(m.color, 7); m.evPts.geometry.setAttribute("position", new THREE.Float32BufferAttribute(new Float32Array(m.events.length * 3), 3));
  scene.add(m.ghost, m.line, m.evPts, m.halo, m.craft);
}
function find(a, x) { let lo = 0, hi = a.length - 1; if (x <= a[0]) return 0; if (x >= a[hi]) return hi - 1;
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (a[mid] <= x) lo = mid; else hi = mid; } return lo; }
function sample(m, t) {
  const i = find(m.t, t), j = Math.min(i + 1, m.t.length - 1);
  const f = j > i ? Math.min(Math.max((t - m.t[i]) / (m.t[j] - m.t[i]), 0), 1) : 0;
  const L = k => m[k][i] + (m[k][j] - m[k][i]) * f;
  const p = new THREE.Vector3(m.pos[3 * i], m.pos[3 * i + 1], m.pos[3 * i + 2])
    .lerp(new THREE.Vector3(m.pos[3 * j], m.pos[3 * j + 1], m.pos[3 * j + 2]), f);
  return {i, p, alt: L("alt"), v: L("v"), q: L("q"), g: m.g[i], m: L("m"), dr: L("dr"),
          ph: m.phases[m.ph[j]] || ""};
}
function place() {
  for (const m of M) {
    for (let i = 0; i < m.t.length; i++) {
      const v = xyz(m.lat[i], m.lon[i], R + m.alt[i] * KM * exag);
      m.pos[3 * i] = v.x; m.pos[3 * i + 1] = v.y; m.pos[3 * i + 2] = v.z;
    }
    m.ghost.geometry.attributes.position.needsUpdate = true;
    m.ghost.geometry.computeBoundingSphere(); m.line.geometry.computeBoundingSphere();
    const ep = m.evPts.geometry.attributes.position;
    m.events.forEach(([, te], k) => { const s = sample(m, te); ep.setXYZ(k, s.p.x, s.p.y, s.p.z); });
    ep.needsUpdate = true; m.evPts.geometry.computeBoundingSphere();
  }
}
place();

// camera: look down on the first quarter of the flown arc
const m0 = M[0], k0 = Math.floor(m0.t.length / 4);
const look = new THREE.Vector3(m0.pos[3 * k0], m0.pos[3 * k0 + 1], m0.pos[3 * k0 + 2]).normalize();
camera.position.copy(look.multiplyScalar(R * 2.9)); controls.target.set(0, 0, 0);

// ---------- side panel ----------
const cards = document.getElementById("cards");
for (const m of M) {
  const c = document.createElement("div"); c.className = "card";
  c.innerHTML = `<h2><span class="sw" style="background:${m.color}"></span>${m.name}<span class="ph"></span></h2>
  <div class="tm"><div data-k="alt">–<span>altitude, km</span></div><div data-k="v">–<span>speed, km/s</span></div>
  <div data-k="q">–<span>dyn. pressure, kPa</span></div><div data-k="g">–<span>acceleration, g</span></div>
  <div data-k="m">–<span>mass, t</span></div><div data-k="dr">–<span>downrange, km</span></div></div>`;
  cards.append(c); m.card = c;
}
for (const id of ["lg1", "lg2"]) document.getElementById(id).innerHTML = M.length > 1
  ? M.map(m => `<span><i style="background:${m.color}"></i>${m.name}</span>`).join("") : "";
const evList = document.getElementById("events");
const allEv = M[0].events.map(([n, te]) => { const li = document.createElement("li");
  li.innerHTML = `${n}<b>${te.toFixed(0)} s</b>`; evList.append(li); return [te, li]; });

function chart(cv, key, t) {
  const r = cv.getBoundingClientRect(), dpr = Math.min(devicePixelRatio, 2);
  if (cv.width !== Math.round(r.width * dpr)) { cv.width = Math.round(r.width * dpr); cv.height = Math.round(r.height * dpr); }
  const c = cv.getContext("2d"); c.setTransform(dpr, 0, 0, dpr, 0, 0); c.clearRect(0, 0, r.width, r.height);
  const nice = x => { const p = Math.pow(10, Math.floor(Math.log10(x))); for (const s of [1, 2, 2.5, 5, 10]) if (s * p >= x) return s * p; };
  const xmax = Math.min(tEnd, Math.max(300, nice(t * 1.25)));
  let ymax = 0; for (const m of M) for (let i = 0; i < m.t.length && m.t[i] <= xmax; i++) ymax = Math.max(ymax, m[key][i]);
  const step = nice((ymax * 1.05 || 1) / 4); ymax = step * 4;
  const L = 34, B = 18, W = r.width - L - 8, H = r.height - B - 6;
  const X = x => L + x / xmax * W, Y = y => 6 + H - y / ymax * H;
  c.font = "11px Segoe UI, system-ui, sans-serif"; c.fillStyle = TK.ink2; c.strokeStyle = TK.grid; c.lineWidth = 1;
  for (let k = 0; k <= 4; k++) { const y = ymax * k / 4; c.beginPath(); c.moveTo(L, Y(y)); c.lineTo(L + W, Y(y)); c.stroke();
    c.textAlign = "right"; c.fillText(+y.toFixed(1), L - 5, Y(y) + 4); }
  for (let k = 0; k <= 4; k++) { c.textAlign = k === 4 ? "right" : "center";
    c.fillText((xmax * k / 4).toFixed(0) + (k === 4 ? " s" : ""), X(xmax * k / 4), r.height - 3); }
  for (const m of M) {
    for (const [full, alpha] of [[true, 0.25], [false, 1]]) {
      c.strokeStyle = m.color; c.globalAlpha = alpha; c.lineWidth = 2; c.lineJoin = "round"; c.beginPath();
      for (let i = 0; i < m.t.length && m.t[i] <= xmax; i++) { if (!full && m.t[i] > t) break;
        i ? c.lineTo(X(m.t[i]), Y(m[key][i])) : c.moveTo(X(m.t[i]), Y(m[key][i])); }
      c.stroke();
    }
    c.globalAlpha = 1; const s = m.s;
    c.beginPath(); c.arc(X(Math.min(t, m.t[m.t.length - 1])), Y(s[key]), 5, 0, 7);
    c.fillStyle = m.color; c.fill(); c.lineWidth = 2; c.strokeStyle = TK.surface; c.stroke();
  }
}

// ---------- time ----------
let t = 0, playing = false, last = performance.now();
const play = document.getElementById("play"), rate = document.getElementById("rate"),
  scrub = document.getElementById("scrub"), follow = document.getElementById("follow"),
  clock = document.getElementById("clock");
play.onclick = () => { if (t >= tEnd) t = 0; playing = !playing; play.textContent = playing ? "Pause" : "Play"; };
scrub.oninput = () => { t = scrub.value / 1000 * tEnd; };
document.getElementById("exag").onchange = e => { exag = e.target.checked ? 5 : 1; place(); };
addEventListener("keydown", e => { if (e.code === "Space" && e.target === document.body) { e.preventDefault(); play.click(); } });
const fmt = s => `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

function resize() { const r = host.getBoundingClientRect(); renderer.setSize(r.width, r.height, false);
  renderer.domElement.style.width = r.width + "px"; renderer.domElement.style.height = r.height + "px";
  camera.aspect = r.width / r.height; camera.updateProjectionMatrix(); }
addEventListener("resize", resize); resize();

function frame(now) {
  const dt = Math.min((now - last) / 1000, 0.5); last = now;
  if (playing) {
    const coasting = M.every(m => /coast/.test(sample(m, t).ph));
    t = Math.min(t + dt * rate.value * (coasting ? 10 : 1), tEnd);
    if (t >= tEnd) { playing = false; play.textContent = "Play"; }
    scrub.value = t / tEnd * 1000;
  }
  for (const m of M) {
    const s = sample(m, t); m.s = s;
    m.line.geometry.setDrawRange(0, s.i + 1);
    m.craft.geometry.attributes.position.setXYZ(0, s.p.x, s.p.y, s.p.z);
    m.craft.geometry.attributes.position.needsUpdate = true; m.craft.geometry.computeBoundingSphere();
    m.evPts.geometry.setDrawRange(0, m.events.filter(e => e[1] <= t).length);
    const q = k => m.card.querySelector(`[data-k=${k}]`).firstChild;
    q("alt").nodeValue = s.alt.toFixed(1); q("v").nodeValue = s.v.toFixed(3);
    q("q").nodeValue = s.q.toFixed(1); q("g").nodeValue = s.g.toFixed(2);
    q("m").nodeValue = s.m.toFixed(2); q("dr").nodeValue = s.dr.toFixed(0);
    m.card.querySelector(".ph").textContent = s.ph;
  }
  allEv.forEach(([te, li]) => li.classList.toggle("done", te <= t));
  clock.firstChild.nodeValue = `T+${fmt(t)}`;
  clock.lastChild.textContent = `${M[0].s.ph}  ·  ×${rate.value}${M.every(m => /coast/.test(m.s.ph)) ? " (coast ×10)" : ""}`;
  if (follow.checked) {
    const d = M[0].s.p.clone().sub(controls.target).multiplyScalar(0.15);
    controls.target.add(d); camera.position.add(d);
  } else controls.target.lerp(new THREE.Vector3(), 0.08);
  controls.update(); renderer.render(scene, camera);
  chart(document.getElementById("c1"), "alt", t); chart(document.getElementById("c2"), "v", t);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
</script></body></html>
"""
