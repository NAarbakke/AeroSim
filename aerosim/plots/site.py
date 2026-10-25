"""AeroSim results site: 3D flight replays in a local frame, an engines page and a hub, all in
the style of the 3D ascent replay (``replay.py``) and all working offline.

    scene_replay_html(models, title, cards, charts, body="booster")   # landing, aircraft, rocket
    engines_html(data)                                                # cycles, regen, hybrid, ...
    hub_html(entries)                                                 # index of the pages

``models`` for a replay: [{"name", "t", "pos" (n, 3) m east-north-up, "channels": {key: array},
"events": [(label, t)], optional "attitude": {"pitch", "roll", "heading"} deg arrays (aircraft),
optional "throttle" 0..1 array and "thrust_dir" (n, 3) unit vectors (rocket plume)}].
"""

from __future__ import annotations

import html

import numpy as np

from aerosim.plots.style import THEMES
from aerosim.plots.web import CHART_JS, CSS, data_tag, importmap_tag, nav, theme_tag, themed

PAGES = [("Overview", "index.html"), ("Rockot ascent", "rockot.html"),
         ("Booster landing", "landing.html"), ("F-16 flight", "f16.html"),
         ("Terrain flights", "terrain.html"),
         ("Hybrid rocket", "hybrid.html"), ("Engines", "engines.html"), ("Engine models", "engines3d.html")]


def _r(a, d):
    return np.round(np.asarray(a, float), d).tolist()


def _thin(n: int, max_points: int) -> np.ndarray:
    return np.unique(np.r_[np.linspace(0, n - 1, min(n, max_points)).astype(int), n - 1])


def _pack_model(m: dict, color: str, max_points: int = 3000) -> dict:
    t = np.asarray(m["t"], float)
    idx = _thin(t.size, max_points)
    pos = np.asarray(m["pos"], float)[idx]
    out = {"name": m["name"], "color": color, "t": _r(t[idx], 3),
           "x": _r(pos[:, 0], 2), "y": _r(pos[:, 1], 2), "z": _r(pos[:, 2], 2),
           "ch": {k: _r(np.asarray(v, float)[idx], 4) for k, v in m["channels"].items()},
           "events": [[lab, float(te)] for lab, te in m.get("events", [])]}
    if "attitude" in m:
        out["att"] = {k: _r(np.asarray(v, float)[idx], 2) for k, v in m["attitude"].items()}
    if "throttle" in m:
        out["thr"] = _r(np.asarray(m["throttle"], float)[idx], 3)
    if "thrust_dir" in m:
        d = np.asarray(m["thrust_dir"], float)[idx]
        out["dir"] = _r(d.ravel(), 3)
    return out


def _page(title: str, current: str, body: str, theme: str, extra_css: str = "") -> str:
    return themed(f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title><style>{CSS}{extra_css}</style></head><body>
{nav(PAGES, current)}
{body}
</body></html>""", theme)


# ------------------------------------------------------------------ 3D replay in a local frame
_SCENE_CSS = r"""
html,body{height:100%}body{display:flex;flex-direction:column}
#app{flex:1;min-height:0;display:grid;grid-template-columns:minmax(0,1fr) 400px;
grid-template-rows:minmax(0,1fr) auto}
#scene{position:relative;min-height:0}#scene canvas{display:block}
#hud{position:absolute;left:20px;top:16px;pointer-events:none;max-width:70%}
#hud h1{margin:0;font-size:20px;font-weight:600;letter-spacing:.2px}
#hud p{margin:2px 0 0;color:var(--ink2);font-size:13px}
#clock{position:absolute;left:20px;bottom:14px;font:600 30px/1 Consolas,"SF Mono",monospace;
pointer-events:none;font-variant-numeric:tabular-nums}
#clock small{display:block;font:500 12px/1.6 "Segoe UI",system-ui,sans-serif;color:var(--ink2)}
#side{grid-row:1/3;grid-column:2;overflow-y:auto;background:var(--surface);
border-left:1px solid var(--grid);padding:16px 18px 20px}
canvas.chart.t{height:150px}
#events{list-style:none;margin:0;padding:0;font-size:12.5px}
#events li{display:flex;gap:8px;padding:3px 0;color:var(--muted);border-bottom:1px solid var(--grid)}
#events li.done{color:var(--ink)}#events li b{font-weight:500;margin-left:auto;
font-variant-numeric:tabular-nums;color:var(--ink2)}
#bar{grid-column:1;display:flex;align-items:center;gap:12px;padding:10px 20px;
background:var(--surface);border-top:1px solid var(--grid);flex-wrap:wrap}
#play{min-width:76px;font-weight:600}#scrub{flex:1;min-width:160px;accent-color:var(--accent)}
@media (max-width:900px){#app{grid-template-columns:1fr;grid-template-rows:60vh auto auto}
#side{grid-row:3;grid-column:1;border-left:0;border-top:1px solid var(--grid)}#bar{grid-row:2}
#clock{font-size:22px}}
"""

_SCENE_JS = r"""
import * as THREE from "three";
import {OrbitControls} from "three/addons/controls/OrbitControls.js";
const D = JSON.parse(document.getElementById("data").textContent), TK = TKC;
const M = D.models, tEnd = Math.max(...M.map(m => m.t[m.t.length - 1]));
// scene frame: x = east, y = up, z = -north, in units of U metres
let lo = [1e99, 1e99, 1e99], hi = [-1e99, -1e99, -1e99];
for (const m of M) for (let i = 0; i < m.t.length; i++) { const p = [m.x[i], m.y[i], m.z[i]];
  for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], p[k]); hi[k] = Math.max(hi[k], p[k]); } }
const span = Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - Math.min(lo[2], 0), 1);
const U = span / 10, S = (x, y, z) => new THREE.Vector3(x / U, z / U, -y / U);
const ctr = S((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (Math.min(lo[2], 0) + hi[2]) / 2);

const host = document.getElementById("scene");
const renderer = new THREE.WebGLRenderer({antialias: true});
renderer.setPixelRatio(Math.min(devicePixelRatio, 2)); host.prepend(renderer.domElement);
const scene = new THREE.Scene(); scene.background = new THREE.Color(TK.page);
const camera = new THREE.PerspectiveCamera(40, 1, 0.001, 500);
const controls = new OrbitControls(camera, renderer.domElement); controls.enableDamping = true;
scene.add(new THREE.AmbientLight(0xffffff, TK.dark ? 0.6 : 0.8));
const sun = new THREE.DirectionalLight(0xffffff, 1.0); sun.position.set(-3, 5, 2); camera.add(sun); scene.add(camera);

// ground: grid around the flown area, pad / start marker
const gsize = Math.ceil(Math.max(hi[0] - lo[0], hi[1] - lo[1], span * 0.6) / U * 1.4);
const grid = new THREE.GridHelper(gsize, 20, new THREE.Color(TK.axis), new THREE.Color(TK.grid));
grid.position.set(ctr.x, 0, ctr.z); scene.add(grid);
const pad = new THREE.Mesh(new THREE.RingGeometry(0.12, 0.18, 48),
  new THREE.MeshBasicMaterial({color: TK.ink, side: THREE.DoubleSide}));
pad.rotation.x = -Math.PI / 2; pad.position.y = 0.002; scene.add(pad);

// vehicle models (not to scale when the flight is large): body axis +y
const vlen = Math.max(D.vehicle.length / U, 0.5), vscale = vlen / (D.vehicle.length / U);
document.getElementById("scale").textContent = vscale > 1.5 ? `vehicle drawn ×${vscale.toFixed(0)}` : "";
function rocketMesh(color, booster) {
  const g = new THREE.Group(), L = vlen, r = L * D.vehicle.diameter / D.vehicle.length;
  const mat = new THREE.MeshPhongMaterial({color, shininess: 30});
  const body = new THREE.Mesh(new THREE.CylinderGeometry(r, r, L, 24), mat); body.position.y = L / 2; g.add(body);
  if (!booster) { const nose = new THREE.Mesh(new THREE.ConeGeometry(r, L * 0.18, 24), mat); nose.position.y = L * 1.09; g.add(nose); }
  else for (let k = 0; k < 4; k++) { const a = k * Math.PI / 2 + Math.PI / 4;
    const leg = new THREE.Mesh(new THREE.CylinderGeometry(r * 0.12, r * 0.12, L * 0.3, 6), mat);
    leg.position.set(Math.cos(a) * r * 1.6, L * 0.1, Math.sin(a) * r * 1.6);
    leg.rotation.set(Math.sin(a) * 0.5, 0, -Math.cos(a) * 0.5); g.add(leg); }
  const fins = new THREE.MeshPhongMaterial({color: TK.ink2});
  for (let k = 0; k < (booster ? 0 : 4); k++) { const f = new THREE.Mesh(new THREE.BoxGeometry(r * 1.6, L * 0.14, r * 0.12), fins);
    f.position.y = L * 0.08; f.rotation.y = k * Math.PI / 2; f.translateX(r * 1.1); g.add(f); }
  const plume = new THREE.Mesh(new THREE.ConeGeometry(r * 0.9, L * 0.6, 20, 1, true),
    new THREE.MeshBasicMaterial({color: 0xffb347, transparent: true, opacity: 0.75, side: THREE.DoubleSide}));
  plume.rotation.x = Math.PI; plume.position.y = -L * 0.3; g.add(plume); g.plume = plume; g.L = L;
  return g;
}
function aircraftMesh(color) {
  const g = new THREE.Group(), L = vlen, mat = new THREE.MeshPhongMaterial({color, shininess: 30, side: THREE.DoubleSide});
  const fus = new THREE.Mesh(new THREE.CylinderGeometry(L * 0.045, L * 0.06, L, 16), mat);
  fus.rotation.z = -Math.PI / 2; g.add(fus);                                    // nose along +x
  const wing = new THREE.Shape(); wing.moveTo(L * 0.15, 0); wing.lineTo(-L * 0.25, L * 0.32);
  wing.lineTo(-L * 0.33, L * 0.32); wing.lineTo(-L * 0.3, 0); wing.lineTo(-L * 0.33, -L * 0.32);
  wing.lineTo(-L * 0.25, -L * 0.32); wing.closePath();
  const w = new THREE.Mesh(new THREE.ShapeGeometry(wing), mat); w.rotation.x = Math.PI / 2; g.add(w);
  const fin = new THREE.Shape(); fin.moveTo(-L * 0.3, 0); fin.lineTo(-L * 0.5, L * 0.22); fin.lineTo(-L * 0.5, 0); fin.closePath();
  g.add(new THREE.Mesh(new THREE.ShapeGeometry(fin), mat));
  const plume = new THREE.Mesh(new THREE.ConeGeometry(L * 0.05, L * 0.4, 16, 1, true),
    new THREE.MeshBasicMaterial({color: 0xffb347, transparent: true, opacity: 0.7, side: THREE.DoubleSide}));
  plume.rotation.z = Math.PI / 2; plume.position.x = -L * 0.7; g.add(plume); g.plume = plume; g.L = L;
  return g;
}
const disc = (() => { const c = document.createElement("canvas"); c.width = c.height = 64;
  const x = c.getContext("2d"); x.fillStyle = "#fff"; x.beginPath(); x.arc(32, 32, 30, 0, 7); x.fill();
  return new THREE.CanvasTexture(c); })();
const dots = (color, size, n) => { const p = new THREE.Points(new THREE.BufferGeometry(),
  new THREE.PointsMaterial({color, size, sizeAttenuation: false, map: disc, transparent: true, alphaTest: 0.5}));
  p.geometry.setAttribute("position", new THREE.Float32BufferAttribute(new Float32Array(n * 3), 3)); return p; };

let exag = 1;
function place() {                      // positions, with the optional altitude exaggeration
  for (const m of M) {
    for (let i = 0; i < m.t.length; i++) { const v = S(m.x[i], m.y[i], m.z[i] * exag);
      m.pos.set([v.x, v.y, v.z], 3 * i); m.gnd.set([v.x, 0.001, v.z], 3 * i); }
    for (const l of [m.ghost, m.line, m.track]) { l.geometry.attributes.position.needsUpdate = true; l.geometry.computeBoundingSphere(); }
    m.events.forEach(([, te], k) => { const s = sample(m, te); m.evPts.geometry.attributes.position.setXYZ(k, s.p.x, s.p.y, s.p.z); });
    m.evPts.geometry.attributes.position.needsUpdate = true;
  }
}
for (const m of M) {
  const n = m.t.length; m.pos = new Float32Array(n * 3); m.gnd = new Float32Array(n * 3);
  const mk = (arr, opa) => { const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.BufferAttribute(arr, 3));
    return new THREE.Line(g, new THREE.LineBasicMaterial({color: m.color, transparent: true, opacity: opa})); };
  m.ghost = mk(m.pos, 0.45); m.line = mk(m.pos, 1); m.track = mk(m.gnd, 0.35);
  m.craft = D.body === "aircraft" ? aircraftMesh(m.color) : rocketMesh(m.color, D.body === "booster");
  m.drop = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]),
    new THREE.LineDashedMaterial({color: TK.ink2, dashSize: 0.08, gapSize: 0.06, transparent: true, opacity: 0.6}));
  m.evPts = dots(m.color, 8, m.events.length);
  scene.add(m.ghost, m.line, m.track, m.craft, m.drop, m.evPts);
}
place();
const ex = document.getElementById("exag");
if (D.exag) ex.onchange = e => { exag = e.target.checked ? D.exag : 1; place(); fit(); };
else ex.parentElement.remove();
function find(a, x) { let lo = 0, hi = a.length - 1; if (x <= a[0]) return 0; if (x >= a[hi]) return hi - 1;
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (a[mid] <= x) lo = mid; else hi = mid; } return lo; }
function sample(m, t) {
  const i = find(m.t, t), j = Math.min(i + 1, m.t.length - 1);
  const f = j > i ? Math.min(Math.max((t - m.t[i]) / (m.t[j] - m.t[i]), 0), 1) : 0;
  const L = a => a[i] + (a[j] - a[i]) * f;
  const P = k => new THREE.Vector3(m.pos[3 * k], m.pos[3 * k + 1], m.pos[3 * k + 2]);
  const p = P(i).lerp(P(j), f), vel = P(j).sub(P(i));
  const s = {i, p, vel, ch: {}}; for (const k in m.ch) s.ch[k] = L(m.ch[k]);
  if (m.att) s.att = {pitch: L(m.att.pitch), roll: L(m.att.roll), heading: m.att.heading[i]};
  if (m.thr) s.thr = L(m.thr);
  if (m.dir) s.dir = S(m.dir[3 * i], m.dir[3 * i + 1], m.dir[3 * i + 2]).normalize();
  return s;
}
const up = new THREE.Vector3(0, 1, 0), rad = Math.PI / 180;
function pose(m, s) {
  const g = m.craft; g.position.copy(s.p);
  if (D.body === "aircraft" && s.att) {
    g.rotation.set(s.att.roll * rad, (90 - s.att.heading) * rad, s.att.pitch * rad, "YZX");
  } else {
    let d = null;
    if (s.dir && s.thr > 0.01) d = s.dir.clone();
    else if (s.vel.lengthSq() > 1e-12) d = D.body === "booster" ? s.vel.clone().negate().normalize() : s.vel.clone().normalize();
    if (d && D.body === "booster" && d.y < 0) d.negate();
    if (d) g.quaternion.setFromUnitVectors(up, d);
  }
  const thr = s.thr ?? 0; g.plume.visible = thr > 0.01; g.plume.scale.set(1, 0.4 + thr, 1);
  const a = m.drop.geometry.attributes.position; a.setXYZ(0, s.p.x, s.p.y, s.p.z); a.setXYZ(1, s.p.x, 0, s.p.z);
  a.needsUpdate = true; m.drop.computeLineDistances();
}

// camera: from the south-east and above, far enough to see the whole flight (refit on exaggeration)
function fit() {
  const zl = Math.min(lo[2], 0) * exag, zh = hi[2] * exag;
  const diag = Math.hypot(hi[0] - lo[0], hi[1] - lo[1], zh - zl) / U;
  const dist = 0.5 * diag / Math.sin(20 * rad) * 1.1 * (D.camera || 1);
  const c = S((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (zl + zh) / 2);
  camera.position.copy(c).add(new THREE.Vector3(0.7, 0.45, 0.75).normalize().multiplyScalar(dist));
  controls.target.copy(c); camera.far = dist * 20; camera.updateProjectionMatrix();
}
fit();

// ---------- side panel ----------
const cards = document.getElementById("cards");
for (const m of M) {
  const c = document.createElement("div"); c.className = "card";
  c.innerHTML = `<h2><span class="sw" style="background:${m.color}"></span>${m.name}<span class="ph"></span></h2>
   <div class="tm">${D.cards.map(k => `<div data-k="${k.k}">–<span>${k.label}</span></div>`).join("")}</div>`;
  cards.append(c); m.card = c;
}
const charts = document.getElementById("charts");
D.charts.forEach((ch, n) => { charts.insertAdjacentHTML("beforeend",
  `<h3>${ch.label}</h3><div class="legend" id="lg${n}"></div><canvas class="chart t" id="tc${n}"></canvas>`);
  legend(document.getElementById("lg" + n), M); });
const evList = document.getElementById("events");
const allEv = M[0].events.map(([n, te]) => { const li = document.createElement("li");
  li.innerHTML = `${n}<b>${te.toFixed(1)} s</b>`; evList.append(li); return [te, li]; });
if (!allEv.length) document.getElementById("evh").remove();
function timeChart(cv, key, t) {
  const [c, w, h] = setup(cv);
  let ymin = 1e99, ymax = -1e99; for (const m of M) for (const v of m.ch[key]) { ymin = Math.min(ymin, v); ymax = Math.max(ymax, v); }
  const [y0, y1, yst] = axisRange(ymin, ymax, 4, D.zero?.includes(key)), [, x1, xst] = axisRange(0, tEnd, 4, true);
  const L = 42, B = 18, T = 6, W = w - L - 10, H = h - B - T, X = x => L + x / x1 * W, Y = y => T + H - (y - y0) / (y1 - y0) * H;
  c.strokeStyle = TK.grid; c.lineWidth = 1; c.fillStyle = TK.ink2;
  for (let y = y0; y <= y1 + yst / 2; y += yst) { c.beginPath(); c.moveTo(L, Y(y)); c.lineTo(L + W, Y(y)); c.stroke();
    c.textAlign = "right"; c.fillText(fmtN(y, yst), L - 6, Y(y) + 4); }
  for (let x = 0; x <= x1 + xst / 2; x += xst) { c.textAlign = x + xst > x1 + xst / 2 ? "right" : "center";
    c.fillText(fmtN(x, xst) + (x + xst > x1 + xst / 2 ? " s" : ""), X(x), h - 3); }
  for (const m of M) {
    for (const [full, alpha] of [[true, 0.25], [false, 1]]) {
      c.strokeStyle = m.color; c.globalAlpha = alpha; c.lineWidth = 2; c.lineJoin = "round"; c.beginPath();
      for (let i = 0; i < m.t.length; i++) { if (!full && m.t[i] > t) break;
        i ? c.lineTo(X(m.t[i]), Y(m.ch[key][i])) : c.moveTo(X(m.t[i]), Y(m.ch[key][i])); }
      c.stroke();
    }
    c.globalAlpha = 1; c.beginPath(); c.arc(X(Math.min(t, m.t[m.t.length - 1])), Y(m.s.ch[key]), 5, 0, 7);
    c.fillStyle = m.color; c.fill(); c.lineWidth = 2; c.strokeStyle = TK.surface; c.stroke();
  }
}

// ---------- time ----------
let t = 0, playing = false, last = performance.now();
const play = document.getElementById("play"), rate = document.getElementById("rate"),
  scrub = document.getElementById("scrub"), follow = document.getElementById("follow"),
  clock = document.getElementById("clock");
for (const r of D.rates) rate.insertAdjacentHTML("beforeend", `<option value="${r}"${r === D.rate ? " selected" : ""}>×${r}</option>`);
play.onclick = () => { if (t >= tEnd) t = 0; playing = !playing; play.textContent = playing ? "Pause" : "Play"; };
scrub.oninput = () => { t = scrub.value / 1000 * tEnd; };
addEventListener("keydown", e => { if (e.code === "Space" && e.target === document.body) { e.preventDefault(); play.click(); } });
function resize() { const r = host.getBoundingClientRect(); renderer.setSize(r.width, r.height, false);
  renderer.domElement.style.width = r.width + "px"; renderer.domElement.style.height = r.height + "px";
  camera.aspect = r.width / r.height; camera.updateProjectionMatrix(); }
addEventListener("resize", resize); resize();
const fmtT = s => tEnd < 120 ? s.toFixed(1).padStart(5, "0") + " s"
  : `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
function frame(now) {
  const dt = Math.min((now - last) / 1000, 0.5); last = now;
  if (playing) { t = Math.min(t + dt * rate.value, tEnd); if (t >= tEnd) { playing = false; play.textContent = "Play"; }
    scrub.value = t / tEnd * 1000; }
  for (const m of M) {
    const s = sample(m, t); m.s = s; pose(m, s);
    m.line.geometry.setDrawRange(0, s.i + 1); m.track.geometry.setDrawRange(0, s.i + 1);
    m.evPts.geometry.setDrawRange(0, m.events.filter(e => e[1] <= t).length);
    for (const k of D.cards) m.card.querySelector(`[data-k="${k.k}"]`).firstChild.nodeValue = s.ch[k.k].toFixed(k.d);
    const ev = m.events.filter(e => e[1] <= t).pop(); m.card.querySelector(".ph").textContent = ev ? ev[0] : D.phase0 || "";
  }
  allEv.forEach(([te, li]) => li.classList.toggle("done", te <= t));
  clock.firstChild.nodeValue = `T+${fmtT(t)}`; clock.lastChild.textContent = `×${rate.value}`;
  if (follow.checked) { const d = M[0].s.p.clone().sub(controls.target).multiplyScalar(0.15);
    controls.target.add(d); camera.position.add(d); }
  controls.update(); renderer.render(scene, camera);
  D.charts.forEach((ch, n) => timeChart(document.getElementById("tc" + n), ch.k, t));
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
"""


def scene_replay_html(models: list[dict], title: str, cards: list[dict], charts: list[dict],
                      body: str, vehicle: dict, subtitle: str = "", rates=(1, 2, 5, 10),
                      rate: float = 1, current: str = "", theme: str = "dark",
                      zero: tuple = (), camera: float = 1.0, phase0: str = "", exag: float = 0) -> str:
    """3D replay of local-frame flights (positions in m, east-north-up from the pad / start).

    cards: [{"k": channel, "label": "altitude, m", "d": decimals}] per model card;
    charts: [{"k": channel, "label": "Altitude (m)"}] time charts; body: "booster" | "rocket" |
    "aircraft"; vehicle: {"length", "diameter"} m (drawn enlarged when the flight is large);
    exag: offer an "Altitude ×exag" toggle (0 = none).
    """
    T = THEMES[theme]
    data = {"models": [_pack_model(m, T["series"][i]) for i, m in enumerate(models)],
            "cards": cards, "charts": charts, "body": body, "vehicle": vehicle,
            "rates": list(rates), "rate": rate, "zero": list(zero), "camera": camera, "phase0": phase0,
            "exag": exag}
    page = f"""<div id="app">
 <div id="scene"><div id="hud"><h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}
  <span id="scale"></span></p></div><div id="clock">T+0<small>&nbsp;</small></div></div>
 <div id="side"><div id="cards"></div><div id="charts"></div><h3 id="evh">Events</h3><ul id="events"></ul></div>
 <div id="bar"><button id="play" aria-label="Play or pause">Play</button>
  <select id="rate" aria-label="Time scale"></select>
  <input id="scrub" type="range" min="0" max="1000" value="0" aria-label="Flight time">
  <label><input type="checkbox" id="follow"> Follow vehicle</label>
  <label><input type="checkbox" id="exag"> Altitude ×{exag}</label></div>
</div>
{theme_tag(theme)}{data_tag(data)}{importmap_tag()}
<script type="module">{CHART_JS}{_SCENE_JS}</script>"""
    return _page(title, current, page, theme, _SCENE_CSS)


# ------------------------------------------------------------------ engines page
_ENG_CSS = r"""
main{max-width:1240px;margin:0 auto;padding:18px 20px 40px}
main>h1{font-size:22px;margin:6px 0 2px}main>p.lead{color:var(--ink2);margin:0 0 18px;max-width:820px}
section{margin:26px 0 8px}section>h2{font-size:17px;margin:0 0 2px}
section>p{color:var(--ink2);margin:0 0 12px;max-width:860px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:12px}
.panel{border:1px solid var(--grid);border-radius:8px;background:var(--surface);padding:12px 14px}
.panel h3{margin:0 0 4px}.panel .cap{font-size:12px;color:var(--ink2);margin:6px 0 0}
#chamber{height:340px;position:relative;border-radius:6px;overflow:hidden}
#chamber canvas{display:block;width:100%;height:100%}.panel{min-width:0}
.cbar{display:flex;align-items:center;gap:8px;font-size:12px;color:var(--ink2)}
.cbar canvas{height:10px;flex:1;min-width:0;width:100%;border-radius:3px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:8px;margin:0 0 12px}
.tile{border:1px solid var(--grid);border-radius:8px;padding:8px 12px;background:var(--surface)}
.tile b{display:block;font-size:20px;font-variant-numeric:tabular-nums}.tile span{font-size:12px;color:var(--ink2)}
"""

_ENG_JS = r"""
import * as THREE from "three";
import {OrbitControls} from "three/addons/controls/OrbitControls.js";
const E = JSON.parse(document.getElementById("data").textContent), C = TKC.series;
const $ = id => document.getElementById(id);
function plot(id, series, o) { legend($(id + "-lg"), series); lineChart($(id), {series, ...o}); }

// propulsion map: fuel specific impulse against Mach
plot("isp", E.map.map((s, i) => ({name: s.name, color: C[i], x: s.mach, y: s.isp, marks: s.mach.length < 3})),
  {xl: "Mach", yl: "Isp, s", yu: "s", digits: 0, xdigits: 2, zero: true});

// liquid cycles
devChart($("dev"), E.cycles.dev.map(r => ({...r, color: C[r.c]})));
$("cyc").innerHTML = `<tr><th>Engine</th><th>Cycle</th><th>Pc, MPa</th><th>Isp vac, s</th>
  <th>Turbine flow</th><th>Turbine PR</th><th>Pump power, MW</th><th>Fuel pump out, MPa</th></tr>` +
  E.cycles.rows.map(r => `<tr><td>${r.name}</td><td class="muted">${r.cycle}</td><td>${r.pc.toFixed(1)}</td>
  <td>${r.isp.toFixed(1)}</td><td>${(r.tf * 100).toFixed(1)} %</td><td>${r.pr.toFixed(2)}</td>
  <td>${r.pw.toFixed(r.pw < 1 ? 2 : 1)}</td><td>${r.pf.toFixed(1)}</td></tr>`).join("");

// regen cooling
const R = E.regen;
plot("rT", [{name: "Hot wall", color: C[1], x: R.x, y: R.t_hot}, {name: "Coolant-side wall", color: C[3], x: R.x, y: R.t_cold},
  {name: "Coolant", color: C[0], x: R.x, y: R.t_cool}], {xl: "axial position, m (throat at 0)", yu: "K", digits: 0, xdigits: 3, zero: true});
plot("rq", [{name: "Heat flux", color: C[1], x: R.x, y: R.q}], {xl: "axial position, m (throat at 0)", yu: "MW/m²", digits: 1, xdigits: 3, zero: true});
(function chamber() {
  const host = $("chamber"), r = new THREE.WebGLRenderer({antialias: true});
  r.setPixelRatio(Math.min(devicePixelRatio, 2)); host.append(r.domElement);
  const sc = new THREE.Scene(); sc.background = new THREE.Color(TKC.surface);
  const cam = new THREE.PerspectiveCamera(35, 1, 0.01, 50), ctl = new OrbitControls(cam, r.domElement);
  ctl.enableDamping = true; sc.add(new THREE.AmbientLight(0xffffff, 0.7));
  const l = new THREE.DirectionalLight(0xffffff, 0.9); l.position.set(2, 3, 4); cam.add(l); sc.add(cam);
  const cx = R.contour_x, cr = R.contour_r, n = cx.length, seg = 96;
  const lo = Math.min(...R.t_hot), hi = Math.max(...R.t_hot);
  const ramp = [new THREE.Color(TKC.dark ? "#4a2410" : "#fbe1cf"), new THREE.Color(TKC.dark ? "#ffd7b8" : "#8a2f08")];
  const col = T => ramp[0].clone().lerp(ramp[1], Math.min(Math.max((T - lo) / (hi - lo), 0), 1));
  const pos = [], cols = [], idx = [];
  const order = R.x.map((x, i) => [x, R.t_hot[i]]).sort((a, b) => a[0] - b[0]);
  const Tat = x => { if (x <= order[0][0]) return order[0][1]; if (x >= order[order.length - 1][0]) return order[order.length - 1][1];
    for (let i = 1; i < order.length; i++) if (order[i][0] >= x) { const [x0, t0] = order[i - 1], [x1, t1] = order[i];
      return t0 + (t1 - t0) * (x - x0) / (x1 - x0); } };
  for (let i = 0; i < n; i++) { const c = col(Tat(cx[i]));
    for (let k = 0; k <= seg; k++) { const a = k / seg * Math.PI * 1.5;          // three-quarter section: see inside
      pos.push(cx[i], cr[i] * Math.cos(a), cr[i] * Math.sin(a)); cols.push(c.r, c.g, c.b); } }
  for (let i = 0; i < n - 1; i++) for (let k = 0; k < seg; k++) { const a = i * (seg + 1) + k, b = a + seg + 1;
    idx.push(a, b, a + 1, b, b + 1, a + 1); }
  const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute("color", new THREE.Float32BufferAttribute(cols, 3)); g.setIndex(idx); g.computeVertexNormals();
  const mesh = new THREE.Mesh(g, new THREE.MeshPhongMaterial({vertexColors: true, side: THREE.DoubleSide, shininess: 20}));
  const span = Math.max(...cx) - Math.min(...cx), mid = (Math.max(...cx) + Math.min(...cx)) / 2;
  mesh.position.x = -mid; sc.add(mesh);
  cam.position.set(span * 0.35, span * 0.7, span * 1.9); ctl.target.set(0, 0, 0);
  const rs = () => { const b = host.getBoundingClientRect(); r.setSize(b.width, b.height, false); cam.aspect = b.width / b.height; cam.updateProjectionMatrix(); };
  addEventListener("resize", rs); rs();
  (function loop() { ctl.update(); r.render(sc, cam); requestAnimationFrame(loop); })();
  const bar = $("cbar"); bar.width = 256; bar.height = 1;
  const x = bar.getContext("2d"); for (let i = 0; i < 256; i++) { x.fillStyle = "#" + col(lo + (hi - lo) * i / 255).getHexString(); x.fillRect(i, 0, 1, 1); }
  $("clo").textContent = lo.toFixed(0) + " K"; $("chi").textContent = hi.toFixed(0) + " K";
})();

// hybrid burn
const H = E.hybrid;
plot("hF", [{name: "Thrust", color: C[0], x: H.t, y: H.thrust}], {xl: "time, s", yu: "kN", digits: 2, xdigits: 1, zero: true});
plot("hP", [{name: "N2O tank", color: C[0], x: H.t, y: H.p_tank}, {name: "Chamber", color: C[1], x: H.t, y: H.pc}],
  {xl: "time, s", yu: "MPa", digits: 2, xdigits: 1, zero: true});
plot("hOF", [{name: "O/F", color: C[2], x: H.t, y: H.of}], {xl: "time, s", digits: 2, xdigits: 1, zero: true});
plot("hR", [{name: "Port radius", color: C[3], x: H.t, y: H.r}], {xl: "time, s", yu: "mm", digits: 1, xdigits: 1});

// ramjet / scramjet
plot("rjI", E.ram.map((s, i) => ({name: s.name, color: C[i], x: s.mach, y: s.isp})), {xl: "Mach", yu: "s", digits: 0, xdigits: 1, zero: true});
plot("rjF", E.ram.map((s, i) => ({name: s.name, color: C[i], x: s.mach, y: s.sf})), {xl: "Mach", yu: "N·s/kg", digits: 0, xdigits: 1, zero: true});
plot("rjT", E.ram.map((s, i) => ({name: s.name, color: C[i], x: s.mach, y: s.t4})), {xl: "Mach", yu: "K", digits: 0, xdigits: 1});

// gas turbines
for (const [id, k] of [["tjF", "turbojet"], ["tfF", "turbofan"]])
  plot(id, E.gt[k].lines.map((s, i) => ({name: s.name, color: C[i], x: s.mach, y: s.thrust, marks: true})),
    {xl: "Mach", yu: "kN", digits: 1, xdigits: 2, zero: true});
for (const [id, k] of [["tjS", "turbojet"], ["tfS", "turbofan"]])
  plot(id, E.gt[k].lines.map((s, i) => ({name: s.name, color: C[i], x: s.mach, y: s.tsfc, marks: true})),
    {xl: "Mach", yu: "lb/(lbf·h)", digits: 3, xdigits: 2});
$("gtv").innerHTML = `<tr><th>Engine</th><th>Quantity</th><th>Model</th><th>Published</th></tr>` +
  E.gt.validation.map(r => `<tr><td>${r.engine}</td><td class="muted">${r.q}</td><td>${r.model}</td><td>${r.pub}</td></tr>`).join("");
"""


def engines_html(data: dict, theme: str = "dark") -> str:
    """Engines page. ``data`` keys: map, cycles, regen, hybrid, ram, gt (see examples/build_site.py)."""
    def panel(id_, title, cap="", h=170):
        return (f'<div class="panel"><h3>{title}</h3><div class="legend" id="{id_}-lg"></div>'
                f'<div class="cw"><canvas class="chart" id="{id_}" style="height:{h}px"></canvas></div>'
                + (f'<p class="cap">{cap}</p>' if cap else "") + "</div>")
    h = data["hybrid"]
    tiles = "".join(f'<div class="tile"><b>{v}</b><span>{k}</span></div>' for k, v in h["tiles"].items())
    body = f"""<main>
<h1>Engines</h1>
<p class="lead">Every engine model in AeroSim on one page: liquid cycles with their turbopump power
balance, a regeneratively cooled chamber from pyskyfire, the hybrid motor's burn, ramjet and
scramjet cycles on Cantera, and pyCycle gas turbines. Hover any chart for values.</p>

<section><h2>Propulsion map</h2>
<p>Fuel specific impulse (thrust per unit fuel flow) against flight Mach: rockets carry their oxidizer
and sit far below the air-breathers, which collect it from the air; each air-breather has a speed band.
Gas turbines at 0.85 power, ramjet and scramjet at stoichiometric fuel on a 50 kPa dynamic-pressure
climb; rocket values are vacuum Isp.</p>
<div class="grid">{panel("isp", "Isp against Mach (s)", h=300)}</div></section>

<section><h2>Liquid engine cycles</h2>
<p>Turbopump power balance (<code>propulsion/liquid/cycle.py</code>): CEA drive gas, CoolProp pump work,
textbook efficiencies. Each dot is the model's deviation from a published figure (hover for the
numbers).</p>
<div class="grid"><div class="panel"><h3>Model against published data</h3>
<div class="cw"><canvas class="chart" id="dev"></canvas></div>
<p class="cap">{html.escape(data["cycles"]["note"])}</p></div>
<div class="panel"><h3>Cycle solutions</h3><table id="cyc"></table></div></div></section>

<section><h2>Regenerative cooling</h2>
<p>{html.escape(data["regen"]["caption"])}</p>
<div class="grid"><div class="panel"><h3>Chamber coloured by hot-wall temperature</h3>
<div id="chamber"></div><div class="cbar"><span id="clo"></span><canvas id="cbar"></canvas><span id="chi"></span></div>
<p class="cap">Drag to rotate, scroll to zoom. Three-quarter section, injector on the left.</p></div>
{panel("rT", "Temperatures along the chamber (K)", h=200)}
{panel("rq", "Hot-side heat flux (MW/m²)", h=200)}</div></section>

<section><h2>Hybrid motor</h2>
<p>{html.escape(h["caption"])}</p><div class="tiles">{tiles}</div>
<div class="grid">{panel("hF", "Vacuum thrust (kN)")}{panel("hP", "Pressures (MPa)")}
{panel("hOF", "Oxidizer-to-fuel ratio")}{panel("hR", "Port radius (mm)")}</div></section>

<section><h2>Ramjet and scramjet</h2>
<p>{html.escape(data["ram_caption"])}</p>
<div class="grid">{panel("rjI", "Fuel specific impulse (s)")}{panel("rjF", "Specific thrust (N·s per kg of air)")}
{panel("rjT", "Burner exit temperature (K)")}</div></section>

<section><h2>Gas turbines (pyCycle)</h2>
<p>{html.escape(data["gt"]["caption"])}</p>
<div class="grid">{panel("tjF", "Turbojet net thrust (kN)")}{panel("tjS", "Turbojet TSFC, lb/(lbf·h)")}
{panel("tfF", "Turbofan net thrust (kN)")}{panel("tfS", "Turbofan TSFC, lb/(lbf·h)")}
<div class="panel"><h3>Design points against published data</h3><table id="gtv"></table></div></div></section>
</main>
{theme_tag(theme)}{data_tag(data)}{importmap_tag()}
<script type="module">{CHART_JS}{_ENG_JS}</script>"""
    return _page("AeroSim engines", "engines.html", body, theme, _ENG_CSS)


# ------------------------------------------------------------------ hub
_HUB_CSS = r"""
main{max-width:1100px;margin:0 auto;padding:22px 20px 40px}
main>h1{font-size:24px;margin:4px 0 2px}main>p{color:var(--ink2);margin:0 0 18px;max-width:780px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px}
a.tile{display:block;text-decoration:none;border:1px solid var(--grid);border-radius:10px;
background:var(--surface);padding:14px 16px;color:var(--ink)}
a.tile:hover{border-color:var(--accent)}a.tile h2{margin:0 0 4px;font-size:16px}
a.tile p{margin:0 0 10px;color:var(--ink2);font-size:13px}
.kv{display:grid;grid-template-columns:repeat(3,1fr);gap:6px}.kv div{font-variant-numeric:tabular-nums;
font-weight:600}.kv span{display:block;font-size:11px;font-weight:400;color:var(--ink2)}
"""


def hub_html(entries: list[dict], theme: str = "dark") -> str:
    """entries: [{"href", "title", "text", "stats": {label: value}}]."""
    tiles = "".join(
        f'<a class="tile" href="{e["href"]}"><h2>{html.escape(e["title"])}</h2><p>{html.escape(e["text"])}</p>'
        f'<div class="kv">' + "".join(f"<div>{html.escape(str(v))}<span>{html.escape(k)}</span></div>"
                                      for k, v in e["stats"].items()) + "</div></a>" for e in entries)
    body = f"""<main><h1>AeroSim results</h1>
<p>Flights and engines modelled in AeroSim. 3D replays run in sped-up time; drag to orbit, scroll to
zoom. Every page works offline.</p><div class="cards">{tiles}</div></main>"""
    return _page("AeroSim results", "index.html", body, theme, _HUB_CSS)
