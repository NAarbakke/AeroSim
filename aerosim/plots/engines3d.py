"""Interactive 3D engine figures for the results site: one page, every engine model.

    engines3d_html([engine_models.liquid_engine(...), engine_models.solid_motor(...), ...])

A three.js cutaway per engine (bodies from ``aerosim.plots.engine_models``), coloured by a
selectable field (gas temperature, pressure, Mach, wall temperature, total temperature...)
with a colour bar; solid and hybrid motors have a burn-time slider that regresses the grain
and drains the tank. Same layout and theme as the 3D replays; works offline.
"""

from __future__ import annotations

from aerosim.plots.site import _page
from aerosim.plots.web import CHART_JS, data_tag, importmap_tag, theme_tag

_CSS = r"""
html,body{height:100%}body{display:flex;flex-direction:column}
#app{flex:1;min-height:0;display:grid;grid-template-columns:minmax(0,1fr) 400px;
grid-template-rows:minmax(0,1fr) auto}
#scene{position:relative;min-height:0}#scene canvas{display:block}
#hud{position:absolute;left:20px;top:16px;pointer-events:none;max-width:70%}
#hud h1{margin:0;font-size:20px;font-weight:600}#hud p{margin:2px 0 0;color:var(--ink2);font-size:13px}
#side{grid-row:1/3;grid-column:2;overflow-y:auto;background:var(--surface);
border-left:1px solid var(--grid);padding:14px 18px 20px}
#list{display:flex;flex-direction:column;gap:2px;margin-bottom:12px}
#list .grp{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;margin:8px 0 2px}
#list button{text-align:left;border:1px solid transparent;background:none;padding:4px 8px}
#list button[aria-pressed=true]{border-color:var(--accent);background:var(--page)}
.kv2{display:grid;grid-template-columns:1fr auto;gap:2px 10px;font-size:13px}
.kv2 span{color:var(--ink2)}.kv2 b{font-weight:600;font-variant-numeric:tabular-nums;text-align:right}
.cbar{display:flex;align-items:center;gap:8px;font-size:12px;color:var(--ink2);margin:6px 0}
.cbar canvas{height:10px;flex:1;min-width:0;width:100%;border-radius:3px}
[hidden]{display:none!important}
#cap{font-size:12.5px;color:var(--ink2);margin:8px 0 0}
#bar{grid-column:1;display:flex;align-items:center;gap:12px;padding:10px 20px;
background:var(--surface);border-top:1px solid var(--grid);flex-wrap:wrap}
#play{min-width:76px;font-weight:600}#scrub{flex:1;min-width:160px;accent-color:var(--accent)}
#tclock{font:600 15px Consolas,monospace;font-variant-numeric:tabular-nums}
@media (max-width:900px){#app{grid-template-columns:1fr;grid-template-rows:60vh auto auto}
#side{grid-row:3;grid-column:1;border-left:0;border-top:1px solid var(--grid)}#bar{grid-row:2}}
"""

_JS = r"""
import * as THREE from "three";
import {OrbitControls} from "three/addons/controls/OrbitControls.js";
const E = JSON.parse(document.getElementById("data").textContent), TK = TKC, $ = id => document.getElementById(id);
const host = $("scene"), renderer = new THREE.WebGLRenderer({antialias: true});
renderer.setPixelRatio(Math.min(devicePixelRatio, 2)); renderer.localClippingEnabled = true; host.prepend(renderer.domElement);
const scene = new THREE.Scene(); scene.background = new THREE.Color(TK.page);
const camera = new THREE.PerspectiveCamera(35, 1, 0.001, 1000), controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; scene.add(new THREE.AmbientLight(0xffffff, TK.dark ? 0.55 : 0.75));
const sun = new THREE.DirectionalLight(0xffffff, 1.0); sun.position.set(2, 3, 4); camera.add(sun); scene.add(camera);
const COL = k => k === "ink2" ? TK.ink2 : k && k[0] === "s" ? TK.series[+k.slice(1)] : (k || TK.ink2);
const ramp = [new THREE.Color(TK.dark ? "#6e3a1a" : "#fbe1cf"), new THREE.Color(TK.dark ? "#ffe3cc" : "#7a2906")];
const clipQuarter = [new THREE.Plane(new THREE.Vector3(0, -1, 0), 0), new THREE.Plane(new THREE.Vector3(0, 0, -1), 0)];
const PLAY_SECONDS = 8, A0 = Math.PI / 2;                                                  // a whole burn plays in 8 s
let eng = null, group = null, field = null, frame = -1, t = 0, playing = false, last = performance.now();

function lathe(b, sweep = 1.5 * Math.PI, n = 72) {      // surface of revolution about x, three-quarter cut
  const pos = [], idx = [], N = b.x.length;
  for (let i = 0; i < N; i++) for (let k = 0; k <= n; k++) { const a = A0 + k / n * sweep;   // open quarter faces +y +z
    pos.push(b.x[i], b.r[i] * Math.cos(a), b.r[i] * Math.sin(a)); }
  for (let i = 0; i < N - 1; i++) for (let k = 0; k < n; k++) { const a = i * (n + 1) + k, c = a + n + 1;
    idx.push(a, c, a + 1, c, c + 1, a + 1); }
  const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3)); g.setIndex(idx);
  g.computeVertexNormals(); g.userData = {n: n + 1}; return g;
}
const fieldMat = () => new THREE.MeshPhongMaterial({vertexColors: true, side: THREE.DoubleSide, shininess: 25});
const solidMat = (c, o = 1, clip = false) => new THREE.MeshPhongMaterial({color: COL(c), side: THREE.DoubleSide,
  shininess: 30, transparent: o < 1, opacity: o, depthWrite: o >= 1,
  clippingPlanes: clip ? clipQuarter : null, clipIntersection: clip});   // clip only the y > 0, z > 0 quarter
function planar(b) {                                    // lower and upper walls across the width
  const g = new THREE.Group(), w = b.w;
  for (const key of ["y0", "y1"]) { const pos = [], idx = [], xs = [];
    for (let i = 0; i < b.x.length; i++) { const y = b[key][i]; if (y == null) continue;
      pos.push(b.x[i], y, -w / 2, b.x[i], y, w / 2); xs.push(i); }
    for (let k = 0; k < xs.length - 1; k++) { const a = 2 * k; idx.push(a, a + 2, a + 1, a + 1, a + 2, a + 3); }
    const geo = new THREE.BufferGeometry(); geo.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    geo.setIndex(idx); geo.computeVertexNormals(); geo.userData = {xs}; g.add(new THREE.Mesh(geo, fieldMat())); }
  const side = [];                                       // far side wall, translucent
  for (let i = 0; i < b.x.length - 1; i++) { if (b.y1[i] == null || b.y1[i + 1] == null) continue;
    side.push(b.x[i], b.y0[i], -w / 2, b.x[i + 1], b.y0[i + 1], -w / 2, b.x[i], b.y1[i], -w / 2,
      b.x[i + 1], b.y0[i + 1], -w / 2, b.x[i + 1], b.y1[i + 1], -w / 2, b.x[i], b.y1[i], -w / 2); }
  const sg = new THREE.BufferGeometry(); sg.setAttribute("position", new THREE.Float32BufferAttribute(side, 3));
  sg.computeVertexNormals(); g.add(new THREE.Mesh(sg, solidMat("ink2", 0.18)));
  return g;
}
function ring(b) {                                      // rotor: hub-to-tip disc with blades
  const g = new THREE.Group(), m = solidMat(b.color);
  g.add(new THREE.Mesh(lathe({x: [b.x, b.x], r: [b.r0, b.r1]}), solidMat(b.color, 0.3)));
  for (let k = 0; k < b.blades; k++) { const a = A0 + (k + 0.5) / b.blades * 1.5 * Math.PI, L = b.r1 - b.r0;
    const bl = new THREE.Mesh(new THREE.BoxGeometry(L * 0.05, L, L * 0.25), m);
    bl.position.set(b.x, (b.r0 + L / 2) * Math.cos(a), (b.r0 + L / 2) * Math.sin(a));
    bl.rotation.x = a - Math.PI / 2; bl.rotateY(0.5); g.add(bl); }
  return g;
}
function grainGeo(b, lv) {                              // propellant cross-section extruded along x
  const L = b.len[lv]; if (L <= 1e-6) return null;
  const s = new THREE.Shape(); s.absarc(0, 0, b.R, 0, Math.PI * 2, false);
  for (const loop of b.holes[lv]) s.holes.push(new THREE.Path(loop.map(p => new THREE.Vector2(p[0], p[1]))));
  const shapes = [s, ...b.islands[lv].map(l => new THREE.Shape(l.map(p => new THREE.Vector2(p[0], p[1]))))];
  const geo = new THREE.ExtrudeGeometry(shapes, {depth: L, bevelEnabled: false, curveSegments: 48});
  geo.rotateY(Math.PI / 2); geo.translate(b.x0[lv], 0, 0); return geo;
}
function build() {
  if (group) { scene.remove(group); group.traverse(o => o.geometry?.dispose()); }
  group = new THREE.Group(); scene.add(group); frame = -1;
  for (const b of eng.bodies) {
    let obj;
    if (b.type === "lathe") obj = new THREE.Mesh(lathe(b), b.f ? fieldMat() : solidMat(b.color, b.opacity ?? 1));
    else if (b.type === "planar") obj = planar(b);
    else if (b.type === "ring") obj = ring(b);
    else if (b.type === "grain") { obj = new THREE.Group(); obj.cache = {}; obj.mat = solidMat(b.color, 1, true); }
    else if (b.type === "tube") obj = new THREE.Mesh(new THREE.BufferGeometry(), solidMat(b.color));
    else if (b.type === "tank") { obj = new THREE.Group();
      obj.add(new THREE.Mesh(lathe({x: [b.x0, b.x0 + b.len], r: [b.R, b.R]}), solidMat("ink2", 0.22)));
      for (const xe of [b.x0, b.x0 + b.len]) obj.add(new THREE.Mesh(lathe({x: [xe, xe], r: [0, b.R]}), solidMat("ink2", 0.22)));
      obj.liq = new THREE.Mesh(new THREE.BufferGeometry(), solidMat("s0", 0.85)); obj.add(obj.liq); }
    b.obj = obj; group.add(obj);
  }
  setFrame(0); colour(); fit();
}
function setFrame(k) {
  if (k === frame) return; frame = k;
  for (const b of eng.bodies) {
    if (b.type === "grain") { const lv = b.frame[Math.min(k, b.frame.length - 1)], o = b.obj;
      o.clear(); if (!(lv in o.cache)) o.cache[lv] = grainGeo(b, lv);
      if (o.cache[lv]) o.add(new THREE.Mesh(o.cache[lv], o.mat)); }
    else if (b.type === "tube") { const r = b.r_in[Math.min(k, b.r_in.length - 1)];
      b.obj.geometry.dispose();
      b.obj.geometry = lathe({x: [b.x0, b.x0, b.x0 + b.len, b.x0 + b.len, b.x0], r: [r, b.R, b.R, r, r]}); }
    else if (b.type === "tank") { const f = Math.max(b.fill[Math.min(k, b.fill.length - 1)], 0.002);
      const x1 = b.x0 + b.len, x0 = x1 - f * b.len, r = b.R * 0.97;               // liquid settles aft
      b.obj.liq.geometry.dispose(); b.obj.liq.geometry = lathe({x: [x0, x0, x1, x1], r: [0, r, r, 0]}); }
  }
  if (eng.time) { $("tclock").textContent = `t = ${eng.time.t[k].toFixed(2)} s`;
    for (const [lab, arr] of Object.entries(eng.time.cards))
      document.querySelector(`[data-tc="${lab}"]`).textContent = (+arr[k]).toFixed(/thrust, N|Kn/.test(lab) ? 0 : 2); }
}
function colour() {
  $("cbwrap").hidden = !field; if (!field) return;
  let lo = Infinity, hi = -Infinity;
  for (const b of eng.bodies) if (b.f && b.f[field]) for (const v of b.f[field]) if (v != null) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  const col = v => ramp[0].clone().lerp(ramp[1], hi > lo ? Math.min(Math.max((v - lo) / (hi - lo), 0), 1) : 0.5);
  for (const b of eng.bodies) {
    if (!b.f || !b.f[field]) continue;
    const meshes = b.type === "planar" ? b.obj.children.slice(0, 2) : [b.obj];
    for (const m of meshes) { const g = m.geometry, cnt = g.attributes.position.count, c = new Float32Array(cnt * 3);
      for (let v = 0; v < cnt; v++) { const i = g.userData.xs ? g.userData.xs[v >> 1] : Math.floor(v / g.userData.n);
        const cc = col(b.f[field][i]); c[3 * v] = cc.r; c[3 * v + 1] = cc.g; c[3 * v + 2] = cc.b; }
      g.setAttribute("color", new THREE.BufferAttribute(c, 3)); }
  }
  const bar = $("cbar"); bar.width = 256; bar.height = 1; const x = bar.getContext("2d");
  for (let i = 0; i < 256; i++) { x.fillStyle = "#" + col(lo + (hi - lo) * i / 255).getHexString(); x.fillRect(i, 0, 1, 1); }
  const f = eng.fields[field], d = f.d ?? 1;
  $("clo").textContent = lo.toFixed(d); $("chi").textContent = `${hi.toFixed(d)} ${f.unit}`;
}
function fit() {
  const box = new THREE.Box3().setFromObject(group), c = box.getCenter(new THREE.Vector3()), s = box.getSize(new THREE.Vector3());
  const d = Math.max(s.x, s.y, s.z) / (2 * Math.tan(17.5 * Math.PI / 180)) * 1.1;
  camera.position.copy(c).add(new THREE.Vector3(0.3, 0.4, 1).normalize().multiplyScalar(d));
  controls.target.copy(c); camera.near = d / 300; camera.far = d * 30; camera.updateProjectionMatrix();
}
function select(i) {
  eng = E.engines[i]; field = eng.field; t = 0; playing = false;
  document.querySelectorAll("#list button").forEach((b, k) => b.setAttribute("aria-pressed", k === i));
  $("title").textContent = eng.name; $("sub").textContent = eng.scale;
  $("cards").innerHTML = Object.entries(eng.cards).map(([k, v]) => `<span>${k}</span><b>${v}</b>`).join("") +
    (eng.time ? Object.keys(eng.time.cards).map(k => `<span>${k}</span><b data-tc="${k}">–</b>`).join("") : "");
  $("field").innerHTML = Object.entries(eng.fields).map(([k, f]) =>
    `<option value="${k}"${k === field ? " selected" : ""}>${f.label}</option>`).join("");
  $("fieldwrap").hidden = !field;
  $("st").innerHTML = eng.stations ? `<tr><th>Station</th><th>T, K</th><th>p, kPa</th></tr>` + eng.stations.map(s =>
    `<tr><td>${s.station} · ${s.name}</td><td>${s.T.toFixed(0)}</td><td>${(s.p / 1e3).toFixed(s.p < 1e5 ? 1 : 0)}</td></tr>`).join("") : "";
  $("sth").hidden = !eng.stations; $("cap").textContent = eng.caption || "";
  $("bar").hidden = !eng.time; $("play").textContent = "Play"; $("scrub").value = 0;
  resize(); build();
}
const list = $("list"); let grp = "";
E.engines.forEach((e, i) => { if (e.group !== grp) { list.insertAdjacentHTML("beforeend", `<div class="grp">${e.group}</div>`); grp = e.group; }
  const b = document.createElement("button"); b.textContent = e.name; b.onclick = () => select(i); list.append(b); });
$("field").onchange = e => { field = e.target.value; colour(); };
$("play").onclick = () => { if (t >= 1) t = 0; playing = !playing; $("play").textContent = playing ? "Pause" : "Play"; };
$("scrub").oninput = e => { t = e.target.value / 1000; };
function resize() { const r = host.getBoundingClientRect(); renderer.setSize(r.width, r.height, false);
  renderer.domElement.style.width = r.width + "px"; renderer.domElement.style.height = r.height + "px";
  camera.aspect = r.width / r.height; camera.updateProjectionMatrix(); }
addEventListener("resize", resize);
function loop(now) {
  const dt = Math.min((now - last) / 1000, 0.5); last = now;
  if (eng.time) {
    if (playing) { t = Math.min(t + dt / PLAY_SECONDS, 1); $("scrub").value = t * 1000;
      if (t >= 1) { playing = false; $("play").textContent = "Play"; } }
    const T = eng.time.t, tt = t * T[T.length - 1]; let k = 0;
    while (k < T.length - 1 && T[k + 1] <= tt) k++;
    setFrame(k);
  }
  controls.update(); renderer.render(scene, camera); requestAnimationFrame(loop);
}
select(0); requestAnimationFrame(loop);
"""


def engines3d_html(engines: list[dict], theme: str = "dark") -> str:
    """One page with a 3D figure per engine (``aerosim.plots.engine_models`` descriptions)."""
    page = f"""<div id="app">
 <div id="scene"><div id="hud"><h1 id="title"></h1><p id="sub"></p></div></div>
 <div id="side"><div id="list" role="group" aria-label="Engine"></div>
  <div class="card"><div class="kv2" id="cards"></div></div>
  <div id="fieldwrap"><label>Colour by <select id="field"></select></label>
   <div class="cbar" id="cbwrap"><span id="clo"></span><canvas id="cbar"></canvas><span id="chi"></span></div></div>
  <h3 id="sth">Stations</h3><table id="st"></table><p id="cap"></p></div>
 <div id="bar" hidden><button id="play" aria-label="Play or pause the burn">Play</button>
  <input id="scrub" type="range" min="0" max="1000" value="0" aria-label="Burn time"><span id="tclock"></span></div>
</div>
{theme_tag(theme)}{data_tag({"engines": engines})}{importmap_tag()}
<script type="module">{CHART_JS}{_JS}</script>"""
    return _page("AeroSim engine models", "engines3d.html", page, theme, _CSS)
