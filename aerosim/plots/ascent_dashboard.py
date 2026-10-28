"""Flight-data dashboard of an orbital ascent, laid out like a paper: a title block, booktabs
tables and figures of lettered sub-plots in the style of LaTeX publication figures (Computer
Modern, framed axes with inward ticks on all four sides, black solid and grey dashed lines, a
frameless key inside the frame). It follows the site's theme and typeface: monochrome in the
academic themes, the theme's series colours otherwise.

    ascent_dashboard_html(ascent_dataset(...), nav=nav(PAGES, "rockot-data.html"))

Every plot shows all models (the first solid, the second dashed); hover a plot for values,
drag the 3D trajectory to turn it.
"""

from __future__ import annotations

import html
import json
from importlib import resources

from aerosim.plots.web import CSS, data_tag, theme_tag, themed

_CSS = r"""
body{font-family:var(--font);font-size:14.5px;font-variant-numeric:lining-nums;background:var(--page)}
:root[data-font=serif] body{font-size:15.5px}
:root:not([data-font=serif]) .sub .cap,:root:not([data-font=serif]) figcaption,:root:not([data-font=serif]) .tcap{font-size:12.5px}
:root:not([data-font=serif]) table.bt{font-size:12.5px}:root:not([data-font=serif]) .tip{font-size:12px}
main{max-width:1180px;margin:0 auto;padding:34px 24px 64px}
header{text-align:center;margin:0 0 30px}
header h1{font-size:25px;font-weight:700;margin:0 0 12px;letter-spacing:.005em}
header .sub{font-size:17px;margin:0 0 4px}header .meta{font-size:13.5px;color:var(--ink2);margin:0}
header .meta a{color:var(--ink2)}
.pair{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,470px),1fr));gap:0 48px;align-items:start}
.pair>*{min-width:0}
h2{font-size:19px;font-weight:700;margin:14px 0 14px}
h2 .n{display:inline-block;min-width:1.6em}
p.text{margin:0 0 12px;text-align:justify;hyphens:auto}
figure{margin:0 0 34px}figcaption{max-width:860px;margin:2px auto 0 !important;text-align:center}
.subs{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,420px),1fr));gap:4px 40px}
.sub{position:relative}
.sub canvas{display:block;width:100%;aspect-ratio:1.62}
.sub canvas.d3{cursor:grab}.sub canvas.d3:active{cursor:grabbing}
.sub canvas.map{aspect-ratio:1.85}
.sub .cap,figcaption{font-size:13px;line-height:1.35}
.sub .cap{text-align:center;margin:2px 0 10px}
figcaption{margin:2px 0 0}figcaption b,.tcap b{font-weight:700}
.tcap{font-size:13px;line-height:1.35;margin:0 0 6px}
.tbl{margin:0 0 30px;overflow-x:auto}
table.bt{border-collapse:collapse;width:100%;font-size:13.5px;font-variant-numeric:tabular-nums}
table.bt th,table.bt td{padding:2px 7px;text-align:right;border:0;white-space:nowrap}
table.bt th:first-child,table.bt td:first-child{text-align:left;padding-left:0}
table.bt th{font-weight:400;color:var(--ink)}
table.bt thead tr:first-child th{border-top:1.4px solid var(--ink);padding-top:4px}
table.bt thead tr:last-child th{border-bottom:.7px solid var(--ink);padding-bottom:3px}
table.bt tbody tr:last-child td{border-bottom:1.4px solid var(--ink);padding-bottom:4px}
table.bt th.grp{text-align:center;padding-bottom:0}
table.bt th.grp span{display:block;border-bottom:.7px solid var(--ink);padding-bottom:1px}
.tip{position:absolute;pointer-events:none;background:var(--page);border:.7px solid var(--ink);
padding:4px 7px;font-size:12.5px;white-space:nowrap;display:none;z-index:5;border-radius:0}
.tip div{display:flex;align-items:center;gap:6px}.tip div.hd{display:block}.tip svg{flex:none}
@media (max-width:620px){main{padding:22px 16px 48px}header h1{font-size:21px}}
"""

_JS = r"""
const $ = id => document.getElementById(id);
const D = JSON.parse($("data").textContent), TK = JSON.parse($("theme").textContent);
// the reader's typeface; sizes are set for Latin Modern, whose glyphs run small, and scaled for the others
const FONT = TK.font, FS = /Latin Modern/.test(FONT) ? 1 : .88, PX = n => (n * FS).toFixed(1) + "px " + FONT;
const INK = TK.ink, GREY = TK.dark ? TK.ink2 : TK.muted, PANE = TK.dark ? "#2a2a2a" : "#e3e3e3",
  FAINT = TK.dark ? "#4a4a4a" : "#b5b5b5";
// line styles in the order pgfplots / matplotlib-pgf papers use: solid black, grey dashed, dotted,
// dash-dot; a colour theme keeps the dashes but draws them in its series colours
const SC = TK.academic ? [INK, GREY, INK, GREY] : TK.series;
const STY = [{c: SC[0], w: 1.3, d: []}, {c: SC[1], w: 1.0, d: [5, 3]}, {c: SC[2], w: 1.0, d: [1.2, 2.2]},
  {c: SC[3], w: 1.0, d: [7, 2.5, 1.2, 2.5]}];
const M = D.models, REF = M[0];
const MINUS = s => s.replace(/^-/, "−");

// ---- text: "$h$ (km)" sets h in italic, as LaTeX math would
function rich(c, s, x, y, align = "left", size = 13) {
  const parts = s.split("$").map((t, i) => [t, i % 2 === 1]);
  const fnt = it => (it ? "italic " : "") + PX(size);
  let w = 0; for (const [t, it] of parts) { c.font = fnt(it); w += c.measureText(t).width; }
  let x0 = align === "center" ? x - w / 2 : align === "right" ? x - w : x;
  c.textAlign = "left";
  for (const [t, it] of parts) { c.font = fnt(it); c.fillText(t, x0, y); x0 += c.measureText(t).width; }
  return w;
}
const nice = x => { const p = Math.pow(10, Math.floor(Math.log10(x)));
  for (const s of [1, 2, 2.5, 5, 10]) if (s * p >= x - 1e-12) return s * p; };
function ticks(lo, hi, n = 5, step) {
  if (hi - lo < 1e-12) { hi = lo + 1; }
  const st = step || nice((hi - lo) / n), a = Math.floor(lo / st + 1e-9) * st, b = Math.ceil(hi / st - 1e-9) * st;
  const m = st / Math.pow(10, Math.floor(Math.log10(st))), sub = Math.abs(m - 2) < 1e-9 ? 4 : 5;
  const major = [], minor = [];
  for (let v = a; v <= b + st / 2; v += st) major.push(+v.toFixed(10));
  for (let v = a; v < b - 1e-9; v += st / sub) minor.push(+v.toFixed(10));
  return {lo: a, hi: b, st, major, minor};
}
const fmt = (v, st) => { const d = Math.max(0, -Math.floor(Math.log10(st) + 1e-9) + (st / Math.pow(10, Math.floor(Math.log10(st))) === 2.5 ? 1 : 0));
  return MINUS((Math.abs(v) < st / 1e6 ? 0 : v).toFixed(Math.min(d, 4))); };
function setup(cv) {
  const r = cv.getBoundingClientRect(), dpr = Math.min(devicePixelRatio || 1, 2);
  if (cv.width !== Math.round(r.width * dpr) || cv.height !== Math.round(r.height * dpr)) {
    cv.width = Math.round(r.width * dpr); cv.height = Math.round(r.height * dpr); }
  const c = cv.getContext("2d"); c.setTransform(dpr, 0, 0, dpr, 0, 0); c.clearRect(0, 0, r.width, r.height);
  return [c, r.width, r.height];
}
const keyLine = (s, w = 22) => `<svg width="${w}" height="6"><line x1="0" y1="3" x2="${w}" y2="3" stroke="${s.c}" stroke-width="${s.w + .2}" stroke-dasharray="${s.d.join(" ")}"/></svg>`;
function onResize(cv, f) { if (window.ResizeObserver) new ResizeObserver(() => f()).observe(cv); else addEventListener("resize", f); }

// ---- 2D plot. o = {series: [{x, y, name, s (style index), key (false = not in key), jump}],
//   xl, yl, xr, yr, ystep, xstep, log, vlines: [x], hlines: [{y, label}], key: "tr"|"tl"|"br"|"bl",
//   xu, yu, xd, yd (hover units and digits), hover (false: none, for x that is not monotonic)}
function plot(id, o) {
  const cv = $(id), tip = document.createElement("div"); tip.className = "tip"; cv.parentElement.append(tip);
  const ser = o.series.map(s => ({...s, st: STY[s.s ?? 0]}));
  let X, Y, box;
  const draw = hx => {
    const [c, w, h] = setup(cv);
    const inX = (s, i) => !o.xr || (s.x[i] >= o.xr[0] && s.x[i] <= o.xr[1]);
    let xs = [], ys = [];
    for (const s of ser) for (let i = 0; i < s.x.length; i++)
      if (s.y[i] != null && Number.isFinite(s.y[i]) && inX(s, i) && (!o.log || s.y[i] > 0)) { xs.push(s.x[i]); ys.push(s.y[i]); }
    const tx = ticks(o.xr ? o.xr[0] : Math.min(...xs), o.xr ? o.xr[1] : Math.max(...xs), 5, o.xstep);
    let ty;
    if (o.log) {
      const a = Math.floor(Math.log10(o.yr ? o.yr[0] : Math.min(...ys))), b = Math.ceil(Math.log10(o.yr ? o.yr[1] : Math.max(...ys)));
      const step = Math.max(1, Math.ceil((b - a) / 5)), major = [], minor = [];
      for (let e = a; e <= b; e += step) major.push(e);
      for (let e = a; e < b; e++) for (let k = 2; k < 10; k++) minor.push(e + Math.log10(k));
      ty = {lo: a, hi: b, major, minor: step === 1 ? minor : [], st: 1};
    } else ty = ticks(o.yr ? o.yr[0] : Math.min(...ys), o.yr ? o.yr[1] : Math.max(...ys), 4, o.ystep);
    if (o.xr && !o.xstep) { tx.lo = o.xr[0]; tx.hi = o.xr[1]; }
    const ylab = v => o.log ? "10" : fmt(v, ty.st);
    c.font = PX(12.5);
    const lw = Math.max(...ty.major.map(v => c.measureText(ylab(v)).width + (o.log ? 7 : 0)));
    const L = lw + (o.yl ? 26 : 10), B = o.xl ? 36 : 22, T = 7, R = 16, W = w - L - R, H = h - B - T;
    const fx = x => L + (x - tx.lo) / (tx.hi - tx.lo) * W;
    const fy = y => T + H - ((o.log ? Math.log10(y) : y) - ty.lo) / (ty.hi - ty.lo) * H;
    X = fx; Y = fy; box = {L, T, W, H, tx};
    // event lines, under the data
    c.lineWidth = .7; c.strokeStyle = FAINT; c.setLineDash([1, 2.5]);
    for (const v of o.vlines || []) if (v > tx.lo && v < tx.hi) { c.beginPath(); c.moveTo(fx(v), T); c.lineTo(fx(v), T + H); c.stroke(); }
    c.setLineDash([]);
    // data, clipped to the frame
    c.save(); c.beginPath(); c.rect(L, T, W, H); c.clip();
    for (const s of ser) {
      c.strokeStyle = s.plain ? FAINT : s.st.c; c.lineWidth = s.lw || s.st.w; c.setLineDash(s.plain ? [] : s.st.d); c.lineJoin = "round";
      c.beginPath(); let pen = false;
      for (let i = 0; i < s.x.length; i++) {
        const y = s.y[i];
        if (y == null || !Number.isFinite(y) || (o.log && y <= 0)) { pen = false; continue; }
        if (pen && s.jump && Math.abs(s.x[i] - s.x[i - 1]) > s.jump) pen = false;
        pen ? c.lineTo(fx(s.x[i]), fy(y)) : c.moveTo(fx(s.x[i]), fy(y)); pen = true;
      }
      c.stroke();
    }
    c.setLineDash([]);
    for (const hl of o.hlines || []) {
      c.strokeStyle = INK; c.lineWidth = .7; c.setLineDash([3, 2]); c.beginPath(); c.moveTo(L, fy(hl.y)); c.lineTo(L + W, fy(hl.y)); c.stroke();
      c.setLineDash([]); c.fillStyle = INK; rich(c, hl.label, L + W - 5, fy(hl.y) - 4, "right", 11.5);
    }
    c.restore();
    // frame and inward ticks on all four sides
    c.strokeStyle = INK; c.lineWidth = .75; c.strokeRect(L + .5, T + .5, W, H);
    const tk = (px, py, dx, dy) => { c.moveTo(px, py); c.lineTo(px + dx, py + dy); };
    c.beginPath();
    for (const v of tx.major) if (v >= tx.lo - 1e-9 && v <= tx.hi + 1e-9) { const p = fx(v); tk(p, T + H, 0, -4.5); tk(p, T, 0, 4.5); }
    for (const v of ty.major) { const p = T + H - (v - ty.lo) / (ty.hi - ty.lo) * H; tk(L, p, 4.5, 0); tk(L + W, p, -4.5, 0); }
    c.stroke(); c.beginPath();
    for (const v of tx.minor) if (v > tx.lo && v < tx.hi) { const p = fx(v); tk(p, T + H, 0, -2.5); tk(p, T, 0, 2.5); }
    for (const v of ty.minor) if (v > ty.lo && v < ty.hi) { const p = T + H - (v - ty.lo) / (ty.hi - ty.lo) * H; tk(L, p, 2.5, 0); tk(L + W, p, -2.5, 0); }
    c.stroke();
    c.fillStyle = INK; c.font = PX(12.5); c.textAlign = "center";
    for (const v of tx.major) if (v >= tx.lo - 1e-9 && v <= tx.hi + 1e-9) c.fillText(o.xfmt ? o.xfmt(v) : fmt(v, tx.st), fx(v), T + H + 15);
    for (const v of ty.major) { const p = T + H - (v - ty.lo) / (ty.hi - ty.lo) * H;
      if (o.log) { c.font = PX(12.5); c.textAlign = "right"; c.fillText("10", L - 13, p + 4);
        c.font = PX(9); c.textAlign = "left"; c.fillText(MINUS(String(v)), L - 12, p - 2); }
      else { c.font = PX(12.5); c.textAlign = "right"; c.fillText(o.yfmt ? o.yfmt(v) : ylab(v), L - 6, p + 4); } }
    if (o.xl) rich(c, o.xl, L + W / 2, h - 4, "center");
    if (o.yl) { c.save(); c.translate(12, T + H / 2); c.rotate(-Math.PI / 2); rich(c, o.yl, 0, 0, "center"); c.restore(); }
    // frameless key inside the frame
    const keyed = ser.filter(s => s.key !== false && s.name);
    if (keyed.length > 1 || o.keyOne) {
      c.font = PX(11.5); const sw = 22, gap = 13;
      const iw = keyed.map(s => sw + 5 + c.measureText(s.name).width);
      const pos = o.key || "tr", oneRow = iw.reduce((a, b) => a + b + gap, -gap) < W - 16;
      const rows = oneRow ? [keyed.map((s, i) => [s, iw[i]])] : keyed.map((s, i) => [[s, iw[i]]]);
      rows.forEach((row, r) => {
        const rw = row.reduce((a, [, w_]) => a + w_ + gap, -gap);
        let x = pos[1] === "r" ? L + W - 8 - rw : L + 8;
        const y = pos[0] === "t" ? T + 11 + r * 15 : T + H - 10 - (rows.length - 1 - r) * 15;
        c.fillStyle = TK.page; c.fillRect(x - 3, y - 7, rw + 6, 14);
        for (const [s, w_] of row) {
          c.strokeStyle = s.st.c; c.lineWidth = s.st.w; c.setLineDash(s.st.d);
          c.beginPath(); c.moveTo(x, y); c.lineTo(x + sw, y); c.stroke(); c.setLineDash([]);
          c.fillStyle = INK; c.textAlign = "left"; c.font = PX(11.5); c.fillText(s.name, x + sw + 5, y + 4); x += w_ + gap;
        }
      });
    }
    // hover: hairline, open markers, values
    if (hx == null || hx < L || hx > L + W || o.hover === false) { tip.style.display = "none"; return; }
    const xv = tx.lo + (hx - L) / W * (tx.hi - tx.lo);
    c.strokeStyle = GREY; c.lineWidth = .6; c.beginPath(); c.moveTo(hx, T); c.lineTo(hx, T + H); c.stroke();
    const rows = [];
    for (const s of ser) {
      if (s.key === false && !s.hover) continue;
      let lo = 0, hi = s.x.length - 1;
      while (hi - lo > 1) { const m = (lo + hi) >> 1; s.x[m] < xv ? lo = m : hi = m; }
      const k = Math.abs(s.x[lo] - xv) < Math.abs(s.x[hi] - xv) ? lo : hi, y = s.y[k];
      if (y == null || !Number.isFinite(y) || (o.log && y <= 0)) continue;
      c.beginPath(); c.arc(fx(s.x[k]), fy(y), 3, 0, 7); c.fillStyle = TK.page; c.fill(); c.strokeStyle = INK; c.lineWidth = .8; c.stroke();
      rows.push(`<div>${keyLine(s.st, 16)}${s.name}: ${MINUS((+y).toFixed(o.yd ?? 2))}${o.yu ? "&thinsp;" + o.yu : ""}</div>`);
    }
    tip.innerHTML = `<div class="hd"><i>${o.xs || "t"}</i>&nbsp;=&nbsp;${MINUS(xv.toFixed(o.xd ?? 0))}${o.xu ? "&thinsp;" + o.xu : ""}</div>` + rows.join("");
    tip.style.display = "block";
    tip.style.left = (hx + 14 + tip.offsetWidth > w ? hx - 14 - tip.offsetWidth : hx + 14) + "px"; tip.style.top = (T + 6) + "px";
  };
  cv.addEventListener("mousemove", e => draw(e.offsetX)); cv.addEventListener("mouseleave", () => draw(null));
  onResize(cv, () => draw(null)); draw(null);
}

// ---- 3D trajectory in a box, matplotlib mplot3d-like: grey back panes with a grid, the ground
// projection dashed. Drag to turn, double-click to reset.
function plot3d(id) {
  const cv = $(id), V0 = {az: -58, el: 24}; let view = {...V0};
  const all = k => M.flatMap(m => m.enu[k]);
  const ax = {x: ticks(Math.min(...all("e")), Math.max(...all("e")), 4), y: ticks(Math.min(...all("n")), Math.max(...all("n")), 4),
    z: ticks(0, Math.max(...all("u")), 4)};
  const half = {x: .8, y: 1.25, z: .62};
  const nx = v => (v - ax.x.lo) / (ax.x.hi - ax.x.lo) * 2 * half.x - half.x;
  const ny = v => (v - ax.y.lo) / (ax.y.hi - ax.y.lo) * 2 * half.y - half.y;
  const nz = v => (v - ax.z.lo) / (ax.z.hi - ax.z.lo) * 2 * half.z - half.z;
  const draw = () => {
    const [c, w, h] = setup(cv);
    const a = view.az * Math.PI / 180, e = view.el * Math.PI / 180;
    const Vv = [Math.cos(e) * Math.sin(a), -Math.cos(e) * Math.cos(a), Math.sin(e)], Rr = [Math.cos(a), Math.sin(a), 0];
    const U = [Vv[1] * Rr[2] - Vv[2] * Rr[1], Vv[2] * Rr[0] - Vv[0] * Rr[2], Vv[0] * Rr[1] - Vv[1] * Rr[0]];
    const dot = (p, q) => p[0] * q[0] + p[1] * q[1] + p[2] * q[2];
    const corners = []; for (const i of [-1, 1]) for (const j of [-1, 1]) for (const k of [-1, 1]) corners.push([i * half.x, j * half.y, k * half.z]);
    const px = corners.map(p => dot(p, Rr)), py = corners.map(p => -dot(p, U)), m = 54;
    const s = Math.min((w - 2 * m) / (Math.max(...px) - Math.min(...px)), (h - 2 * m) / (Math.max(...py) - Math.min(...py)));
    const cx = w / 2 - s * (Math.max(...px) + Math.min(...px)) / 2, cy = h / 2 - s * (Math.max(...py) + Math.min(...py)) / 2;
    const P = p => [cx + s * dot(p, Rr), cy - s * dot(p, U)];
    const line = (p, q) => { const [a1, b1] = P(p), [a2, b2] = P(q); c.moveTo(a1, b1); c.lineTo(a2, b2); };
    const bx = Vv[0] > 0 ? -half.x : half.x, by = Vv[1] > 0 ? -half.y : half.y, bz = Vv[2] > 0 ? -half.z : half.z;   // back panes
    const fx_ = -bx, fy_ = -by;
    // panes: grid lines, then their edges
    c.lineWidth = .6; c.strokeStyle = PANE; c.beginPath();
    for (const v of ax.y.major) { line([bx, ny(v), -half.z], [bx, ny(v), half.z]); line([-half.x, ny(v), bz], [half.x, ny(v), bz]); }
    for (const v of ax.x.major) { line([nx(v), by, -half.z], [nx(v), by, half.z]); line([nx(v), -half.y, bz], [nx(v), half.y, bz]); }
    for (const v of ax.z.major) { line([bx, -half.y, nz(v)], [bx, half.y, nz(v)]); line([-half.x, by, nz(v)], [half.x, by, nz(v)]); }
    c.stroke();
    c.strokeStyle = FAINT; c.lineWidth = .7; c.beginPath();
    const rect = pts => { for (let i = 0; i < 4; i++) line(pts[i], pts[(i + 1) % 4]); };
    rect([[bx, -half.y, -half.z], [bx, half.y, -half.z], [bx, half.y, half.z], [bx, -half.y, half.z]]);
    rect([[-half.x, by, -half.z], [half.x, by, -half.z], [half.x, by, half.z], [-half.x, by, half.z]]);
    rect([[-half.x, -half.y, bz], [half.x, -half.y, bz], [half.x, half.y, bz], [-half.x, half.y, bz]]);
    c.stroke();
    // ground projection (dashed, grey) and the trajectories
    const path = (m_, z) => { c.beginPath(); m_.enu.e.forEach((v, i) => { const [a1, b1] = P([nx(v), ny(m_.enu.n[i]), z == null ? nz(m_.enu.u[i]) : z]); i ? c.lineTo(a1, b1) : c.moveTo(a1, b1); }); c.stroke(); };
    c.strokeStyle = FAINT; c.lineWidth = .8; c.setLineDash([3, 2.5]); path(REF, -half.z); c.setLineDash([]);
    M.forEach((m_, i) => { const st = STY[i]; c.strokeStyle = st.c; c.lineWidth = st.w + .15; c.setLineDash(st.d); path(m_); c.setLineDash([]); });
    // tick labels on the front edges of the floor and on the right-hand vertical edge
    const ctr = P([0, 0, 0]);
    const labels = (pts, vals, t, title) => {
      const mid = P(pts(0.5)), d = [mid[0] - ctr[0], mid[1] - ctr[1]], n_ = Math.hypot(...d) || 1, u = [d[0] / n_, d[1] / n_];
      c.fillStyle = INK; c.font = PX(12); c.textAlign = "center";
      c.strokeStyle = INK; c.lineWidth = .7; c.beginPath();
      vals.forEach(v => { const q = P(pts(v)); c.moveTo(q[0], q[1]); c.lineTo(q[0] + u[0] * 4, q[1] + u[1] * 4); }); c.stroke();
      vals.forEach(v => { const q = P(pts(v)); c.fillText(fmt(t.inv(v), t.st), q[0] + u[0] * 15, q[1] + u[1] * 15 + 4); });
      rich(c, title, mid[0] + u[0] * 36, mid[1] + u[1] * 36 + 4, "center", 12.5);
    };
    const frac = (t, v) => (v - t.lo) / (t.hi - t.lo);
    const xT = {...ax.x, inv: f => ax.x.lo + f * (ax.x.hi - ax.x.lo)}, yT = {...ax.y, inv: f => ax.y.lo + f * (ax.y.hi - ax.y.lo)},
      zT = {...ax.z, inv: f => ax.z.lo + f * (ax.z.hi - ax.z.lo)};
    labels(f => [-half.x + 2 * half.x * f, fy_, -half.z], ax.x.major.map(v => frac(ax.x, v)), xT, "east (km)");
    labels(f => [fx_, -half.y + 2 * half.y * f, -half.z], ax.y.major.map(v => frac(ax.y, v)), yT, "north (km)");
    const e1 = [fx_, by], e2 = [bx, fy_], pick = P([...e1, 0])[0] > P([...e2, 0])[0] ? e1 : e2;
    labels(f => [pick[0], pick[1], -half.z + 2 * half.z * f], ax.z.major.map(v => frac(ax.z, v)), zT, "$h$ (km)");
  };
  let drag = null;
  cv.addEventListener("pointerdown", e => { drag = [e.clientX, e.clientY, view.az, view.el]; cv.setPointerCapture(e.pointerId); });
  cv.addEventListener("pointermove", e => { if (!drag) return;
    view.az = drag[2] - (e.clientX - drag[0]) * .4; view.el = Math.max(2, Math.min(88, drag[3] + (e.clientY - drag[1]) * .3)); draw(); });
  cv.addEventListener("pointerup", () => drag = null);
  cv.addEventListener("dblclick", () => { view = {...V0}; draw(); });
  onResize(cv, draw); draw();
}

// ---- the figures
const SEP = REF.events.filter(e => /burnout|burn1_end|burn2_end/.test(e.key)).map(e => e.t);
const tIns = Math.max(...M.map(m => m.summary.t_insertion)), TA = [0, Math.ceil(tIns / 100) * 100];
const S = (k, x = "t", src = "series") => M.map((m, i) => ({name: m.name, s: i, x: m[src][x], y: m[src][k]}));
const T_ = {xl: "time $t$ (s)", xu: "s", vlines: SEP};

plot("f-h", {...T_, series: S("h"), xr: TA, yl: "altitude $h$ (km)", yu: "km", yd: 1, key: "br"});
plot3d("f-3d");
plot("f-v", {...T_, series: S("v"), xr: TA, yl: "speed $v$ (km/s)", yu: "km/s", yd: 3, key: "br"});
plot("f-g", {...T_, series: S("fpa"), xr: TA, yl: "flight-path angle $γ$ (deg)", yu: "deg", yd: 1});
plot("f-q", {...T_, series: S("q"), xr: [0, 150], yl: "dynamic pressure $q$ (kPa)", yu: "kPa", yd: 1, vlines: []});
plot("f-M", {...T_, series: S("mach"), xr: [0, 150], yl: "Mach number $M$", yd: 2, key: "br", vlines: []});
plot("f-a", {...T_, series: S("acc"), xr: TA, yl: "acceleration $a$/$g$₀", yd: 2});
plot("f-m", {...T_, series: S("m"), xr: TA, yl: "mass $m$ (t)", yu: "t", yd: 2});
plot("f-F", {...T_, series: S("thrust", "t", "physics"), xr: TA, yl: "thrust $F$ (kN)", yu: "kN", yd: 0, log: true});
plot("f-D", {...T_, series: S("drag", "t", "physics"), xr: [0, 150], yl: "drag $D$ (kN)", yu: "kN", yd: 1, vlines: []});
plot("f-L", {...T_, xr: TA, yl: "velocity loss $Δv$ (m/s)", yu: "m/s", yd: 0, key: "tl",
  series: [["gravity", "Gravity"], ["drag", "Drag"], ["pressure", "Back-pressure"]].map(([k, n], i) =>
    ({name: n, s: i, x: REF.physics.t, y: REF.physics.loss[k]}))});
plot("f-Q", {...T_, series: S("heat", "t", "physics"), xr: [100, 300], log: true, yr: [10, 1e6], vlines: [],
  yl: "heat flux $½ρv³$ (W/m²)", yu: "W/m²", yd: 0,
  hlines: [{y: D.fairing_heat_flux, label: "fairing limit"}]});
const TRACK = M.map((m, i) => ({name: m.name, s: i, x: m.series.lon, y: m.series.lat, jump: 180}));
// coastlines: plain grey hairlines, not in the key
const COAST = D.coast.map(l => ({x: l.map(p => p[0]), y: l.map(p => p[1]), key: false, plain: true, lw: .45, jump: 180}));
plot("f-gt", {series: [...COAST, ...TRACK], hover: false, xr: [-180, 180], yr: [-90, 90], xstep: 60, ystep: 30,
  xl: "longitude (deg)", yl: "latitude (deg)", xs: "λ", xu: "deg", xd: 1, yu: "deg", yd: 1, key: "br"});
plot("f-dr", {series: M.map((m, i) => ({name: m.name, s: i, x: m.series.dr.filter((_, k) => m.series.t[k] <= tIns),
  y: m.series.h.filter((_, k) => m.series.t[k] <= tIns)})), xl: "ground range (km)", yl: "altitude $h$ (km)",
  xs: "range", xu: "km", yu: "km", yd: 1, key: "br"});

$("json").addEventListener("click", e => { e.preventDefault();
  const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([$("data").textContent], {type: "application/json"}));
  a.download = "ascent-flight-data.json"; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000); });
"""

def _coast() -> list:
    d = json.loads(resources.files("aerosim.plots").joinpath("data/coastline_110m.json").read_text(encoding="utf-8"))
    return [[[round(x, 1), round(y, 1)] for x, y in ln] for ln in d["lines"] if len(ln) > 3]


def _f(v, d=1):
    return "–" if v is None else f"{v:,.{d}f}".replace(",", " ").replace("-", "−")


def _table(head: str, rows: list[list[str]]) -> str:
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="bt"><thead>{head}</thead><tbody>{body}</tbody></table>'


def _summary_rows(models) -> list[list[str]]:
    S = [m["summary"] for m in models]
    L = [s["losses"] for s in S]
    rows = [("Lift-off mass", "t", [_f(s["liftoff_mass"], 2) for s in S]),
            ("Payload", "t", [_f(s["payload"], 2) for s in S]),
            ("Mach 1", "s", [_f(s["t_mach1"], 1) for s in S]),
            ("Maximum dynamic pressure", "kPa", [_f(s["max_q"], 1) for s in S]),
            ("&ensp;at time", "s", [_f(s["t_max_q"], 0) for s in S]),
            ("&ensp;at altitude", "km", [_f(s["h_max_q"], 2) for s in S]),
            ("Peak acceleration", "<i>g</i><sub>0</sub>", [_f(s["max_g"], 2) for s in S]),
            ("Transfer-orbit insertion", "s", [_f(s["t_insertion"], 1) for s in S]),
            ("Ideal velocity, all burns", "m/s", [_f(s["dv_ideal"], 0) for s in S]),
            ("Gravity loss", "m/s", [_f(x["gravity"], 0) for x in L]),
            ("Drag loss", "m/s", [_f(x["drag"], 0) for x in L]),
            ("Back-pressure loss", "m/s", [_f(x["pressure"], 0) for x in L]),
            ("Final orbit, perigee × apogee", "km", [f'{_f(s["perigee"], 0)} × {_f(s["apogee"], 0)}' for s in S]),
            ("Inclination", "deg", [_f(s["inc"], 2) for s in S]),
            ("Period", "min", [_f(s["period"], 2) for s in S]),
            ("Breeze-KM propellant reserve", "kg", [_f(s["propellant_left"], 0) for s in S])]
    return [[q, f'<span class="muted">{u}</span>', *v] for q, u, v in rows]


def _event_rows(models) -> list[list[str]]:
    keys = [e["key"] for e in models[0]["events"]]
    out = []
    for k in keys:
        ev = [next((e for e in m["events"] if e["key"] == k), None) for m in models]
        r = [ev[0]["label"]]
        for e in ev:
            r += ["–"] * 3 if e is None else [_f(e["t"], 1), _f(e["h"], 1), _f(e["v"], 3)]
        out.append(r)
    return out


def _orbit_rows(models) -> list[list[str]]:
    out = []
    for o in models[0]["orbits"]:
        r = [o["label"]]
        for m in models:
            p = next((x for x in m["orbits"] if x["key"] == o["key"]), None)
            r += ["–"] * 3 if p is None else [_f(p["perigee"], 0), _f(p["apogee"], 0), _f(p["inc"], 2)]
        out.append(r)
    return out


def _grouped_head(first: str, models, cols: list[str]) -> str:
    grp = "".join(f'<th class="grp" colspan="{len(cols)}"><span>{html.escape(m["name"])}</span></th>' for m in models)
    sub = "".join(f"<th>{c}</th>" for _ in models for c in cols)
    return f"<tr><th></th>{grp}</tr><tr><th>{first}</th>{sub}</tr>"


def ascent_dashboard_html(data: dict, *, nav: str = "", subtitle: str = "", report_href: str | None = None,
                          fairing_heat_flux: float = 1135.0) -> str:
    """The dashboard page for ``ascent_dataset`` output (see ``aerosim.plots.ascent_data``)."""
    models = data["models"]
    sty = ["solid", "dashed", "dotted", "dash-dotted"]
    key = " and ".join(f'{html.escape(m["name"]).lower()} ({sty[i]})' for i, m in enumerate(models))
    links = [f'<a href="{html.escape(report_href)}">PDF report</a>'] if report_href else []
    links.append('<a href="#" id="json">flight data (JSON)</a>')
    ref = models[0]["summary"]

    def fig(n, subs, cap):
        return (f'<figure><div class="subs">' + "".join(
            f'<div class="sub"><canvas id="{i}"{" class=" + cls if cls else ""}></canvas><p class="cap"><b>({chr(97 + k)})</b> {c}</p></div>'
            for k, (i, c, cls) in enumerate(subs)) + f'</div><figcaption><b>Figure {n}:</b> {cap}</figcaption></figure>')

    head_sum = "<tr><th>Quantity</th><th></th>" + "".join(f"<th>{html.escape(m['name'])}</th>" for m in models) + "</tr>"
    t1 = (f'<div class="tbl"><p class="tcap"><b>Table 1:</b> Summary of the ascent. Losses are integrated over the '
          f'powered flight to transfer-orbit insertion.</p>{_table(head_sum, _summary_rows(models))}</div>')
    t2 = (f'<div class="tbl"><p class="tcap"><b>Table 3:</b> Event timeline: time <i>t</i> (s), altitude <i>h</i> (km) '
          f'and Earth-relative speed <i>v</i> (km/s).</p>'
          f'{_table(_grouped_head("Event", models, ["<i>t</i>", "<i>h</i>", "<i>v</i>"]), _event_rows(models))}</div>')
    t3 = (f'<div class="tbl"><p class="tcap"><b>Table 2:</b> Osculating orbit after each staging event: perigee and '
          f'apogee altitude <i>h</i><sub>p</sub>, <i>h</i><sub>a</sub> (km) and inclination <i>i</i> (deg). A negative '
          f'perigee is a suborbital arc.</p>'
          f'{_table(_grouped_head("After", models, ["<i>h</i><sub>p</sub>", "<i>h</i><sub>a</sub>", "<i>i</i>"]), _orbit_rows(models))}</div>')
    dot = "Dotted verticals mark stage cut-offs of the first model."
    body = f"""<main>
<header><h1>{html.escape(data.get("title") or "Ascent flight data")}</h1>
<p class="sub">{html.escape(subtitle or data.get("target", ""))}</p>
<p class="meta">Models: {key} · {" · ".join(links)}</p></header>
<h2><span class="n">1</span>Summary</h2>
<div class="pair">{t1}<div>{t3}</div></div>
{t2}
<h2><span class="n">2</span>Trajectory</h2>
{fig(1, [("f-h", "Altitude to transfer-orbit insertion.", ""), ("f-3d", "Trajectory over the ground, with its projection. Drag to turn.", "d3")],
     f"Ascent from lift-off to Breeze-KM cut-off at {_f(ref['t_insertion'], 0)} s. {dot}")}
{fig(2, [("f-v", "Earth-relative speed.", ""), ("f-g", "Flight-path angle above the local horizon.", "")],
     "Velocity. The flight-path angle falls through zero as Breeze-KM flies the transfer-orbit insertion.")}
{fig(3, [("f-gt", "Ground track over the full flight.", "map"), ("f-dr", "Altitude against ground range.", "")],
     "Ground track (Natural Earth coastlines) and the ascent profile over the ground.")}
<h2><span class="n">3</span>Atmospheric flight</h2>
{fig(4, [("f-q", "Dynamic pressure.", ""), ("f-M", "Mach number.", "")],
     f"The first stage through the dense atmosphere: maximum dynamic pressure of {_f(ref['max_q'], 1)} kPa at "
     f"{_f(ref['t_max_q'], 0)} s, {_f(ref['t_max_q'] - ref['t_mach1'], 0)} s after Mach 1.")}
{fig(5, [("f-D", "Aerodynamic drag.", ""), ("f-Q", "Free-molecular heat flux after stage 1.", "")],
     f"Aerodynamic loads. The fairing is released when ½ρ<i>v</i>³ falls below {_f(fairing_heat_flux, 0)} W/m².")}
<h2><span class="n">4</span>Propulsion and performance</h2>
{fig(6, [("f-F", "Thrust, logarithmic scale.", ""), ("f-m", "Vehicle mass.", "")],
     "Thrust, with the back-pressure loss of each stage, and mass through staging and fairing jettison.")}
{fig(7, [("f-a", "Non-gravitational acceleration.", ""), ("f-L", f"Velocity losses, {html.escape(models[0]['name']).lower()}.", "")],
     "Axial load on the payload, and the velocity lost to gravity, drag and nozzle back-pressure.")}
</main>
{theme_tag("paper")}{data_tag({**data, "coast": _coast(), "fairing_heat_flux": fairing_heat_flux})}
<script>{_JS}</script>"""
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(data.get("title") or "Ascent flight data")}</title><style>{CSS}{_CSS}</style></head><body>
{nav}
{body}
</body></html>"""
    return themed(page, "paper")
