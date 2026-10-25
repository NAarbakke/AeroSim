"""Shared pieces of AeroSim's HTML pages: theme tokens, CSS, a canvas chart library, the offline
three.js import map. Pages built from these look like the 3D ascent replay (``replay.py``).

Charts follow ``aerosim.plots.style``: categorical colours in fixed order, 2 px lines, hairline
grid, ink-coloured text, a legend whenever there are two or more series, and a hover crosshair
with a tooltip on every line chart.
"""

from __future__ import annotations

import html
import json
import math

from aerosim.plots.replay import _importmap
from aerosim.plots.style import THEMES

CSS = r"""
:root{--surface:__SURFACE__;--page:__PAGE__;--ink:__INK__;--ink2:__INK2__;--muted:__MUTED__;
--grid:__GRID__;--axis:__AXIS__;--accent:__SERIES0__}
*{box-sizing:border-box}html,body{margin:0;background:var(--page);color:var(--ink);
font:14px/1.45 "Segoe UI",system-ui,-apple-system,Helvetica,Arial,sans-serif}
a{color:var(--ink)}
nav.top{display:flex;align-items:center;gap:6px 18px;flex-wrap:wrap;padding:10px 20px;
background:var(--surface);border-bottom:1px solid var(--grid);font-size:13px}
nav.top b{font-weight:650;margin-right:8px}
nav.top a{color:var(--ink2);text-decoration:none;padding:2px 0;border-bottom:2px solid transparent}
nav.top a:hover{color:var(--ink)}nav.top a[aria-current]{color:var(--ink);border-color:var(--accent)}
.card{border:1px solid var(--grid);border-radius:8px;padding:10px 12px;margin-bottom:10px;
background:var(--surface)}
.card h2{margin:0 0 6px;font-size:13px;font-weight:600;display:flex;align-items:center;gap:8px}
.sw{width:12px;height:12px;border-radius:3px;flex:none}
.card .ph{margin-left:auto;font-weight:500;color:var(--ink2)}
.tm{display:grid;grid-template-columns:repeat(3,1fr);gap:4px 10px}
.tm div{font-variant-numeric:tabular-nums;font-size:15px;font-weight:600}
.tm span{display:block;font-size:11px;font-weight:400;color:var(--ink2)}
h3{margin:16px 0 4px;font-size:13px;font-weight:600}
.legend{display:flex;flex-wrap:wrap;gap:4px 14px;font-size:12px;color:var(--ink2);margin-bottom:2px}
.legend i{display:inline-block;width:14px;height:3px;border-radius:2px;margin-right:5px;
vertical-align:middle}
.cw{position:relative}canvas.chart{width:100%;height:170px;display:block}
.tip{position:absolute;pointer-events:none;background:var(--page);border:1px solid var(--axis);
border-radius:6px;padding:5px 8px;font-size:12px;white-space:nowrap;display:none;z-index:5;
font-variant-numeric:tabular-nums}
.tip i{display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:6px}
button,select{font:inherit;color:var(--ink);background:var(--page);border:1px solid var(--axis);
border-radius:6px;padding:5px 12px;cursor:pointer}
button:focus-visible,select:focus-visible,input:focus-visible,a:focus-visible{outline:2px solid
var(--accent);outline-offset:2px}
label{display:flex;align-items:center;gap:6px;color:var(--ink2);font-size:13px;cursor:pointer}
table{border-collapse:collapse;width:100%;font-size:12.5px;font-variant-numeric:tabular-nums}
th,td{padding:4px 8px;text-align:right;border-bottom:1px solid var(--grid)}
th:first-child,td:first-child{text-align:left}th{color:var(--ink2);font-weight:500}
td.good{color:var(--ink)}td.dev{color:var(--ink2)}
.muted{color:var(--ink2)}
"""

CHART_JS = r"""
const TKC = JSON.parse(document.getElementById("theme").textContent);
const niceStep = x => { const p = Math.pow(10, Math.floor(Math.log10(x)));
  for (const s of [1, 2, 2.5, 5, 10]) if (s * p >= x - 1e-12) return s * p; };
function axisRange(lo, hi, n = 4, zero = false) {
  if (zero) lo = Math.min(lo, 0);
  if (lo < 0 && -lo < 0.02 * (hi - lo)) lo = 0;              // touchdown round-off is not a new tick
  if (hi - lo < 1e-12) { hi = lo + 1; }
  const st = niceStep((hi - lo) / n); return [Math.floor(lo / st) * st, Math.ceil(hi / st - 1e-9) * st, st];
}
const fmtN = (v, st) => { const d = Math.max(0, -Math.floor(Math.log10(st) + 1e-9));
  return (Math.abs(v) < st / 2 ? 0 : v).toFixed(Math.min(d, 4)); };
function setup(cv) {
  const r = cv.getBoundingClientRect(), dpr = Math.min(devicePixelRatio, 2);
  if (cv.width !== Math.round(r.width * dpr) || cv.height !== Math.round(r.height * dpr)) {
    cv.width = Math.round(r.width * dpr); cv.height = Math.round(r.height * dpr); }
  const c = cv.getContext("2d"); c.setTransform(dpr, 0, 0, dpr, 0, 0); c.clearRect(0, 0, r.width, r.height);
  c.font = "11px Segoe UI, system-ui, sans-serif"; return [c, r.width, r.height];
}
// static line chart: o = {series:[{name,color,x,y,dash}], xl, yl, zero, xr, yr, digits}
function lineChart(cv, o) {
  const draw = hx => {
    const [c, w, h] = setup(cv);
    let xs = [], ys = [];
    for (const s of o.series) for (let i = 0; i < s.x.length; i++) if (Number.isFinite(s.y[i])) { xs.push(s.x[i]); ys.push(s.y[i]); }
    const [x0, x1, xst] = o.xr || axisRange(Math.min(...xs), Math.max(...xs));
    const [y0, y1, yst] = o.yr || axisRange(Math.min(...ys), Math.max(...ys), 4, o.zero);
    const L = 42, B = 30, T = 8, W = w - L - 10, H = h - B - T;
    const X = x => L + (x - x0) / (x1 - x0) * W, Y = y => T + H - (y - y0) / (y1 - y0) * H;
    c.strokeStyle = TKC.grid; c.lineWidth = 1; c.fillStyle = TKC.ink2;
    for (let y = y0; y <= y1 + yst / 2; y += yst) { c.beginPath(); c.moveTo(L, Y(y)); c.lineTo(L + W, Y(y)); c.stroke();
      c.textAlign = "right"; c.fillText(fmtN(y, yst), L - 6, Y(y) + 4); }
    for (let x = x0; x <= x1 + xst / 2; x += xst) { c.textAlign = "center"; c.fillText(fmtN(x, xst), X(x), T + H + 14); }
    c.textAlign = "right"; c.fillText(o.xl || "", L + W, h - 2);
    c.save(); c.translate(10, T); c.textAlign = "left"; c.restore();
    for (const s of o.series) {
      c.strokeStyle = s.color; c.lineWidth = 2; c.lineJoin = "round"; c.setLineDash(s.dash ? [5, 4] : []);
      c.beginPath(); let pen = false;
      for (let i = 0; i < s.x.length; i++) { if (!Number.isFinite(s.y[i])) { pen = false; continue; }
        pen ? c.lineTo(X(s.x[i]), Y(s.y[i])) : c.moveTo(X(s.x[i]), Y(s.y[i])); pen = true; }
      c.stroke(); c.setLineDash([]);
      if (s.marks) for (let i = 0; i < s.x.length; i++) if (Number.isFinite(s.y[i])) {
        c.beginPath(); c.arc(X(s.x[i]), Y(s.y[i]), 4, 0, 7); c.fillStyle = s.color; c.fill();
        c.lineWidth = 2; c.strokeStyle = TKC.surface; c.stroke(); }
    }
    if (hx != null) {
      const xv = x0 + (hx - L) / W * (x1 - x0);
      c.strokeStyle = TKC.axis; c.lineWidth = 1; c.beginPath(); c.moveTo(hx, T); c.lineTo(hx, T + H); c.stroke();
      const rows = [];
      for (const s of o.series) { let k = -1, d = 1e99;
        for (let i = 0; i < s.x.length; i++) { const e = Math.abs(s.x[i] - xv); if (e < d && Number.isFinite(s.y[i])) { d = e; k = i; } }
        if (k < 0) continue;
        c.beginPath(); c.arc(X(s.x[k]), Y(s.y[k]), 5, 0, 7); c.fillStyle = s.color; c.fill();
        c.lineWidth = 2; c.strokeStyle = TKC.surface; c.stroke();
        rows.push(`<div><i style="background:${s.color}"></i>${s.name}: <b>${(+s.y[k]).toFixed(o.digits ?? 2)}</b> ${o.yu || ""}</div>`);
        if (!rows.x) rows.x = s.x[k];
      }
      tip.innerHTML = `<div class="muted">${o.xl || "x"} ${(+rows.x).toFixed(o.xdigits ?? 2)}</div>` + rows.join("");
      tip.style.display = "block"; tip.style.left = Math.min(hx + 12, w - tip.offsetWidth - 4) + "px"; tip.style.top = "6px";
    } else tip.style.display = "none";
  };
  const tip = document.createElement("div"); tip.className = "tip"; cv.parentElement.append(tip);
  cv.onmousemove = e => draw(e.offsetX); cv.onmouseleave = () => draw(null);
  draw(null); addEventListener("resize", () => draw(null));
}
function legend(el, series) { el.innerHTML = series.length > 1 ? series.map(s =>
  `<span><i style="background:${s.color}${s.dash ? ";opacity:.6" : ""}"></i>${s.name}</span>`).join("") : ""; }
// horizontal dot plot of percent deviation: rows = [{label, value, color, note}]
function devChart(cv, rows, lim = 20) {
  const r0 = cv.getBoundingClientRect(); cv.style.height = (rows.length * 26 + 34) + "px";
  const draw = hy => {
    const [c, w, h] = setup(cv);
    const L = Math.min(250, w * 0.45), W = w - L - 16, T = 6, X = v => L + (v + lim) / (2 * lim) * W;
    c.strokeStyle = TKC.grid; c.fillStyle = TKC.ink2; c.textAlign = "center";
    for (let v = -lim; v <= lim; v += lim / 2) { c.beginPath(); c.moveTo(X(v), T); c.lineTo(X(v), h - 22); c.stroke();
      c.fillText((v > 0 ? "+" : "") + v + " %", X(v), h - 6); }
    c.strokeStyle = TKC.axis; c.lineWidth = 1.5; c.beginPath(); c.moveTo(X(0), T); c.lineTo(X(0), h - 22); c.stroke();
    rows.forEach((r, i) => { const y = T + 13 + i * 26;
      if (hy != null && Math.abs(hy - y) < 13) { c.fillStyle = TKC.grid; c.fillRect(0, y - 13, w, 26); }
      c.fillStyle = TKC.ink; c.textAlign = "left"; c.fillText(r.label, 4, y + 4);
      const v = Math.max(-lim, Math.min(lim, r.value));
      c.strokeStyle = r.color; c.lineWidth = 2; c.beginPath(); c.moveTo(X(0), y); c.lineTo(X(v), y); c.stroke();
      c.beginPath(); c.arc(X(v), y, 5, 0, 7); c.fillStyle = r.color; c.fill(); c.lineWidth = 2;
      c.strokeStyle = TKC.surface; c.stroke();
      c.fillStyle = TKC.ink2; c.textAlign = r.value >= 0 ? "left" : "right";
      c.fillText((r.value > 0 ? "+" : "") + r.value.toFixed(1) + " %", X(v) + (r.value >= 0 ? 9 : -9), y + 4); });
    if (hy != null) { const i = Math.round((hy - T - 13) / 26); const r = rows[i];
      if (r) { tip.innerHTML = `<b>${r.label}</b><div class="muted">${r.note}</div>`; tip.style.display = "block";
        tip.style.left = "8px"; tip.style.top = (T + 26 + i * 26) + "px"; } else tip.style.display = "none";
    } else tip.style.display = "none";
  };
  const tip = document.createElement("div"); tip.className = "tip"; cv.parentElement.append(tip);
  cv.onmousemove = e => draw(e.offsetY); cv.onmouseleave = () => draw(null);
  draw(null); addEventListener("resize", () => draw(null));
}
"""


def themed(text: str, theme: str = "dark") -> str:
    """Fill ``__TOKEN__`` placeholders from the theme (``__SERIES0__`` = first series colour)."""
    T = THEMES[theme]
    text = text.replace("__SERIES0__", T["series"][0])
    for k, v in T.items():
        if isinstance(v, str):
            text = text.replace(f"__{k.upper()}__", v)
    return text


def nav(pages: list[tuple[str, str]], current: str) -> str:
    """Top navigation bar: [(label, href), ...]; ``current`` is the active href."""
    links = "".join(f'<a href="{h}"{" aria-current=page" if h == current else ""}>{html.escape(n)}</a>'
                    for n, h in pages)
    return f'<nav class="top"><b>AeroSim</b>{links}</nav>'


def _finite(o):
    """JSON has no NaN / Infinity: non-finite numbers become null (a gap in a chart)."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_finite(v) for v in o]
    return o


def data_tag(obj, id_: str = "data") -> str:
    text = json.dumps(_finite(obj), separators=(",", ":"), allow_nan=False)
    return f'<script type="application/json" id="{id_}">' + text.replace("</", "<\\/") + "</script>"


def theme_tag(theme: str = "dark") -> str:
    T = THEMES[theme]
    return data_tag({**{k: v for k, v in T.items() if isinstance(v, str)}, "series": T["series"],
                     "dark": theme == "dark"}, "theme")


def importmap_tag() -> str:
    return f'<script type="importmap">{_importmap()}</script>'
