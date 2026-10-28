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
from aerosim.plots.style import DARK_THEMES, FONTS, SITE_THEMES, THEMES, default_font

CSS = r"""
*{box-sizing:border-box}html,body{margin:0;background:var(--page);color:var(--ink);
font:14px/1.45 var(--font)}
:root[data-font=serif] body{font-size:15px;font-variant-numeric:lining-nums}
:root[data-theme=carbon] h3,:root[data-theme=paper] h3{font-weight:700;letter-spacing:.01em}
:root[data-theme=carbon] .card,:root[data-theme=carbon] button,:root[data-theme=carbon] select{border-radius:2px}
:root[data-theme=paper] .card,:root[data-theme=paper] .panel,:root[data-theme=paper] .tile,
:root[data-theme=paper] button,:root[data-theme=paper] select{border-radius:0}
:root[data-theme=paper] nav.top{border-bottom-color:#000}
:root[data-theme=paper] nav.top a[aria-current]{border-color:#000}
nav.top .switch{margin-left:auto;display:flex;gap:8px;white-space:nowrap}
nav.top select{padding:2px 8px;font-size:12px}
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
// carbon: pgfplots-like charts - framed axes, inward ticks on all four sides, no grid, serif
// labels, thin lines told apart by dash pattern as well as colour
const ACAD = !!TKC.academic, CFONT = TKC.font || "system-ui, sans-serif";
const DASHES = [[], [6, 3], [1.5, 2.5], [8, 3, 1.5, 3], [3, 2], [10, 4], [1, 4], [5, 2, 1, 2]];
const dashOf = i => ACAD ? DASHES[i % DASHES.length] : [];
const LW = ACAD ? 1.25 : 2, KEYIN = !!TKC.key_inside, GHOST = KEYIN ? 0.12 : ACAD ? 0.3 : 0.25;
const lwOf = i => !ACAD ? 2 : TKC.weights ? TKC.weights[Math.min(i, TKC.weights.length - 1)] : LW;
// paper: the key sits inside the frame, top right, without a box (as pgfplots / matplotlib place
// it); the HTML legend above the chart is then hidden. items = [{name, color, dash}]
function insideKey(cv, c, L, T, W, items) {
  if (!KEYIN) return;
  const lg = [cv.previousElementSibling, cv.parentElement && cv.parentElement.previousElementSibling]
    .find(e => e && e.classList && e.classList.contains("legend"));
  if (lg) lg.style.display = "none";
  if (items.length < 2) return;
  c.save(); c.font = "11.5px " + CFONT; c.textBaseline = "middle";
  const rows = [[]], gap = 12, sw = 22, maxW = W - 12; let w = 0;
  for (const it of items) { const iw = sw + 4 + c.measureText(it.name).width;
    if (w > 0 && w + gap + iw > maxW) { rows.push([]); w = 0; } rows[rows.length - 1].push([it, iw]); w += (w ? gap : 0) + iw; }
  rows.forEach((row, r) => { let x = L + W - 6 - row.reduce((a, [, iw], k) => a + iw + (k ? gap : 0), 0); const y = T + 10 + r * 15;
    c.fillStyle = TKC.page; c.globalAlpha = 0.85; c.fillRect(x - 3, y - 7, row.reduce((a, [, iw], k) => a + iw + (k ? gap : 0), 0) + 6, 14); c.globalAlpha = 1;
    for (const [it, iw] of row) { c.strokeStyle = it.color; c.lineWidth = it.lw || 1.2; c.setLineDash(it.dash || []);
      c.beginPath(); c.moveTo(x, y); c.lineTo(x + sw, y); c.stroke(); c.setLineDash([]);
      c.fillStyle = TKC.ink; c.textAlign = "left"; c.fillText(it.name, x + sw + 4, y + 0.5); x += iw + gap; } });
  c.restore();
}
// grid line (default) or a pair of inward ticks on both sides (academic), y at pixel py
function yTick(c, L, W, py) { c.beginPath();
  if (ACAD) { c.strokeStyle = TKC.ink; c.lineWidth = 0.75; c.moveTo(L, py); c.lineTo(L + 4, py); c.moveTo(L + W - 4, py); c.lineTo(L + W, py); }
  else { c.strokeStyle = TKC.grid; c.lineWidth = 1; c.moveTo(L, py); c.lineTo(L + W, py); }
  c.stroke(); }
function xTick(c, T, H, px) { if (!ACAD) return; c.beginPath(); c.strokeStyle = TKC.ink; c.lineWidth = 0.75;
  c.moveTo(px, T + H); c.lineTo(px, T + H - 4); c.moveTo(px, T); c.lineTo(px, T + 4); c.stroke(); }
function frameBox(c, L, T, W, H) { if (!ACAD) return; c.strokeStyle = TKC.ink; c.lineWidth = 0.75; c.setLineDash([]); c.strokeRect(L + 0.5, T + 0.5, W, H); }
const tickInk = () => ACAD ? TKC.ink : TKC.ink2;
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
  c.font = (ACAD ? "12.5px " : "11px ") + CFONT; return [c, r.width, r.height];
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
    c.fillStyle = tickInk();
    for (let y = y0; y <= y1 + yst / 2; y += yst) { yTick(c, L, W, Y(y));
      c.textAlign = "right"; c.fillText(fmtN(y, yst), L - 6, Y(y) + 4); }
    for (let x = x0; x <= x1 + xst / 2; x += xst) { xTick(c, T, H, X(x)); c.textAlign = "center"; c.fillText(fmtN(x, xst), X(x), T + H + 14); }
    c.textAlign = "right"; c.fillText(o.xl || "", L + W, h - 2);
    c.save(); c.translate(10, T); c.textAlign = "left"; c.restore();
    o.series.forEach((s, si) => {
      c.strokeStyle = s.color; c.lineWidth = lwOf(si); c.lineJoin = "round"; c.setLineDash(s.dash ? [5, 4] : dashOf(si));
      c.beginPath(); let pen = false;
      for (let i = 0; i < s.x.length; i++) { if (!Number.isFinite(s.y[i])) { pen = false; continue; }
        pen ? c.lineTo(X(s.x[i]), Y(s.y[i])) : c.moveTo(X(s.x[i]), Y(s.y[i])); pen = true; }
      c.stroke(); c.setLineDash([]);
      if (s.marks) for (let i = 0; i < s.x.length; i++) if (Number.isFinite(s.y[i])) {
        c.beginPath(); c.arc(X(s.x[i]), Y(s.y[i]), 4, 0, 7); c.fillStyle = s.color; c.fill();
        c.lineWidth = 2; c.strokeStyle = TKC.surface; c.stroke(); }
    });
    frameBox(c, L, T, W, H);
    insideKey(cv, c, L, T, W, o.series.map((s, si) => ({name: s.name, color: s.color, dash: s.dash ? [5, 4] : dashOf(si), lw: lwOf(si)})));
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
function legend(el, series) { el.innerHTML = series.length > 1 ? series.map((s, i) => ACAD
  ? `<span><svg width="22" height="6" aria-hidden="true" style="margin-right:5px;vertical-align:middle"><line x1="0" y1="3" x2="22" y2="3" stroke="${s.color}" stroke-width="1.4" stroke-dasharray="${(s.dash ? [5, 4] : dashOf(i)).join(" ")}"/></svg>${s.name}</span>`
  : `<span><i style="background:${s.color}${s.dash ? ";opacity:.6" : ""}"></i>${s.name}</span>`).join("") : ""; }
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


def _flat(theme: str) -> dict:
    T = THEMES[theme]
    return {**{k: v for k, v in T.items() if isinstance(v, (str, bool, list))},
            "dark": theme in DARK_THEMES, "name": theme}


def _font_faces() -> str:
    """@font-face rules for the site's typefaces (Latin Modern and B612 subsets), embedded so they
    work offline and from file://."""
    import base64
    from importlib import resources
    out = []
    for fam, f, style, weight in (("Latin Modern Roman", "lmroman10-regular", "normal", 400),
                                  ("Latin Modern Roman", "lmroman10-italic", "italic", 400),
                                  ("Latin Modern Roman", "lmroman10-bold", "normal", 700),
                                  ("Latin Modern Mono", "lmmono10-regular", "normal", 400),
                                  ("B612", "B612-Regular", "normal", 400),
                                  ("B612", "B612-Italic", "italic", 400),
                                  ("B612", "B612-Bold", "normal", 700)):
        b = resources.files("aerosim.plots").joinpath(f"data/fonts/{f}.woff").read_bytes()
        out.append(f'@font-face{{font-family:"{fam}";font-style:{style};font-weight:{weight};'
                   f'font-display:swap;src:url(data:font/woff;base64,{base64.b64encode(b).decode()}) format("woff")}}')
    return "".join(out)


def theme_head(default: str = "dark") -> str:
    """Put in <head>: every switchable theme's tokens as CSS variables, the carbon fonts, and a
    script that applies the theme stored by the nav toggle (localStorage, if available) before the
    page renders. Pages keep reading their ``theme`` JSON tag; ``theme_tag`` points it at the
    chosen theme."""
    def block(name):
        T = THEMES[name]
        v = ";".join(f"--{k}:{T[k]}" for k in ("surface", "page", "ink", "ink2", "muted", "grid", "axis", "font", "mono"))
        s = ";".join(f"--s{i}:{c}" for i, c in enumerate(T["series"]))
        return f"{v};{s};--accent:{T['series'][0]}"
    names = SITE_THEMES if default in SITE_THEMES else (default,)
    css = f":root{{{block(default)}}}" + "".join(f":root[data-theme={n}]{{{block(n)}}}" for n in names)
    # the typeface choice comes after the themes so it overrides their --font / --mono
    css += "".join(f":root[data-font={k}]{{--font:{v['font']};--mono:{v['mono']}}}" for k, v in FONTS.items())
    # B612 runs wide: a tighter nav keeps it on one line as far down as the other faces
    css += (":root[data-font=technical] body{font-size:13.5px;letter-spacing:.005em}"
            ":root[data-font=technical] nav.top{font-size:12px;letter-spacing:0;column-gap:15px}"
            ":root[data-font=technical] nav.top select{font-size:11.5px}"
            ":root[data-font=technical] #clock,:root[data-font=technical] #gauges div,"
            ":root[data-font=technical] #tclock,:root[data-font=technical] .tm div"
            "{font-variant-numeric:tabular-nums;letter-spacing:0}")
    themes = json.dumps({n: {**_flat(n), "font_key": default_font(n)} for n in names}, separators=(",", ":"))
    fonts = json.dumps({k: {"font": v["font"], "mono": v["mono"]} for k, v in FONTS.items()},
                       separators=(",", ":"))
    js = (f"(function(){{var T={themes},F={fonts},d={json.dumps(default)},t=d,f='';"
          "try{t=localStorage.getItem('aerosim-theme')||d;f=localStorage.getItem('aerosim-font')||f}catch(e){}"
          "if(!T[t])t=d;var th=Object.assign({},T[t]);if(!F[f])f=th.font_key;"
          "th.font=F[f].font;th.mono=F[f].mono;document.documentElement.setAttribute('data-font',f);"
          "document.documentElement.setAttribute('data-theme',t);window.AEROSIM_THEME=th;})();")
    return f'<style id="aerosim-themes">{_font_faces()}{css}</style><script>{js}</script>'


def themed(text: str, theme: str = "dark") -> str:
    """Point ``__TOKEN__`` placeholders at the theme's CSS variables (``__SERIES0__`` = the accent)
    so the page follows the theme chosen at run time; a full page also gets ``theme_head``."""
    T = THEMES[theme]
    text = text.replace("__SERIES0__", "var(--accent)")
    for k, v in T.items():
        if isinstance(v, str):
            text = text.replace(f"__{k.upper()}__", f"var(--{k})")
    if "<head>" in text and 'id="aerosim-themes"' not in text:
        text = text.replace("<head>", "<head>" + theme_head(theme), 1)
    return text


def nav(pages: list[tuple[str, str]], current: str) -> str:
    """Top navigation bar: [(label, href), ...]; ``current`` is the active href. The select at
    the right switches the site's theme (stored in localStorage; the page reloads)."""
    links = "".join(f'<a href="{h}"{" aria-current=page" if h == current else ""}>{html.escape(n)}</a>'
                    for n, h in pages)
    opts = "".join(f'<option value="{n}">{n.capitalize()}</option>' for n in SITE_THEMES)
    switch = ('<span class="switch"><select class="theme" aria-label="Theme" onchange="try{localStorage.setItem(\'aerosim-theme\','
              f'this.value)}}catch(e){{}}location.reload()">{opts}</select>'
              '<script>document.currentScript.previousElementSibling.value='
              'document.documentElement.getAttribute("data-theme")||"dark"</script>')
    fopts = "".join(
        f'<option value="{k}">{html.escape(v["label"])}</option>' for k, v in FONTS.items())
    switch += ('<select class="font" aria-label="Typeface" onchange="try{localStorage.setItem(\'aerosim-font\','
               f'this.value)}}catch(e){{}}location.reload()">{fopts}</select>'
               '<script>document.currentScript.previousElementSibling.value='
               'document.documentElement.getAttribute("data-font")</script></span>')
    return f'<nav class="top"><b>AeroSim</b>{links}{switch}</nav>'


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
    """The page's ``theme`` JSON (tokens for scripts), switched to the run-time theme if one was
    chosen (see ``theme_head``)."""
    return data_tag(_flat(theme), "theme") + (
        '<script>(function(){var e=document.getElementById("theme");'
        'if(window.AEROSIM_THEME&&e)e.textContent=JSON.stringify(window.AEROSIM_THEME)})()</script>')


def importmap_tag(gltf: bool = False) -> str:
    return f'<script type="importmap">{_importmap(gltf)}</script>'


def write_page(path, text: str, tries: int = 20) -> None:
    """Write a page atomically: a temporary sibling, then a rename over the target. On Windows a
    freshly written large file is often held for a moment by a scanner or indexer, so the rename
    is retried briefly instead of failing the build (and a failed build never leaves half a page)."""
    import os
    import time
    from pathlib import Path
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    for k in range(tries):
        try:
            os.replace(tmp, path)
            return
        except OSError:
            if k == tries - 1:
                raise
            time.sleep(0.25)
