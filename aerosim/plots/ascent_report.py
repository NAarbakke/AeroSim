"""LaTeX report of an orbital ascent: a two-column article in Computer Modern with figures drawn
by matplotlib's PGF backend at column width (their text is set by LaTeX in the document's own
font), booktabs tables and a text whose numbers all come from the flight data.

    data = ascent_dataset({...}, site=(62.8, 40.3), target="...")       # aerosim.plots.ascent_data
    pdf = build_report(data, "docs/reports/rockot", name="Rockot_Ascent_Report",
                       title="...", models_text={"First principles": "...", ...})

``build_report`` writes ``build/report.tex`` and ``build/figures/*.pgf`` under the output folder
and compiles them with ``latexmk -lualatex`` (TeX Live or MiKTeX); ``compile=False`` stops after
writing the sources. The pieces are usable on their own: ``report_figures`` (the .pgf figures)
and ``report_tex`` (the LaTeX text).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from importlib import resources
from pathlib import Path

import numpy as np

from aerosim.core import AerosimError, require
from aerosim.plots.style import rc

COLUMN = 3.29          # in: column width of the article layout below (A4, 18 mm margins, 7 mm gap)
FIG_H = 2.05
STYLES = [{"color": "#000000", "lw": 1.0, "ls": "-"}, {"color": "#6e6e6e", "lw": 0.85, "ls": (0, (4.5, 2.5))},
          {"color": "#000000", "lw": 0.85, "ls": (0, (1, 1.6))}, {"color": "#6e6e6e", "lw": 0.85, "ls": "-."}]
GREY = "#b4b4b4"


# ------------------------------------------------------------------ figures
def _separations(ref) -> list[float]:
    return [e["t"] for e in ref["events"] if any(k in e["key"] for k in ("burnout", "burn1_end", "burn2_end"))]


def _new(plt, h=FIG_H):
    fig, ax = plt.subplots(figsize=(COLUMN, h), constrained_layout=True)
    return fig, ax


def _vlines(ax, xs, lo, hi):
    for x in xs:
        if lo < x < hi:
            ax.axvline(x, color=GREY, lw=0.5, ls=(0, (1, 2)), zorder=0)


def _lines(ax, models, key, src="series", tr=None):
    """One line per model against time; only the samples inside ``tr`` (so the y-axis autoscales
    to the plotted window), plus one either side so the line reaches the frame."""
    for i, m in enumerate(models):
        x, y = np.asarray(m[src]["t"], float), np.asarray(m[src][key], float)
        if tr is not None:
            k = np.nonzero((x >= tr[0]) & (x <= tr[1]))[0]
            k = np.arange(max(k[0] - 1, 0), min(k[-1] + 2, x.size)) if k.size else k
            x, y = x[k], y[k]
        ax.plot(x, y, label=m["name"], **STYLES[i])


def _time_axes(ax, xr, sep, ylabel):
    ax.set_xlim(*xr)
    ax.set_xlabel(r"time $t$ (s)")
    ax.set_ylabel(ylabel)
    _vlines(ax, sep, *xr)


def report_figures(data: dict, out: Path) -> list[str]:
    """Write every figure as a .pgf file (to ``\\input`` in the report); returns their names."""
    mpl = require("matplotlib", "plot")
    plt = require("matplotlib.pyplot", "plot")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    models, ref = data["models"], data["models"][0]
    sep = _separations(ref)
    t_ins = max(m["summary"]["t_insertion"] for m in models)
    ta = (0, float(np.ceil(t_ins / 100) * 100))
    names = []

    def save(fig, name):
        fig.savefig(out / f"{name}.pgf", backend="pgf")
        plt.close(fig)
        names.append(name)

    with plt.rc_context(rc("paper", pgf=True) | {"legend.fontsize": 7.5, "font.size": 8.5, "axes.labelsize": 8.5,
                                                  "xtick.labelsize": 8, "ytick.labelsize": 8}):
        for key, name, ylabel, xr, loc, src in (
                ("h", "altitude", r"altitude $h$ (km)", ta, "lower right", "series"),
                ("v", "speed", r"speed $v$ (km/s)", ta, "lower right", "series"),
                ("fpa", "fpa", r"flight-path angle $\gamma$ (deg)", ta, "upper right", "series"),
                ("q", "q", r"dynamic pressure $q$ (kPa)", (0, 150), "upper right", "series"),
                ("mach", "mach", r"Mach number $M$", (0, 150), "lower right", "series"),
                ("drag", "drag", r"drag $D$ (kN)", (0, 150), "upper right", "physics"),
                ("m", "mass", r"mass $m$ (t)", ta, "upper right", "series"),
                ("acc", "accel", r"acceleration $a/g_0$", ta, "upper right", "series")):
            fig, ax = _new(plt)
            _lines(ax, models, key, src, xr)
            _time_axes(ax, xr, sep if xr == ta else [], ylabel)
            ax.set_ylim(bottom=0 if key not in ("fpa",) else None)
            ax.legend(loc=loc)
            save(fig, name)

        fig, ax = _new(plt)                                     # thrust, log scale
        _lines(ax, models, "thrust", "physics", ta)
        ax.set_yscale("log")
        _time_axes(ax, ta, sep, r"thrust $F$ (kN)")
        ax.legend(loc="center right")
        save(fig, "thrust")

        fig, ax = _new(plt)                                     # free-molecular heat flux
        _lines(ax, models, "heat", "physics", (100, 300))
        ax.set_yscale("log")
        ax.set_ylim(10, 1e6)
        _time_axes(ax, (100, 300), [], r"heat flux $\frac{1}{2}\rho v^3$ (W/m$^2$)")
        lim = data.get("fairing_heat_flux", 1135.0)
        ax.axhline(lim, color="#000000", lw=0.6, ls=(0, (3, 2)))
        ax.annotate("fairing limit", (0.98, lim), xycoords=("axes fraction", "data"), ha="right", va="bottom",
                    fontsize=7.5, xytext=(0, 1.5), textcoords="offset points")
        ax.legend(loc="upper right")
        save(fig, "heat")

        fig, ax = _new(plt)                                     # velocity losses, reference model
        for i, (k, lab) in enumerate((("gravity", "Gravity"), ("drag", "Drag"), ("pressure", "Back-pressure"))):
            ax.plot(ref["physics"]["t"], ref["physics"]["loss"][k], label=lab, **STYLES[i])
        _time_axes(ax, ta, sep, r"velocity loss $\Delta v$ (m/s)")
        ax.set_ylim(bottom=0)
        ax.legend(loc="center right")
        save(fig, "losses")

        fig, ax = _new(plt)                                     # altitude against ground range
        for i, m in enumerate(models):
            t = np.asarray(m["series"]["t"])
            keep = t <= t_ins
            ax.plot(np.asarray(m["series"]["dr"])[keep], np.asarray(m["series"]["h"])[keep], label=m["name"], **STYLES[i])
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("ground range (km)")
        ax.set_ylabel(r"altitude $h$ (km)")
        ax.legend(loc="lower right")
        save(fig, "range")

        fig, ax = _new(plt, COLUMN * 0.56)                      # ground track
        coast = json.loads(resources.files("aerosim.plots").joinpath("data/coastline_110m.json").read_text(encoding="utf-8"))
        for ln in coast["lines"]:
            a = np.asarray(ln)
            ax.plot(a[:, 0], a[:, 1], color=GREY, lw=0.3, ls="-")
        for i, m in enumerate(models):
            lon, lat = np.asarray(m["series"]["lon"], float), np.asarray(m["series"]["lat"], float)
            lon = np.where(np.r_[False, np.abs(np.diff(lon)) > 180], np.nan, lon)    # break at the date line
            ax.plot(lon, lat, label=m["name"], **STYLES[i])
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_xticks(range(-180, 181, 60))
        ax.set_yticks(range(-90, 91, 30))
        ax.set_aspect("equal")
        ax.set_xlabel("longitude (deg)")
        ax.set_ylabel("latitude (deg)")
        ax.legend(loc="center right", fontsize=6.5)
        save(fig, "track")

        fig = plt.figure(figsize=(COLUMN, 2.55))                # 3D trajectory over the ground
        ax = fig.add_subplot(projection="3d", computed_zorder=False)
        fig.subplots_adjust(left=0, right=0.94, bottom=0.04, top=1.0)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color((1, 1, 1, 0))
            axis.pane.set_edgecolor(GREY)
            axis._axinfo["grid"].update(color="#e3e3e3", linewidth=0.5)
            axis._axinfo["tick"].update(inward_factor=0.0, outward_factor=0.3)
            axis.line.set_linewidth(0.5)
            axis.set_minor_locator(mpl.ticker.NullLocator())
            axis.set_major_locator(mpl.ticker.MaxNLocator(4))
        e, n = np.asarray(ref["enu"]["e"]), np.asarray(ref["enu"]["n"])
        ax.plot(e, n, np.zeros_like(e), color=GREY, lw=0.7, ls=(0, (3, 2)))
        for i, m in enumerate(models):
            ax.plot(m["enu"]["e"], m["enu"]["n"], m["enu"]["u"], **STYLES[i])
        ax.set_xlabel("east (km)", labelpad=-4)
        ax.set_ylabel("north (km)", labelpad=-2)
        ax.zaxis.set_rotate_label(False)
        ax.set_zlabel(r"$h$ (km)", labelpad=-2)
        ax.tick_params(pad=-2, labelsize=7)
        ax.set_zlim(bottom=0)
        ax.set_box_aspect((1.0, 1.6, 0.75), zoom=0.92)
        ax.view_init(elev=22, azim=-40)
        save(fig, "trajectory3d")
    return names


# ------------------------------------------------------------------ text
_TEX_ESC = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
            "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "\\": r"\textbackslash{}",
            "°": r"$^\circ$", "×": r"$\times$"}


def tex(s: str) -> str:
    """Escape plain text for LaTeX."""
    return "".join(_TEX_ESC.get(c, c) for c in str(s))


def _n(v, d=1) -> str:
    return "--" if v is None else rf"\num{{{v:.{d}f}}}"


def _head_cell(name: str) -> str:
    """Column head, broken over two lines when long."""
    a, _, b = name.partition(" ")
    return rf"\begin{{tabular}}[b]{{@{{}}r@{{}}}}{tex(a)}\\{tex(b)}\end{{tabular}}" if len(name) > 14 and b else tex(name)


def _summary_table(models) -> str:
    S = [m["summary"] for m in models]
    rows = [("Lift-off mass", "t", [_n(s["liftoff_mass"], 2) for s in S]),
            ("Payload", "t", [_n(s["payload"], 2) for s in S]),
            ("Mach 1", "s", [_n(s["t_mach1"], 1) for s in S]),
            ("Max.\\ dynamic pressure", "kPa", [_n(s["max_q"], 1) for s in S]),
            ("\\quad at time", "s", [_n(s["t_max_q"], 0) for s in S]),
            ("\\quad at altitude", "km", [_n(s["h_max_q"], 2) for s in S]),
            ("Peak acceleration", "$g_0$", [_n(s["max_g"], 2) for s in S]),
            ("Insertion (transfer)", "s", [_n(s["t_insertion"], 1) for s in S]),
            ("Ideal $\\Delta v$, all burns", "m/s", [_n(s["dv_ideal"], 0) for s in S]),
            ("Gravity loss", "m/s", [_n(s["losses"]["gravity"], 0) for s in S]),
            ("Drag loss", "m/s", [_n(s["losses"]["drag"], 0) for s in S]),
            ("Back-pressure loss", "m/s", [_n(s["losses"]["pressure"], 0) for s in S]),
            ("Perigee", "km", [_n(s["perigee"], 1) for s in S]),
            ("Apogee", "km", [_n(s["apogee"], 1) for s in S]),
            ("Inclination", "deg", [_n(s["inc"], 2) for s in S]),
            ("Period", "min", [_n(s["period"], 2) for s in S]),
            ("Propellant reserve", "kg", [_n(s["propellant_left"], 0) for s in S])]
    cols = "l l " + " ".join("r" for _ in models)
    head = "Quantity & & " + " & ".join(_head_cell(m["name"]) for m in models)
    body = "\n".join(rf"{q} & {u} & " + " & ".join(v) + r" \\" for q, u, v in rows)
    return (r"\begin{table}[t]\centering\small" "\n"
            r"\caption{Summary of the ascent. Losses are integrated over the powered flight to transfer-orbit insertion; "
            r"the reserve is the Breeze-KM propellant left in the final orbit.}\label{tab:summary}" "\n"
            rf"\begin{{tabular}}{{{cols}}}\toprule" "\n" + head + r" \\ \midrule" "\n" + body + "\n"
            r"\bottomrule\end{tabular}\end{table}")


def _grouped(models, first, cols, rows, caption, label) -> str:
    k = len(cols)
    spec = "l" + "".join(" " + "r" * k for _ in models)
    grp = " & ".join(rf"\multicolumn{{{k}}}{{c}}{{{tex(m['name'])}}}" for m in models)
    rules = "".join(rf"\cmidrule(lr){{{2 + i * k}-{1 + (i + 1) * k}}}" for i in range(len(models)))
    sub = " & ".join(cols * len(models))
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    return (r"\begin{table*}[t]\centering\small" "\n" rf"\caption{{{caption}}}\label{{{label}}}" "\n"
            rf"\begin{{tabular}}{{{spec}}}\toprule" "\n" rf" & {grp} \\ {rules}" "\n" rf"{first} & {sub} \\ \midrule" "\n"
            + body + "\n" r"\bottomrule\end{tabular}\end{table*}")


def _event_rows(models) -> list[list[str]]:
    out = []
    for e0 in models[0]["events"]:
        r = [tex(e0["label"])]
        for m in models:
            e = next((x for x in m["events"] if x["key"] == e0["key"]), None)
            r += ["--"] * 5 if e is None else [_n(e["t"], 1), _n(e["h"], 1), _n(e["v"], 3), _n(e["fpa"], 1), _n(e["m"], 2)]
        out.append(r)
    return out


def _orbit_rows(models) -> list[list[str]]:
    out = []
    for o0 in models[0]["orbits"]:
        r = [tex(o0["label"])]
        for m in models:
            o = next((x for x in m["orbits"] if x["key"] == o0["key"]), None)
            r += ["--"] * 4 if o is None else [_n(o["perigee"], 0), _n(o["apogee"], 0), _n(o["inc"], 2), _n(o["ecc"], 4)]
        out.append(r)
    return out


def _figure(subs, caption, label) -> str:
    parts = []
    for i, (name, cap) in enumerate(subs):
        parts.append(r"\begin{subfigure}{\linewidth}\centering" "\n" rf"\input{{figures/{name}.pgf}}" "\n"
                     rf"\caption{{{cap}}}\label{{{label}{chr(97 + i)}}}" "\n" r"\end{subfigure}")
    return (r"\begin{figure}[tbp]\centering" "\n" + "\n\\par\\medskip\n".join(parts) + "\n"
            rf"\caption{{{caption}}}\label{{{label}}}" "\n" r"\end{figure}")


def _pct(a, b) -> str:
    return _n(100 * (a / b - 1), 1) + r"\,\%"


def report_tex(data: dict, *, title: str, subtitle: str = "", author: str = "", abstract: str = "",
               models_text: dict | None = None, date: str = r"\today") -> str:
    """The report's LaTeX source. ``models_text``: {model name: LaTeX sentence(s) describing it}."""
    models, ref = data["models"], data["models"][0]
    S0 = ref["summary"]
    others = models[1:]
    sep_ev = {e["key"]: e for e in ref["events"]}
    fair = sep_ev.get("fairing_jettison")
    models_text = models_text or {}
    model_par = " ".join(rf"The \emph{{{tex(m['name']).lower()}}} model ({['solid', 'dashed', 'dotted', 'dash-dotted'][i]} "
                         rf"lines in every figure) {models_text.get(m['name'], '')}".rstrip() + ("" if models_text.get(m['name'], '').endswith('.') else ".")
                         for i, m in enumerate(models))
    fhf = data.get("fairing_heat_flux", 1135.0)
    site = data.get("site", {})
    stages = ", ".join(rf"{tex(s['name'])} (\qty{{{s['thrust_vac']:.1f}}}{{kN}})" for s in ref["stages"])

    cmp = []
    for m in others:
        S = m["summary"]
        cmp.append(
            rf"Against the {tex(ref['name']).lower()} model, the {tex(m['name']).lower()} model lifts off "
            rf"{_n(abs(S['liftoff_mass'] - S0['liftoff_mass']), 2)}\,t {'heavier' if S['liftoff_mass'] > S0['liftoff_mass'] else 'lighter'}, "
            rf"peaks at {_n(S['max_q'], 1)}\,kPa of dynamic pressure ({_pct(S['max_q'], S0['max_q'])}) and loses "
            rf"{_n(S['losses']['gravity'], 0)}\,m/s to gravity and {_n(S['losses']['drag'], 0)}\,m/s to drag, "
            rf"against {_n(S0['losses']['gravity'], 0)} and {_n(S0['losses']['drag'], 0)}\,m/s. "
            rf"It reaches a {_n(S['perigee'], 0)}\,$\times$\,{_n(S['apogee'], 0)}\,km orbit with "
            rf"{_n(S['propellant_left'], 0)}\,kg of Breeze-KM propellant in reserve, against "
            rf"{_n(S0['perigee'], 0)}\,$\times$\,{_n(S0['apogee'], 0)}\,km and {_n(S0['propellant_left'], 0)}\,kg.")
    fair_txt = (rf"The fairing is released at {_n(fair['t'], 1)}\,s and {_n(fair['h'], 1)}\,km, once the "
                rf"free-molecular heat flux $\tfrac12\rho v^3$ falls below \qty{{{fhf:.0f}}}{{W/m^2}} "
                r"(Figure~\ref{fig:aerob}). ") if fair else ""
    ins = sep_ev.get(next((k for k in sep_ev if k.endswith("burn1_end")), ""), None)
    ins_txt = (rf"Breeze-KM cuts off at {_n(ins['t'], 1)}\,s, {_n(ins['dr'], 0)}\,km down range, on a transfer "
               rf"orbit whose apogee sits at the target altitude; ") if ins else ""
    final = [o for o in ref["orbits"] if o["key"] == "final"]
    final_txt = (rf"the second burn raises the perigee to {_n(final[0]['perigee'], 0)}\,km, leaving an orbit of "
                 rf"eccentricity {_n(final[0]['ecc'], 4)} at {_n(final[0]['inc'], 2)}$^\circ$ "
                 rf"(Table~\ref{{tab:orbits}}).") if final else "."

    head = rf"""\documentclass[10pt,a4paper,twocolumn]{{article}}
\usepackage[T1]{{fontenc}}
\usepackage{{lmodern}}
\usepackage[a4paper,margin=18mm,columnsep=7mm]{{geometry}}
\usepackage{{amsmath,graphicx,booktabs,pgf}}
\providecommand{{\mathdefault}}[1]{{#1}}% used by matplotlib's .pgf figures
\usepackage{{siunitx}}
\sisetup{{group-minimum-digits=4,group-digits=integer,group-separator={{\,}},detect-all}}
\usepackage{{caption,subcaption}}
\captionsetup{{font=small,labelfont=bf,labelsep=colon}}
\captionsetup[sub]{{font=small,labelfont=bf,labelformat=parens,labelsep=space}}
\captionsetup[table]{{position=top,skip=4pt}}
\usepackage{{microtype}}
\usepackage[hidelinks]{{hyperref}}
\setlength{{\textfloatsep}}{{12pt plus 2pt minus 2pt}}
\setlength{{\tabcolsep}}{{4.5pt}}
\begin{{document}}
\twocolumn[{{\centering
{{\Large\bfseries {tex(title)}\par}}\vspace{{9pt}}
{{\large {tex(subtitle)}\par}}\vspace{{4pt}}
{{\small {tex(author) + (r" \textperiodcentered{} " if author else "")}{date}\par}}\vspace{{12pt}}
{(r"{\small\begin{minipage}{0.84\textwidth}\textbf{Abstract.} " + abstract + r"\end{minipage}\par}\vspace{16pt}") if abstract else ""}
}}]
"""
    body = rf"""
\section{{Introduction}}
This report documents a simulated ascent from {tex(data.get('launch_site', 'the launch site'))}
({_n(site.get('lat', 0), 1)}$^\circ$\,N, {_n(site.get('lon', 0), 1)}$^\circ$\,E) to {tex(data.get('target', 'orbit'))}.
The vehicle has three stages: {stages}, vacuum thrust, carrying a {_n(S0['payload'], 2)}\,t payload.
{model_par}
Every number and figure below is generated from the simulation output by \texttt{{aerosim.plots.ascent\_report}};
the same data drive the interactive dashboard.

\section{{Method}}
The vehicle is a point mass flown in an Earth-centred inertial frame over a rotating, oblate Earth,
\begin{{equation}}
\ddot{{\mathbf r}} = \mathbf g_{{J_2}}(\mathbf r) + \frac{{F\,\hat{{\mathbf u}} - D\,\hat{{\mathbf v}}_\mathrm{{rel}}}}{{m}},
\qquad \dot m = -\dot m_\mathrm{{stage}},
\end{{equation}}
with $J_2$ gravity, drag $D = \tfrac12\rho v^2 C_D(M) A$ on the Earth-relative velocity in a co-rotating
US Standard Atmosphere, and thrust corrected for nozzle back-pressure,
\begin{{equation}}
F = F_\mathrm{{vac}} - p_a A_e .
\end{{equation}}
Mass flow is constant per stage and staging is instantaneous. The flight programme is a vertical rise,
a pitch kick and a gravity turn for stage~1, a linear pitch programme for stage~2, and a closed-loop
climb-rate law for the first Breeze-KM burn, which ends when the apogee reaches the target altitude;
a second burn at apogee circularises the orbit. Velocity losses are integrated along the trajectory as
\begin{{equation}}
\begin{{gathered}}
\Delta v_D = \int \frac{{D}}{{m}}\,\mathrm dt,\qquad
\Delta v_p = \int \frac{{p_a A_e}}{{m}}\,\mathrm dt,\\
\Delta v_g = \int g \sin\gamma \,\mathrm dt,
\end{{gathered}}
\end{{equation}}
the gravity loss counted while the engines burn.

\section{{Results}}
\subsection{{Trajectory}}
{_summary_table(models)}
Table~\ref{{tab:summary}} collects the main figures of both flights. The vehicle passes Mach~1 at
{_n(S0['t_mach1'], 1)}\,s and maximum dynamic pressure, {_n(S0['max_q'], 1)}\,kPa, at {_n(S0['t_max_q'], 0)}\,s
and {_n(S0['h_max_q'], 1)}\,km (Figure~\ref{{fig:atm}}). Acceleration peaks at {_n(S0['max_g'], 2)}\,$g_0$
just before stage~1 burns out (Figure~\ref{{fig:loadsa}}). {fair_txt}{ins_txt}{final_txt}

{_figure([("altitude", "Altitude to transfer-orbit insertion."), ("trajectory3d", "Trajectory over the ground (azimuthal equidistant about the pad), with its projection.")],
         "Ascent from lift-off to Breeze-KM cut-off. Dotted verticals mark the stage cut-offs of the first model.", "fig:traj")}
{_figure([("speed", "Earth-relative speed."), ("fpa", "Flight-path angle above the local horizon.")],
         "Velocity through the powered ascent.", "fig:vel")}

\subsection{{Atmospheric flight}}
Drag and dynamic pressure are confined to the first 150\,s (Figures~\ref{{fig:atm}} and \ref{{fig:aero}}).
Drag climbs steeply through the transonic rise in $C_D$ and peaks near maximum dynamic pressure.

{_figure([("q", "Dynamic pressure."), ("mach", "Mach number.")], "The first stage through the dense atmosphere.", "fig:atm")}
{_figure([("drag", "Aerodynamic drag."), ("heat", "Free-molecular heat flux after stage~1.")],
         "Aerodynamic loads. The fairing goes once the heat flux falls below the limit (dashed).", "fig:aero")}

\subsection{{Propulsion and losses}}
Ideal (vacuum rocket-equation) velocity summed over all burns is {_n(S0['dv_ideal'], 0)}\,m/s. Of it,
{_n(S0['losses']['gravity'], 0)}\,m/s is lost to gravity, {_n(S0['losses']['drag'], 0)}\,m/s to drag and
{_n(S0['losses']['pressure'], 0)}\,m/s to nozzle back-pressure (Figure~\ref{{fig:loadsb}}). The gravity loss
falls back late in the Breeze-KM burn, where the flight-path angle turns negative and gravity adds speed.

{_figure([("thrust", "Thrust, logarithmic scale."), ("mass", "Vehicle mass.")],
         "Thrust, with the back-pressure loss of each stage, and mass through staging and fairing jettison.", "fig:prop")}
{_figure([("accel", "Non-gravitational acceleration."), ("losses", rf"Velocity losses, {tex(ref['name']).lower()} model.")],
         "Axial load on the payload, and velocity lost to gravity, drag and back-pressure.", "fig:loads")}

\subsection{{Orbit}}
{_grouped(models, "After", [r"$h_p$", r"$h_a$", r"$i$", r"$e$"], _orbit_rows(models),
          r"Osculating orbit after each staging event: perigee and apogee altitude $h_p$, $h_a$ (km), inclination $i$ (deg) "
          r"and eccentricity $e$. A negative perigee is a suborbital arc.", "tab:orbits")}
{_grouped(models, "Event", [r"$t$", r"$h$", r"$v$", r"$\gamma$", r"$m$"], _event_rows(models),
          r"Event timeline: time $t$ (s), altitude $h$ (km), Earth-relative speed $v$ (km/s), flight-path angle "
          r"$\gamma$ (deg) and mass $m$ (t).", "tab:events")}
Tables~\ref{{tab:orbits}} and \ref{{tab:events}} follow the orbit as it is built up stage by stage, and
Figure~\ref{{fig:ground}} the ground track over the whole flight.

{_figure([("track", "Ground track over the full flight."), ("range", "Altitude against ground range.")],
         "Ground track (Natural Earth coastlines) and the ascent profile over the ground.", "fig:ground")}

\section{{Discussion}}
{" ".join(cmp) if cmp else "A single model was flown."}
Both flights rest on the same equations; the differences above come from the vehicle data and the
flight programme each model uses.

\end{{document}}
"""
    return head + body


# ------------------------------------------------------------------ build
def build_report(data: dict, out_dir, *, name: str = "Ascent_Report", compile: bool = True, **text) -> Path:
    """Write figures and LaTeX under ``out_dir/build`` and compile to ``out_dir/<name>.pdf``.
    ``text``: keyword arguments of ``report_tex`` (title, subtitle, author, abstract, models_text)."""
    out = Path(out_dir)
    build = out / "build"
    report_figures(data, build / "figures")
    src = build / f"{name}.tex"
    src.write_text(report_tex(data, **text), encoding="utf-8", newline="\n")
    (build / "flight-data.json").write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    if not compile:
        return src
    exe = shutil.which("latexmk")
    if exe is None:
        raise AerosimError("latexmk not found: install TeX Live or MiKTeX, or call build_report(..., compile=False)")
    run = subprocess.run([exe, "-lualatex", "-interaction=nonstopmode", "-halt-on-error", "-file-line-error",
                          src.name], cwd=build, capture_output=True, text=True, encoding="utf-8", errors="replace")
    pdf = build / f"{name}.pdf"
    if run.returncode != 0 or not pdf.exists():
        tail = "\n".join(l for l in run.stdout.splitlines() if l.startswith(("!", "./")) or "Error" in l)[-2000:]
        raise AerosimError(f"LaTeX failed (see {build / (name + '.log')}):\n{tail}")
    shutil.copyfile(pdf, out / pdf.name)
    return out / pdf.name
