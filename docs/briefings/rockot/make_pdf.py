"""Technical-note PDF of the Rockot ascent model (performance figures only).

    python docs/briefings/rockot/make_briefing.py   # data.json + figures
    python docs/briefings/rockot/make_pdf.py        # -> Rockot_Ascent_Briefing.pdf
"""

from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

HERE = Path(__file__).resolve().parent
BUILD = HERE / "build"
FIG = BUILD / "figures"
OUT = HERE / "Rockot_Ascent_Briefing.pdf"

FONTS = Path("C:/Windows/Fonts")
pdfmetrics.registerFont(TTFont("Serif", str(FONTS / "cambria.ttc"), subfontIndex=0))
for name, file in (("Serif-Bold", "cambriab.ttf"), ("Serif-Italic", "cambriai.ttf"),
                   ("Serif-BoldItalic", "cambriaz.ttf")):
    pdfmetrics.registerFont(TTFont(name, str(FONTS / file)))
pdfmetrics.registerFontFamily("Serif", normal="Serif", bold="Serif-Bold", italic="Serif-Italic",
                              boldItalic="Serif-BoldItalic")

W = A4[0] - 50 * mm
INK = colors.black


def style(name, **kw):
    base = {"fontName": "Serif", "fontSize": 10, "leading": 13.5, "textColor": INK}
    return ParagraphStyle(name, **{**base, **kw})


S = {
    "title": style("title", fontName="Serif-Bold", fontSize=15, leading=19, alignment=TA_CENTER),
    "byline": style("byline", fontSize=10, alignment=TA_CENTER, spaceBefore=4, spaceAfter=14),
    "abs": style("abs", fontSize=9.5, leading=12.5, alignment=TA_JUSTIFY, leftIndent=24, rightIndent=24),
    "h1": style("h1", fontName="Serif-Bold", fontSize=11, spaceBefore=12, spaceAfter=4, keepWithNext=1),
    "body": style("body", alignment=TA_JUSTIFY, spaceAfter=6),
    "cap": style("cap", fontSize=9, leading=11.5, alignment=TA_JUSTIFY, spaceBefore=3, spaceAfter=8),
    "cell": style("cell", fontSize=9, leading=11),
    "cellr": style("cellr", fontSize=9, leading=11, alignment=2),
    "cellc": style("cellc", fontSize=9, leading=11, alignment=TA_CENTER),
    "ref": style("ref", fontSize=9, leading=11.5, leftIndent=16, firstLineIndent=-16, spaceAfter=2),
}


def n0(v):
    return f"{round(v):,}"


def n1(v):
    return f"{v:,.1f}".replace("-", "−")


def table(rows, widths, caption, head=1, spans=()):
    """Three-rule (booktabs) table, caption above; first column left, the rest right.
    ``spans`` = column ranges (c0, c1) of the first header row merged and centred."""
    def cell(i, j, c):
        if i == 0 and any(a <= j <= b for a, b in spans):
            return Paragraph(str(c), S["cellc"])
        return Paragraph(str(c), S["cell"] if j == 0 else S["cellr"])
    data = [[cell(i, j, c) for j, c in enumerate(r)] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, hAlign="CENTER")
    t.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 1.0, INK), ("LINEBELOW", (0, head - 1), (-1, head - 1), 0.5, INK),
        ("LINEBELOW", (0, -1), (-1, -1), 1.0, INK), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        *[("SPAN", (a, 0), (b, 0)) for a, b in spans],
        *[("LINEBELOW", (a, 0), (b, 0), 0.5, INK) for a, b in spans]]))
    return KeepTogether([Paragraph(caption, S["cap"]), t, Spacer(1, 8)])


def figure(name, caption, width=W, heading=None):
    from PIL import Image as PILImage
    w, h = PILImage.open(FIG / name).size
    return KeepTogether([*([heading] if heading else []),
                         Image(str(FIG / name), width=width, height=width * h / w),
                         Paragraph(caption, S["cap"])])


def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Serif", 9)
    canvas.drawCentredString(A4[0] / 2, 12 * mm, str(doc.page))
    canvas.restoreState()


def build():
    D = json.loads((BUILD / "data.json").read_text(encoding="utf-8"))
    ev, st, v = D["events"], D["vehicle"]["stages"], D["vehicle"]
    cmp = {r["event"]: r for r in D["comparison"]}
    tr, fin = D["orbits"]["transfer"], D["orbits"]["final"]
    pa = D["peak_accel_g"]
    dv, tw = v["ideal_dv"], v["liftoff_tw"]
    s1, s2 = cmp["Stage 1_burnout"], cmp["Stage 2_burnout"]
    err = lambda r: 100 * (r["model"]["speed"] / r["guide"]["speed"] - 1)
    story = []

    story += [
        Paragraph("Ascent Performance of the Rockot/Breeze-KM Launch Vehicle from Plesetsk", S["title"]),
        Paragraph("AeroSim technical note &nbsp;·&nbsp; October 2026", S["byline"]),
        Paragraph(
            "<b>Abstract.</b> A three-degree-of-freedom ascent model of Rockot/Breeze-KM is calibrated "
            f"against the reference mission of the User's Guide [1]: {n0(v['payload'])} kg to a 650 km "
            f"circular orbit at 86.58°. Stage-1 and stage-2 burnout speeds agree within "
            f"{abs(err(s1)):.1f} % and {abs(err(s2)):.1f} %. Maximum dynamic pressure is "
            f"{n1(D['max_q']['q_kpa'])} kPa and peak axial load {n1(pa['Stage 1'])} g. The upper stage "
            f"inserts into a {n0(fin['perigee_km'])} × {n0(fin['apogee_km'])} km orbit at "
            f"{fin['inclination']:.2f}° with {n0(D['propellant_left'])} kg of propellant remaining.",
            S["abs"]),
    ]

    # 1. vehicle
    story += [Paragraph("1&nbsp;&nbsp;Vehicle", S["h1"]), Paragraph(
        "Engine, specific impulse and burn-time data are from [1], dry masses from [2]. Stage-1 and "
        "stage-2 propellant loads are fitted (Section 4); thrust follows from load, burn time and "
        f"specific impulse. Lift-off mass is {n1(v['liftoff_mass'] / 1e3)} t and lift-off "
        f"thrust-to-weight ratio {tw:.2f}.", S["body"])]
    dash = "–"
    story.append(table([
        ["", "Stage 1", "Stage 2", "Breeze-KM"],
        ["Engines", "RD-0233 ×3, RD-0234", "RD-0235, RD-0236 ×4", "S5.98M"],
        ["Thrust, vacuum (kN)", *[n0(s["thrust_vac_kN"]) for s in st]],
        ["<i>I</i><sub>sp</sub>, vacuum (s)", *[n1(s["isp_vac"]) for s in st]],
        ["<i>I</i><sub>sp</sub>, sea level (s)", *[n1(s["isp_sl"]) if s["isp_sl"] else dash for s in st]],
        ["Burn time (s)", n0(st[0]["burn_time"]), n0(st[1]["burn_time"]), f"≤ {n0(st[2]['burn_time'])}"],
        ["Propellant (kg)", *[n0(s["propellant"]) for s in st]],
        ["Dry mass (kg)", *[n0(s["dry"]) for s in st]],
        ["Ideal Δ<i>v</i>, vacuum (m/s)", *[n0(x) for x in dv]],
    ], [W * 0.31, W * 0.23, W * 0.23, W * 0.23],
        f"<b>Table 1.</b> Stage parameters. Ideal Δ<i>v</i> totals {n0(sum(dv))} m/s "
        "(fairing carried to stage-2 burnout)."))

    # 2. performance
    q_g = cmp["max_q"]["guide"]
    story += [Paragraph("2&nbsp;&nbsp;Mission performance", S["h1"]), table([
        ["", "Model", "User's Guide [1]"],
        ["Payload (kg)", n0(v["payload"]), "1,570"],
        ["Lift-off mass (t)", n1(v["liftoff_mass"] / 1e3), "107"],
        ["Max. dynamic pressure (kPa)", f"{n1(D['max_q']['q_kpa'])} at {n0(D['max_q']['time'])} s",
         f"{n1(q_g['q'] / 1e3)} at {q_g['time']} s"],
        ["Peak axial load, stage 1 (g)", n1(pa["Stage 1"]), "6.2–8.0"],
        ["Peak axial load, stage 2 (g)", n1(pa["Stage 2"]), "≤ 3.0"],
        ["Peak axial load, Breeze-KM (g)", f"{max(pa['Breeze-KM burn 1'], pa['Breeze-KM burn 2']):.2f}", dash],
        ["Transfer orbit (km)", f"{n0(tr['perigee_km'])} × {n0(tr['apogee_km'])}", "160 × 635"],
        ["Final orbit (km)", f"{n0(fin['perigee_km'])} × {n0(fin['apogee_km'])}", "650 × 650"],
        ["Inclination (deg)", f"{fin['inclination']:.2f}", "86.58"],
        ["Lift-off to transfer orbit (s)", n0(ev["Breeze-KM_burn1_end"]), "882"],
        ["Breeze-KM propellant remaining (kg)", n0(D["propellant_left"]), dash],
    ], [W * 0.46, W * 0.27, W * 0.27], "<b>Table 2.</b> Reference mission, Plesetsk "
        f"({D['site']['lat']}° N, {D['site']['lon']}° E).")]

    # 3. trajectory
    story += [figure(
        "report_ascent.png", "<b>Figure 1.</b> Ascent to the end of Breeze-KM burn 1. Line: model; "
        "points: waypoints of [1, fig. 3-6]. Vertical lines mark max. <i>q</i>, stage-1 separation, "
        "fairing jettison and stage-2 separation.",
        heading=Paragraph("3&nbsp;&nbsp;Trajectory", S["h1"]))]
    labels = {"max_q": "Max. <i>q</i>", "Stage 1_burnout": "Stage-1 sep.",
              "fairing_jettison": "Fairing jettison", "Stage 2_burnout": "Stage-2 sep.",
              "Breeze-KM_burn1_end": "Burn-1 cutoff"}
    rows = [["Event", "<i>t</i> (s)", "", "<i>h</i> (km)", "", "<i>v</i> (m/s)", "", "γ (deg)", ""],
            ["", *["Model", "Guide"] * 4]]
    for r in D["comparison"]:
        m, g = r["model"], r["guide"]
        rows.append([labels[r["event"]], n0(m["time"]), n0(g["time"]), n1(m["altitude"] / 1e3),
                     n1(g["altitude"] / 1e3), n0(m["speed"]), n0(g["speed"]), n1(m["fpa"]), n1(g["fpa"])])
    story.append(table(rows, [W * 0.2] + [W * 0.1] * 8,
                       "<b>Table 3.</b> State at flight events: model against [1, fig. 3-6]. "
                       "<i>v</i>: Earth-relative speed; γ: flight-path angle.", head=2,
                       spans=((1, 2), (3, 4), (5, 6), (7, 8))))

    # 4. calibration and limitations
    story += [Paragraph("4&nbsp;&nbsp;Calibration and limitations", S["h1"]), Paragraph(
        "Stage-1 and stage-2 propellant loads, pitch-kick angle, stage-2 pitch rate and the Breeze-KM "
        "climb-rate law were fitted by least squares to ten waypoints of [1]; eight are met within "
        "tolerance. The two misses, lift-off mass "
        f"(+{100 * (v['liftoff_mass'] / 107e3 - 1):.1f} %) and burn-1 cutoff "
        f"(+{n0(ev['Breeze-KM_burn1_end'] - 882)} s), indicate that the model needs more energy than "
        "the flight vehicle, consistent with its simplified insertion steering and generic drag curve. "
        "The model is a point mass with constant thrust per stage, J<sub>2</sub> gravity and an "
        "exponential upper atmosphere [3].", S["body"])]

    story += [Paragraph("References", S["h1"]), *[Paragraph(t, S["ref"]) for t in (
        "[1] Eurockot Launch Services, <i>Rockot User's Guide</i>, EHB0003, Issue 5, Rev. 0, 2011.",
        "[2] M. Wade, “Rokot,” <i>Encyclopedia Astronautica</i>, astronautix.com.",
        "[3] D. A. Vallado, <i>Fundamentals of Astrodynamics and Applications</i>, table 8-4.",
    )]]

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=25 * mm, rightMargin=25 * mm,
                            topMargin=22 * mm, bottomMargin=22 * mm,
                            title="Ascent Performance of the Rockot/Breeze-KM Launch Vehicle",
                            author="AeroSim")
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    print("wrote", OUT)


if __name__ == "__main__":
    build()
