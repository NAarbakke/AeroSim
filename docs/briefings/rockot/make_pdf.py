"""PDF edition of the Rockot briefing (same content and numbers as the deck).

    python docs/briefings/rockot/make_briefing.py   # data.json + figures
    python docs/briefings/rockot/make_pdf.py        # -> Rockot_Ascent_Briefing.pdf
"""

from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
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
for name, file in (("Body", "calibri.ttf"), ("Body-Bold", "calibrib.ttf"),
                   ("Head", "arialbd.ttf")):
    pdfmetrics.registerFont(TTFont(name, str(FONTS / file)))
pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold")

NAVY, ICE, INK2, MUTED, GRID = (colors.HexColor(h) for h in
                                ("#14213D", "#EEF2F7", "#52514E", "#898781", "#E1E0D9"))
W = A4[0] - 36 * mm

S = {
    "title": ParagraphStyle("title", fontName="Head", fontSize=24, leading=29, textColor=colors.white),
    "sub": ParagraphStyle("sub", fontName="Body", fontSize=12.5, leading=17, textColor=ICE),
    "h1": ParagraphStyle("h1", fontName="Head", fontSize=15, leading=19, textColor=NAVY,
                         spaceBefore=10, spaceAfter=6, keepWithNext=1),
    "body": ParagraphStyle("body", fontName="Body", fontSize=10.5, leading=14.5, alignment=TA_LEFT,
                           spaceAfter=4),
    "bullet": ParagraphStyle("bullet", fontName="Body", fontSize=10.5, leading=14.5, leftIndent=12,
                             bulletIndent=2, spaceAfter=3),
    "cap": ParagraphStyle("cap", fontName="Body", fontSize=8.5, leading=11, textColor=MUTED,
                          spaceAfter=6),
    "statv": ParagraphStyle("statv", fontName="Head", fontSize=19, leading=23, textColor=NAVY),
    "statl": ParagraphStyle("statl", fontName="Body", fontSize=9.5, leading=12),
    "statn": ParagraphStyle("statn", fontName="Body", fontSize=8, leading=10, textColor=MUTED),
    "cell": ParagraphStyle("cell", fontName="Body", fontSize=9.5, leading=12, textColor=INK2),
    "cellb": ParagraphStyle("cellb", fontName="Body-Bold", fontSize=9.5, leading=12, textColor=NAVY),
}


def n0(v):
    return f"{round(v):,}"


def n1(v):
    return f"{v:,.1f}"


def bullets(items):
    return [Paragraph(t, S["bullet"], bulletText="•") for t in items]


def figure(name, width=W, caption=None):
    from PIL import Image as PILImage
    w, h = PILImage.open(FIG / name).size
    out = [Image(str(FIG / name), width=width, height=width * h / w)]
    if caption:
        out.append(Paragraph(caption, S["cap"]))
    return out


def table(rows, widths, header=True, align_right_from=None):
    data = [[Paragraph(str(c), S["cellb"] if (header and i == 0) else S["cell"]) for c in r]
            for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, hAlign="LEFT")
    style = [("GRID", (0, 0), (-1, -1), 0.5, GRID), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), ICE))
    t.setStyle(TableStyle(style))
    return t


def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Body", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 10 * mm, "Rockot / Breeze-KM ascent model  |  AeroSim  |  "
                      "data: Rockot User's Guide EHB0003 Iss. 5 (2011)")
    canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, str(doc.page))
    canvas.restoreState()


def build():
    D = json.loads((BUILD / "data.json").read_text())
    ev = D["events"]
    cmp = {r["event"]: r for r in D["comparison"]}
    s1, s2, b1 = cmp["Stage 1_burnout"], cmp["Stage 2_burnout"], cmp["Breeze-KM_burn1_end"]
    tr, fin = D["orbits"]["transfer"], D["orbits"]["final"]
    st = D["vehicle"]["stages"]
    pct = lambda m, g: f"{100 * (m / g - 1):+.1f} %"
    story = []

    # cover band
    cover = Table([[Paragraph("Rockot / Breeze-KM from Plesetsk", S["title"])],
                   [Paragraph("Ascent trajectory model of the User's Guide reference mission: "
                              "650 km circular orbit, 86.58° inclination  |  AeroSim  |  October 2026",
                              S["sub"])]], colWidths=[W])
    cover.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), NAVY),
                               ("LEFTPADDING", (0, 0), (-1, -1), 12),
                               ("TOPPADDING", (0, 0), (0, 0), 14),
                               ("BOTTOMPADDING", (0, -1), (-1, -1), 14)]))
    story += [cover, Spacer(1, 10)]

    # 1. bottom line
    story.append(Paragraph("The model flies the guide's reference mission to orbit", S["h1"]))
    q_g = D["comparison"][0]["guide"]
    stats = [
        (f"{n0(D['vehicle']['payload'])} kg", "payload to 650 km circular",
         "User's Guide fig. 3-1, i = 86.4°"),
        (f"{n1(D['max_q']['q_kpa'])} kPa", f"max dynamic pressure at {n0(D['max_q']['time'])} s",
         f"guide: {q_g['q'] / 1e3} kPa at {q_g['time']} s"),
        (f"{n0(ev['Breeze-KM_burn1_end'])} s", "lift-off to transfer orbit",
         "guide: 882 s (end of Breeze-KM burn 1)"),
    ]
    cells = [[[Paragraph(v, S["statv"]), Paragraph(lbl, S["statl"]), Paragraph(nt, S["statn"])]
              for v, lbl, nt in stats]]
    st_t = Table(cells, colWidths=[W / 3 - 2] * 3)
    st_t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), ICE),
                              ("LINEAFTER", (0, 0), (1, 0), 3, colors.white),
                              ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                              ("LEFTPADDING", (0, 0), (-1, -1), 10)]))
    story += [st_t, Spacer(1, 8)]
    story += bullets([
        (f"Stage 1 and 2 burnout speeds within {pct(s1['model']['speed'], s1['guide']['speed'])} and "
        f"{pct(s2['model']['speed'], s2['guide']['speed'])} of the guide's trajectory (fig. 3-6)."),
        (f"Breeze-KM burn 1 reaches a {n0(tr['perigee_km'])} x {n0(tr['apogee_km'])} km transfer orbit "
        f"(guide: 160 x 635 km); burn 2 circularises at {n0(fin['perigee_km'])} x "
        f"{n0(fin['apogee_km'])} km, i = {fin['inclination']:.2f}°."),
        ("Stage 1/2 propellant masses are not published; they were fitted to the guide's waypoints. "
        f"The fitted vehicle weighs {n1(D['vehicle']['liftoff_mass'] / 1000)} t at lift-off against "
        "the guide's 107 t."),
    ])

    # 2. vehicle
    story.append(Paragraph("Three hypergolic stages; the top one restarts", S["h1"]))
    rows = [["", "Stage 1", "Stage 2", "Breeze-KM"],
            ["Engines", "3 x RD-0233 + RD-0234", "RD-0235 + 4 RD-0236", "S5.98M, 20 kN"],
            ["Thrust, vacuum", *[f"{n0(x['thrust_vac_kN'])} kN" for x in st]],
            ["Isp, vacuum", *[f"{n1(x['isp_vac'])} s" for x in st]],
            ["Burn time", f"{n0(st[0]['burn_time'])} s", f"{n0(st[1]['burn_time'])} s", "up to 1,000 s"],
            ["Propellant", *[f"{n0(x['propellant'])} kg" for x in st]],
            ["Dry mass", *[f"{n0(x['dry'])} kg" for x in st]]]
    story.append(table(rows, [W * 0.22, W * 0.26, W * 0.26, W * 0.26]))
    story.append(Paragraph(
        "Isp, burn times and Breeze-KM data: User's Guide tables 2-2 to 2-8. Stage 1/2 propellant is "
        "fitted (section 7), so their thrust follows from it at the guide's burn time; the guide quotes "
        "2,070 kN and 256 kN vacuum. Dry masses: Encyclopedia Astronautica.", S["cap"]))

    # 3. site
    story += [
        Paragraph("Launch from Plesetsk, northbound into a near-polar orbit", S["h1"]),
        *bullets([
            (f"Plesetsk Cosmodrome, {D['site']['lat']}° N {D['site']['lon']}° E, about 200 km south "
            "of Archangel (guide §3.2). Allowed azimuths 4.8° (i = 86.4°) to 76.2° (i = 63.2°); "
            "341.5° for sun-synchronous orbits."),
            (f"Target plane i = {D['program']['target_inclination']}°: the model launches along the "
            "rotation-corrected azimuth and yaws out residual out-of-plane velocity."),
        ]),
        *figure("ground_track.png", W * 0.78, "AeroSim model output (aerosim.plots.ground_track)."),
    ]

    # 4. timeline
    story.append(Paragraph("About fifteen minutes from lift-off to the transfer orbit", S["h1"]))
    tl = [("Lift-off", 0, 0), ("Max q", ev["max_q"], 50), ("Stage 1 separation", ev["Stage 1_burnout"], 122),
          ("Fairing jettison", ev["fairing_jettison"], 173), ("Stage 2 separation", ev["Stage 2_burnout"], 305),
          ("End of Breeze-KM burn 1", ev["Breeze-KM_burn1_end"], 882),
          ("Breeze-KM burn 2 start", ev["Breeze-KM_burn2_start"], 4360)]
    story.append(table([["Event", "Model (s)", "Guide (s)"]] +
                       [[e, n0(m), f"{g:,}"] for e, m, g in tl], [W * 0.5, W * 0.25, W * 0.25]))
    story.append(Paragraph(
        f"Burn 2 lasts {n0(ev['Breeze-KM_burn2_end'] - ev['Breeze-KM_burn2_start'])} s in the model "
        "(guide: about 39 s); the guide deploys the spacecraft at 5,861 s.", S["cap"]))

    # 5. profile
    pa = D["peak_accel_g"]
    story.append(Paragraph("Ascent profile tracks the guide's published waypoints", S["h1"]))
    story += figure("ascent_overview.png", W, "Blue line: AeroSim model. Orange points: User's Guide "
                    "fig. 3-6 waypoints. Grey lines: max q, stage 1 separation, fairing jettison, "
                    "stage 2 separation.")
    story += bullets([
        "Max q, both stagings and the end of burn 1 sit on or near the guide's points.",
        (f"Peak axial acceleration {n1(pa['Stage 1'])} g at stage 1 burnout; guide table 5-1 gives "
        f"6.2-8.0 g. Stage 2 peaks at {n1(pa['Stage 2'])} g (guide: up to 3.0 g)."),
    ])

    # 6. comparison
    story.append(Paragraph("Model against the guide at each event", S["h1"]))
    story += figure("comparison.png", W)
    labels = {"max_q": "Max q", "Stage 1_burnout": "S1 sep", "fairing_jettison": "Fairing",
              "Stage 2_burnout": "S2 sep", "Breeze-KM_burn1_end": "Burn 1 end"}
    rows = [["Event", "Time model / guide (s)", "Altitude (km)", "Speed (m/s)", "Flight-path angle (deg)"]]
    for r in D["comparison"]:
        m, g = r["model"], r["guide"]
        rows.append([labels[r["event"]], f"{n0(m['time'])} / {n0(g['time'])}",
                     f"{n1(m['altitude'] / 1e3)} / {n1(g['altitude'] / 1e3)}",
                     f"{n0(m['speed'])} / {n0(g['speed'])}", f"{n1(m['fpa'])} / {n1(g['fpa'])}"])
    story.append(table(rows, [W * 0.16, W * 0.21, W * 0.21, W * 0.21, W * 0.21]))

    # 7. insertion
    story.append(Paragraph("Breeze-KM burns twice to reach the 650 km class orbit", S["h1"]))
    story += figure("trajectory.png", W)
    story.append(table([["", "Perigee", "Apogee", "Inclination"],
                        ["Transfer orbit, model", f"{n0(tr['perigee_km'])} km", f"{n0(tr['apogee_km'])} km",
                         f"{tr['inclination']:.2f}°"],
                        ["Transfer orbit, guide", "160 km", "635 km", "86.58°"],
                        ["Final orbit, model", f"{n0(fin['perigee_km'])} km", f"{n0(fin['apogee_km'])} km",
                         f"{fin['inclination']:.2f}°"],
                        ["Deployment orbit, guide", "650 km", "650 km", "86.58°"]],
                       [W * 0.4, W * 0.2, W * 0.2, W * 0.2]))
    story.append(Spacer(1, 6))
    story += bullets([
        ("Burn 1 is closed-loop: pitch holds a commanded climb rate while the 20 kN engine builds "
        "speed. Burn 2 fires at apogee, prograde, and stops when perigee reaches the transfer apogee."),
        (f"Burn 1 runs {n0(ev['Breeze-KM_burn1_end'] - ev['Breeze-KM_burn1_start'])} s against the "
        f"guide's 577 s and ends lower ({n0(b1['model']['altitude'] / 1e3)} km vs 251 km): the model's "
        f"steering is less efficient. Breeze-KM propellant left: {n0(D['propellant_left'])} kg of "
        f"{n0(st[2]['propellant'])} kg."),
        ("The guide's own figure is not fully consistent: its burn-2 altitude reads 665.6 km, its "
        "deployment orbit 650 km, and its final-orbit box (499 x 506 km, 88.999°) appears carried over "
        "from another mission. The model targets the 635.3 km transfer apogee and circularises there."),
    ])

    # 8. calibration
    story.append(Paragraph("Two masses and five steering settings were fitted", S["h1"]))
    cal = D.get("calibration") or {"stage_propellant": [x["propellant"] for x in st], "residuals": []}
    story.append(table([
        ["Quantity", "Source value", "Used in model", "Basis"],
        ["Stage 1 propellant", "71,455 kg (Astronautix)", f"{n0(cal['stage_propellant'][0])} kg", "fitted"],
        ["Stage 2 propellant", "10,710 kg (Astronautix)", f"{n0(cal['stage_propellant'][1])} kg", "fitted"],
        ["Breeze-KM propellant / dry", "4,975 / 1,320 kg", "4,975 / 1,320 kg", "User's Guide t. 2-8"],
        ["Payload", "1,570 kg", f"{n0(D['vehicle']['payload'])} kg", "User's Guide fig. 3-1"],
        ["Fairing", "not published", f"{n0(D['vehicle']['fairing'])} kg", "estimate"],
        ["Lift-off mass", "107 t", f"{n1(D['vehicle']['liftoff_mass'] / 1000)} t", "result"],
    ], [W * 0.28, W * 0.28, W * 0.2, W * 0.24]))
    story.append(Spacer(1, 6))
    story += bullets([
        ("The guide gives engines, burn times, Breeze-KM masses and lift-off mass, but not stage 1/2 "
        "propellant. Published stage masses leave stage 1 about 300 m/s slow at burnout against the "
        "guide's own trajectory."),
        ("A least-squares fit adjusts both propellant loads plus pitch kick, stage 2 pitch and "
        "Breeze-KM climb rate to ten guide targets. Mass flow is propellant / burn time at constant "
        "thrust, so thrust scales with the fitted load."),
    ])
    if cal["residuals"]:
        met = sum(abs(r) <= 1.5 for r in cal["residuals"])
        story.append(Paragraph(
            f"<b>{met} of {len(cal['residuals'])} targets met</b> within 1.5 tolerance units (25 m/s "
            "speed, 2-4 km altitude, 1° angle, 3 km perigee). Missed: Breeze-KM burn 1 ends "
            f"{n0(ev['Breeze-KM_burn1_end'] - 882)} s later than the guide, and the fitted vehicle is "
            f"{n1(D['vehicle']['liftoff_mass'] / 1000 - 107)} t heavier at lift-off. Both point the same "
            "way: the model needs more energy than the real vehicle to fly the same trajectory, most "
            "likely from its simple insertion steering and generic drag.", S["body"]))

    # 9. limits + next steps, sources
    story.append(Paragraph("Limits and next steps", S["h1"]))
    lim = table([["What the model leaves out", "Next steps"],
                 ["3-DOF point mass: no attitude dynamics or control-loop limits",
                  "Stage drop zones and impact points (guide: 990-1,100 km for stage 1)"],
                 ["Generic slender-body drag curve, not Rockot wind-tunnel data",
                  "Payload-vs-orbit sweep to reproduce guide fig. 3-1"],
                 ["Constant thrust per stage; no throttling, tail-off or vernier phases",
                  "SSO mission: 341.5° azimuth, parking orbit and plane change"],
                 ["Gravity to J2 only; exponential upper atmosphere",
                  "Monte Carlo on thrust, Isp and winds for dispersion"]], [W / 2, W / 2])
    story.append(lim)
    story.append(Paragraph("Sources and reproduction", S["h1"]))
    story += bullets([
        ("Eurockot Launch Services, <i>Rockot User's Guide</i>, EHB0003, Issue 5, Revision 0, August "
        "2011: tables 2-1 to 2-8 and 5-1, section 3, figures 3-1 and 3-6. Copy in docs/references/rockot/."),
        "Encyclopedia Astronautica, Rokot (astronautix.com/r/rokot.html): stage gross and empty masses.",
        ("D. Vallado, <i>Fundamentals of Astrodynamics and Applications</i>, table 8-4: upper-atmosphere "
        "density."),
        ("Model: configs/examples/rocket/orbital/rockot.yaml; calibration: examples/rockot_ascent.py; "
        "this briefing: docs/briefings/rockot/make_briefing.py then make_pdf.py and build/deck.js."),
    ])

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=18 * mm,
                            title="Rockot / Breeze-KM ascent from Plesetsk", author="AeroSim")
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    print("wrote", OUT)


if __name__ == "__main__":
    build()
