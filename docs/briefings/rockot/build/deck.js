// Rockot / Breeze-KM ascent performance: slide edition of the technical note (make_pdf.py).
// Reads data.json + figures/ written by make_briefing.py.
//   NODE_PATH=./node_modules node deck.js   ->  ../Rockot_Ascent_Briefing.pptx
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require("./apply_theme.js");

const D = JSON.parse(fs.readFileSync(path.join(__dirname, "data.json"), "utf8"));
const FIG = (f) => path.join(__dirname, "figures", f);
const OUT = path.join(__dirname, "..", "Rockot_Ascent_Briefing.pptx");

const THEME = {
  name: "AeroSim Note",
  headFontFace: "Cambria",
  bodyFontFace: "Cambria",
  colors: {
    dk1: "000000", lt1: "FFFFFF", dk2: "1A1A1A", lt2: "F2F2F2",
    accent1: "2A78D6", accent2: "EB6834", accent3: "1BAF7A", accent4: "EDA100",
    accent5: "6250D6", accent6: "E34948", hlink: "2A78D6", folHlink: "6250D6",
  },
};
const INK = "000000", MUTED = "595959";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";                       // 13.33 x 7.5 in
pres.title = "Ascent Performance of the Rockot/Breeze-KM Launch Vehicle";
pres.author = "AeroSim";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

// ---------- layouts ----------
const pageNo = { x: 12.3, y: 7.0, w: 0.5, h: 0.3, fontSize: 11, color: MUTED, align: "right" };
pres.defineSlideMaster({
  title: "TITLE", background: { color: C.background1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 1.2, y: 1.6, w: 10.9, h: 1.6,
        fontSize: 36, bold: true, color: C.text1, align: "center", valign: "bottom", margin: 0 },
      text: "Title" } },
    { placeholder: { options: { name: "body", type: "body", x: 1.2, y: 3.35, w: 10.9, h: 0.5,
        fontSize: 18, color: C.text1, align: "center", valign: "top", margin: 0 }, text: "Byline" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT", background: { color: C.background1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 0.45, w: 11.7, h: 0.7,
        fontSize: 28, bold: true, color: C.text1, align: "left", valign: "middle", margin: 0 }, text: "Title" } },
  ],
  slideNumber: pageNo,
});

// ---------- helpers ----------
const n0 = (v) => Math.round(v).toLocaleString("en-US");
const n1 = (v) => (Math.round(v * 10) / 10).toFixed(1).replace("-", "−");
const ev = D.events, v = D.vehicle, st = v.stages, pa = D.peak_accel_g;
const cmp = Object.fromEntries(D.comparison.map((r) => [r.event, r]));
const tr = D.orbits.transfer, fin = D.orbits.final;
const DASH = "–";

function text(slide, t, o) {
  slide.addText(t, { isTextBox: true, margin: 0, fontSize: 18, color: C.text1, valign: "top", ...o });
}
function caption(slide, t, o) {
  text(slide, t, { fontSize: 14, color: MUTED, ...o });
}

// Three-rule (booktabs) table: rule above the header, below it, and below the last row.
// Cells may be strings or { text, options } (e.g. colspan); first column left, the rest right.
function table(slide, rows, o) {
  const head = o.head || 1, last = rows.length - 1;
  const rule = (pt) => ({ type: "solid", pt, color: INK });
  const none = { type: "none" };
  const data = rows.map((r, i) => r.map((c, j) => {
    const cell = typeof c === "string" ? { text: c, options: {} } : c;
    const top = i === 0 ? rule(1.25) : none;
    const bottom = i === head - 1 ? rule(0.75) : i === last ? rule(1.25) : none;
    return { text: cell.text, options: { align: j === 0 ? "left" : "right", color: INK,
      border: [top, none, bottom, none], ...cell.options } };
  }));
  slide.addTable(data, { fontFace: "Cambria", fontSize: 16, margin: [0.03, 0.08, 0.03, 0.08], valign: "middle",
    ...o, head: undefined });
}

// ---------- 1. title + abstract ----------
pres.addSection({ title: "Overview" });
let s = pres.addSlide({ masterName: "TITLE", sectionTitle: "Overview" });
s.addText("Ascent Performance of the Rockot/Breeze-KM Launch Vehicle from Plesetsk", { placeholder: "title" });
s.addText("AeroSim technical note  ·  October 2026", { placeholder: "body" });
const err = (r) => Math.abs(100 * (r.model.speed / r.guide.speed - 1)).toFixed(1);
text(s, [
  { text: "Abstract. ", options: { bold: true } },
  { text: `A three-degree-of-freedom ascent model is calibrated against the User's Guide reference mission [1]: ` +
    `${n0(v.payload)} kg to a 650 km circular orbit at 86.58°. Stage-1 and stage-2 burnout speeds agree within ` +
    `${err(cmp["Stage 1_burnout"])} % and ${err(cmp["Stage 2_burnout"])} %. Maximum dynamic pressure is ` +
    `${n1(D.max_q.q_kpa)} kPa and peak axial load ${n1(pa["Stage 1"])} g. The upper stage inserts into a ` +
    `${n0(fin.perigee_km)} × ${n0(fin.apogee_km)} km orbit at ${fin.inclination.toFixed(2)}° with ` +
    `${n0(D.propellant_left)} kg of propellant remaining.` },
], { x: 2.2, y: 4.4, w: 8.9, h: 2.0, fontSize: 16, align: "justify" });

// ---------- 2. stage parameters ----------
pres.addSection({ title: "Results" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.addText("Stage parameters", { placeholder: "title" });
const dv = v.ideal_dv;
table(s, [
  ["", "Stage 1", "Stage 2", "Breeze-KM"],
  ["Engines", "RD-0233 ×3, RD-0234", "RD-0235, RD-0236 ×4", "S5.98M"],
  ["Thrust, vacuum (kN)", ...st.map((x) => n0(x.thrust_vac_kN))],
  ["Isp, vacuum (s)", ...st.map((x) => n1(x.isp_vac))],
  ["Isp, sea level (s)", ...st.map((x) => (x.isp_sl ? n1(x.isp_sl) : DASH))],
  ["Burn time (s)", n0(st[0].burn_time), n0(st[1].burn_time), `≤ ${n0(st[2].burn_time)}`],
  ["Propellant (kg)", ...st.map((x) => n0(x.propellant))],
  ["Dry mass (kg)", ...st.map((x) => n0(x.dry))],
  ["Ideal Δv, vacuum (m/s)", ...dv.map(n0)],
], { x: 0.8, y: 1.55, w: 11.7, colW: [3.6, 2.7, 2.7, 2.7], rowH: 0.46 });
caption(s, `Ideal Δv totals ${n0(dv.reduce((a, b) => a + b, 0))} m/s (fairing carried to stage-2 burnout). ` +
  `Lift-off mass ${n1(v.liftoff_mass / 1000)} t; thrust-to-weight ${v.liftoff_tw.toFixed(2)}. ` +
  "Engine and Isp data [1]; dry masses [2]; stage-1/2 propellant fitted (slide 6).",
  { x: 0.8, y: 5.95, w: 11.7, h: 0.7 });

// ---------- 3. mission performance ----------
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.addText("Mission performance", { placeholder: "title" });
const qg = cmp.max_q.guide;
table(s, [
  ["", "Model", "User's Guide [1]"],
  ["Payload (kg)", n0(v.payload), "1,570"],
  ["Lift-off mass (t)", n1(v.liftoff_mass / 1000), "107"],
  ["Max. dynamic pressure (kPa)", `${n1(D.max_q.q_kpa)} at ${n0(D.max_q.time)} s`, `${n1(qg.q / 1000)} at ${qg.time} s`],
  ["Peak axial load, stage 1 (g)", n1(pa["Stage 1"]), "6.2–8.0"],
  ["Peak axial load, stage 2 (g)", n1(pa["Stage 2"]), "≤ 3.0"],
  ["Peak axial load, Breeze-KM (g)", Math.max(pa["Breeze-KM burn 1"], pa["Breeze-KM burn 2"]).toFixed(2), DASH],
  ["Transfer orbit (km)", `${n0(tr.perigee_km)} × ${n0(tr.apogee_km)}`, "160 × 635"],
  ["Final orbit (km)", `${n0(fin.perigee_km)} × ${n0(fin.apogee_km)}`, "650 × 650"],
  ["Inclination (deg)", fin.inclination.toFixed(2), "86.58"],
  ["Lift-off to transfer orbit (s)", n0(ev["Breeze-KM_burn1_end"]), "882"],
  ["Breeze-KM propellant remaining (kg)", n0(D.propellant_left), DASH],
], { x: 0.8, y: 1.45, w: 8.9, colW: [4.3, 2.3, 2.3], rowH: 0.4 });
caption(s, `Reference mission from Plesetsk (${D.site.lat}° N, ${D.site.lon}° E).`,
  { x: 0.8, y: 6.4, w: 8.9, h: 0.4 });

// ---------- 4. trajectory figure ----------
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.addText("Ascent trajectory", { placeholder: "title" });
s.addImage({ path: FIG("report_ascent.png"), x: 1.95, y: 1.3, w: 9.42, h: 5.3,
  altText: "Altitude, speed, flight-path angle, dynamic pressure, acceleration and mass versus time: " +
    "model line with User's Guide waypoints" });
caption(s, "Line: model; points: waypoints of [1, fig. 3-6]. Vertical lines: max. q, stage-1 separation, " +
  "fairing jettison, stage-2 separation.", { x: 1.95, y: 6.65, w: 9.42, h: 0.35, fontSize: 13 });

// ---------- 5. state at events ----------
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.addText("State at flight events", { placeholder: "title" });
const labels = { max_q: "Max. q", "Stage 1_burnout": "Stage-1 sep.", fairing_jettison: "Fairing jettison",
  "Stage 2_burnout": "Stage-2 sep.", "Breeze-KM_burn1_end": "Burn-1 cutoff" };
const grp = (t) => ({ text: t, options: { colspan: 2, align: "center" } });
const rows = [
  ["", grp("t (s)"), grp("h (km)"), grp("v (m/s)"), grp("γ (deg)")],
  ["Event", ...Array(4).fill(["Model", "Guide"]).flat()],
  ...D.comparison.map((r) => {
    const m = r.model, g = r.guide;
    return [labels[r.event], n0(m.time), n0(g.time), n1(m.altitude / 1000), n1(g.altitude / 1000),
      n0(m.speed), n0(g.speed), n1(m.fpa), n1(g.fpa)];
  }),
];
table(s, rows, { x: 0.8, y: 1.6, w: 11.7, colW: [2.5, ...Array(8).fill(1.15)], rowH: 0.48, head: 2 });
caption(s, "Model against [1, fig. 3-6]. v: Earth-relative speed; γ: flight-path angle.",
  { x: 0.8, y: 5.25, w: 11.7, h: 0.4 });

// ---------- 6. calibration, limitations, references ----------
pres.addSection({ title: "Method" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
s.addText("Calibration and limitations", { placeholder: "title" });
const items = [
  "Fitted by least squares to ten waypoints of [1]: stage-1/2 propellant, pitch kick, stage-2 pitch rate, " +
    "Breeze-KM climb-rate law. Eight of ten met.",
  `Misses: lift-off mass +${(100 * (v.liftoff_mass / 107e3 - 1)).toFixed(1)} %, burn-1 cutoff ` +
    `+${n0(ev["Breeze-KM_burn1_end"] - 882)} s. The model needs more energy than the flight vehicle.`,
  "Point mass, constant thrust per stage, J2 gravity, generic drag, exponential upper atmosphere [3].",
];
s.addText(items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1 } })),
  { isTextBox: true, x: 0.8, y: 1.5, w: 11.7, h: 2.8, margin: 0, fontSize: 18, color: C.text1,
    valign: "top", paraSpaceAfter: 10 });
const refs = [
  "[1] Eurockot Launch Services, Rockot User's Guide, EHB0003, Issue 5, Rev. 0, 2011.",
  "[2] M. Wade, “Rokot,” Encyclopedia Astronautica, astronautix.com.",
  "[3] D. A. Vallado, Fundamentals of Astrodynamics and Applications, table 8-4.",
];
s.addShape(pres.shapes.LINE, { x: 0.8, y: 4.95, w: 3.0, h: 0, line: { color: INK, width: 0.75 } });
text(s, "References", { x: 0.8, y: 5.1, w: 11.7, h: 0.4, fontSize: 16, bold: true });
text(s, refs.join("\n"), { x: 0.8, y: 5.55, w: 11.7, h: 1.2, fontSize: 14, paraSpaceAfter: 4 });

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
