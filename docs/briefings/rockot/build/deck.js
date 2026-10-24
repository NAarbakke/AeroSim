// Rockot / Breeze-KM ascent briefing. Reads data.json + figures/ written by make_briefing.py.
//   node deck.js   ->  ../Rockot_Ascent_Briefing.pptx
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require("./apply_theme.js");   // run with NODE_PATH=./node_modules

const D = JSON.parse(fs.readFileSync(path.join(__dirname, "data.json"), "utf8"));
const FIG = (f) => path.join(__dirname, "figures", f);
const OUT = path.join(__dirname, "..", "Rockot_Ascent_Briefing.pptx");

const THEME = {
  name: "Plesetsk Night",
  headFontFace: "Arial",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "0B0B0B", lt1: "FFFFFF", dk2: "14213D", lt2: "EEF2F7",
    accent1: "2A78D6", accent2: "EB6834", accent3: "1BAF7A", accent4: "EDA100",
    accent5: "6250D6", accent6: "E34948", hlink: "2A78D6", folHlink: "6250D6",
  },
};
const HEX = { navy: "14213D", ice: "EEF2F7", blue: "2A78D6", orange: "EB6834", grid: "E1E0D9",
              ink2: "52514E", muted: "898781", white: "FFFFFF" };

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";                       // 13.33 x 7.5 in
pres.title = "Rockot / Breeze-KM ascent from Plesetsk";
pres.author = "AeroSim";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

// ---------- layouts ----------
const FOOT = "Rockot / Breeze-KM ascent model  |  AeroSim  |  data: Rockot User's Guide EHB0003 Iss. 5 (2011)";
pres.defineSlideMaster({
  title: "TITLE_DARK", background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.2, w: 11.7, h: 1.5,
        fontSize: 44, bold: true, color: C.background1, valign: "bottom", margin: 0 },
      text: "Title" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 3.9, w: 11.0, h: 1.4,
        fontSize: 20, color: C.background2, valign: "top", margin: 0 }, text: "Subtitle" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT", background: { color: C.background1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.9,
        fontSize: 32, bold: true, color: C.text2, valign: "middle", margin: 0 }, text: "Title" } },
    { text: { text: FOOT, options: { x: 0.6, y: 7.0, w: 10.5, h: 0.3, fontSize: 10,
        color: C.text1, transparency: 45, margin: 0 } } },
  ],
  slideNumber: { x: 12.3, y: 7.0, w: 0.5, h: 0.3, fontSize: 10, color: HEX.muted, align: "right" },
});
pres.defineSlideMaster({
  title: "CLOSING_DARK", background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.9,
        fontSize: 32, bold: true, color: C.background1, valign: "middle", margin: 0 },
      text: "Title" } },
  ],
  slideNumber: { x: 12.3, y: 7.0, w: 0.5, h: 0.3, fontSize: 10, color: HEX.ice, align: "right" },
});

// ---------- helpers ----------
const n0 = (v) => Math.round(v).toLocaleString("en-US");
const n1 = (v) => (Math.round(v * 10) / 10).toFixed(1);
const ev = (k) => D.events[k];
const cmp = Object.fromEntries(D.comparison.map((r) => [r.event, r]));
const pct = (m, g) => `${m >= g ? "+" : ""}${(100 * (m / g - 1)).toFixed(1)} %`;

// keep numbers with their units on one line: "20 kN" never breaks as "20 / kN"
const nb = (t) => typeof t === "string"
  ? t.replace(/(\d) (kN|km\/s|km|kg|kPa|m\/s|s|g|t|N)\b/g, "$1 $2").replace(/(\d)-(\d)/g, "$1‑$2")
  : t;
function text(slide, t, o) {
  t = nb(t);
  slide.addText(t, { isTextBox: true, margin: 0, fontSize: 14, color: C.text1, valign: "top", ...o });
}
function card(slide, x, y, w, h, name) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08,
    fill: { color: C.background2 }, line: { color: C.background2 }, objectName: name });
}
function stat(slide, x, y, w, value, label, note) {
  card(slide, x, y, w, 1.9, "stat " + label);
  text(slide, value, { x: x + 0.3, y: y + 0.25, w: w - 0.6, h: 0.8, fontSize: 36, bold: true,
    color: C.text2, fontFace: THEME.headFontFace, valign: "middle", objectName: "stat value" });
  text(slide, label, { x: x + 0.3, y: y + 1.05, w: w - 0.6, h: 0.4, fontSize: 14, color: C.text1 });
  if (note) text(slide, note, { x: x + 0.3, y: y + 1.42, w: w - 0.6, h: 0.35, fontSize: 11, color: HEX.muted });
}
function bullets(slide, items, o) {
  slide.addText(items.map((t, i) => ({ text: nb(t), options: { bullet: true, breakLine: i < items.length - 1 } })),
    { isTextBox: true, margin: 0, fontSize: 15, color: C.text1, valign: "top", paraSpaceAfter: 8, ...o });
}
const chartText = { catAxisLabelColor: HEX.ink2, valAxisLabelColor: HEX.ink2,
  catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", catAxisLabelFontSize: 11,
  valAxisLabelFontSize: 11, valGridLine: { color: HEX.grid, size: 0.75 }, catGridLine: { style: "none" },
  catAxisLineShow: true, catAxisLineColor: "C3C2B7", valAxisLineShow: false,
  titleFontFace: "+mj-lt", titleFontSize: 14, titleColor: HEX.navy, showTitle: true,
  legendFontFace: "+mn-lt", legendFontSize: 11, legendColor: HEX.ink2,
  dataLabelFontFace: "+mn-lt", dataLabelFontSize: 10, dataLabelColor: HEX.ink2 };

// ---------- 1. title ----------
pres.addSection({ title: "Overview" });
let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Overview" });
s.addText("Rockot / Breeze-KM from Plesetsk", { placeholder: "title" });
s.addText("Ascent trajectory model of the User's Guide reference mission:\n650 km circular orbit, 86.58° inclination  |  AeroSim  |  October 2026",
  { placeholder: "body" });
s.addNotes("Briefing on an AeroSim model of the Rockot / Breeze-KM ascent from Plesetsk, " +
  "built from the Rockot User's Guide (Eurockot, EHB0003 Issue 5, 2011) and calibrated to its reference trajectory.");

// ---------- 2. bottom line ----------
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Overview" });
s.addText("The model flies the guide's reference mission to orbit", { placeholder: "title" });
const s1 = cmp["Stage 1_burnout"], s2 = cmp["Stage 2_burnout"], b1 = cmp["Breeze-KM_burn1_end"];
stat(s, 0.6, 1.5, 3.85, `${n0(D.vehicle.payload)} kg`, "payload to 650 km circular", "User's Guide fig. 3-1, i = 86.4°");
stat(s, 4.74, 1.5, 3.85, `${n1(D.max_q.q_kpa)} kPa`, `max dynamic pressure at ${n0(D.max_q.time)} s`,
  `guide: ${D.comparison[0].guide.q / 1e3} kPa at ${D.comparison[0].guide.time} s`);
stat(s, 8.88, 1.5, 3.85, `${n0(ev("Breeze-KM_burn1_end"))} s`, "lift-off to transfer orbit",
  `guide: ${b1.guide.time} s (end of Breeze-KM burn 1)`);
const tr = D.orbits.transfer, fin = D.orbits.final;
bullets(s, [
  `Stage 1 and 2 burnout speeds within ${pct(s1.model.speed, s1.guide.speed)} and ${pct(s2.model.speed, s2.guide.speed)} of the guide's trajectory (fig. 3-6).`,
  `Breeze-KM burn 1 reaches a ${n0(tr.perigee_km)} x ${n0(tr.apogee_km)} km transfer orbit (guide: 160 x 635 km); burn 2 circularises at ${n0(fin.perigee_km)} x ${n0(fin.apogee_km)} km, i = ${fin.inclination.toFixed(2)}°.`,
  "Stage 1/2 propellant masses are not published; they were fitted to the guide's waypoints. The fitted vehicle weighs " +
    `${n1(D.vehicle.liftoff_mass / 1000)} t at lift-off against the guide's 107 t.`,
], { x: 0.6, y: 3.75, w: 12.1, h: 2.9 });

// ---------- 3. vehicle ----------
pres.addSection({ title: "Vehicle and mission" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Vehicle and mission" });
s.addText("Three hypergolic stages; the top one restarts", { placeholder: "title" });
const stageNotes = [
  ["3 x RD-0233 + RD-0234", "converted SS-19 stage, 17.2 m long"],
  ["RD-0235 + 4 RD-0236 verniers", "3.9 m long, verniers steer"],
  ["S5.98M, 20 kN", "up to 8 ignitions, 12 x 13 N + 4 x 400 N thrusters"],
];
D.vehicle.stages.forEach((st, i) => {
  const x = 0.6 + i * 4.14;
  card(s, x, 1.5, 3.85, 4.2, "stage card " + st.name);
  text(s, st.name, { x: x + 0.3, y: 1.75, w: 3.25, h: 0.5, fontSize: 22, bold: true, color: C.text2,
    fontFace: THEME.headFontFace });
  text(s, stageNotes[i][0], { x: x + 0.3, y: 2.3, w: 3.25, h: 0.35, fontSize: 13, color: HEX.ink2 });
  const rows = [
    ["Thrust, vacuum", `${n0(st.thrust_vac_kN)} kN`],
    ["Isp, vacuum", `${n1(st.isp_vac)} s`],
    ["Burn time", i === 2 ? "up to 1,000 s" : `${n0(st.burn_time)} s`],
    ["Propellant", `${n0(st.propellant)} kg`],
    ["Dry mass", `${n0(st.dry)} kg`],
  ];
  s.addTable(rows.map((r) => [{ text: r[0], options: { color: HEX.ink2 } },
                              { text: r[1], options: { bold: true, align: "right", color: HEX.navy } }]),
    { x: x + 0.3, y: 2.85, w: 3.25, colW: [1.7, 1.55], fontSize: 13, fontFace: "Calibri",
      border: { type: "none" }, rowH: 0.42, margin: 0.02 });
  text(s, stageNotes[i][1], { x: x + 0.3, y: 5.05, w: 3.25, h: 0.5, fontSize: 11, color: HEX.muted });
});
text(s, "Isp, burn times and Breeze-KM data: User's Guide tables 2-2 to 2-8. Stage 1/2 propellant is fitted (slide 9), so their thrust follows from it at the guide's burn time; the guide quotes 2,070 kN and 256 kN vacuum. Dry masses: Encyclopedia Astronautica.",
  { x: 0.6, y: 6.0, w: 12.1, h: 0.6, fontSize: 12, color: HEX.muted });

// ---------- 4. site + ground track ----------
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Vehicle and mission" });
s.addText("Launch from Plesetsk, northbound into a near-polar orbit", { placeholder: "title" });
bullets(s, [
  `Plesetsk Cosmodrome, ${D.site.lat}° N ${D.site.lon}° E, about 200 km south of Archangel (guide §3.2).`,
  "Allowed azimuths from 4.8° (i = 86.4°) to 76.2° (i = 63.2°); 341.5° for sun-synchronous orbits.",
  `Target plane i = ${D.program.target_inclination}°: the model launches along the rotation-corrected azimuth and yaws out residual out-of-plane velocity.`,
  "Ground track runs north over the Barents Sea and the pole, then down the far side of the Earth.",
], { x: 0.6, y: 1.5, w: 4.6, h: 5.2 });
s.addImage({ path: FIG("ground_track.png"), x: 5.5, y: 1.45, w: 7.25, h: 3.44,
  altText: "Ground track of the modelled Rockot ascent from Plesetsk on a latitude-longitude grid" });
text(s, "AeroSim model output (aerosim.plots.ground_track)", { x: 5.5, y: 4.95, w: 7.2, h: 0.3, fontSize: 11, color: HEX.muted });

// ---------- 5. timeline ----------
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Vehicle and mission" });
s.addText("About fifteen minutes from lift-off to the transfer orbit", { placeholder: "title" });
const tl = [
  ["Lift-off", 0, 0], ["Max q", ev("max_q"), 50], ["Stage 1 sep", ev("Stage 1_burnout"), 122],
  ["Fairing", ev("fairing_jettison"), 173], ["Stage 2 sep", ev("Stage 2_burnout"), 305],
  ["Burn 1 end", ev("Breeze-KM_burn1_end"), 882],
];
const tx0 = 1.4, tx1 = 11.9, tmax = Math.max(...tl.map((e) => Math.max(e[1], e[2]))), ty = 3.6;
s.addShape(pres.shapes.LINE, { x: tx0, y: ty, w: tx1 - tx0, h: 0, line: { color: "C3C2B7", width: 1.5 } });
tl.forEach(([name, tm, tg], i) => {
  const x = tx0 + (tx1 - tx0) * Math.sqrt(tm / tmax);     // sqrt time scale: early events spread
  s.addShape(pres.shapes.OVAL, { x: x - 0.1, y: ty - 0.1, w: 0.2, h: 0.2, fill: { color: C.accent1 },
    line: { color: C.background1, width: 2 } });
  const up = i % 2 === 0;
  text(s, name, { x: x - 0.8, y: up ? ty - 1.35 : ty + 0.35, w: 1.6, h: 0.4, fontSize: 15, bold: true,
    color: C.text2, align: "center" });
  text(s, `${n0(tm)} s model`, { x: x - 0.8, y: up ? ty - 0.95 : ty + 0.75, w: 1.6, h: 0.3, fontSize: 12,
    color: HEX.ink2, align: "center" });
  text(s, `${tg} s guide`, { x: x - 0.8, y: up ? ty - 0.65 : ty + 1.05, w: 1.6, h: 0.3, fontSize: 12,
    color: HEX.muted, align: "center" });
});
text(s, "Time axis is square-root scaled so the first two minutes are readable.", { x: 0.6, y: 5.4, w: 8, h: 0.3, fontSize: 11, color: HEX.muted });
card(s, 0.6, 5.85, 12.1, 0.95, "after burn 1");
text(s, `Then: coast to apogee, Breeze-KM burn 2 at ${n0(ev("Breeze-KM_burn2_start"))} s for ${n0(ev("Breeze-KM_burn2_end") - ev("Breeze-KM_burn2_start"))} s (guide: 4,360 s, about 39 s) circularises the orbit; the guide deploys the spacecraft at 5,861 s.`,
  { x: 0.9, y: 6.0, w: 11.5, h: 0.7, fontSize: 14, valign: "middle" });

// ---------- 6. profile (plots output) ----------
pres.addSection({ title: "Results" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.addText("Ascent profile tracks the guide's published waypoints", { placeholder: "title" });
s.addImage({ path: FIG("ascent_overview.png"), x: 0.6, y: 1.35, w: 9.6, h: 5.4,
  altText: "Six small multiples of altitude, speed, flight-path angle, dynamic pressure, acceleration and mass versus time, model line with User's Guide waypoints" });
bullets(s, [
  "Blue line: AeroSim model. Orange points: User's Guide fig. 3-6 waypoints.",
  "Max q, both stagings and the end of burn 1 sit on or near the guide's points.",
  `Peak axial acceleration ${n1(D.peak_accel_g["Stage 1"])} g at stage 1 burnout; guide table 5-1 gives 6.2-8.0 g.`,
  `Stage 2 peaks at ${n1(D.peak_accel_g["Stage 2"])} g (guide: up to 3.0 g).`,
], { x: 10.45, y: 1.5, w: 2.3, h: 5.2, fontSize: 13 });

// ---------- 7. comparison (native charts + table) ----------
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.addText("Model against the guide at each event", { placeholder: "title" });
const evs = D.comparison;
const labels = evs.map((r) => ({ "max_q": "Max q", "Stage 1_burnout": "S1 sep", "fairing_jettison": "Fairing",
  "Stage 2_burnout": "S2 sep", "Breeze-KM_burn1_end": "Burn 1 end" }[r.event]));
s.addChart(pres.charts.BAR, [
  { name: "AeroSim model", labels, values: evs.map((r) => +(r.model.speed / 1000).toFixed(2)) },
  { name: "User's Guide", labels, values: evs.map((r) => +(r.guide.speed / 1000).toFixed(2)) },
], { x: 0.6, y: 1.4, w: 6.0, h: 3.5, barDir: "col", barGapWidthPct: 60, chartColors: [HEX.blue, HEX.orange],
  title: "Earth-relative speed (km/s)", showLegend: true, legendPos: "t", showValue: false,
  valAxisMinVal: 0, ...chartText });
s.addChart(pres.charts.BAR, [
  { name: "AeroSim model", labels, values: evs.map((r) => +(r.model.altitude / 1000).toFixed(1)) },
  { name: "User's Guide", labels, values: evs.map((r) => +(r.guide.altitude / 1000).toFixed(1)) },
], { x: 6.75, y: 1.4, w: 6.0, h: 3.5, barDir: "col", barGapWidthPct: 60, chartColors: [HEX.blue, HEX.orange],
  title: "Altitude (km)", showLegend: true, legendPos: "t", valAxisMinVal: 0, ...chartText });
const hdr = ["Event", "Time model / guide (s)", "Altitude (km)", "Speed (m/s)", "Flight-path angle (deg)"]
  .map((h) => ({ text: h, options: { bold: true, color: HEX.navy, fill: { color: HEX.ice } } }));
const trows = evs.map((r, i) => [labels[i], `${n0(r.model.time)} / ${n0(r.guide.time)}`,
  `${n1(r.model.altitude / 1000)} / ${n1(r.guide.altitude / 1000)}`, `${n0(r.model.speed)} / ${n0(r.guide.speed)}`,
  `${n1(r.model.fpa)} / ${n1(r.guide.fpa)}`]);
s.addTable([hdr, ...trows], { x: 0.6, y: 5.0, w: 12.1, colW: [2.1, 2.5, 2.5, 2.5, 2.5], fontSize: 12,
  fontFace: "Calibri", color: HEX.ink2, rowH: 0.27, border: { type: "solid", color: HEX.grid, pt: 0.75 }, margin: 0.04 });

// ---------- 8. trajectory + orbit ----------
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.addText("Breeze-KM burns twice to reach the 650 km class orbit", { placeholder: "title" });
s.addImage({ path: FIG("trajectory.png"), x: 0.6, y: 1.4, w: 8.0, h: 3.2,
  altText: "Altitude versus downrange from lift-off to the end of the first Breeze-KM burn with events marked" });
const orbRows = [
  ["", "Perigee", "Apogee", "Inclination"],
  ["Transfer orbit, model", `${n0(tr.perigee_km)} km`, `${n0(tr.apogee_km)} km`, `${tr.inclination.toFixed(2)}°`],
  ["Transfer orbit, guide", "160 km", "635 km", "86.58°"],
  ["Final orbit, model", `${n0(fin.perigee_km)} km`, `${n0(fin.apogee_km)} km`, `${fin.inclination.toFixed(2)}°`],
  ["Deployment orbit, guide", "650 km", "650 km", "86.58°"],
];
s.addTable(orbRows.map((r, i) => r.map((c, j) => ({ text: c, options: {
  bold: i === 0 || j === 0, color: i === 0 ? HEX.navy : HEX.ink2, fill: i === 0 ? { color: HEX.ice } : undefined,
  align: j === 0 ? "left" : "right" } }))),
  { x: 0.6, y: 4.85, w: 8.0, colW: [3.2, 1.6, 1.6, 1.6], fontSize: 13, fontFace: "Calibri", rowH: 0.34,
    border: { type: "solid", color: HEX.grid, pt: 0.75 }, margin: 0.05 });
card(s, 8.95, 1.4, 3.8, 5.25, "orbit notes");
text(s, "Insertion", { x: 9.25, y: 1.65, w: 3.2, h: 0.4, fontSize: 18, bold: true, color: C.text2, fontFace: THEME.headFontFace });
bullets(s, [
  "Burn 1 is closed-loop: pitch holds a commanded climb rate while the 20 kN engine builds speed.",
  "Burn 2 fires at apogee, prograde, and stops when perigee reaches the transfer apogee.",
  `Burn 1 runs ${n0(ev("Breeze-KM_burn1_end") - ev("Breeze-KM_burn1_start"))} s against the guide's 577 s and ends lower (${n0(b1.model.altitude / 1000)} km vs 251 km): the model's steering is less efficient.`,
  `Breeze-KM propellant left: ${n0(D.propellant_left)} kg of ${n0(D.vehicle.stages[2].propellant)} kg.`,
], { x: 9.25, y: 2.15, w: 3.25, h: 4.4, fontSize: 13 });
s.addNotes("The guide's transfer orbit is 160 x 635.3 km yet its burn-2 altitude reads 665.6 km and its deployment orbit 650 km; " +
  "the final-orbit box (499 x 506 km, 88.999 deg) does not match this mission and looks carried over from another figure. " +
  "The model targets the transfer apogee and circularises there.");

// ---------- 9. calibration ----------
pres.addSection({ title: "Method" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
s.addText("Two masses and five steering settings were fitted", { placeholder: "title" });
const cal = D.calibration || { stage_propellant: D.vehicle.stages.map((x) => x.propellant) };
const calRows = [
  ["Quantity", "Source value", "Used in model", "Basis"],
  ["Stage 1 propellant", "71,455 kg (Astronautix)", `${n0(cal.stage_propellant[0])} kg`, "fitted"],
  ["Stage 2 propellant", "10,710 kg (Astronautix)", `${n0(cal.stage_propellant[1])} kg`, "fitted"],
  ["Breeze-KM propellant / dry", "4,975 / 1,320 kg", "4,975 / 1,320 kg", "User's Guide t. 2-8"],
  ["Payload", "1,570 kg", `${n0(D.vehicle.payload)} kg`, "User's Guide fig. 3-1"],
  ["Fairing", "not published", `${n0(D.vehicle.fairing)} kg`, "estimate"],
  ["Lift-off mass", "107 t", `${n1(D.vehicle.liftoff_mass / 1000)} t`, "result"],
];
s.addTable(calRows.map((r, i) => r.map((c, j) => ({ text: c, options: { bold: i === 0, color: i === 0 ? HEX.navy : HEX.ink2,
  fill: i === 0 ? { color: HEX.ice } : undefined } }))),
  { x: 0.6, y: 1.45, w: 7.6, colW: [2.3, 2.1, 1.6, 1.6], fontSize: 13, fontFace: "Calibri", rowH: 0.38,
    border: { type: "solid", color: HEX.grid, pt: 0.75 }, margin: 0.05 });
bullets(s, [
  "The guide gives engines, burn times, Breeze-KM masses and lift-off mass, but not stage 1/2 propellant.",
  "Published stage masses leave stage 1 about 300 m/s slow at burnout against the guide's own trajectory.",
  "A least-squares fit adjusts both propellant loads plus pitch kick, stage 2 pitch and Breeze-KM climb rate to ten guide targets.",
  "Mass flow is propellant / burn time at constant thrust, so thrust scales with the fitted load.",
], { x: 8.5, y: 1.45, w: 4.25, h: 3.6, fontSize: 13 });
if (D.calibration) {
  const r = D.calibration.residuals.map(Math.abs);
  const met = r.filter((x) => x <= 1.5).length;
  card(s, 0.6, 4.65, 12.1, 1.95, "fit quality");
  text(s, `${met} of ${r.length} targets met`, { x: 0.9, y: 4.85, w: 4.0, h: 0.6, fontSize: 26, bold: true,
    color: C.text2, fontFace: THEME.headFontFace, valign: "middle" });
  text(s, "within 1.5 tolerance units (25 m/s speed, 2-4 km altitude, 1° angle, 3 km perigee)",
    { x: 0.9, y: 5.45, w: 4.0, h: 0.9, fontSize: 12, color: HEX.muted });
  text(s, `Missed: Breeze-KM burn 1 ends ${n0(ev("Breeze-KM_burn1_end") - 882)} s later than the guide, and the fitted vehicle is ` +
    `${n1(D.vehicle.liftoff_mass / 1000 - 107)} t heavier at lift-off. Both point the same way: the model needs more energy ` +
    "than the real vehicle to fly the same trajectory, most likely from its simple insertion steering and generic drag.",
    { x: 5.2, y: 4.9, w: 7.2, h: 1.5, fontSize: 14, valign: "middle" });
}

// ---------- 10. limits + next steps ----------
s = pres.addSlide({ masterName: "CLOSING_DARK", sectionTitle: "Method" });
s.addText("Limits and next steps", { placeholder: "title" });
const cols = [
  ["What the model leaves out", ["3-DOF point mass: no attitude dynamics or control-loop limits",
    "Generic slender-body drag curve, not Rockot wind-tunnel data",
    "Constant thrust per stage; no throttling, tail-off or vernier phases",
    "Spherical-harmonic gravity to J2 only; exponential upper atmosphere"]],
  ["Next steps", ["Add stage drop zones and impact points (guide: 990-1,100 km for stage 1)",
    "Payload-vs-orbit sweep to reproduce guide fig. 3-1",
    "SSO mission: 341.5° azimuth, parking orbit and plane change",
    "Monte Carlo on thrust, Isp and winds for dispersion"]],
];
cols.forEach(([head, items], i) => {
  const x = 0.6 + i * 6.2;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.5, w: 5.9, h: 5.0, rectRadius: 0.08,
    fill: { color: C.background1, transparency: 92 }, line: { color: C.background1, transparency: 80 },
    objectName: "closing card " + i });
  text(s, head, { x: x + 0.35, y: 1.8, w: 5.2, h: 0.5, fontSize: 22, bold: true, color: C.background1,
    fontFace: THEME.headFontFace });
  bullets(s, items, { x: x + 0.35, y: 2.5, w: 5.2, h: 3.8, color: C.background2, fontSize: 15 });
});

// ---------- 11. sources ----------
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
s.addText("Sources and reproduction", { placeholder: "title" });
bullets(s, [
  "Eurockot Launch Services, Rockot User's Guide, EHB0003, Issue 5, Revision 0, August 2011: tables 2-1 to 2-8, section 3, figures 3-1 and 3-6. Copy in docs/references/rockot/.",
  "Encyclopedia Astronautica, Rokot (astronautix.com/r/rokot.html): stage gross and empty masses.",
  "D. Vallado, Fundamentals of Astrodynamics and Applications, table 8-4: upper-atmosphere density.",
  "Model: AeroSim configs/examples/rocket/orbital/rockot.yaml; calibration: examples/rockot_ascent.py.",
  "This briefing: python docs/briefings/rockot/make_briefing.py, then node docs/briefings/rockot/build/deck.js.",
], { x: 0.6, y: 1.5, w: 12.1, h: 5.2, fontSize: 15 });

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
