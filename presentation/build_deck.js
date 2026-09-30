// Builds the SCAI / Information Society 2026 talk deck (16:9, ~12 min + Q&A).
// Usage: node presentation/build_deck.js [--fig1 fig1.png] [--fig2 fig2_tsne.png] [--out deck.pptx]
// fig1 must be a PNG render of paper/figures/fig1_descriptive.pdf.
const path = require("path");
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");

function arg(name, fallback) {
  const i = process.argv.indexOf(`--${name}`);
  return i > -1 ? process.argv[i + 1] : fallback;
}
const ROOT = path.resolve(__dirname, "..");
const FIG1 = arg("fig1", path.join(__dirname, "fig1_descriptive.png"));
const FIG2 = arg("fig2", path.join(ROOT, "paper/figures/fig2_tsne.png"));
const OUT = arg("out", path.join(__dirname, "IS2026_SCAI_ET_representations.pptx"));

// Palette: deep ink (dominant), light mist, amber accent (gaze heat), muted slate.
const INK = "16324F";
const MIST = "EEF3F8";
const AMBER = "E0A23B";
const SLATE = "5B6B7B";
const RED = "B2182B";
const WHITE = "FFFFFF";
const HEAD = "Cambria";
const BODY = "Calibri";

async function icon(Comp, color, size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(
    React.createElement(Comp, { color: `#${color}`, size: String(size) })
  );
  const png = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + png.toString("base64");
}

// Icon inside a filled circle — the deck's recurring motif ("pupil" dots).
async function iconDot(slide, Comp, x, y, d, fill, fg) {
  slide.addShape("ellipse", { x, y, w: d, h: d, fill: { color: fill }, line: { color: fill } });
  const pad = d * 0.26;
  slide.addImage({ data: await icon(Comp, fg), x: x + pad, y: y + pad, w: d - 2 * pad, h: d - 2 * pad });
}

function text(slide, t, opts) {
  slide.addText(t, { isTextBox: true, fontFace: BODY, color: INK, margin: 0, valign: "top", ...opts });
}

function bullets(items, size = 16, color = INK) {
  return items.map((it, i) => ({
    text: it,
    options: { bullet: true, breakLine: i < items.length - 1, fontSize: size, color, paraSpaceAfter: 8 },
  }));
}

async function main() {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
  pres.author = "Tomi Božak, Zoja Anžur, Gašper Slapničar";
  pres.title = "Comparison of Eye-Tracking Representations for Engagement Inference in Knowledge Workers";

  const footer = "Božak, Anžur & Slapničar · SCAI, Information Society 2026";

  // ---- Slide masters (they show up as reusable layouts in PowerPoint) ----
  pres.defineSlideMaster({
    title: "IS_DARK",
    background: { color: INK },
    objects: [],
  });
  const contentObjects = [
    { text: { text: footer, options: { x: 0.6, y: 7.0, w: 8, h: 0.3, fontFace: BODY, fontSize: 10, color: "8A96A3", margin: 0 } } },
    {
      placeholder: {
        options: { name: "title", type: "title", x: 0.6, y: 0.45, w: 12.1, h: 0.85, fontFace: HEAD, fontSize: 32, bold: true, color: INK, valign: "middle", margin: 0 },
        text: "",
      },
    },
  ];
  const slideNum = { x: 12.2, y: 7.0, w: 0.5, h: 0.3, fontFace: BODY, fontSize: 10, color: "8A96A3", align: "right" };
  pres.defineSlideMaster({ title: "IS_CONTENT", background: { color: WHITE }, objects: contentObjects, slideNumber: slideNum });
  pres.defineSlideMaster({
    title: "IS_TITLE_BODY",
    background: { color: WHITE },
    slideNumber: { ...slideNum },
    objects: [
      ...contentObjects,
      {
        placeholder: {
          options: { name: "body", type: "body", x: 0.6, y: 1.6, w: 12.1, h: 5.1, fontFace: BODY, fontSize: 18, color: INK, margin: 0 },
          text: "",
        },
      },
    ],
  });

  // ---------------- 1. Title ----------------
  let s = pres.addSlide({ masterName: "IS_DARK" });
  // Decorative "eye": two concentric circles on the right.
  s.addShape("ellipse", { x: 9.3, y: 1.6, w: 3.4, h: 3.4, fill: { color: "1F4467" }, line: { color: "1F4467" } });
  s.addShape("ellipse", { x: 10.2, y: 2.5, w: 1.6, h: 1.6, fill: { color: AMBER }, line: { color: AMBER } });
  s.addShape("ellipse", { x: 10.65, y: 2.95, w: 0.7, h: 0.7, fill: { color: INK }, line: { color: INK } });
  text(s, "SCAI · Information Society 2026 · Ljubljana", { x: 0.8, y: 0.8, w: 8, h: 0.4, fontSize: 16, color: AMBER, bold: true });
  text(s, "Comparison of Eye-Tracking Representations for Engagement Inference in Knowledge Workers", {
    x: 0.8, y: 1.45, w: 8.2, h: 2.3, fontFace: HEAD, fontSize: 38, bold: true, color: WHITE, valign: "middle",
  });
  text(s, "Tomi Božak, Zoja Anžur, Gašper Slapničar", { x: 0.8, y: 4.3, w: 8, h: 0.45, fontSize: 22, color: WHITE, bold: true });
  text(s, "Jožef Stefan Institute · Jožef Stefan International Postgraduate School", { x: 0.8, y: 4.8, w: 8.2, h: 0.4, fontSize: 16, color: "C9D6E3" });
  text(s, "29th International Multiconference Information Society · 5–9 October 2026", { x: 0.8, y: 6.4, w: 8.5, h: 0.4, fontSize: 14, color: "C9D6E3" });
  s.addNotes(
`TIME: ~0:30 (cumulative 0:30)
• Good morning/afternoon, I'm Tomi Božak from the Jožef Stefan Institute; joint work with Zoja Anžur and Gašper Slapničar.
• One-sentence pitch: before you train any model on eye-tracking data you have to decide how to represent a gaze window — we compared four common options under one fair, leakage-safe protocol.
• Spoiler to keep in mind: the simplest option, the raw signal, did best — but everything was close to chance.`);

  // ---------------- 2. Motivation ----------------
  s = pres.addSlide({ masterName: "IS_CONTENT" });
  s.addText("Why eye tracking for engagement at work?", { placeholder: "title" });
  const rows = [
    [fa.FaSyncAlt, "Engagement fluctuates", "Momentary immersion changes within a working day and is barely visible in application logs."],
    [fa.FaEye, "Gaze is continuous and non-intrusive", "A screen-mounted tracker gives pupil size, gaze position, fixations, saccades and blinks without any user action."],
    [fa.FaCogs, "Adaptive systems need the state", "Interactive systems that adapt to the user need evidence of their momentary psychological state."],
  ];
  for (let i = 0; i < rows.length; i++) {
    const y = 1.75 + i * 1.6;
    await iconDot(s, rows[i][0], 0.6, y, 0.9, INK, WHITE);
    text(s, rows[i][1], { x: 1.8, y: y - 0.02, w: 5.7, h: 0.4, fontSize: 20, bold: true });
    text(s, rows[i][2], { x: 1.8, y: y + 0.42, w: 5.7, h: 0.9, fontSize: 15, color: SLATE });
  }
  s.addShape("roundRect", { x: 8.1, y: 1.75, w: 4.6, h: 4.4, fill: { color: MIST }, line: { color: MIST }, rectRadius: 0.15 });
  text(s, "Target (experience sampling)", { x: 8.5, y: 2.05, w: 3.9, h: 0.4, fontSize: 14, bold: true, color: SLATE });
  text(s, "“Right now I am immersed in my work”", { x: 8.5, y: 2.6, w: 3.9, h: 1.6, fontFace: HEAD, fontSize: 26, italic: true, color: INK, valign: "middle" });
  text(s, "Rated in situ on a 0–6 scale, several times per working day. We call it the engagement rating.", { x: 8.5, y: 4.45, w: 3.9, h: 1.3, fontSize: 15, color: INK });
  s.addNotes(
`TIME: ~1:00 (cumulative 1:30)
• Knowledge work: engagement — specifically immersion in the task — changes during the day, and logs of app usage tell us little about it.
• Eye tracking is attractive because it is continuous and passive: pupil size, fixations, saccades, blinks, gaze position — the user does nothing.
• Our target comes from experience sampling: participants are prompted during the day and answer "Right now I am immersed in my work" on a 0–6 scale — rated in the moment, not recalled later.
• That single item is what we try to predict from gaze.`);

  // ---------------- 3. Four representations ----------------
  s = pres.addSlide({ masterName: "IS_CONTENT" });
  s.addText("Before any model: how do you represent a gaze window?", { placeholder: "title" });
  const reps = [
    [fa.FaWaveSquare, "Raw signals", "600-d", "5 channels (pupil L/R, gaze x/y, eye–screen distance), resampled to 120 samples and flattened.", "Lowest"],
    [fa.FaRulerCombined, "Handcrafted", "181 → 58–60", "Ocular blink, fixation, saccade, pupil, distance and missingness statistics; filtered per fold.", "Medium"],
    [fa.FaEye, "GazeMAE", "256-d", "Pretrained gaze-specific autoencoder on position + velocity; kept frozen.", "High"],
    [fa.FaBrain, "MOMENT", "1,024-d", "Pretrained general time-series foundation model (MOMENT-1-large); kept frozen.", "High"],
  ];
  const cw = 2.85, gap = 0.23;
  for (let i = 0; i < 4; i++) {
    const x = 0.6 + i * (cw + gap);
    s.addShape("roundRect", { x, y: 1.65, w: cw, h: 4.3, fill: { color: MIST }, line: { color: MIST }, rectRadius: 0.12 });
    await iconDot(s, reps[i][0], x + 0.3, 1.95, 0.8, INK, WHITE);
    text(s, reps[i][1], { x: x + 0.3, y: 2.95, w: cw - 0.6, h: 0.45, fontSize: 19, bold: true });
    text(s, reps[i][2], { x: x + 0.3, y: 3.42, w: cw - 0.6, h: 0.45, fontFace: HEAD, fontSize: 22, bold: true, color: AMBER });
    text(s, reps[i][3], { x: x + 0.3, y: 3.95, w: cw - 0.6, h: 1.4, fontSize: 13.5, color: SLATE });
    text(s, `Pipeline cost: ${reps[i][4]}`, { x: x + 0.3, y: 5.42, w: cw - 0.6, h: 0.35, fontSize: 13, bold: true });
  }
  text(s, "They differ in effort, storage and compute — yet are rarely compared on the same windows, labels and split.", {
    x: 0.6, y: 6.2, w: 12.1, h: 0.6, fontSize: 16, italic: true, color: INK,
  });
  s.addNotes(
`TIME: ~1:15 (cumulative 2:45)
• A window of gaze data can reach a classifier in at least four ways — go left to right.
• Raw: just the five tracker channels, resampled and flattened — no domain knowledge, high dimensional.
• Handcrafted: the classic ocular features — blinks, fixations, saccades, pupil — interpretable, from our earlier work.
• Two learned options, both used frozen, out of the box: GazeMAE is gaze-specific, MOMENT is a general time-series foundation model.
• Key point: costs differ a lot, but published results use different windows, labels and — most importantly — validation protocols, so they cannot be compared.`);

  // ---------------- 4. Research question ----------------
  s = pres.addSlide({ masterName: "IS_DARK" });
  text(s, "Research question", { x: 0.8, y: 0.9, w: 11, h: 0.45, fontSize: 18, bold: true, color: AMBER });
  text(s, "Which eye-tracking representation supports the most reliable inference of self-reported momentary engagement in unseen knowledge workers?", {
    x: 0.8, y: 1.5, w: 11.7, h: 2.0, fontFace: HEAD, fontSize: 32, bold: true, color: WHITE, valign: "middle",
  });
  const same = [
    [fa.FaClone, "Identical 3-s windows"],
    [fa.FaBullseye, "Identical binary target"],
    [fa.FaUserSlash, "Identical leave-one-subject-out folds"],
  ];
  for (let i = 0; i < 3; i++) {
    const x = 0.8 + i * 4.05;
    await iconDot(s, same[i][0], x, 4.45, 0.85, AMBER, INK);
    text(s, same[i][1], { x: x + 1.05, y: 4.45, w: 2.8, h: 0.85, fontSize: 18, color: WHITE, bold: true, valign: "middle" });
  }
  text(s, "Controlled comparison: only the representation changes.", { x: 0.8, y: 5.95, w: 11.7, h: 0.5, fontSize: 18, color: "C9D6E3", italic: true });
  s.addNotes(
`TIME: ~0:30 (cumulative 3:15)
• Read the question: which representation best supports inferring momentary engagement in people the model has never seen?
• "Unseen" is the point — a deployed system meets new users, so we test on held-out participants.
• Everything else is held fixed: same windows, same binary target, same folds. Only the representation changes.`);

  // ---------------- 5. Data ----------------
  s = pres.addSlide({ masterName: "IS_CONTENT" });
  s.addText("Data: knowledge workers in the wild", { placeholder: "title" });
  const stats = [
    ["17", "participants"],
    ["657", "full-workday recordings"],
    ["140,531", "modelled 3-s windows"],
    ["60 Hz", "Tobii tracker, 5 channels"],
  ];
  for (let i = 0; i < 4; i++) {
    const x = 0.6 + i * 3.08;
    s.addShape("roundRect", { x, y: 1.7, w: 2.85, h: 1.9, fill: { color: MIST }, line: { color: MIST }, rectRadius: 0.12 });
    text(s, stats[i][0], { x: x + 0.1, y: 1.85, w: 2.65, h: 1.0, fontFace: HEAD, fontSize: 44, bold: true, color: INK, align: "center", valign: "middle" });
    text(s, stats[i][1], { x: x + 0.1, y: 2.9, w: 2.65, h: 0.5, fontSize: 15, color: SLATE, align: "center" });
  }
  s.addText(
    bullets([
      "Recorded at participants' own workplace during their normal routine (multimodal study; eye tracking only here).",
      "Experience-sampling prompts several times per day; each answer labels the preceding 15 minutes of recording.",
      "3-s windows, 2.25-s step (25 % overlap); kept if ≥ 30 % valid frames and ≥ 32 valid frames.",
      "Modelled windows ≈ 2 % of all recorded windows; 2,136–25,447 per participant. Dataset not yet public.",
    ], 18),
    { x: 0.6, y: 4.0, w: 12.1, h: 2.7, fontFace: BODY, color: INK, margin: 0, valign: "top", isTextBox: true }
  );
  s.addNotes(
`TIME: ~1:00 (cumulative 4:15)
• Real-world data: knowledge workers recorded at their own desks over full working days — not a lab task.
• 17 participants, 657 workday recordings, 19–79 recordings per person.
• Labels: each experience-sampling answer is assigned to the 15 minutes before the prompt; only those labelled intervals are used.
• Windows are 3 s with 25 % overlap; quality thresholds are permissive because tracking loss is frequent in unconstrained office work.
• Result: about 140 thousand windows — roughly 2 % of everything recorded. Data collection is ongoing; dataset not public yet.`);

  // ---------------- 6. Descriptive + target ----------------
  s = pres.addSlide({ masterName: "IS_CONTENT" });
  s.addText("Engagement ratings are person-specific", { placeholder: "title" });
  // fig1 aspect 233.24 x 110.73
  const f1w = 7.4, f1h = f1w * 110.73 / 233.24;
  s.addImage({ path: FIG1, x: 0.6, y: 1.75, w: f1w, h: f1h });
  text(s, "(a) gaze density on the normalised display · (b) rating distribution; red ticks = the 17 participant means.", {
    x: 0.6, y: 1.75 + f1h + 0.15, w: f1w, h: 0.6, fontSize: 12, color: SLATE, italic: true,
  });
  await iconDot(s, fa.FaUserFriends, 8.45, 1.8, 0.75, RED, WHITE);
  text(s, "Means span 2.2–4.9", { x: 9.4, y: 1.85, w: 3.3, h: 0.65, fontSize: 22, bold: true, color: RED, valign: "middle" });
  s.addText(
    bullets([
      "An absolute threshold would mostly encode who the participant is.",
      "Inside each fold, centre every training participant's ratings on their own mean.",
      "Centre the held-out participant with the mean of training means — no test labels used.",
      "Positive = centred rating > 0 → 57.9 % positive overall (0–95 % per participant).",
    ], 15),
    { x: 8.45, y: 2.85, w: 4.3, h: 3.9, fontFace: BODY, color: INK, margin: 0, valign: "top", isTextBox: true }
  );
  s.addNotes(
`TIME: ~1:15 (cumulative 5:30)
• Left: gaze concentrates at the screen centre but covers the whole display — 94.7 % of samples are on screen.
• Right: the rating distribution; the red ticks are the individual participant means — from 2.2 to 4.9 on a 0–6 scale.
• So one person's "4" is another person's "2". An absolute cut-off would mostly learn identity.
• We therefore centre ratings per participant, inside each fold. The held-out person is centred with the average of the training participants' means, so no test labels leak into the threshold.
• Binary target: above vs. below centre; 57.9 % positive overall, but 0–95 % depending on the person.`);

  // ---------------- 7. Protocol ----------------
  s = pres.addSlide({ masterName: "IS_CONTENT" });
  s.addText("Leakage-safe evaluation protocol", { placeholder: "title" });
  const steps = [
    [fa.FaClock, "Windows", "3 s, joined on one window ID across all four exports"],
    [fa.FaLayerGroup, "4 representations", "Raw · Handcrafted · GazeMAE · MOMENT"],
    [fa.FaFilter, "Per-fold preprocessing", "Filtering, imputation, scaling fitted on training participants only"],
    [fa.FaProjectDiagram, "3 predictors", "Majority baseline · Random forest (300 trees) · MLP"],
    [fa.FaUserSlash, "Leave-one-subject-out", "17 folds; test participant never seen in training"],
  ];
  const sw = 2.2, sg = 0.275;
  for (let i = 0; i < steps.length; i++) {
    const x = 0.6 + i * (sw + sg);
    await iconDot(s, steps[i][0], x + (sw - 1.1) / 2, 1.85, 1.1, i === 4 ? AMBER : INK, i === 4 ? INK : WHITE);
    text(s, steps[i][1], { x, y: 3.15, w: sw, h: 0.75, fontSize: 17, bold: true, align: "center", valign: "middle" });
    text(s, steps[i][2], { x, y: 3.95, w: sw, h: 1.2, fontSize: 13.5, color: SLATE, align: "center" });
    if (i < steps.length - 1) {
      s.addShape("rightArrow", { x: x + sw - 0.03, y: 2.28, w: sg + 0.06, h: 0.24, fill: { color: AMBER }, line: { color: AMBER } });
    }
  }
  s.addShape("roundRect", { x: 0.6, y: 5.45, w: 12.1, h: 1.25, fill: { color: MIST }, line: { color: MIST }, rectRadius: 0.12 });
  text(s, [
    { text: "Primary metric: balanced accuracy", options: { bold: true, breakLine: true } },
    { text: "Chance is fixed at 0.500 in every fold, unlike plain accuracy, which tracks each participant's base rate. Secondary: accuracy, macro-F1, ROC-AUC. Unweighted mean over participants (16 two-class folds).", options: { color: SLATE } },
  ], { x: 0.9, y: 5.6, w: 11.5, h: 1.0, fontSize: 15, valign: "middle" });
  s.addNotes(
`TIME: ~1:00 (cumulative 6:30)
• Walk the pipeline left to right.
• All four exports are joined on one window identifier — every model sees exactly the same windows.
• Everything data-dependent — feature filtering, imputation, standardisation — is fitted on training participants only.
• Classifiers: majority baseline, random forest, and an MLP. Frozen encoders: only the classifier on top is trained.
• Leave-one-subject-out: 17 folds, the test person is never in training.
• Why balanced accuracy: positive rates differ strongly between people; balanced accuracy has chance at 0.5 everywhere. One participant had only one class, so 16 folds for balanced accuracy.`);

  // ---------------- 8. Results chart ----------------
  s = pres.addSlide({ masterName: "IS_CONTENT" });
  s.addText("Raw signals ranked first — but all results are near chance", { placeholder: "title" });
  s.addChart(pres.charts.BAR, [
    { name: "Random forest", labels: ["Raw", "Handcrafted", "GazeMAE", "MOMENT"], values: [0.036, 0.008, 0.010, 0.001] },
    { name: "MLP", labels: ["Raw", "Handcrafted", "GazeMAE", "MOMENT"], values: [0.034, 0.021, 0.005, -0.003] },
  ], {
    x: 0.5, y: 1.55, w: 7.9, h: 5.2, barDir: "col", barGapWidthPct: 60,
    chartColors: [INK, AMBER],
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "+0.000;-0.000;0.000", dataLabelFontSize: 11, dataLabelColor: INK,
    valAxisMinVal: -0.01, valAxisMaxVal: 0.045, valAxisMajorUnit: 0.01, valAxisLabelFormatCode: "+0.00;-0.00;0",
    valAxisLabelColor: SLATE, catAxisLabelColor: INK, catAxisLabelFontSize: 14, valAxisLabelFontSize: 11,
    valGridLine: { color: "E3E8EE", size: 0.75 }, catGridLine: { style: "none" },
    showValAxisTitle: true, valAxisTitle: "Balanced accuracy − 0.500 (chance)", valAxisTitleColor: SLATE, valAxisTitleFontSize: 12,
    showLegend: true, legendPos: "t", legendFontSize: 13, legendColor: INK,
  });
  const calls = [
    ["0.536", "best balanced accuracy", "raw + random forest"],
    ["0.553", "best ROC-AUC", "raw + MLP"],
    ["0.075", "SD across participants", "> the 0.036 margin over chance"],
  ];
  for (let i = 0; i < 3; i++) {
    const y = 1.65 + i * 1.72;
    s.addShape("roundRect", { x: 8.8, y, w: 3.9, h: 1.5, fill: { color: i === 2 ? "FBEFD9" : MIST }, line: { color: i === 2 ? "FBEFD9" : MIST }, rectRadius: 0.12 });
    text(s, calls[i][0], { x: 9.05, y: y + 0.15, w: 1.8, h: 1.2, fontFace: HEAD, fontSize: 34, bold: true, color: i === 2 ? "9A5B00" : INK, valign: "middle" });
    text(s, [
      { text: calls[i][1], options: { bold: true, breakLine: true } },
      { text: calls[i][2], options: { color: SLATE } },
    ], { x: 10.85, y: y + 0.15, w: 1.75, h: 1.2, fontSize: 13, valign: "middle" });
  }
  s.addNotes(
`TIME: ~1:45 (cumulative 8:15)
• The chart shows balanced accuracy minus chance — so zero means no better than guessing.
• Raw signals are highest with both classifiers: 0.536 balanced accuracy with the random forest, 0.534 with the MLP; the MLP on raw also has the best AUC, 0.553, and best macro-F1.
• Handcrafted features come next overall — 0.521 with the MLP. GazeMAE is just above chance; MOMENT is at chance with both classifiers.
• Be honest about the size: the best margin is 3.6 points, and the spread across participants (SD 0.075) is larger than that margin and larger than the gap between raw and handcrafted.
• Both classifiers put raw first and MOMENT last; the two middle ones swap — so we don't claim an order between handcrafted and GazeMAE.
• (If asked: full table is in the backup slides.)`);

  // ---------------- 9. t-SNE ----------------
  s = pres.addSlide({ masterName: "IS_CONTENT" });
  s.addText("What the representations encode: people, not engagement", { placeholder: "title" });
  const f2w = 9.0, f2h = f2w * 1420 / 2800;
  s.addImage({ path: FIG2, x: 0.6, y: 1.6, w: f2w, h: f2h });
  text(s, "t-SNE of a common sample of 5,000 windows. Top: coloured by engagement rating. Bottom: coloured by participant.", {
    x: 0.6, y: 1.6 + f2h + 0.12, w: f2w, h: 0.5, fontSize: 12, color: SLATE, italic: true,
  });
  await iconDot(s, fa.FaTimes, 9.95, 1.75, 0.65, SLATE, WHITE);
  text(s, "No region follows the engagement rating in any representation.", { x: 10.75, y: 1.7, w: 2.0, h: 1.6, fontSize: 14.5 });
  await iconDot(s, fa.FaUserFriends, 9.95, 3.6, 0.65, RED, WHITE);
  text(s, "Participant patches: clear in raw, weaker in handcrafted and GazeMAE, near-uniform in MOMENT.", { x: 10.75, y: 3.55, w: 2.0, h: 2.2, fontSize: 14.5 });
  text(s, "Exploratory view — not evidence of separability.", { x: 9.95, y: 6.0, w: 2.8, h: 0.7, fontSize: 12.5, italic: true, color: SLATE });
  s.addNotes(
`TIME: ~1:00 (cumulative 9:15)
• Same four representations, projected to 2-D with t-SNE, same 5,000 windows.
• Top row, coloured by engagement: no region follows the rating — in raw or in the learned embeddings.
• Bottom row, same projections coloured by participant: raw forms clear per-person patches; handcrafted and GazeMAE weaker; MOMENT almost uniform.
• Message: the most visible structure in gaze windows is the person, not their engagement — which is exactly why subject-disjoint evaluation matters.
• Caveat: projections are exploratory; we don't draw conclusions about separability from them.`);

  // ---------------- 10. Discussion ----------------
  s = pres.addSlide({ masterName: "IS_CONTENT" });
  s.addText("Why no gain from frozen encoders — and what to use", { placeholder: "title" });
  s.addShape("roundRect", { x: 0.6, y: 1.65, w: 5.9, h: 5.05, fill: { color: MIST }, line: { color: MIST }, rectRadius: 0.12 });
  await iconDot(s, fa.FaExchangeAlt, 0.9, 1.95, 0.75, INK, WHITE);
  text(s, "Pretraining mismatch", { x: 1.85, y: 1.95, w: 4.4, h: 0.75, fontSize: 21, bold: true, valign: "middle" });
  s.addText(
    bullets([
      "GazeMAE: pretrained on controlled stimulus-viewing corpora.",
      "MOMENT: pretrained on generic time-series benchmarks.",
      "Office gaze: day-long, spontaneous, with dropouts, idle stretches and heterogeneous displays.",
      "Our result concerns frozen transfer only — fine-tuning is untested.",
    ], 17),
    { x: 0.9, y: 2.95, w: 5.35, h: 3.6, fontFace: BODY, color: INK, margin: 0, valign: "top", isTextBox: true }
  );
  const guide = [
    [fa.FaWaveSquare, INK, WHITE, "Raw signals", "Simplest pipeline: no feature engineering, checkpoint or encoder pass."],
    [fa.FaRulerCombined, INK, WHITE, "Handcrafted features", "Close behind and interpretable; effective on related targets in earlier work."],
    [fa.FaBrain, "D6DCE3", INK, "Frozen general encoder", "Not supported as a default choice here."],
  ];
  text(s, "Practical guidance", { x: 6.95, y: 1.7, w: 5.7, h: 0.5, fontSize: 21, bold: true });
  for (let i = 0; i < 3; i++) {
    const y = 2.45 + i * 1.45;
    await iconDot(s, guide[i][0], 6.95, y, 0.75, guide[i][1], guide[i][2]);
    text(s, guide[i][3], { x: 7.9, y: y - 0.05, w: 4.8, h: 0.4, fontSize: 17, bold: true });
    text(s, guide[i][4], { x: 7.9, y: y + 0.38, w: 4.8, h: 0.85, fontSize: 14, color: SLATE });
  }
  s.addNotes(
`TIME: ~1:15 (cumulative 10:30)
• Our interpretation for the encoders: a domain mismatch. GazeMAE learned from controlled viewing experiments; MOMENT from generic time-series benchmarks. Neither has seen day-long, spontaneous office gaze with dropouts and different monitors.
• Important scope: we only tested frozen encoders — the out-of-the-box setting a practitioner tries first. Fine-tuning may change the picture.
• Practical guidance: start with raw windows — cheapest pipeline. Keep handcrafted features in mind when you need interpretability. Don't assume a frozen general-purpose encoder will help.`);

  // ---------------- 11. Limitations & future work ----------------
  s = pres.addSlide({ masterName: "IS_CONTENT" });
  s.addText("Limitations and next steps", { placeholder: "title" });
  const lim = [
    "One unpublished, still-growing dataset; 17 participants.",
    "Target: one noisy self-report item, assigned to a 15-min interval and binarised.",
    "Overlapping windows and repeated labels shrink the effective sample size.",
    "One windowing scheme, one seed, one QC rule set, two public checkpoints.",
  ];
  const nxt = [
    "Fine-tune the encoders on office gaze — starting with GazeMAE.",
    "Calibrated personalisation for new users.",
    "Longer windows and aggregated targets matching how engagement is reported.",
    "Validate the comparison on further datasets.",
  ];
  const cols = [[fa.FaExclamationTriangle, "Limitations", lim, "D6DCE3", INK], [fa.FaRocket, "Next steps", nxt, AMBER, INK]];
  for (let c = 0; c < 2; c++) {
    const x = 0.6 + c * 6.2;
    s.addShape("roundRect", { x, y: 1.65, w: 5.9, h: 5.05, fill: { color: c === 0 ? MIST : "FBEFD9" }, line: { color: c === 0 ? MIST : "FBEFD9" }, rectRadius: 0.12 });
    await iconDot(s, cols[c][0], x + 0.3, 1.95, 0.75, cols[c][3], cols[c][4]);
    text(s, cols[c][1], { x: x + 1.25, y: 1.95, w: 4.3, h: 0.75, fontSize: 21, bold: true, valign: "middle" });
    s.addText(bullets(cols[c][2], 17.5), { x: x + 0.3, y: 2.95, w: 5.3, h: 3.6, fontFace: BODY, color: INK, margin: 0, valign: "top", isTextBox: true });
  }
  s.addNotes(
`TIME: ~1:00 (cumulative 11:30)
• Limitations, briefly: a single, still-growing dataset of 17 people; the label is one noisy self-report spread over 15 minutes; overlapping windows mean the effective sample is smaller than 140k.
• Also one windowing scheme, one seed and two specific checkpoints — the ranking is preliminary guidance, not a definitive benchmark.
• Next: fine-tune the encoders on office gaze (GazeMAE came closest to handcrafted), personalised calibration for new users, and longer windows / aggregated targets that match how people actually report engagement.`);

  // ---------------- 12. Take-home ----------------
  s = pres.addSlide({ masterName: "IS_DARK" });
  text(s, "Take-home messages", { x: 0.8, y: 0.7, w: 11, h: 0.8, fontFace: HEAD, fontSize: 36, bold: true, color: WHITE, valign: "middle" });
  const take = [
    "Under a leakage-safe, subject-disjoint protocol, raw gaze windows gave the highest observed result (0.536 balanced accuracy, 0.553 ROC-AUC).",
    "Frozen GazeMAE and MOMENT embeddings did not improve on raw signals, despite higher preprocessing and compute cost.",
    "Cross-person engagement inference from 3-s office gaze is hard: between-participant spread exceeds the differences between representations.",
  ];
  for (let i = 0; i < 3; i++) {
    const y = 1.85 + i * 1.3;
    s.addShape("ellipse", { x: 0.8, y, w: 0.75, h: 0.75, fill: { color: AMBER }, line: { color: AMBER } });
    text(s, String(i + 1), { x: 0.8, y, w: 0.75, h: 0.75, fontFace: HEAD, fontSize: 24, bold: true, color: INK, align: "center", valign: "middle" });
    text(s, take[i], { x: 1.85, y: y - 0.05, w: 10.7, h: 1.1, fontSize: 18, color: WHITE, valign: "middle" });
  }
  text(s, "Thank you — questions?", { x: 0.8, y: 5.9, w: 7, h: 0.7, fontFace: HEAD, fontSize: 30, bold: true, color: AMBER, valign: "middle" });
  text(s, "tb85088@student.uni-lj.si", { x: 7.9, y: 5.95, w: 4.7, h: 0.6, fontSize: 16, color: "C9D6E3", align: "right", valign: "middle" });
  s.addNotes(
`TIME: ~0:45 (cumulative ~12:15)
• Three things to remember.
• One: under a fair, subject-disjoint protocol, the simplest representation — raw gaze windows — gave the best observed results.
• Two: frozen pretrained encoders did not help, despite costing more.
• Three: the task itself is hard in the wild — person-to-person variation is bigger than the differences between representations.
• Thank the audience and the funders (ARIS N1-0319, SNSF 214991 — Weave project). Invite questions.`);

  // ---------------- Backup ----------------
  s = pres.addSlide({ masterName: "IS_CONTENT" });
  s.addText("Backup: full leave-one-subject-out results", { placeholder: "title" });
  const hdr = ["", "Bal. acc.", "Δ", "SD", "Macro-F1", "ROC-AUC", "Acc."].map((t) => ({
    text: t, options: { bold: true, color: WHITE, fill: { color: INK }, align: t ? "center" : "left" },
  }));
  const data = [
    ["Majority baseline", ".500", "—", ".000", ".326", ".500", ".524"],
    ["Raw, RF", ".536", "+.036", ".075", ".464", ".550", ".528"],
    ["Raw, MLP", ".534", "+.034", ".050", ".472", ".553", ".542"],
    ["Handcrafted, RF", ".508", "+.008", ".035", ".418", ".527", ".528"],
    ["Handcrafted, MLP", ".521", "+.021", ".032", ".451", ".535", ".531"],
    ["GazeMAE, RF", ".510", "+.010", ".027", ".441", ".514", ".540"],
    ["GazeMAE, MLP", ".505", "+.005", ".031", ".447", ".508", ".532"],
    ["MOMENT, RF", ".501", "+.001", ".013", ".380", ".504", ".521"],
    ["MOMENT, MLP", ".497", "−.003", ".010", ".362", ".504", ".507"],
  ];
  const best = { 1: 1, 2: 1, 4: 2, 5: 2, 6: 2 }; // column -> best row index
  const body = data.map((r, ri) => r.map((c, ci) => ({
    text: c,
    options: { bold: best[ci] === ri, align: ci ? "center" : "left", fill: { color: ri % 2 ? WHITE : MIST }, color: INK },
  })));
  s.addTable([hdr, ...body], {
    x: 0.6, y: 1.6, w: 12.1, colW: [3.1, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5], fontFace: BODY, fontSize: 15, rowH: 0.42,
    border: { type: "solid", pt: 0.5, color: "D6DCE3" }, valign: "middle",
  });
  text(s, "Balanced accuracy is primary (chance 0.500); Δ = paired improvement over the majority baseline; SD across participants. All metrics except accuracy over the 16 two-class folds. Dimensionality: raw 600, handcrafted 58–60, GazeMAE 256, MOMENT 1,024. Best per metric in bold.", {
    x: 0.6, y: 5.95, w: 12.1, h: 0.8, fontSize: 13, color: SLATE, italic: true,
  });
  s.addNotes(
`BACKUP — show only if asked.
• Table 1 of the paper. Bold = best per metric column.
• GazeMAE RF has the second-best plain accuracy (.540), but plain accuracy partly reflects each participant's base rate, which is why it is secondary.
• Rounded gap raw RF vs best handcrafted = .536 − .521 = .015 (0.0156 at full precision).`);

  s = pres.addSlide({ masterName: "IS_CONTENT" });
  s.addText("Backup: representation details and sanity check", { placeholder: "title" });
  s.addText(
    bullets([
      "Raw: 5 channels linearly resampled to 120 samples (whole 3-s window), flattened to 600-d; gaze normalised to [0,1] per display resolution.",
      "Handcrafted: offline event detection (blinks from 90–400 ms gaps, I-DT fixations, saccades between fixations); per-fold filter drops >60 % missing, zero variance, correlation > 0.8 → 58–60 of 181 features.",
      "GazeMAE: public micro–macro checkpoints, position + velocity at 500 Hz, rescaled to 35 px per degree of visual angle; 128 + 128 = 256-d.",
      "MOMENT-1-large: input length 512, mean pooling, 1,024-d; RevIN standardises each channel per window.",
      "MLP: 1028–512–256–128–64, GELU, layer norm, Adam (lr 2e-4, wd 1e-3), early stopping on a 20 % stratified split. Single seed.",
    ], 16),
    { x: 0.6, y: 1.6, w: 7.6, h: 5.2, fontFace: BODY, color: INK, margin: 0, valign: "top", isTextBox: true }
  );
  s.addShape("roundRect", { x: 8.6, y: 1.6, w: 4.1, h: 5.1, fill: { color: MIST }, line: { color: MIST }, rectRadius: 0.12 });
  text(s, "Persistence check", { x: 8.9, y: 1.85, w: 3.5, h: 0.45, fontSize: 18, bold: true });
  text(s, "0.960", { x: 8.9, y: 2.4, w: 3.5, h: 0.9, fontFace: HEAD, fontSize: 44, bold: true, color: INK });
  text(s, "balanced accuracy of carrying the last label forward within recordings (1,434 blocks, 13 two-class participants).", { x: 8.9, y: 3.35, w: 3.5, h: 1.3, fontSize: 13.5, color: SLATE });
  text(s, "Labels are constant over each 15-min interval by construction — not comparable with the cross-person results.", { x: 8.9, y: 4.8, w: 3.5, h: 1.6, fontSize: 13.5, italic: true });
  s.addNotes(
`BACKUP — show only if asked.
• Likely questions: why GazeMAE rescaling (the encoder expects 35 px per degree; displays and viewing distances differ between participants); why MOMENT needs no rescaling (RevIN per-window standardisation).
• Physical monitor sizes are not in the data — standard monitor sizes per resolution were assumed, combined with measured eye–screen distance.
• Persistence: 0.999 accuracy, 0.960 balanced accuracy, 0.965 macro-F1 — shows labels are highly autocorrelated because one answer covers 15 minutes; it uses the participant's own labels, so it is not a competitor to the LOSO models.`);

  await pres.writeFile({ fileName: OUT });
  console.log("wrote", OUT);
}

main().catch((e) => { console.error(e); process.exit(1); });
