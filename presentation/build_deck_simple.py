"""Build the simple SKUI 2026 talk deck on top of the SKUI 2024 presentation.

The 2024 deck supplies the master, logo footer, title slide, eye-tracker photo
and window-segmentation diagram; everything else is plain, editable PowerPoint
content (native chart, table and shapes).

Usage:
    python presentation/build_deck_simple.py --template SKUI_2024_presentation.pptx \
        [--fig1 presentation/fig1_descriptive.png] [--fig2 paper/figures/fig2_tsne.png] \
        [--out presentation/SKUI2026_ET_representations.pptx]
"""

import argparse
import copy
from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
BLACK = RGBColor(0, 0, 0)
GREY = RGBColor(0x59, 0x59, 0x59)
GREEN = RGBColor(0x70, 0xAD, 0x47)
ORANGE = RGBColor(0xF4, 0xB1, 0x42)
BLUE = RGBColor(0x44, 0x72, 0xC4)
PASTELS = ["E2EFDA", "FFF2CC", "DDEBF7", "FCE4D6"]

# Short speaker bullets, also written to talk_notes.md.
NOTES: dict[str, list[str]] = {}


def set_title(slide, text: str) -> None:
    """Title in the 2024 style: bold red."""
    slide.shapes.title.text = text
    run = slide.shapes.title.text_frame.paragraphs[0].runs[0]
    run.font.bold = True
    run.font.color.rgb = RGBColor(0xFF, 0x00, 0x00)


def title_only(prs: Presentation, layout):
    """Content layout (keeps the logo footer) with the body placeholder removed."""
    slide = prs.slides.add_slide(layout)
    body = slide.placeholders[1]._element
    body.getparent().remove(body)
    return slide


def drop_unused_images(slide) -> None:
    """Drop image relationships left behind by removed shapes."""
    xml = slide._element.xml
    for rid, rel in list(slide.part.rels.items()):
        if rel.reltype.endswith("/image") and f'"{rid}"' not in xml:
            slide.part.drop_rel(rid)


def delete_slide(prs: Presentation, slide) -> None:
    """Remove a slide and its relationship from the presentation."""
    lst = prs.slides._sldIdLst
    for sld_id in list(lst):
        if prs.part.related_part(sld_id.rId) is slide.part:
            prs.part.drop_rel(sld_id.rId)
            lst.remove(sld_id)
            return


def set_order(prs: Presentation, slides: list) -> None:
    """Reorder the slide list to match ``slides``."""
    lst = prs.slides._sldIdLst
    by_part = {prs.part.related_part(s.rId): s for s in lst}
    for s in list(lst):
        lst.remove(s)
    for slide in slides:
        lst.append(by_part[slide.part])


def remove_shapes(slide, names: set[str]) -> None:
    """Delete top-level shapes by name."""
    for sh in list(slide.shapes):
        if sh.name in names:
            sh._element.getparent().remove(sh._element)


def fill_body(tf, items: list[tuple[int, str]]) -> None:
    """Write (level, text) bullets into a body placeholder, 24/20 pt like the 2024 deck."""
    tf.clear()
    for i, (level, text) in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.level = level
        run = p.add_run()
        run.text = text
        run.font.size = Pt(24 if level == 0 else 20)


def textbox(slide, x, y, w, h, text, size=16, bold=False, italic=False, color=BLACK, align=PP_ALIGN.LEFT):
    """Add a plain text box."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    lines = text.split("\n")
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run()
        r.text = line
        r.font.size, r.font.bold, r.font.italic, r.font.color.rgb = Pt(size), bold, italic, color
    return tb


def box(slide, x, y, w, h, text, fill, size=18, bold=False):
    """Rounded rectangle with centred black text."""
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.fill.solid()
    shp.fill.fore_color.rgb = RGBColor.from_string(fill)
    shp.line.color.rgb = RGBColor(0xBF, 0xBF, 0xBF)
    tf = shp.text_frame
    tf.word_wrap = True
    for i, line in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = line
        r.font.size = Pt(size if i == 0 else size - 4)
        r.font.bold = bold if i == 0 else False
        r.font.color.rgb = BLACK if i == 0 else GREY
    return shp


def arrow(slide, x1, y1, x2, y2):
    """Straight grey arrow."""
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = RGBColor(0x7F, 0x7F, 0x7F)
    c.line.width = Pt(1.75)
    ln = c.line._get_or_add_ln()
    tail = ln.makeelement("{http://schemas.openxmlformats.org/drawingml/2006/main}tailEnd", {"type": "triangle"})
    ln.append(tail)


def add_slide_number(slide, template_box) -> None:
    """Copy the 2024 deck's slide-number field box."""
    slide.shapes._spTree.append(copy.deepcopy(template_box))


def notes(slide, key: str, bullets: list[str]) -> None:
    """Store talk bullets and write them to the slide notes."""
    NOTES[key] = bullets
    slide.notes_slide.notes_text_frame.text = "\n".join(f"- {b}" for b in bullets)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True, type=Path)
    ap.add_argument("--fig1", type=Path, default=ROOT / "presentation/fig1_descriptive.png")
    ap.add_argument("--fig2", type=Path, default=ROOT / "paper/figures/fig2_tsne.png")
    ap.add_argument("--out", type=Path, default=ROOT / "presentation/SKUI2026_ET_representations.pptx")
    ap.add_argument("--notes-md", type=Path, default=ROOT / "presentation/talk_notes.md")
    args = ap.parse_args()

    prs = Presentation(str(args.template))
    old = list(prs.slides)
    title_s, intro_s, prep_s = old[0], old[1], old[3]
    num_box = next(sh for sh in prep_s.shapes if sh.name == "TextBox 7")._element
    num_box = copy.deepcopy(num_box)
    L_CONTENT = prs.slide_layouts[1]

    # ---- 1. Title (2024 title slide, new text) ----
    sh = {s.name: s for s in title_s.shapes}
    t = sh["Title 1"].text_frame.paragraphs[0].runs[0]
    t.text = "Comparison of Eye-Tracking Representations for Engagement Inference in Knowledge Workers"
    t.font.size = Pt(40)
    subtitle, date_box = [s for s in title_s.shapes if s.name == "Subtitle 2"]
    authors = subtitle.text_frame.paragraphs[2].runs
    for run, txt in zip(authors, ["Tomi Božak", ", ", "Zoja", " ", "Anžur", " and ", "Gašper", " ", "Slapničar"]):
        run.text = txt
    date_box.text_frame.paragraphs[0].runs[0].text = "SKUI 2026,"
    date_box.text_frame.paragraphs[1].runs[0].text = "October 2026, Ljubljana"
    notes(title_s, "1. Title", [
        "Thank the chair; I'm Tomi Božak from the Department of Intelligent Systems, JSI.",
        "Joint work with Zoja Anžur and Gašper Slapničar.",
        "Topic: which way of representing eye-tracking data works best for predicting engagement at work.",
    ])

    # ---- 2. Introduction (2024 slide, keep only the eye-tracker photo) ----
    remove_shapes(intro_s, {"Group 4", "Rectangle: Rounded Corners 12", "Curved Connector 6"})
    ish = {s.name: s for s in intro_s.shapes}
    for name in ("Picture 4", "Rectangle: Rounded Corners 13", "TextBox 18"):
        ish[name].top = ish[name].top - Inches(1.9)
        ish[name].left = ish[name].left - Inches(0.35)
    drop_unused_images(intro_s)
    ish["TextBox 18"].text_frame.paragraphs[0].runs[0].text = "Screen-mounted eye tracker"
    for r in ish["TextBox 18"].text_frame.paragraphs[0].runs[1:]:
        r.text = ""
    fill_body(ish["Content Placeholder 2"].text_frame, [
        (0, "Momentary engagement of knowledge workers"),
        (1, "Changes during the day, hard to see in logs"),
        (0, "Eye tracking: continuous and contact-free"),
        (0, "How to represent a gaze window?"),
        (1, "Raw signal, handcrafted features or pretrained embeddings"),
        (1, "Rarely compared under the same conditions"),
    ])
    notes(intro_s, "2. Introduction", [
        "Engagement (immersion in work) changes during the day; application logs show little of it.",
        "A screen-mounted eye tracker records gaze and pupil size continuously, without the user doing anything.",
        "Before training any model we must choose how to represent a window of gaze data.",
        "Options: raw signal, handcrafted features, or embeddings from pretrained models.",
        "Published results use different data, labels and splits, so they can't be compared directly — that's our gap.",
    ])

    # ---- 3. Four representations (simple diagram) ----
    rep_s = title_only(prs, L_CONTENT)
    set_title(rep_s, "Four Representations")
    box(rep_s, 0.9, 3.25, 2.6, 1.4, "3-s gaze window\n5 channels, 60 Hz", "F2F2F2", size=20, bold=True)
    reps = [("Raw signal", "600 values"), ("Handcrafted features", "58–60 features"),
            ("GazeMAE (frozen)", "256-d embedding"), ("MOMENT (frozen)", "1,024-d embedding")]
    for i, (name, dim) in enumerate(reps):
        y = 1.75 + i * 1.2
        box(rep_s, 4.6, y, 3.8, 0.95, f"{name}\n{dim}", PASTELS[i], size=20, bold=True)
        arrow(rep_s, 3.5, 3.95, 4.6, y + 0.475)
        arrow(rep_s, 8.4, y + 0.475, 9.5, 3.95)
    box(rep_s, 9.5, 3.25, 2.9, 1.4, "Classifier\nRandom Forest / MLP", "F2F2F2", size=20, bold=True)
    add_slide_number(rep_s, num_box)
    notes(rep_s, "3. Four representations", [
        "Every model sees the same 3-second windows: pupil size (L/R), gaze x/y, eye–screen distance.",
        "Raw: the signal itself, resampled and flattened — no domain knowledge.",
        "Handcrafted: blinks, fixations, saccades, pupil statistics — interpretable, from our earlier work.",
        "GazeMAE: pretrained on eye-tracking data; MOMENT: general time-series foundation model. Both used frozen, out of the box.",
        "On top: the same classifiers — random forest and an MLP.",
    ])

    # ---- 4. Data ----
    data_s = prs.slides.add_slide(L_CONTENT)
    set_title(data_s, "Data")
    body = data_s.placeholders[1]
    body.width = Inches(6.0)
    fill_body(body.text_frame, [
        (0, "Knowledge workers at their own desks"),
        (1, "17 participants, 657 full workdays"),
        (0, "Eye tracker, 60 Hz"),
        (1, "Pupil size, gaze x/y, eye–screen distance"),
        (0, "Self-report several times a day"),
        (1, "“Right now I am immersed in my work”"),
        (1, "0–6 scale, labels the preceding 15 min"),
    ])
    w = 5.9
    data_s.shapes.add_picture(str(args.fig1), Inches(7.1), Inches(2.3), width=Inches(w))
    textbox(data_s, 7.1, 2.3 + w * 110.73 / 233.24 + 0.1, w, 0.6,
            "(a) gaze density on screen   (b) engagement ratings;\nred ticks = participant means", size=14, italic=True, color=GREY)
    add_slide_number(data_s, num_box)
    notes(data_s, "4. Data", [
        "Real-world data: knowledge workers recorded at their own desks during normal workdays.",
        "17 participants, 657 full workdays; eye tracker at 60 Hz.",
        "Several times a day they answered a short questionnaire; our target: “Right now I am immersed in my work”, 0–6.",
        "Each answer labels the 15 minutes before it.",
        "Left plot: gaze mostly in the screen centre. Right: ratings; red ticks show each participant's mean — very different between people.",
    ])

    # ---- 5. Windows and target (2024 preprocessing slide with window diagram) ----
    remove_shapes(prep_s, {"TextBox 3"})
    psh = {s.name: s for s in prep_s.shapes}
    psh["Title 1"].text_frame.paragraphs[0].runs[0].text = "Windows and Target"
    for r in psh["Title 1"].text_frame.paragraphs[0].runs[1:]:
        r.text = ""
    fill_body(psh["Content Placeholder 2"].text_frame, [
        (0, "3-s windows, 25 % overlap"),
        (1, "Quality filter → 140,531 windows"),
        (0, "Ratings are person-specific"),
        (1, "Participant means: 2.2–4.9"),
        (0, "Binary target"),
        (1, "Above vs. below the participant's mean"),
        (1, "Centring fitted without test labels"),
    ])
    notes(prep_s, "5. Windows and target", [
        "Recordings cut into 3-second windows with 25 % overlap; windows with too much tracking loss removed → about 140 thousand windows.",
        "Ratings differ a lot between people (means 2.2–4.9), so an absolute threshold would mostly learn who the person is.",
        "So we centre ratings per participant and predict above vs. below their own mean.",
        "For the test participant we use the average of the training participants' means — no test labels leak in.",
    ])

    # ---- 6. Methodology with LOSO diagram ----
    meth_s = prs.slides.add_slide(L_CONTENT)
    set_title(meth_s, "Methodology")
    body = meth_s.placeholders[1]
    body.width = Inches(6.2)
    fill_body(body.text_frame, [
        (0, "Majority baseline, Random Forest, MLP"),
        (0, "Leave-One-Subject-Out CV"),
        (1, "17 folds"),
        (1, "Preprocessing fitted on training subjects only"),
        (0, "Main measure: balanced accuracy"),
        (1, "Chance = 0.5"),
    ])
    frame = meth_s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(7.6), Inches(1.95), Inches(4.9), Inches(4.3))
    frame.fill.background()
    frame.line.color.rgb = BLACK
    frame.line.width = Pt(2.25)
    rows = [("P01", GREEN), ("P02", GREEN), ("P03", GREEN), ("⋮", None), ("P16", GREEN), ("P17", ORANGE)]
    for i, (lab, col) in enumerate(rows):
        y = 2.2 + i * 0.62
        textbox(meth_s, 7.75, y, 0.8, 0.45, lab, size=16)
        if col is not None:
            bar = meth_s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(8.6), Inches(y + 0.02), Inches(3.6), Inches(0.42))
            bar.fill.solid()
            bar.fill.fore_color.rgb = col
            bar.line.fill.background()
    textbox(meth_s, 7.6, 6.3, 4.9, 0.4, "green = train, orange = test (repeat for every subject)", size=14, italic=True, color=GREY)
    add_slide_number(meth_s, num_box)
    notes(meth_s, "6. Methodology", [
        "Three predictors on every representation: majority baseline, random forest and an MLP.",
        "Leave-one-subject-out: train on 16 people (green), test on the one left out (orange); repeat 17 times.",
        "Imputation, scaling and feature selection are fitted on training people only — no leakage.",
        "Main measure is balanced accuracy, because class balance differs per person; chance is always 0.5.",
    ])

    # ---- 7. Results chart ----
    res_s = title_only(prs, L_CONTENT)
    set_title(res_s, "Results")
    cd = CategoryChartData()
    cd.categories = ["Raw", "Handcrafted", "GazeMAE", "MOMENT"]
    cd.add_series("Random Forest", (0.536, 0.508, 0.510, 0.501))
    cd.add_series("MLP", (0.534, 0.521, 0.505, 0.497))
    gf = res_s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.9), Inches(1.75), Inches(7.6), Inches(4.9), cd)
    ch = gf.chart
    ch.has_legend = True
    ch.legend.position = XL_LEGEND_POSITION.TOP
    ch.legend.include_in_layout = False
    ch.legend.font.size = Pt(16)
    for s, col in zip(ch.series, (BLUE, ORANGE)):
        s.format.fill.solid()
        s.format.fill.fore_color.rgb = col
    pl = ch.plots[0]
    pl.gap_width = 60
    pl.has_data_labels = True
    dl = pl.data_labels
    dl.number_format, dl.number_format_is_linked = "0.000", False
    dl.position = XL_LABEL_POSITION.OUTSIDE_END
    dl.font.size = Pt(13)
    va = ch.value_axis
    va.minimum_scale, va.maximum_scale, va.major_unit = 0.46, 0.56, 0.02
    va.crosses_at = 0.5  # bars start at chance level
    va.tick_labels.font.size = Pt(14)
    va.tick_labels.number_format, va.tick_labels.number_format_is_linked = "0.00", False
    va.major_gridlines.format.line.color.rgb = RGBColor(0xD9, 0xD9, 0xD9)
    va.has_title = True
    va.axis_title.text_frame.text = "Balanced accuracy (chance = 0.5)"
    va.axis_title.text_frame.paragraphs[0].runs[0].font.size = Pt(14)
    ca = ch.category_axis
    ca.tick_labels.font.size = Pt(16)
    from pptx.enum.chart import XL_TICK_LABEL_POSITION
    ca.tick_label_position = XL_TICK_LABEL_POSITION.LOW
    textbox(res_s, 8.9, 2.3, 3.9, 3.8,
            "Best: raw signal\n0.536 balanced accuracy\n0.553 ROC-AUC\n\nAll close to chance\n\nSD across subjects: 0.075",
            size=20)
    add_slide_number(res_s, num_box)
    notes(res_s, "7. Results", [
        "Bars start at 0.5 = chance; above the line is better than guessing.",
        "Raw signal is best with both classifiers: 0.536 with random forest, 0.534 with MLP (best AUC 0.553).",
        "Handcrafted next (0.521 with MLP); GazeMAE just above chance; MOMENT at chance.",
        "Honest part: everything is close to chance, and the spread across participants (SD 0.075) is bigger than the differences.",
        "Full table with all metrics on the backup slide.",
    ])

    # ---- 8. t-SNE ----
    tsne_s = title_only(prs, L_CONTENT)
    set_title(tsne_s, "People, Not Engagement")
    w = 8.9
    tsne_s.shapes.add_picture(str(args.fig2), Inches((13.333 - w) / 2), Inches(1.6), width=Inches(w))
    textbox(tsne_s, 0.9, 1.6 + w * 1420 / 2800 + 0.05, 11.5, 0.45,
            "t-SNE, 5,000 windows. Top: by engagement → no structure.   Bottom: by participant → clusters.",
            size=16, italic=True, color=GREY, align=PP_ALIGN.CENTER)
    add_slide_number(tsne_s, num_box)
    notes(tsne_s, "8. t-SNE", [
        "Same four representations projected to 2-D.",
        "Top row coloured by engagement rating: no region follows engagement in any representation.",
        "Bottom row coloured by participant: clear clusters in raw, weaker in handcrafted and GazeMAE, almost none in MOMENT.",
        "So the strongest signal in gaze is who the person is, not how engaged they are — this is why testing on unseen people matters.",
    ])

    # ---- 9. Conclusion ----
    conc_s = prs.slides.add_slide(L_CONTENT)
    set_title(conc_s, "Conclusion and Discussion")
    fill_body(conc_s.placeholders[1].text_frame, [
        (0, "Raw signal best, handcrafted close"),
        (1, "Frozen GazeMAE and MOMENT did not help"),
        (0, "Engagement across people is hard to predict"),
        (1, "Results near chance, large subject variability"),
        (0, "Practical: start with raw or handcrafted"),
        (0, "Future work"),
        (1, "Fine-tune encoders on office gaze"),
        (1, "Personalisation, longer windows"),
    ])
    add_slide_number(conc_s, num_box)
    notes(conc_s, "9. Conclusion", [
        "Under a fair, subject-disjoint comparison, the simplest input — raw signal — gave the best result; handcrafted close behind.",
        "Frozen pretrained encoders did not help: likely a mismatch — trained on lab eye-tracking or generic time series, not day-long office gaze.",
        "Predicting engagement in new people from 3-s windows in real office work is hard.",
        "Practical advice: start with raw or handcrafted features; don't expect a frozen encoder to help.",
        "Next: fine-tune the encoders, personalise to new users, try longer windows.",
        "Thank you — happy to take questions.",
    ])

    # ---- 10. Backup table ----
    tab_s = title_only(prs, L_CONTENT)
    set_title(tab_s, "Results: All Metrics")
    data = [
        ["", "Bal. acc.", "SD", "Macro-F1", "ROC-AUC", "Acc."],
        ["Majority baseline", ".500", ".000", ".326", ".500", ".524"],
        ["Raw, RF", ".536", ".075", ".464", ".550", ".528"],
        ["Raw, MLP", ".534", ".050", ".472", ".553", ".542"],
        ["Handcrafted, RF", ".508", ".035", ".418", ".527", ".528"],
        ["Handcrafted, MLP", ".521", ".032", ".451", ".535", ".531"],
        ["GazeMAE, RF", ".510", ".027", ".441", ".514", ".540"],
        ["GazeMAE, MLP", ".505", ".031", ".447", ".508", ".532"],
        ["MOMENT, RF", ".501", ".013", ".380", ".504", ".521"],
        ["MOMENT, MLP", ".497", ".010", ".362", ".504", ".507"],
    ]
    best = {(2, 1), (3, 3), (3, 4), (3, 5)}
    tbl = tab_s.shapes.add_table(len(data), 6, Inches(0.9), Inches(1.7), Inches(11.5), Inches(4.4)).table
    tbl.columns[0].width = Inches(3.5)
    for c in range(1, 6):
        tbl.columns[c].width = Inches(1.6)
    for r, row in enumerate(data):
        for c, val in enumerate(row):
            cell = tbl.cell(r, c)
            cell.text = val
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
            for run in p.runs:
                run.font.size = Pt(16)
                run.font.bold = r == 0 or (r, c) in best
    textbox(tab_s, 0.9, 6.25, 11.5, 0.4,
            "LOSO, mean over held-out participants; SD of balanced accuracy across participants. Best per metric in bold.",
            size=14, italic=True, color=GREY)
    add_slide_number(tab_s, num_box)
    notes(tab_s, "10. Backup: all metrics", [
        "Only if asked. Paper Table 1.",
        "GazeMAE RF has good plain accuracy (.540), but plain accuracy partly reflects each person's class balance — that's why balanced accuracy is primary.",
    ])

    # Delete unused 2024 slides last, so new slide part names never collide with kept ones.
    for s in old:
        if s not in (title_s, intro_s, prep_s):
            delete_slide(prs, s)
    set_order(prs, [title_s, intro_s, rep_s, data_s, prep_s, meth_s, res_s, tsne_s, conc_s, tab_s])
    prs.save(str(args.out))

    md = ["# SKUI 2026 talk — what to say", "",
          "Target: about 12 minutes (~1 min per slide) plus questions. Slide 10 is backup.", ""]
    for key in sorted(NOTES, key=lambda k: int(k.split(".")[0])):
        md += [f"## {key}", *[f"- {b}" for b in NOTES[key]], ""]
    args.notes_md.write_text("\n".join(md), encoding="utf-8")
    print("wrote", args.out, "and", args.notes_md)


if __name__ == "__main__":
    main()
