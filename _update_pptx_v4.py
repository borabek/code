# -*- coding: utf-8 -*-
"""Build ConnectionPointDetector_Presentation_v4.pptx from v3.

Keeps slides 1..23 (history + architecture) and the existing branding/theme untouched.
Rewrites the closing section (slides 24..30) with the CURRENT honest numbers, and appends
new slides for the findings made since 27.07: two regimes, the measurement-discipline
lessons, the deployment gap, and the robot-only view.

Every number here comes from a receipt in results/ -- see NUMBERS below for the source of each.
Language: English only. Style: plain logic, no jargon inflation.
"""
import copy, os
from pptx import Presentation
from pptx.util import Emu

SRC = "ConnectionPointDetector_Presentation_v3.pptx"
OUT = "ConnectionPointDetector_Presentation_v4.pptx"
DATE = "WiringRobot.ConnectionPointDetector  |  Friedhelm Loh Group  R&D RAS  |  29.07.2026"

# --- where each number comes from ---------------------------------------------------------------
# seg 0.858 / 0.680      results/scheffler_seg (locked 11-part test, opened 17.07)
# CP-F1 0.760 weighted   36-part running-pipeline sample, regime-weighted at 10.5% high-CP
# low 0.775 / high 0.626 same sample, split by manufacturer CP count
# WORK OOF 0.828         development frame (GroupKFold by family)
# holdout 0.7601         results/ locked holdout, single shot, SPENT
# gate nested            results/pitstop2_gate_nested.json (29.07)
# ceiling 0.95           oracle-gate on current candidates (not reachable from geometry)

TITLE, SUB, BODY, FOOT = "TextBox 1", "TextBox 3", "TextBox 4", "TextBox 5"


def dup_slide(prs, index):
    source = prs.slides[index]
    new = prs.slides.add_slide(source.slide_layout)
    for shp in list(new.shapes):
        shp._element.getparent().remove(shp._element)
    for shp in source.shapes:
        new.shapes._spTree.append(copy.deepcopy(shp._element))
    return new


def by_name(slide, name):
    for sh in slide.shapes:
        if sh.name == name:
            return sh
    return None


def set_line(shape, text):
    """Replace a single-line shape's text, keeping the first run's formatting."""
    tf = shape.text_frame
    p = tf.paragraphs[0]
    if p.runs:
        p.runs[0].text = text
        for r in p.runs[1:]:
            r._r.getparent().remove(r._r)
    else:
        p.add_run().text = text
    for extra in list(tf.paragraphs[1:]):
        extra._p.getparent().remove(extra._p)


def set_bullets(shape, items, marker="*  "):
    """Rewrite a bullet body. items = list of strings, or (marker, text) pairs.

    Formatting is cloned from the FIRST bullet already on the slide, so the deck's
    colours/sizes survive: bullet marker run (bold, accent) + text run (grey).
    """
    tf = shape.text_frame
    src_p = tf.paragraphs[0]
    runs = src_p.runs
    mark_r = copy.deepcopy(runs[0]._r) if runs else None
    text_r = copy.deepcopy(runs[1]._r) if len(runs) > 1 else copy.deepcopy(runs[0]._r)
    src_pPr = copy.deepcopy(src_p._p.find(
        "{http://schemas.openxmlformats.org/drawingml/2006/main}pPr"))

    for p in list(tf.paragraphs):
        p._p.getparent().remove(p._p)

    for it in items:
        mk, txt = it if isinstance(it, tuple) else (marker, it)
        p = tf.add_paragraph()
        if src_pPr is not None:
            p._p.insert(0, copy.deepcopy(src_pPr))
        if mark_r is not None:
            r = copy.deepcopy(mark_r)
            r.find("{http://schemas.openxmlformats.org/drawingml/2006/main}t").text = mk
            p._p.append(r)
        r2 = copy.deepcopy(text_r)
        r2.find("{http://schemas.openxmlformats.org/drawingml/2006/main}t").text = txt
        p._p.append(r2)


def write(slide, title, sub, bullets, marker="*  "):
    set_line(by_name(slide, TITLE), title)
    set_line(by_name(slide, SUB), sub)
    set_bullets(by_name(slide, BODY), bullets, marker)
    set_line(by_name(slide, FOOT), DATE)


# ------------------------------------------------------------------------------------------------
prs = Presentation(SRC)
TEMPLATE = 23                      # slide 24 -- the layout every closing slide reuses

CONTENT = [
    # ---- 24  Current status -------------------------------------------------------------------
    ("9  Current Status (Jul 2026)",
     "Where the System Stands Today",
     ["Task is semantic segmentation of the terminal surface, not keypoint regression -- "
      "this pivot is what made the results defensible",
      "Pipeline: STEP -> isotropic remesh (6000 vertices) -> 4-model DiffusionNet ensemble -> "
      "opening extraction -> wire/tool gate -> robot-ready connection points",
      "Segmentation (the thesis's own metric): accuracy 0.858, mean IoU 0.680 vs the reference "
      "thesis 0.514 -- the benchmark goal is met and exceeded",
      "Connection points, running pipeline, corpus-weighted: CP-F1 0.760",
      "Robot contract unchanged: entry point + approach vector + insertion axis + depth",
      "All numbers below are leakage-free (GroupKFold by product family, out-of-fold only)"]),

    # ---- 25  Three frames ---------------------------------------------------------------------
    ("10  How to Read Our Numbers",
     "One Metric, Three Frames -- Always Stated Explicitly",
     ["Development (WORK, out-of-fold): CP-F1 0.828 -- used to compare ideas, NOT a product claim",
      "Running pipeline (what the robot actually executes today): CP-F1 0.760",
      "Locked holdout (single shot, never reused): CP-F1 0.760 -- this is the product number",
      "Development measurement runs about 3x more optimistic than the holdout. We learned this "
      "the hard way: a change that gained +0.012 in development lost 0.029 on the holdout",
      "Rule we now follow: quote the frame with every number, and never iterate on the holdout"]),

    # ---- 26  Two regimes ----------------------------------------------------------------------
    ("11  The Two Regimes",
     "One Average Was Hiding Two Different Problems",
     [("*  ", "Terminals split by how many connection points they have, and the two halves fail "
              "for OPPOSITE reasons"),
      ("*  ", "Low-CP parts (under 8 CPs) = 89.5% of the catalog: CP-F1 0.775, limited by "
              "PRECISION -- the model finds openings that are not wire entries"),
      ("*  ", "High-CP parts (8 or more) = 10.5%: CP-F1 0.626, limited by RECALL -- the openings "
              "are too small to survive the mesh at a fixed vertex budget"),
      ("*  ", "Root cause on the high-CP side is arithmetic, not intelligence: 82.7% of their "
              "openings hold fewer than 30 vertices, and the filter required 30"),
      ("*  ", "Reporting rule that came out of this: never quote a flat average over a skewed "
              "sample -- weight by the real catalog mix, or the headline is wrong by 0.10")]),

    # ---- 27  What worked ----------------------------------------------------------------------
    ("12  What Worked",
     "Solid, Defensible Results",
     [("OK  ", "Segmentation beats the reference thesis on its own metric: 0.680 vs 0.514 mean IoU"),
      ("OK  ", "All five classes reach the thesis's own production bar (IoU >= 0.50), which the "
               "reference itself missed on 3 of 5 classes"),
      ("OK  ", "Uniform isotropic remeshing closed the CAD-vs-scan domain gap -- the single "
               "change that made training transfer at all"),
      ("OK  ", "Wire/tool gate (random forest on structural features): precision 0.61 -> 0.75"),
      ("OK  ", "Growing the corpus 914 -> 1753 parts: +0.011 CP-F1. The data lever is alive, "
               "but the curve is clearly flattening"),
      ("OK  ", "Every drawing the robot produces is verified by an INDEPENDENT auditor that "
               "re-reads the file from disk; non-compliant output is deleted, not shipped")]),

    # ---- 28  What did not work ------------------------------------------------------------------
    ("13  What Did Not Work",
     "Dead Ends We Ruled Out by Measurement, Not by Opinion",
     [("X  ", "CAD pseudo-labels as the training target -- they agree with human ground truth at "
              "only F1 0.46. A data ceiling; no model or tuning closes it"),
      ("X  ", "Pure geometry as a stand-alone detector -- best F1 0.400 against 0.82 for the ML "
              "branch. It can FIND openings (recall 0.86) but cannot say which one takes a wire"),
      ("X  ", "Human labelling as the next lever -- a blind test showed the MODEL beats the human "
              "annotator (86.5% vs 79.5% on the same 156 openings)"),
      ("X  ", "Threshold tuning -- now formally exhausted: with family-out nested selection the "
              "deployed thresholds are already optimal (gain +0.000 on high-CP)"),
      ("X  ", "Ten further levers measured and killed, each against a kill criterion written "
               "BEFORE the measurement: B-rep features, channel profiling, finer tessellation, "
               "hole-preserving remesh, adaptive density, high-CP fine-tuning, and others")]),

    # ---- 29  Bugs -------------------------------------------------------------------------------
    ("14  Bugs That Cost Us the Most",
     "All Four Had the Same Signature: Failing Silently",
     [("!  ", "A missing geometry package made every ray/containment call throw -- and a broad "
              "'except: continue' swallowed it. Two geometry helpers returned their input "
              "unchanged and looked successful. Replaced with our own ray intersection code"),
      ("!  ", "The visualisation's self-check used that same broken call and printed '8/8 passed'. "
              "Verification cannot live inside the code it verifies"),
      ("!  ", "A mesher handle was not released on failure, so ONE unmeshable file made every "
              "later part crawl: 3 hours of zero progress with no error message"),
      ("!  ", "Remeshing was non-deterministic ACROSS PROCESSES. Our determinism test ran three "
              "times in ONE process and therefore tested nothing"),
      ("!  ", "Six measurement artifacts came from our own diagnostic code, never from the data. "
              "Standing rule: when production and diagnostics disagree, suspect the diagnostics")]),

    # ---- 30  Ceilings ---------------------------------------------------------------------------
    ("15  The Ceilings -- What Actually Limits Us",
     "Measured Limits, Not Guesses",
     ["Wire entry vs tool opening is a FUNCTIONAL distinction, not a geometric one. The reference "
      "thesis merges them into one class by design, so geometry cannot separate them (AUC 0.61)",
      "Information wall: letting the model see the very parts it is scored on gains only +0.003, "
      "while +0.151 remains to the candidate ceiling. The answer is not in what we show the model",
      "Perfect-gate ceiling is 0.894-0.95, but the surviving false positives are real openings "
      "that simply do not take a wire -- geometrically identical to true ones",
      "Practical ceiling for the low-CP majority is therefore reached; the honest remaining lever "
      "is the high-CP minority and more DATA, not more model",
      "The 0.85 target was our own, not the thesis's. The thesis benchmark (0.514 mean IoU) is met "
      "and exceeded; CP-F1 is now a product improvement axis"]),
]

for i, (t, s, b) in enumerate(CONTENT):
    idx = 23 + i
    if idx < len(prs.slides._sldIdLst):
        sl = prs.slides[idx]
    else:
        sl = dup_slide(prs, TEMPLATE)
    marker = "*  "
    if isinstance(b[0], tuple):
        write(sl, t, s, b)
    else:
        write(sl, t, s, b, marker)

# ---- new slides appended after the rewritten block ---------------------------------------------
NEW = [
    ("16  Deployment Gap -- Measured vs Delivered",
     "The Robot Was Not Getting What We Had Already Measured",
     [("!  ", "A config-only 'product change' turned out to be a lie: the robot code never read "
              "the new setting. It was measured, recorded, and never actually deployed"),
      ("!  ", "A routing component was trained on one input and fed a different one at runtime, "
              "so it stayed switched OFF exactly on the parts that needed it"),
      ("OK  ", "After wiring both correctly: high-CP CP-F1 0.515 -> 0.626, low-CP untouched at "
               "0.775, corpus-weighted 0.748 -> 0.760"),
      ("OK  ", "Worst affected part went from CP-F1 0.273 to 0.737 with no model change at all"),
      ("*  ", "Lesson: a measured improvement is not a delivered improvement. We now verify the "
              "running pipeline, not the configuration file")]),

    ("17  What the Robot Sees",
     "Robot-Only View Is Now the Default Output",
     ["On a new, unseen terminal there IS no manufacturer reference -- so the default drawing "
      "shows only the connection points the robot itself placed, in one colour",
      "The comparison view (manufacturer vs robot, colour-coded hit/miss) is still available, "
      "but it is a diagnostic tool for us, not the product output",
      "Each connection point is drawn as a needle that passes through the body and protrudes on "
      "both sides, so it stays visible on 8 mm thin terminals in any viewer",
      "Every file is checked against a 9-rule drawing contract by an independent auditor that "
      "re-reads it from disk; a file that fails is deleted rather than shown",
      "The auditor is itself mutation-tested: we corrupt the record on purpose and confirm it "
      "still refuses to pass (7 of 7 caught)"]),

    ("18  Next Steps",
     "Where the Remaining Value Is",
     ["More data, targeted at the high-CP minority -- the only lever still measurably alive",
      "Adaptive vertex budget so small openings survive the mesh (the 30-vertex arithmetic problem)",
      "A wire-specific signal to separate wire entries from tool openings -- this is the one "
      "thing that would move the precision ceiling, and geometry cannot supply it",
      "Retrain the wire/tool gate on the grown corpus: it is still trained on 823 parts while "
      "the corpus now holds 1753",
      "Keep the discipline: kill criteria written before the measurement, regime-split reporting, "
      "and the holdout spent exactly once"]),
]

for t, s, b in NEW:
    sl = dup_slide(prs, TEMPLATE)
    write(sl, t, s, b)

# refresh the footer date on every slide so the deck is internally consistent
for sl in prs.slides:
    f = by_name(sl, FOOT)
    if f is not None and f.has_text_frame and "ConnectionPointDetector" in f.text_frame.text:
        set_line(f, DATE)

prs.save(OUT)
print(f"{OUT} written -- {len(prs.slides)} slides")
