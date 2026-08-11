# -*- coding: utf-8 -*-
"""Update the ConnectionPointDetector deck: keep slides 1-16 as the June history,
append a new 'Latest Developments (Jul 2026)' section, and refresh the summary.
All numbers are honest / work-in-progress. Saves to a NEW file (original untouched)."""
import copy
from pptx import Presentation

SRC = "ConnectionPointDetector_Presentation_NEW.pptx"
OUT = "ConnectionPointDetector_Presentation_v2.pptx"
DATE = "R&D RAS  |  15.07.2026"

prs = Presentation(SRC)


def dup_slide(prs, index):
    """Deep-copy an existing slide (preserves theme, footer, layout)."""
    source = prs.slides[index]
    new = prs.slides.add_slide(source.slide_layout)
    for shp in list(new.shapes):            # strip layout placeholders
        shp._element.getparent().remove(shp._element)
    for shp in source.shapes:
        new.shapes._spTree.append(copy.deepcopy(shp._element))
    return new


def set_text(shape, text):
    """Set a shape's text keeping the first run's formatting."""
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


def by_name(slide, name):
    for sh in slide.shapes:
        if sh.name == name:
            return sh
    return None


def fill(slide, section, title, left_head, left_bul, right_head, right_bul, page):
    """Fill a slide duplicated from slide 17 (Summary two-column layout)."""
    set_text(by_name(slide, "TextBox 1"), section)
    set_text(by_name(slide, "TextBox 2"), title)
    set_text(by_name(slide, "TextBox 4"), left_head)
    # left bullets live in TextBox 5..11 (7 boxes)
    for i, name in enumerate([f"TextBox {n}" for n in range(5, 12)]):
        sh = by_name(slide, name)
        if sh is None:
            continue
        set_text(sh, ("▪  " + left_bul[i]) if i < len(left_bul) else "")
    set_text(by_name(slide, "TextBox 14"), right_head)
    # right bullets live in TextBox 15..20 (6 boxes)
    for i, name in enumerate([f"TextBox {n}" for n in range(15, 21)]):
        sh = by_name(slide, name)
        if sh is None:
            continue
        set_text(sh, ("▶  " + right_bul[i]) if i < len(right_bul) else "")
    # footer date + page number
    d = by_name(slide, "TextBox 28")
    if d:
        set_text(d, DATE)
    pg = by_name(slide, "TextBox 29")
    if pg:
        set_text(pg, str(page))


SEC = "7  Latest Developments"

# ---- the six new slides (each duplicated from slide 17) --------------------------
NEW = [
    dict(section=SEC, title="Architecture Pivot — Jun → Jul 2026",
         lh="Earlier approach (Jun)",
         lb=["Render 6 orthographic views of the mesh",
             "YOLOv6 2D detection → 3D crop recovery",
             "DiffusionNet spectral segmentation",
             "Required WSL / Linux GPU",
             "Learned from rendered 2D images"],
         rh="Current approach (Jul)",
         rb=["STEP B-rep → gmsh tessellation",
             "CAD-direct opening extraction = labels",
             "HierPoint per-vertex 3D regression",
             "Native Windows GPU — no WSL",
             "Learns from 3D geometry directly"]),
    dict(section=SEC, title="Current Pipeline — CAD-Direct + HierPoint",
         lh="Pipeline stages",
         lb=["STEP file → gmsh tessellation (0.5 mm)",
             "step_openings.py: cylinder/opening → pseudo-labels",
             "HierPoint: FPS pooling + EdgeConv, 7-ch head",
             "Gaussian-vote decode → connection points",
             "Robot-ready JSON (entry_point, approach_vector …)"],
         rh="Scale engineering",
         rb=["Spatial patching of large meshes (full density)",
             "Disk-cached pooling hierarchies (lazy load)",
             "Source-free training — fits a 31 GB box",
             "int32 kNN graphs, chunked forward pass",
             "Single laptop GPU (NVIDIA T1200, 4 GB)"]),
    dict(section=SEC, title="WSCAD Training Corpus",
         lh="Corpus (v8)",
         lb=["3033 terminal-block parts (WSCAD-Universe STEP)",
             "CAD-direct labels: 25,529 connection points",
             "Geometry-grouped split — no train/val leakage",
             "12,556 spatial-patch training graphs"],
         rh="Evaluation protocol",
         rb=["Metric: Jaccard accuracy = TP / (TP+FP+FN)",
             "Frozen val (278) / held-out test (284)",
             "Scored vs CAD labels — no human-GT, no leakage",
             "5 mm match tolerance, single-shot test"]),
    dict(section=SEC, title="Results — Convergence (Work in Progress)",
         lh="v31 convergence on WSCAD",
         lb=["F1: ep5 0.706 · ep15 0.771 · ep25 0.804 — climbing",
             "Full-coverage Jaccard 0.55 → 0.67, target ~0.74",
             "Reference: v28 milestone F1 0.898 (earlier corpus)",
             "Training ongoing — not a final number"],
         rh="Post-training levers (no retrain)",
         rb=["Octahedral TTA (axis-preserving)   +0.01–0.03",
             "Snapshot ensemble (multi-epoch)    +0.01–0.02",
             "Decode calibration (threshold / NMS)",
             "Locked operating point → single-shot test"]),
    dict(section=SEC, title="Selective Prediction — Automation vs Coverage",
         lh="Concept",
         lb=["Not every part is equally certain",
             "Per-part risk score → abstain on hard parts",
             "Accepted → automated; rest → human review",
             "Report accuracy + coverage + automation yield"],
         rh="Measured (v31, in progress)",
         rb=["≥ 0.90 Jaccard on least-risky ~40–50 % of parts",
             "Oracle upper bound: 0.90 at ~50 % coverage",
             "Roadmap: better risk score + convergence → 55–65 %",
             "Honest rule: < 50 % ⇒ human-review routing"]),
    dict(section=SEC, title="Key Finding — Ground Truth Sets the Ceiling",
         lh="What we measured",
         lb=["CAD pseudo-labels vs 9-part human GT: F1 ≈ 0.46",
             "Labels over-mark (mounting/screw/vent holes)",
             "… also under-mark and mis-locate a mid-bore",
             "The model already reproduces CAD labels well"],
         rh="What it means (de-risks the roadmap)",
         rb=["Human-aligned 0.90 needs human labels, not more tuning",
             "Next lever = label quality, not architecture",
             "Plan: 100–150 human-labeled parts as true benchmark",
             "Full-coverage vs selective reported separately"]),
]

# duplicate slide 17 for each new slide; page numbers 18..23
for k, spec in enumerate(NEW):
    s = dup_slide(prs, 16)
    fill(s, spec["section"], spec["title"], spec["lh"], spec["lb"],
         spec["rh"], spec["rb"], 18 + k)

# ---- refresh the original Summary (slide 17) and move it to the very end ---------
summ = prs.slides[16]
fill(summ,
     "8  Summary",
     "Status & Roadmap",
     "Achieved (Jun → Jul 2026)",
     ["CAD-direct labelling: STEP → gmsh → openings",
      "HierPoint backbone on native Windows GPU",
      "WSCAD corpus: 3033 parts, leakage-free split",
      "Honest metric: Jaccard vs CAD labels, no human-GT",
      "Selective prediction + TTA/ensemble harness",
      "54 automated tests green across the toolchain"],
     "Roadmap (next)",
     ["Finish v31 convergence + TTA + snapshot ensemble",
      "Improved selective risk score → higher coverage",
      "Human-labeled benchmark → path to aligned 0.90",
      "v32 regularization run (octahedral augment + WD)",
      "Locked single-shot test, then robot integration"],
     24)

# move the summary slide element to the end of the deck
sldIdLst = prs.slides._sldIdLst
ids = list(sldIdLst)
sldIdLst.remove(ids[16])
sldIdLst.append(ids[16])

# ---- agenda (slide 2): relabel item 6 to include the new section ----------------
ag = prs.slides[1]
lbl = by_name(ag, "Rectangle 13")     # "Summary"
if lbl:
    set_text(lbl, "Summary  +  Latest Developments (Jul)")

prs.save(OUT)
print("saved:", OUT, "| slides:", len(prs.slides.__iter__.__self__._sldIdLst))
