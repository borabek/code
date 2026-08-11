# -*- coding: utf-8 -*-
"""Slide copy for the Bora Bayrakci deck.

VOICE: an engineer showing his own work to a buyer. First person, short sentences, plain
words. No consultant phrasing ("buys you", "on the table", "due diligence") and no long
dash-heavy clauses -- that cadence is what made the first draft read as machine-written.

HONESTY RULE: the headline segmentation figure is the SHIPPED model (63.3%), not the best
model ever trained (68.1%). Both are shown. A buyer who asks "is that the model you ship?"
must get a yes.
"""
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN


def build(ctx):
    new_slide = ctx["new_slide"]; txt = ctx["txt"]; bullets = ctx["bullets"]; table = ctx["table"]
    NAVY, CYAN, GREY, DARK, LIGHT = ctx["NAVY"], ctx["CYAN"], ctx["GREY"], ctx["DARK"], ctx["LIGHT"]
    GREEN, RED, AUTHOR = ctx["GREEN"], ctx["RED"], ctx["AUTHOR"]

    # ---- 1  TITLE ------------------------------------------------------------------------------
    s = new_slide(dark_bg=True)
    txt(s, 1.00, 2.20, 11.30, 0.85, "WiringRobot.ConnectionPointDetector", 40, NAVY, True)
    bar = s.shapes.add_shape(1, Inches(1.00), Inches(3.15), Inches(11.30), Inches(0.04))
    bar.fill.solid(); bar.fill.fore_color.rgb = CYAN
    bar.line.fill.background(); bar.shadow.inherit = False
    txt(s, 1.00, 3.27, 11.30, 0.52,
        "Finding wire entry points on terminal blocks, straight from the CAD file", 18, GREY)
    txt(s, 1.00, 4.05, 11.30, 0.40, AUTHOR, 16, DARK, True)
    txt(s, 1.00, 4.40, 11.30, 0.35, "R&D RAS  |  Friedhelm Loh Group  |  29 July 2026", 12, LIGHT)

    # ---- 2  THE PRODUCT ------------------------------------------------------------------------
    s = new_slide("1  The Product", "What It Does")
    txt(s, 0.60, 1.50, 5.80, 0.35, "The problem", 15, RED, True)
    bullets(s, 0.60, 1.90, 5.80, 2.4, [
        ("", "A wiring robot has to know where to push the wire in."),
        ("", "Today someone teaches every terminal type by hand."),
        ("", "There are thousands of types in the catalogue. That does not scale."),
    ], size=14, gap=9)
    txt(s, 6.90, 1.50, 5.80, 0.35, "What my system does", 15, GREEN, True)
    bullets(s, 6.90, 1.90, 5.80, 2.4, [
        ("", "You give it one CAD file. Nothing else."),
        ("", "It returns every wire opening: where it is, which way it points, how deep."),
        ("", "It takes seconds, and it gives the same answer every time."),
    ], size=14, gap=9)
    txt(s, 0.60, 4.45, 12.10, 0.35, "Why a simple CAD script cannot do this", 15, NAVY, True)
    bullets(s, 0.60, 4.85, 12.10, 1.6, [
        ("", "A wire opening and a screwdriver opening have the same shape. Only their purpose "
             "differs, and shape alone cannot tell you the purpose."),
        ("", "The manufacturer marks the contact seat, which sits 5 to 25 mm inside the part. "
             "The robot needs the mouth on the surface. My system converts one to the other."),
    ], size=14, gap=9)

    # ---- 3  PIPELINE ---------------------------------------------------------------------------
    s = new_slide("2  How It Works", "The Pipeline")
    steps = [("STEP file", "CAD input"), ("Tessellation", "gmsh, 0.5 mm"),
             ("Remesh", "6 000 vertices"), ("Segmentation", "4 x DiffusionNet"),
             ("Find openings", "connected components"), ("Wire / tool filter", "random forest"),
             ("Robot output", "point + axis + depth")]
    x = 0.42
    for i, (a, b) in enumerate(steps):
        box = s.shapes.add_shape(5, Inches(x), Inches(2.15), Inches(1.62), Inches(1.05))
        box.fill.solid()
        box.fill.fore_color.rgb = NAVY if i in (3, 5) else RGBColor(0xED, 0xF3, 0xF8)
        box.line.color.rgb = NAVY; box.shadow.inherit = False
        tf = box.text_frame; tf.word_wrap = True
        p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
        r = p.add_run(); r.text = a
        r.font.size = Pt(11); r.font.bold = True; r.font.name = "Calibri"
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF) if i in (3, 5) else NAVY
        p2 = tf.add_paragraph(); p2.alignment = PP_ALIGN.CENTER
        r2 = p2.add_run(); r2.text = b
        r2.font.size = Pt(9); r2.font.name = "Calibri"
        r2.font.color.rgb = RGBColor(0xE8, 0xF2, 0xF8) if i in (3, 5) else GREY
        if i < len(steps) - 1:
            txt(s, x + 1.63, 2.47, 0.24, 0.4, ">", 12, CYAN, True)
        x += 1.86
    txt(s, 0.60, 3.55, 12.10, 0.35, "Blue boxes are learned. The rest is fixed geometry.",
        15, NAVY, True)
    bullets(s, 0.60, 3.98, 12.10, 2.3, [
        ("Remesh:", " I rebuild every CAD surface at the same point density the network was "
                    "trained on. Before I did this, the model did not work on real CAD files at "
                    "all. This one step is what made it work."),
        ("Wire / tool filter:", " the network can find an opening but cannot say whether a wire "
                                "or a screwdriver goes in. So I trained a second, small model "
                                "just for that. It raised precision from 61% to 75%."),
        ("Everything else is deterministic:", " same file in, same coordinates out."),
    ], size=14, gap=10)

    # ---- 4  MODELS -----------------------------------------------------------------------------
    s = new_slide("2  How It Works", "The Models I Used")
    # Satirlar sikisikti: satir yuksekligi 0.36 -> 0.46 ve hucre metinleri kisaltildi.
    y = table(s, 0.60, 1.52, 12.10, [
        ["Component", "Method", "Function", "Training data"],
        ["Segmentation", "DiffusionNet", "labels every surface vertex", "102 labelled parts"],
        ["Ensemble", "4 models, combined", "recovers openings one model misses", "4 training runs"],
        ["Opening extraction", "connected components", "groups vertices into one opening", "none"],
        ["Insertion axis", "ray probing", "gives the robot its approach vector", "none"],
        ["Wire / tool gate", "Random Forest", "removes openings that take no wire", "1 041 parts"],
        ["Regime router", "Random Forest", "selects settings per part", "1 906 parts"],
    ], [2.55, 2.55, 4.60, 2.40], row_h=0.46)
    txt(s, 0.60, y + 0.28, 12.10, 0.35, "Two different kinds of training data", 15, NAVY, True)
    bullets(s, 0.60, y + 0.70, 12.10, 1.3, [
        ("102 labelled parts are hand-painted vertex by vertex.", " That is the only way to train "
         "this kind of network, and it costs hours per part."),
        ("1 906 catalogue parts are point lists from the manufacturer.", " They train the gate and "
         "carry every score in this deck."),
    ], size=14, gap=9)

    # ---- 5  GLOSSARY ---------------------------------------------------------------------------
    s = new_slide("3  Terms", "The Words I Use in This Deck")
    table(s, 0.60, 1.50, 6.10, [
        ["Term", "Definition"],
        ["Connection point (CP)", "one wire opening: position + direction"],
        ["Mesh / vertex", "the 3D surface as triangles; a vertex is a corner"],
        ["Segmentation", "labelling every vertex with what it is"],
        ["Remeshing", "rebuilding the surface at a chosen point density"],
        ["Ensemble", "several models voting together"],
    ], [2.60, 3.50], size=13, row_h=0.42)
    table(s, 7.10, 1.50, 5.60, [
        ["Metric", "Definition"],
        ["Precision", "of the openings I report, how many are real"],
        ["Recall", "of the real openings, how many I find"],
        ["F1", "precision and recall combined into one score"],
        ["IoU", "how well my labels overlap the correct ones"],
        ["Dice", "same idea as IoU, scored a little more softly"],
    ], [2.10, 3.50], size=13, row_h=0.42)
    txt(s, 0.60, 4.85, 12.10, 0.35, "Two words about how I test", 15, NAVY, True)
    bullets(s, 0.60, 5.25, 12.10, 1.4, [
        ("Out-of-fold:", " a part is only ever scored by a model that never saw it in training."),
        ("Hold-out:", " a set of parts I locked away before training and opened once, at the end. "
                      "I did not tune anything against it."),
    ], size=14, gap=8)

    # ---- 11  VS MASTER THESIS ------------------------------------------------------------------
    s = new_slide("4  Comparison", "Compared to the Master Thesis")
    table(s, 0.60, 1.45, 12.10, [
        ["", "Master thesis", "My system"],
        ["Mean IoU (my shipped model)", "51.4%", "63.3%"],
        ["Classes over the 50% usable bar", "not reported per class", "4 of 5   (best model: 5 of 5)"],
        ["Training data", "synthetic plus a little real", "1 906 real catalogue parts"],
        ["Testing", "one split", "out-of-fold, families kept apart, locked hold-out"],
        ["What comes out", "a labelled surface", "point, axis, approach direction, depth"],
        ["Connection-point F1", "not measured", "76.0%"],
    ], [3.30, 4.40, 4.40])
    txt(s, 0.60, 4.60, 12.10, 0.35, "How I got from the thesis result to mine", 15, NAVY, True)
    bullets(s, 0.60, 5.00, 12.10, 1.7, [
        ("1.", " I rebuilt every mesh at one fixed density. The thesis trained and tested on "
               "similar meshes; real CAD files look different, and that gap was the whole problem."),
        ("2.", " I replaced CAD-derived labels with real catalogue data. CAD labels only agree "
               "with a human 46% of the time, so they capped the result."),
        ("3.", " I grew the data from 914 to 1 753 parts and added the wire / tool filter, which "
               "the thesis does not have at all."),
    ], size=14, gap=8)

    # ---- 7  THESIS GAPS AS A PRODUCT ------------------------------------------------------------
    s = new_slide("4  Comparison", "What the Thesis Does Not Solve for a Product")
    bullets(s, 0.60, 1.50, 12.10, 5.1, [
        ("No coordinates for the robot.", " The thesis outputs a labelled surface. A robot cannot "
         "drive to a colour. I derive the opening mouth, the insertion axis, the approach vector "
         "and the depth, so the output is an instruction the robot can execute."),
        ("Wire and tool openings are one class by design.", " The thesis class Contact covers both "
         "the contact and the screwdriver slot. A robot following it would push wire into a tool "
         "opening. I added a separate classifier for exactly this decision: precision 60.7% to 75.1%."),
        ("The model does not survive real CAD files.", " Training and test meshes in the thesis "
         "look alike; production STEP files do not. I rebuild every surface at one fixed vertex "
         "density before inference, and that is what made the model transfer at all."),
        ("One split can hide a leak.", " Terminal variants differ by millimetres, so a sibling part "
         "on the other side of a random split inflates the score. I group by product family, score "
         "out-of-fold, and keep a hold-out that was opened once."),
        ("The product metric is never measured.", " The thesis reports IoU only, which says how "
         "well the surface is labelled, not whether the openings are found. I measure CP-F1 "
         "against manufacturer data on 1 906 parts: 76.0%."),
    ], size=14, gap=13)
    # ---- 6  PARAMETERS -------------------------------------------------------------------------
    s = new_slide("5  Settings", "The Settings I Run With")
    table(s, 0.60, 1.60, 5.70, [
        ["Geometry parameter", "Value"],
        ["Tessellation deflection", "0.5 mm"],
        ["Vertices per part after remesh", "6 000"],
        ["Spectral basis (k_eig)", "96"],
        ["Minimum opening size", "10 vertices"],
        ["Vertex confidence mask", "30%"],
        ["Terminal merge radius", "3.0 mm"],
    ], [3.60, 2.10])
    table(s, 6.90, 1.60, 5.80, [
        ["Decision parameter", "Value"],
        ["Ensemble members", "4"],
        ["Votes required to keep a candidate", "1"],
        ["Gate threshold, low CP density", "35%"],
        ["Gate threshold, high CP density", "25%"],
        ["Auto-accept confidence", "50%"],
        ["Matching tolerance", "6% of part diagonal"],
    ], [3.70, 2.10])
    bullets(s, 0.60, 5.20, 12.10, 1.4, [
        ("I did not pick these thresholds by eye.", " I searched them with the part families kept "
         "apart, so the search could not cheat. It landed on the value already in the product in "
         "5 out of 5 folds."),
    ], size=14)

    # ---- 7  RESULTS: SEGMENTATION --------------------------------------------------------------
    s = new_slide("6  Results", "Segmentation: How Well It Labels the Surface")
    table(s, 0.60, 1.42, 12.10, [
        ["Mean IoU  (metric used by the master thesis)", "Score", "Difference"],
        ["Master thesis", "51.4%", "-"],
        ["My model, the one in the product", "63.3%", "+11.9 points"],
        ["My best model, not shipped yet", "68.1%", "+16.7 points"],
    ], [5.60, 3.00, 3.50])
    # SINIF KIRILIMI: "5 of 5" iddiasi URUN modeli icin YANLIStI (SnapPoint %45.4). Dogrusu 4/5,
    # ve barajin altinda kalan tek sinif kablolamayla ilgisi olmayan sinif -- bunu acikca yaziyorum.
    table(s, 0.60, 3.05, 6.10, [
        ["Class (product model)", "IoU", "Above 50%"],
        ["Housing  (the body)", "77.3%", "yes"],
        ["CableEntry  (wire opening)", "68.8%", "yes"],
        ["Contact  (metal contact)", "64.6%", "yes"],
        ["LabelSurface  (label area)", "64.2%", "yes"],
        ["SnapPoint  (rail clip)", "45.4%", "no"],
    ], [3.20, 1.40, 1.50], size=13, row_h=0.33)
    txt(s, 7.10, 3.05, 5.60, 0.33, "What the 50% line means", 14, NAVY, True)
    bullets(s, 7.10, 3.42, 5.60, 2.4, [
        ("", "50% IoU is the bar the thesis itself calls usable: above it, the labelled area "
             "matches the true area more than it misses it."),
        ("", "4 of my 5 classes clear it. The one that does not is SnapPoint, the clip that "
             "holds the terminal on the rail. It has nothing to do with finding wire openings."),
        ("", "CableEntry, the class the robot actually depends on, sits at 68.8%."),
    ], size=13, gap=7)
    txt(s, 0.60, 5.35, 6.10, 0.33, "Accuracy 83.1%   ·   Dice 74.6%", 14, DARK, True)
    bullets(s, 0.60, 5.72, 6.10, 1.0, [
        ("", "Same classes and same metric as the thesis. I quote the model I ship, not my best one."),
    ], size=13, gap=6)

    # ---- 8  RESULTS: CONNECTION POINTS ---------------------------------------------------------
    s = new_slide("6  Results", "Connection Points: How Well It Finds the Openings")
    table(s, 0.60, 1.50, 12.10, [
        ["Evaluation frame", "CP-F1", "Definition"],
        ["The running system", "76.0%", "what the robot gets today"],
        ["Locked hold-out, opened once", "76.0%", "a clean second confirmation"],
        ["My development number", "82.8%", "internal only, I do not quote it as the result"],
        ["Ceiling with a perfect filter", "89.4%", "the best this design could ever reach"],
    ], [4.30, 2.00, 5.80])
    txt(s, 0.60, 4.05, 12.10, 0.35, "What the wire / tool filter is worth", 15, NAVY, True)
    bullets(s, 0.60, 4.45, 12.10, 1.9, [
        ("Precision went from 60.7% to 75.1%", " and recall barely moved (68.4% to 65.1%). "
         "So three out of four picks are now correct instead of three out of five."),
        ("I report the hold-out number, not the development one.", " Development numbers in this "
         "field run about three times too optimistic, and mine did too."),
    ], size=14, gap=9)

    # ---- 9  TWO REGIMES ------------------------------------------------------------------------
    s = new_slide("6  Results", "Where It Works Well, and Where It Does Not")
    # Rejim adlari INSAN diliyle: "low-CP / high-CP" ic terminolojidir, aliciya bir sey anlatmaz.
    table(s, 0.60, 1.60, 12.10, [
        ["", "Low CP density\nfewer than 8 connection points",
         "High CP density\n8 connection points or more"],
        ["Share of catalogue", "89.5%", "10.5%"],
        ["Representative types", "single- and double-level feed-through",
         "distribution and marshalling blocks"],
        ["CP-F1", "77.5%", "62.6%"],
        ["Limiting factor", "precision: openings that carry no wire",
         "recall: openings below the size threshold"],
        ["Operating mode", "automatic", "automatic with review"],
    ], [3.30, 4.40, 4.40])
    bullets(s, 0.60, 4.95, 12.10, 1.9, [
        ("Low CP density covers 89.5% of the catalogue", ", so the system is usable across most "
         "part numbers today."),
        ("The high-density deficit had an arithmetic cause.", " 82.7% of those openings occupied "
         "fewer than 30 mesh vertices while the extraction filter required 30. Lowering the "
         "threshold for this regime moved it from 55.0% to 77.1%."),
        ("Scores are weighted by the true catalogue distribution.", " An unweighted average over "
         "my test sample would have overstated the result by up to 10 points."),
    ], size=14, gap=8)

    # ---- 10  GOOD vs BAD -----------------------------------------------------------------------
    s = new_slide("7  Examples", "One Part It Gets Right, One It Gets Wrong")
    txt(s, 0.60, 1.50, 5.80, 0.4, "Right:  PXC 3213608", 17, GREEN, True)
    bullets(s, 0.60, 1.95, 5.80, 2.6, [
        ("", "4 openings out of 4 found, nothing invented. F1 100%."),
        ("", "Direction error 0 degrees, position error 1.3 mm. The robot needs under 2 mm."),
        ("", "Round, recessed openings on one face, well separated."),
    ], size=14, gap=9)
    txt(s, 6.90, 1.50, 5.80, 0.4, "Wrong:  PXC 3071356", 17, RED, True)
    bullets(s, 6.90, 1.95, 5.80, 2.6, [
        ("", "0 out of 4 found, and 2 wrong picks. F1 0%."),
        ("", "Spring-clamp openings with no round hole. The network does not react to them."),
        ("", "The two wrong picks are screw openings that look the same as wire openings."),
    ], size=14, gap=9)
    txt(s, 0.60, 4.55, 12.10, 0.35, "I show the failure on purpose", 15, NAVY, True)
    bullets(s, 0.60, 4.95, 12.10, 1.5, [
        ("", "Because I know why it fails. It is the wire-versus-tool confusion, not random error."),
        ("", "And I checked that it is not a lack of data. Letting the model train on the exact "
             "parts it is tested on only gains 0.3 points."),
    ], size=14, gap=8)

    # ---- 12  NEXT STEPS ------------------------------------------------------------------------
    s = new_slide("8  Next", "What I Would Do Next")
    table(s, 0.60, 1.55, 12.10, [
        ["Next step", "Expected effect", "Status"],
        ["Retrain the filter on all 1 906 parts", "it still runs on 1 041", "running now"],
        ["Turn on two modes I already measured", "+2.8 points, code is ready", "ready"],
        ["Try my better segmentation model", "4.8 IoU points higher, never tested for F1", "untested"],
        ["Higher mesh resolution for high-CP-density parts", "recovers sub-threshold openings", "designed"],
        ["A signal that is not geometric", "the only way past the precision ceiling", "open"],
    ], [5.30, 4.40, 2.40])
    txt(s, 0.60, 4.85, 12.10, 0.35, "And what I will not claim", 15, NAVY, True)
    bullets(s, 0.60, 5.25, 12.10, 1.4, [
        ("", "The ceiling is 89.4%. The wrong picks that survive are real openings that simply do "
             "not take a wire, and no threshold setting removes them."),
        ("", "Tuning thresholds is finished. What is left is more data and a new kind of signal."),
    ], size=14, gap=8)
