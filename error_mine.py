"""Error mining for a CP checkpoint: WHERE do the FP/FN live?

Runs inference once over a manifest-pinned part list, decodes at the deploy
threshold, Hungarian-matches vs GT (5mm -- identical to the metrics used
everywhere), then reports errors grouped by product family (pulled from the
STEP PRODUCT header via the catalog number, since part_nr alone doesn't say
what the part IS) and lists the worst parts. Feeds the accuracy-90 roadmap:
tells us whether the residual ~130 val errors are systematic (a family the
model can't see -> data/synth lever) or diffuse (label noise / capacity).

Usage:
  python error_mine.py --gt-dir "C:\\...\\JSON" --extra-source wscad_corpus_v3 \\
      --parts-file _v27_val.txt --ckpt checkpoints/cp_hp_v28_ftc3_best.ckpt \\
      --thr 0.25 --stp-dir all_wscad_stp --device cuda
"""
import argparse
import glob
import logging
import os
import re
from collections import defaultdict

import numpy as np

logging.basicConfig(level=logging.ERROR)


_PRODUCT_RE = re.compile(r"PRODUCT\s*\(\s*'([^']*)'")
_FILENAME_RE = re.compile(r"FILE_NAME\s*\(\s*'([^']*)'", re.I)
_CAT_RE = re.compile(r"wscaduniverse_([0-9-]+)_")


def _family_of(name):
    """Product name -> family token. 'UK 5 N' -> UK, 'PT 2,5-QUATTRO' -> PT."""
    name = re.sub(r"[-_]select$", "", str(name).strip(), flags=re.I)
    tok = re.split(r"[_ ]", name)[0].upper()
    # a bare catalog number is not a family -- reject it rather than invent one
    return tok if tok and not tok.isdigit() else ""


def build_family_map(stp_dir):
    """catalog number -> product family (e.g. UK / PT / ST), read from the STEP.

    The previous version read only the first 300_000 chars and looked only for
    PRODUCT. In a STEP file the DATA section is dominated by geometry entities
    (millions of CARTESIAN_POINTs on our big parts), so PRODUCT routinely sits far
    beyond that window -- which is why ~75% of parts landed in "?", and worse, why
    the ones that did were BIASED: the biggest, hardest parts were exactly the ones
    that got dropped. A family breakdown built on that map cannot answer the only
    question it exists to answer ("is the error concentrated or spread?").

    Now: scan the WHOLE file (chunked, early-exit on the first PRODUCT), and fall
    back to the HEADER's FILE_NAME, which is in the first few KB and usually carries
    the product name too. Returns (map, stats) so the caller can PRINT its own
    coverage -- a breakdown over a map that resolved half the corpus is a lie, and
    the tool should say so instead of quietly printing "?" rows.
    """
    fam, stats = {}, {"files": 0, "by_product": 0, "by_filename": 0, "unresolved": []}
    for f in sorted(glob.glob(os.path.join(stp_dir, "*.stp"))):
        m = _CAT_RE.search(os.path.basename(f))
        if not m:
            continue
        cat = m.group(1)
        stats["files"] += 1
        head, hit = "", ""
        try:
            with open(f, errors="ignore") as fh:
                while True:
                    chunk = fh.read(1 << 20)          # 1MB at a time
                    if not chunk:
                        break
                    head += chunk
                    pm = _PRODUCT_RE.search(head)
                    if pm:
                        hit = _family_of(pm.group(1))
                        if hit:
                            stats["by_product"] += 1
                        break
                    # keep only the tail so a PRODUCT split across chunks still matches
                    if not hit and len(head) > (1 << 20):
                        head = head[-4096:]
                if not hit:                            # fall back to the HEADER
                    fh.seek(0)
                    fm = _FILENAME_RE.search(fh.read(8192))
                    if fm:
                        hit = _family_of(fm.group(1))
                        if hit:
                            stats["by_filename"] += 1
        except OSError:
            continue
        if hit:
            fam[cat] = hit
        else:
            stats["unresolved"].append(cat)
    return fam, stats


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", required=True, dest="gt_dir")
    ap.add_argument("--extra-source", action="append", default=[],
                    dest="extra_sources")
    ap.add_argument("--parts-file", required=True, dest="parts_file")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--thr", type=float, default=0.25)
    ap.add_argument("--stp-dir", default="all_wscad_stp", dest="stp_dir")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-gpu-verts", type=int, default=14000,
                    dest="max_gpu_verts")
    ap.add_argument("--worst", type=int, default=15)
    args = ap.parse_args(argv)

    import json_dataset as jd
    import cp_regressor as cpr
    import cp_targets as ct
    import metrics as mcp
    import train_cp as tc

    fam_map, fam_stats = build_family_map(args.stp_dir)
    n_res = fam_stats["by_product"] + fam_stats["by_filename"]
    cov = 100.0 * n_res / max(fam_stats["files"], 1)
    print(f"family map: {n_res}/{fam_stats['files']} STEP files resolved ({cov:.1f}%)"
          f"  [PRODUCT {fam_stats['by_product']}, FILE_NAME {fam_stats['by_filename']}]")
    if cov < 90.0:
        print("  WARNING: coverage below 90% -- the '?' bucket is big enough to HIDE "
              "the concentration this report exists to find. Treat the table as "
              "indicative only.")
    want = {l.strip() for l in open(args.parts_file, encoding="utf-8")
            if l.strip()}
    parts = [p for p in jd.iter_parts(args.gt_dir) if str(p.part_nr) in want]
    for src in args.extra_sources:
        parts += [p for p in jd.iter_parts(src) if str(p.part_nr) in want]
    print(f"parts: {len(parts)}/{len(want)}")

    model, meta, backbone = cpr.load_model(args.ckpt, device=args.device)
    rows = []
    for p in parts:
        _, gt_pts, gt_dirs = jd.dedup_connection_points(p)
        if not len(gt_pts):
            continue
        V = np.asarray(p.vertices, float)
        Vn, _, scale = cpr.normalize_vertices(V)
        arr = cpr.infer_knngraph(model, meta, Vn, device=args.device,
                                 max_gpu_verts=args.max_gpu_verts,
                                 offset_scale=scale,
                                 patch=cpr.is_patch_part(p.part_nr),
                                 part_nr=p.part_nr)
        preds = ct.decode_predictions(V, arr, heatmap_thresh=args.thr,
                                      nms_radius_mm=tc._nms_radius(None, 5.0),
                                      min_votes=1)
        rep = mcp.keypoint_report(preds, gt_pts, gt_dirs, dist_thresh_mm=5.0)
        m = re.search(r"wscaduniverse_([0-9-]+)_", str(p.part_nr))
        fam = fam_map.get(m.group(1), "?") if m else \
            str(p.part_nr).split(".")[0]
        rows.append((str(p.part_nr), fam, len(gt_pts), rep["tp"], rep["fp"],
                     rep["fn"]))

    by_fam = defaultdict(lambda: [0, 0, 0, 0, 0])  # parts, gt, tp, fp, fn
    for _, fam, ngt, tp, fp, fn in rows:
        a = by_fam[fam]
        a[0] += 1; a[1] += ngt; a[2] += tp; a[3] += fp; a[4] += fn
    print(f"\n=== errors by product family (thr {args.thr}) ===")
    print(f"{'family':<12}{'parts':>6}{'GT':>5}{'TP':>5}{'FP':>5}{'FN':>5}"
          f"{'F1':>8}{'err/part':>9}")
    ranked = sorted(by_fam.items(), key=lambda kv: -(kv[1][3] + kv[1][4]))
    tot_err = sum(a[3] + a[4] for _, a in ranked) or 1
    tot_tp = sum(a[2] for _, a in ranked)
    cum = 0
    for fam, (np_, gt, tp, fp, fn) in ranked:
        f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0
        share = 100.0 * (fp + fn) / tot_err
        cum += share
        print(f"{fam:<12}{np_:>6}{gt:>5}{tp:>5}{fp:>5}{fn:>5}{100*f1:>7.1f}%"
              f"{(fp+fn)/np_:>9.2f}   {share:5.1f}% of all error (cum {cum:5.1f}%)")

    # THE VERDICT. This report exists to answer exactly one question, and printing a
    # table and leaving the reader to squint at it is how that question stays
    # unanswered. accuracy(Jaccard) = TP/(TP+FP+FN); the counterfactual below is the
    # honest ceiling of "just drop/fix the worst family", NOT a promise.
    worst_fam, wa = ranked[0]
    w_err = wa[3] + wa[4]
    base_acc = tot_tp / (tot_tp + tot_err)
    rest_tp = tot_tp - wa[2]
    rest_err = tot_err - w_err
    acc_wo = rest_tp / max(rest_tp + rest_err, 1)
    print(f"\n=== VERDICT ===")
    print(f"  overall accuracy (Jaccard) = {base_acc:.4f}   [target 0.90 needs F1 0.947]")
    print(f"  worst family '{worst_fam}' carries {100.0*w_err/tot_err:.1f}% of ALL error "
          f"from {wa[0]} part(s)")
    print(f"  accuracy if that family were removed entirely: {acc_wo:.4f} "
          f"({acc_wo-base_acc:+.4f})")
    top3 = sum(a[3] + a[4] for _, a in ranked[:3])
    print(f"  top-3 families hold {100.0*top3/tot_err:.1f}% of the error")
    if 100.0 * top3 / tot_err > 60.0:
        print("  => CONCENTRATED. Targeted work (scope-exclude / relabel / add parts for "
              "these families) is worth far more than any tuning lever. Check FIRST "
              "whether they are even terminal blocks.")
    else:
        print("  => SPREAD EVENLY. No family fix will move the number much: this is an "
              "architecture/label wall. Do NOT burn 17-34h on v32/seeds expecting 0.90 "
              "-- the remaining points are in the LABELS.")

    print(f"\n=== worst {args.worst} parts ===")
    rows.sort(key=lambda r: -(r[4] + r[5]))
    for pn, fam, ngt, tp, fp, fn in rows[:args.worst]:
        if fp + fn == 0:
            break
        print(f"  {pn:<52} {fam:<10} GT={ngt:<3} TP={tp:<3} FP={fp:<3} FN={fn}")
    tot_fp = sum(r[4] for r in rows); tot_fn = sum(r[5] for r in rows)
    clean = sum(1 for r in rows if r[4] + r[5] == 0)
    print(f"\nTOTAL: FP={tot_fp} FN={tot_fn}; error-free parts "
          f"{clean}/{len(rows)} ({100*clean/len(rows):.0f}%)")


if __name__ == "__main__":
    main()
