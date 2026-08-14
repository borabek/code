"""Visual QA for the CAD-direct detector: GLB per part, RED = CAD detections
(frame-transformed into the labeled corpus frame), GREEN = human GT.

The raw step_openings output lives in the STEP frame, which does NOT match the
labeled corpus frame (axes permuted + translated -- see cad_eval.align_frames).
Feeding it straight to export_glb.py would draw the markers in the wrong place
and make the detector look broken; this script applies the same alignment
cad_eval scores with, so what you SEE is exactly what was SCORED.

Usage:
  python cad_glb_qa.py --preds _cad_pxc_auto.json --gt-dir "C:\\...\\JSON" \\
      --step-dir _cad_eval_pxc --out-dir glb_cad_qa
Then double-click the .glb files (Windows 3D Viewer).
"""
import os
import json
import glob
import argparse
import logging

import numpy as np

import json_dataset as jd
import cad_eval as ce

logger = logging.getLogger(__name__)


def main(argv=None):
    ap = argparse.ArgumentParser(description="GLB visual QA: CAD preds (red) vs "
                                             "human GT (green), frame-aligned")
    ap.add_argument("--preds", required=True, help="step_openings --out JSON")
    ap.add_argument("--gt-dir", required=True, dest="gt_dir")
    ap.add_argument("--step-dir", required=True, dest="step_dir")
    ap.add_argument("--out-dir", default="glb_cad_qa", dest="out_dir")
    ap.add_argument("--align-tol", type=float, default=2.0, dest="align_tol")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    import step_to_json as sj
    import export_glb as eg

    with open(args.preds, encoding="utf-8") as fh:
        preds_doc = json.load(fh)
    gt_index = ce.build_gt_index(args.gt_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    written = 0
    for part in preds_doc.get("parts", []):
        cat = ce.catalog_nr(part["part_nr"]) or str(part["part_nr"])
        gt = gt_index.get(cat)
        if gt is None:
            continue
        steps = glob.glob(os.path.join(args.step_dir, f"*{cat}*.st*p"))
        if not steps:
            continue
        V_step, _ = sj.load_any_mesh(steps[0], deflection=0.3)
        V_gt = np.asarray(gt.vertices, float)
        R, t, res = ce.align_frames(V_step, V_gt)
        if res > args.align_tol:
            logger.warning("%s: alignment residual %.2fmm > %.1f -- skipped "
                           "(mismatched geometry)", cat, res, args.align_tol)
            continue
        cad_nodes = []
        for cp in part.get("connection_points", []):
            pt = R @ np.asarray(cp["entry_point"], float) + t
            dv = R @ np.asarray(cp["approach_vector"], float)
            cad_nodes.append({"entry_point": pt.tolist(),
                              "approach_vector": dv.tolist(),
                              "confidence_score": 1.0})
        _, gt_pts, gt_dirs = jd.dedup_connection_points(gt)
        gt_nodes = [{"entry_point": p.tolist(), "approach_vector": d.tolist()}
                    for p, d in zip(np.asarray(gt_pts, float),
                                    np.asarray(gt_dirs, float))]
        F = np.asarray(gt.faces, dtype=np.uint32) if len(gt.faces) else \
            np.zeros((0, 3), np.uint32)
        glb = eg._GlbBuilder()
        eg.build_scene(glb, V_gt, F, ml_cps=cad_nodes, cad_cps=gt_nodes)
        out = os.path.join(args.out_dir, f"{gt.part_nr}_cad_vs_gt.glb")
        glb.write(out)
        logger.info("  %s: %d CAD det (red) vs %d GT (green), align %.2fmm -> %s",
                    gt.part_nr, len(cad_nodes), len(gt_nodes), res, out)
        written += 1
    print(f"\n{written} GLB file(s) -> {args.out_dir}  (double-click to open in "
          f"Windows 3D Viewer; RED = CAD detector, GREEN = human ground truth)")


if __name__ == "__main__":
    main()
