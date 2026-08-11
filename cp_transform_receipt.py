# -*- coding: utf-8 -*-
"""OBJ<->STEP frame receipt (audit P1 #8). The whole CP pipeline derives points in the publisher
OBJ frame, but the human-CP labels and the robot target live in the exact WSCAD STEP frame. This
verifies, per part, that the two are the SAME frame, by comparing the raw OBJ vertex bbox to the
STEP B-rep bbox (gmsh). Records center residual + extent residual + SHAs. A part whose center
residual exceeds --max-center-mm is auto-flagged for manual review (frame mismatch).

For the Scheffler corpus the transform is IDENTITY (measured: center residual ~0, extent differs
by the ~0.5mm remesh envelope), so no per-part transform is needed -- but the receipt PROVES it
instead of assuming it.

Usage: .venv/Scripts/python.exe cp_transform_receipt.py [--max-center-mm 2.0]
"""
import argparse, os, glob, json, hashlib
import numpy as np
import gmsh

CORPUS = "wscad_corpus_scheffler_exact"


def obj_bbox(p):
    vs = [[float(x) for x in ln.split()[1:4]] for ln in open(p) if ln.startswith("v ")]
    v = np.array(vs, float); return v.min(0), v.max(0)


def sha16(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-center-mm", type=float, default=2.0)
    ap.add_argument("--out", default="results/cp_transform_receipt.json")
    a = ap.parse_args()
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
    rows = []
    for d in sorted(glob.glob(os.path.join(CORPUS, "*", "*"))):
        pid = os.path.basename(d); split = os.path.basename(os.path.dirname(d))
        obj = os.path.join(d, f"{pid}.obj"); stp = os.path.join(d, f"{pid}.stp")
        if not (os.path.exists(obj) and os.path.exists(stp)):
            continue
        try:
            omin, omax = obj_bbox(obj)
            gmsh.open(stp); b = gmsh.model.getBoundingBox(-1, -1); gmsh.clear()
            smin, smax = np.array(b[:3]), np.array(b[3:])
            cres = float(np.abs((omax + omin) / 2 - (smax + smin) / 2).max())
            eres = float(np.abs((omax - omin) - (smax - smin)).max())
            rows.append({"part_id": pid, "split": split,
                         "center_residual_mm": round(cres, 3), "extent_residual_mm": round(eres, 3),
                         "same_frame": cres <= a.max_center_mm,
                         "obj_sha16": sha16(obj), "step_sha16": sha16(stp)})
        except Exception as e:
            rows.append({"part_id": pid, "split": split, "error": str(e)[:80], "same_frame": False})
    gmsh.finalize()
    ok = [r for r in rows if r.get("same_frame")]
    flagged = [r["part_id"] for r in rows if not r.get("same_frame")]
    out = {"frozen": "2026-07-18", "transform": "identity (OBJ frame == STEP frame)",
           "max_center_residual_mm": a.max_center_mm, "n_parts": len(rows), "n_same_frame": len(ok),
           "worst_center_residual_mm": round(max((r.get("center_residual_mm", 0) for r in rows), default=0), 3),
           "worst_extent_residual_mm": round(max((r.get("extent_residual_mm", 0) for r in rows), default=0), 3),
           "flagged_for_manual_review": flagged, "rows": rows}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"{len(rows)} parts | same_frame {len(ok)}/{len(rows)} | worst center {out['worst_center_residual_mm']}mm "
          f"extent {out['worst_extent_residual_mm']}mm | flagged {len(flagged)}")
    if flagged:
        print("  flagged:", flagged[:20])
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
