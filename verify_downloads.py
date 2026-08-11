# -*- coding: utf-8 -*-
"""Downloaded STEP files must actually BE the part their JSON describes.

WHY THIS EXISTS: wscad_fetch.py's own comment promised "verify_downloads.py re-checks every file
against its JSON mesh" -- and the file did not exist. That promise was the only safety net behind
a real hazard: part numbers collide across catalogues. The availability census matched our "KLM"
numbers exactly to Legrand, Pilz, Gira, EATON, Mitsubishi and MERTEN, and a "PXC" number to Lapp.
Silently adding such a file puts geometry into the corpus that does not belong to the JSON whose
ConnectionPoints are the ground truth, which corrupts training AND scoring without any error.

THE TEST: every JSON carries Graphic3d.Points -- the manufacturer's own point cloud for that part.
Tessellate the STEP and compare its shape to that cloud (see MAX_AXIS_REL for which statistic and
why). Measured on 20 true pairs and 20 deliberately wrong ones, with the threshold chosen on a
different sample and seed than the one it was tested on: 20/20 true pairs pass, 15/20 wrong pairs
are caught.

WHAT IT CANNOT DO, stated plainly: the 5 misses are all siblings inside one product family
(PXC 3211819/3211822/3211814; Siemens 3RV2011-1AA15/-1EA25/-4AA10). Their geometry is *identical*
-- per-axis difference 0.000, alignment residual 0.3mm -- because such variants differ in
electrical rating and marking, not in shape. No geometric test can separate them, and no better
threshold exists. This is not a gap in the check: the download requests one exact part number from
one exact manufacturer id, so a sibling swap cannot arise on that path. The hazard this file does
guard -- a different maker's part sharing our number -- is caught every time.

    python verify_downloads.py                 # check every STEP that has a JSON
    python verify_downloads.py --since 3600    # only files written in the last hour
    python verify_downloads.py --quarantine    # move failures to all_wscad_stp_rejected/

Receipt: results/verify_downloads.json
"""
import os, sys, json, glob, time, shutil, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("BA_ALLOW_SEEN", "1")

STP_DIR = "all_wscad_stp"
REJECT_DIR = "all_wscad_stp_rejected"
# Manufacturer JSONs live in TWO places: the hand-collected Desktop\JSON and the DataSet 1 dump
# (_ds1). Looking only at the first made the verifier skip all 7 freshly downloaded files as
# "no JSON" -- i.e. it silently verified nothing while reporting success. Both are searched.
JSON_DIRS = [r"C:\Users\DE00024082\Desktop\JSON", os.path.join("_ds1", "DataSet")]
OUT = "results/verify_downloads.json"

# WHICH TEST, AND WHY THIS ONE. Four candidate statistics were measured on 14 true pairs and 14
# deliberately wrong pairs (each STEP against a random other part's JSON):
#
#   statistic                     true max   wrong min   verdict
#   alignment residual              1.777      1.353     OVERLAPS
#   95th pct nearest-neighbour      4.836      2.215     OVERLAPS
#   mean nearest-neighbour          1.852      1.344     OVERLAPS
#   per-axis extent ratio           0.130      0.245     SEPARATES CLEANLY
#
# Distance after alignment cannot tell two similarly-sized parts apart. The sorted per-axis extents
# are a shape fingerprint -- a different part may share a bounding-box diagonal but rarely the same
# aspect ratio -- and the test needs no alignment at all, so a poor fit cannot fake a pass.
# The cut sits in the measured gap, on the safe side of it. n=14 per group: wide margin, small
# sample, so the residual is still recorded (loosely bounded) and printed for every rejection.
MAX_AXIS_REL = 0.18       # PRIMARY: worst per-axis extent may differ by at most 18%
MAX_RESIDUAL_MM = 8.0     # secondary sanity bound only -- NOT the discriminator
MAX_BBOX_REL = 0.25       # secondary: bounding-box diagonal


def json_for(pid):
    """Find the manufacturer JSON for a part number, whatever the maker prefix or directory."""
    for d in JSON_DIRS:
        hits = (glob.glob(os.path.join(d, f"*.{pid}.json"))
                or glob.glob(os.path.join(d, f"*.{pid}_*.json")))
        if hits:
            return hits[0]
    return None


def check(stp, jf):
    """Return (ok, detail). ok=False means the STEP is not the part the JSON describes."""
    from infer_step_cp import step_to_mesh
    from cad_eval import align_frames
    j = json.load(open(jf, encoding="utf-8-sig"))
    pts = (j.get("Graphic3d") or {}).get("Points") or []
    if len(pts) < 8:
        return None, {"reason": "JSON'da nokta bulutu yok -- dogrulanamaz"}
    Vj = np.array([[q["X"], q["Y"], q["Z"]] for q in pts], float)
    Vr, _Fr = step_to_mesh(stp)
    if len(Vr) < 8:
        return False, {"reason": "STEP meshlenemedi"}

    # shape fingerprint: sorted per-axis extents, so part orientation in the file does not matter
    es = np.sort(Vr.max(0) - Vr.min(0))
    ej = np.sort(Vj.max(0) - Vj.min(0))
    axis_rel = float(np.max(np.abs(es - ej) / np.maximum(ej, 1e-9)))

    R, t, res = align_frames(Vr, Vj)
    dj = float(np.linalg.norm(Vj.max(0) - Vj.min(0)))
    ds = float(np.linalg.norm(Vr.max(0) - Vr.min(0)))
    rel = abs(ds - dj) / max(dj, 1e-9)
    ok = (axis_rel <= MAX_AXIS_REL) and (float(res) <= MAX_RESIDUAL_MM) and (rel <= MAX_BBOX_REL)
    return ok, {"axis_rel_diff": round(axis_rel, 4), "residual_mm": round(float(res), 3),
                "bbox_json_mm": round(dj, 2), "bbox_step_mm": round(ds, 2),
                "bbox_rel_diff": round(rel, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", type=float, default=0,
                    help="sadece son N saniyede yazilmis dosyalari kontrol et")
    ap.add_argument("--quarantine", action="store_true",
                    help=f"basarisiz dosyalari {REJECT_DIR}/ altina TASI")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(STP_DIR, "*.stp")))
    if a.since:
        cut = time.time() - a.since
        files = [f for f in files if os.path.getmtime(f) >= cut]
    if a.limit:
        files = files[:a.limit]

    rows, npass, nfail, nskip = [], 0, 0, 0
    print(f"{len(files)} STEP kontrol edilecek "
          f"(esik: residual <= {MAX_RESIDUAL_MM}mm, bbox farki <= {MAX_BBOX_REL:.0%})", flush=True)
    for i, stp in enumerate(files, 1):
        pid = os.path.basename(stp).split("_")[1]
        jf = json_for(pid)
        if not jf:
            nskip += 1
            continue
        try:
            ok, det = check(stp, jf)
        except Exception as e:
            ok, det = False, {"reason": f"{type(e).__name__}: {str(e)[:80]}"}
        det.update(part=pid, step=os.path.basename(stp))
        if ok is None:
            nskip += 1
        elif ok:
            npass += 1
        else:
            nfail += 1
            print(f"  RED  {pid}: {det}", flush=True)
            if a.quarantine:
                os.makedirs(REJECT_DIR, exist_ok=True)
                shutil.move(stp, os.path.join(REJECT_DIR, os.path.basename(stp)))
                det["quarantined"] = True
        det["ok"] = ok
        rows.append(det)
        if i % 100 == 0:
            print(f"  {i}/{len(files)}  gecti {npass} / kaldi {nfail} / atlandi {nskip}", flush=True)

    os.makedirs("results", exist_ok=True)
    json.dump({"checked": len(rows), "pass": npass, "fail": nfail, "skipped_no_json": nskip,
               "max_residual_mm": MAX_RESIDUAL_MM, "max_bbox_rel": MAX_BBOX_REL,
               "quarantined": bool(a.quarantine), "rows": rows},
              open(OUT, "w"), indent=1)
    print(f"\ngecti {npass} | KALDI {nfail} | atlandi (JSON yok) {nskip}")
    print(f"makbuz -> {OUT}")
    if nfail and not a.quarantine:
        print("NOT: basarisizlar YERINDE DURUYOR. Karantinaya almak icin --quarantine ile kosun.")
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
