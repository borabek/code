# -*- coding: utf-8 -*-
"""Flag likely ACCESSORIES (end-plates / partition plates / covers) in a labelling batch, so the
annotator skips the parts with no connection points. EPLAN confirmed 1050100000 is a WAP end-plate
(no CPs); the gap-based selection let several such accessories in.

Reliable, model-free, absolute signal: surface-complexity = mesh surface area / convex-hull area.
A flat slab (end-plate) is ~convex -> ratio near 1. A real terminal has wire tunnels and clamp
cavities -> lots of interior surface -> ratio well above 1. Also report the thinness (a WAP plate is
a thin wafer). Anchor: 1050100000 is a known end-plate.
"""
import os, sys, glob, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from region_label_helper import load_obj

try:
    from scipy.spatial import ConvexHull
except Exception:
    ConvexHull = None


def mesh_area(V, F):
    tri = V[F]
    return float(0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum())


def complexity(V, F):
    ma = mesh_area(V, F)
    if ConvexHull is None:
        return None
    try:
        ha = float(ConvexHull(V).area)
    except Exception:
        return None
    return ma / max(ha, 1e-9)


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "_label_targets_2"
    rows = []
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(d):
            continue
        pid = os.path.basename(os.path.normpath(d))
        of = os.path.join(d, f"{pid}.obj")
        if not os.path.exists(of):
            continue
        V, F = load_obj(of)
        ext = np.sort(V.max(0) - V.min(0))[::-1]
        cx = complexity(V, F)
        thin = float(ext[2] / ext[0])          # wafer plates are very thin
        rows.append((pid, cx if cx is not None else 0.0, thin, ext))
    # threshold: end-plate anchor 1050100000 sets the low-complexity floor
    anc = next((r[1] for r in rows if r[0] == "1050100000"), None)
    thr = (anc * 1.15) if anc else 1.15   # 1050100000 (end-plate) ~1.00 -> terminals need clear cavities
    rows.sort(key=lambda r: r[1])
    ancs = f"{anc:.2f}" if anc is not None else "n/a (not in this batch)"
    print(f"anchor 1050100000 (known end-plate) complexity = {ancs} -> accessory if complexity < {thr:.2f}\n")
    print(f"{'part':14s} {'complexity':>10s} {'thin':>6s} {'ext(mm)':>16s}  verdict")
    skip, label = [], []
    for pid, cx, thin, ext in rows:
        acc = cx < thr
        v = "ACCESSORY -> SKIP" if acc else "terminal -> label"
        (skip if acc else label).append(pid)
        print(f"{pid:14s} {cx:10.2f} {thin:6.3f} {str(ext.round(0).astype(int)):>16s}  {v}")
    print(f"\n{len(label)} to LABEL, {len(skip)} likely ACCESSORIES to skip")
    print("SKIP:", " ".join(skip))
    mf = os.path.join(root, "batch2_manifest.json")
    if os.path.exists(mf):
        m = json.load(open(mf))
        cxmap = {r[0]: round(r[1], 2) for r in rows}
        for p in m["parts"]:
            p["surface_complexity"] = cxmap.get(p["part_id"])
            p["triage"] = "skip_accessory" if cxmap.get(p["part_id"], 9) < thr else "label"
        json.dump(m, open(mf, "w"), indent=1)
        print(f"-> wrote triage into {mf}")


if __name__ == "__main__":
    main()
