# Classical (NO-ML) detector for recessed openings / sockets on a connector mesh
# -> connection-point pseudo-labels.
#
# The connection points on these connectors are recessed openings in the part's
# TOP face. A geometric detector finds them WITHOUT any training data, which
# (a) can beat a data-starved ML model on a narrow part family, and (b) turns
# unlabelled .stp parts into LABELLED data to fine-tune the ML model on.
#
# Method (top-down height map):
#   - the part's THINNEST axis is the face normal (depth) axis;
#   - project all vertices onto the face plane and grid it; each cell's TOP
#     surface is the MAX depth there (what you would see looking down);
#   - the dominant top-face sits at a high percentile of those cell maxima;
#   - cells whose top surface is well BELOW that level are 'inside an opening'
#     (the top face is locally recessed/absent) -- this ignores the always-present
#     bottom face and the flush top face, isolating the actual cavities;
#   - 4-connected recessed cells form one opening each -> CP at its centroid on
#     the top-face level, outward approach = +depth axis.
#
# Usage:
#   python detect_openings.py part.stp --out openings.json
#   python detect_openings.py real_json --out openings.json          # a directory
#   python detect_openings.py part.stp --out o.json --depth-frac 0.15 # tune sensitivity

import os
import json
import argparse
import logging

import numpy as np
from scipy import ndimage

import step_to_json as sj

logger = logging.getLogger(__name__)


def detect_openings(V, cell_mm=1.5, depth_frac=0.22, min_area_mm2=20.0, top_pct=80.0):
    """Recessed openings in the top face via a top-down height map. Returns a list
    of {entry_point, approach_vector, n_votes} dicts (one per opening)."""
    V = np.asarray(V, dtype=np.float64)
    if len(V) < 16:
        return []
    ext = V.max(0) - V.min(0)
    a = int(np.argmin(ext))                       # depth / face-normal axis
    others = [i for i in range(3) if i != a]
    thickness = float(ext[a]) or 1.0
    fa = V[:, others]                              # face-plane coords (N, 2)
    depth = V[:, a]

    mn = fa.min(0)
    gi = np.floor((fa - mn) / cell_mm).astype(int)
    gw, gh = int(gi[:, 0].max()) + 1, int(gi[:, 1].max()) + 1
    topmap = np.full((gw, gh), -np.inf)
    np.maximum.at(topmap, (gi[:, 0], gi[:, 1]), depth)   # per-cell top surface
    occupied = np.isfinite(topmap)
    if int(occupied.sum()) < 8:
        return []
    top_level = float(np.percentile(topmap[occupied], top_pct))
    recessed = occupied & (topmap < top_level - depth_frac * thickness)

    lab, n = ndimage.label(recessed)              # 4-connected recessed clusters
    cell_area = cell_mm * cell_mm
    sign = 1.0 if top_level >= float(depth.mean()) else -1.0
    out = []
    for c in range(1, n + 1):
        cells = np.argwhere(lab == c)
        if len(cells) * cell_area < min_area_mm2:
            continue
        cen = cells.mean(0)
        center = np.empty(3)
        center[a] = top_level
        center[others[0]] = mn[0] + (cen[0] + 0.5) * cell_mm
        center[others[1]] = mn[1] + (cen[1] + 0.5) * cell_mm
        direction = np.zeros(3)
        direction[a] = sign
        out.append({"entry_point": [float(x) for x in center],
                    "approach_vector": [float(x) for x in direction],
                    "n_votes": int(len(cells))})
    return out


def run(source, **kw):
    parts_out = []
    for part in sj.iter_parts_any(source):
        ops = detect_openings(part.vertices, **kw)
        logger.info("  %-30s %d opening(s)", part.part_nr, len(ops))
        parts_out.append({"part_nr": part.part_nr,
                          "n_vertices": int(part.n_vertices),
                          "n_detected": len(ops),
                          "connection_points": ops})
    return {"detector": "geometric_pocket", "params": kw, "parts": parts_out}


def to_abb_labels(openings):
    """Detected openings -> ABB ConnectionPoints, so the geometric pseudo-labels
    can train / fine-tune the ML model."""
    return [{"Index": i, "Name": f"OPEN-{i}",
             "Point": {"X": o["entry_point"][0], "Y": o["entry_point"][1],
                       "Z": o["entry_point"][2]},
             "InsertDirection": {"X": o["approach_vector"][0],
                                 "Y": o["approach_vector"][1],
                                 "Z": o["approach_vector"][2]}}
            for i, o in enumerate(openings)]


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Geometric (no-ML) recessed-opening detector -> CP pseudo-labels")
    ap.add_argument("source", help=".stp/.stl/.obj/.json or a directory")
    ap.add_argument("--out", required=True, help="output predictions JSON")
    ap.add_argument("--cell-mm", type=float, default=1.5, help="height-map cell size (mm)")
    ap.add_argument("--depth-frac", type=float, default=0.22,
                    help="a cell is 'recessed' if its top surface is this fraction of "
                         "the part thickness below the top-face level (lower = shallower)")
    ap.add_argument("--min-area-mm2", type=float, default=20.0,
                    help="discard openings smaller than this area")
    ap.add_argument("--top-pct", type=float, default=80.0,
                    help="percentile of cell-top-surfaces taken as the top-face level")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    res = run(args.source, cell_mm=args.cell_mm, depth_frac=args.depth_frac,
              min_area_mm2=args.min_area_mm2, top_pct=args.top_pct)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    tot = sum(p["n_detected"] for p in res["parts"])
    print(f"\n{len(res['parts'])} part(s), {tot} opening(s) total -> {args.out}")


def _selftest():
    """A synthetic plate with two recessed sockets -> detector finds two openings."""
    import make_samples_varied as mv
    V, F = mv.grid_plate(60, 60, 120.0, 120.0)
    mv.add_socket(V, (40, 40), 14.0, 10.0)
    mv.add_socket(V, (80, 80), 14.0, 10.0)
    ops = detect_openings(V, cell_mm=2.0, depth_frac=0.12, min_area_mm2=20.0)
    assert len(ops) >= 2, f"expected >=2 openings, got {len(ops)}"
    print("detect_openings selftest OK: found", len(ops), "recessed opening(s)")


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        _selftest()
    else:
        main()
