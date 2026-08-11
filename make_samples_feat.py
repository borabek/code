# Generate synthetic ABB-style parts whose connection points sit on a
# DISTINCTIVE geometric feature (a raised boss / socket), unlike make_samples.py
# whose CPs are at geometrically indistinguishable spots on a flat plate.
#
# WHY: a geometry-based detector (knngraph / diffusionnet) learns a mapping
# "local geometry -> is this a connection point?". On make_samples.py's flat
# plate every vertex has the same neighbourhood, so no such mapping exists and
# the model collapses to a flat heatmap (TP=0). Here each CP is the APEX of a
# smooth raised boss, so its neighbourhood (a curved bump) is clearly different
# from the flat background -- the model can learn "bump apex -> CP" and actually
# fire. This also mimics real connectors, where connection points ARE openings /
# sockets / raised terminals rather than arbitrary points on a flat face.
#
# Still synthetic (not a substitute for the real corpus), but now LEARNABLE so
# the end-to-end demo produces real, non-zero detections.
#
# Usage:
#   python make_samples_feat.py ./synthetic_feat 300
import json
import os
import sys

import numpy as np


def grid_plate(nx, ny, w, h):
    """Triangulated open plate in the z=0 plane, size w x h mm -> (V, F)."""
    xs = np.linspace(0, w, nx)
    ys = np.linspace(0, h, ny)
    X, Y = np.meshgrid(xs, ys)
    V = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
    F = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            a = j * nx + i
            b = a + 1
            c = a + nx
            d = c + 1
            F.append([a, b, c])
            F.append([b, d, c])
    return V, np.asarray(F, dtype=np.int64)


def add_boss(V, center_xy, radius, height):
    """Raise a smooth Gaussian boss around center_xy (in place).

    z(d) = height * exp(-d^2 / (2 (radius/2)^2)); the apex (nearest vertex to the
    centre) becomes the highest point of a local bump -- a neighbourhood the
    geometric model can distinguish from the flat plate. Returns (apex_index,
    apex_point, outward_direction). The boss points up, so the outward insert
    direction is +z."""
    d = np.linalg.norm(V[:, :2] - np.asarray(center_xy), axis=1)
    s = max(radius, 1e-6) / 2.0
    V[:, 2] += height * np.exp(-(d ** 2) / (2.0 * s * s))
    vi = int(np.argmin(d))
    return vi, V[vi].copy(), np.array([0.0, 0.0, 1.0])


def _place_centers(rng, w, h, k, min_sep, margin=0.18, tries=200):
    """Sample k boss centres inside the plate, at least min_sep apart."""
    centers = []
    for _ in range(k * tries):
        if len(centers) >= k:
            break
        cx = rng.uniform(margin * w, (1 - margin) * w)
        cy = rng.uniform(margin * h, (1 - margin) * h)
        if all(np.hypot(cx - px, cy - py) >= min_sep for px, py in centers):
            centers.append((cx, cy))
    return centers


def make_part(idx, rng):
    # finer grid than make_samples so each boss spans several vertices
    w = float(rng.uniform(90, 150))
    h = float(rng.uniform(90, 150))
    nx = int(rng.integers(40, 52))
    ny = int(rng.integers(40, 52))
    V, F = grid_plate(nx, ny, w, h)

    k = int(rng.integers(2, 5))                     # 2-4 connection points
    radius = float(rng.uniform(10.0, 16.0))         # boss footprint (mm)
    height = float(rng.uniform(6.0, 12.0))          # boss height (mm)
    centers = _place_centers(rng, w, h, k, min_sep=max(3.0 * radius, 35.0))

    cps = []
    for c, (cx, cy) in enumerate(centers):
        vi, p, ddir = add_boss(V, (cx, cy), radius, height)
        cps.append({
            "Index": c,
            "Name": f"X{idx}-{c}",
            "Point": {"X": float(p[0]), "Y": float(p[1]), "Z": float(p[2])},
            "InsertDirection": {"X": float(ddir[0]), "Y": float(ddir[1]),
                                "Z": float(ddir[2])},
        })

    zmin, zmax = float(V[:, 2].min()), float(V[:, 2].max())
    return {
        "PartNr": f"SYNF.{idx:03d}",
        "Graphic3d": {
            "Points": [{"X": float(x), "Y": float(y), "Z": float(z)}
                       for x, y, z in V],
            "Indices": F.ravel().tolist(),
        },
        "BoundingBox": {"Dimension": {"X": w, "Y": h, "Z": zmax - zmin},
                        "Location": {"X": 0.0, "Y": 0.0, "Z": zmin}},
        "ConnectionPoints": cps,
    }


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    out = argv[0] if len(argv) > 0 else "synthetic_feat"
    n = int(argv[1]) if len(argv) > 1 else 300
    seed = int(argv[2]) if len(argv) > 2 else 0
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(seed)
    n_cps = 0
    for i in range(n):
        obj = make_part(i, rng)
        n_cps += len(obj["ConnectionPoints"])
        with open(os.path.join(out, f"part_{i:04d}.json"), "w", encoding="utf-8") as fh:
            json.dump(obj, fh)
    print(f"wrote {n} parts ({n_cps} connection points total) -> {os.path.abspath(out)}")


def _selftest():
    """One part: every CP must be the local z-apex of its boss (distinctive)."""
    rng = np.random.default_rng(0)
    obj = make_part(0, rng)
    import json_dataset as jd
    part = jd.part_from_dict(obj)
    assert part.n_cps >= 2
    V = part.vertices
    for cp in part.cp_points:
        d = np.linalg.norm(V[:, :2] - cp[:2], axis=1)
        near = d < 8.0
        # the CP's z is at/above the local max around it -> it's a real apex
        assert cp[2] >= V[near, 2].max() - 1e-6, "CP is not the local apex"
    print("make_samples_feat selftest OK:", part.part_nr, part.n_vertices,
          "verts", part.n_cps, "cps; CPs are boss apexes (distinctive)")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        _selftest()
    else:
        main()
