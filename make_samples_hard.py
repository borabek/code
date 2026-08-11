# Discriminative synthetic parts: each part carries CONNECTION POINTS (smooth
# raised bosses, the only labelled feature) AND non-CP DECOYS -- sharp cones,
# recessed pits, and a raised rectangular block (corners/edges, like a plain
# box). The decoys are distinctive geometry that is NOT a connection point, so
# the model is forced to learn that a CP is a SPECIFIC feature (a smooth round
# boss apex with an outward normal), not "any bump / corner / distinctive spot".
#
# WHY this exists (vs make_samples_feat.py): make_samples_feat made the model
# FIRE (CPs are learnable bosses), but every distinctive feature there WAS a CP,
# so the model over-generalised to "anything sticking out" and produced false
# positives on a plain box's corners. Mixing in unlabelled decoys -- especially
# the box-like block -- teaches discrimination, cutting those false positives.
#
# Still synthetic, but now both LEARNABLE and DISCRIMINATIVE.
#
# Usage:
#   python make_samples_hard.py ./synthetic_hard 400
import json
import os
import sys

import numpy as np


def grid_plate(nx, ny, w, h):
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


# ---- feature primitives (all edit V[:,2] in place) ----

def add_boss(V, c, radius, height):
    """Smooth Gaussian boss (a CONNECTION POINT). Returns (apex_idx, apex_pt)."""
    d = np.linalg.norm(V[:, :2] - np.asarray(c), axis=1)
    s = max(radius, 1e-6) / 2.0
    V[:, 2] += height * np.exp(-(d ** 2) / (2.0 * s * s))
    vi = int(np.argmin(d))
    return vi, V[vi].copy()


def add_cone(V, c, radius, height):
    """Sharp linear cone (DECOY -- pointy, unlike the smooth boss)."""
    d = np.linalg.norm(V[:, :2] - np.asarray(c), axis=1)
    z = np.clip(1.0 - d / max(radius, 1e-6), 0.0, None) * height
    V[:, 2] += z


def add_pit(V, c, radius, depth):
    """Recessed Gaussian pit (DECOY -- points the other way)."""
    d = np.linalg.norm(V[:, :2] - np.asarray(c), axis=1)
    s = max(radius, 1e-6) / 2.0
    V[:, 2] -= depth * np.exp(-(d ** 2) / (2.0 * s * s))


def add_block(V, c, half_w, half_h, height):
    """Raise a rectangular mesa (DECOY -- flat top, sharp edges/corners, like a
    plain box). This is the geometry that previously caused box false-positives."""
    x, y = V[:, 0], V[:, 1]
    inside = (np.abs(x - c[0]) <= half_w) & (np.abs(y - c[1]) <= half_h)
    V[inside, 2] = np.maximum(V[inside, 2], height)


def add_rail(V, x0, half_w, height):
    """A long raised RAIL/ridge running the full length at x=x0 (DECOY). Real
    connectors have side rails/edges; on the real parts these long raised
    features were the main FALSE-POSITIVE source (the model fired on them as if
    they were connection points). Adding them as unlabelled background teaches
    the model 'a long rail is NOT a connection point'."""
    inside = np.abs(V[:, 0] - x0) <= half_w
    V[inside, 2] = np.maximum(V[inside, 2], height)


def _place(rng, w, h, n, min_sep, margin=0.16, tries=300):
    pts = []
    for _ in range(n * tries):
        if len(pts) >= n:
            break
        cx = rng.uniform(margin * w, (1 - margin) * w)
        cy = rng.uniform(margin * h, (1 - margin) * h)
        if all(np.hypot(cx - px, cy - py) >= min_sep for px, py in pts):
            pts.append((cx, cy))
    return pts


def make_part(idx, rng):
    w = float(rng.uniform(100, 160))
    h = float(rng.uniform(100, 160))
    nx = int(rng.integers(44, 56))
    ny = int(rng.integers(44, 56))
    V, F = grid_plate(nx, ny, w, h)

    # ~12% of parts are NEGATIVES: decoys only, zero connection points.
    negative = rng.random() < 0.12
    k_cp = 0 if negative else int(rng.integers(2, 5))
    k_decoy = int(rng.integers(2, 5))
    radius = float(rng.uniform(10.0, 16.0))

    # lay out CP + decoy centres on one shared grid so nothing overlaps
    centers = _place(rng, w, h, k_cp + k_decoy, min_sep=max(3.0 * radius, 38.0))
    cp_centers = centers[:k_cp]
    decoy_centers = centers[k_cp:]

    # one raised rectangular block on ~55% of parts (corners/edges = box-like)
    if rng.random() < 0.55:
        bx = rng.uniform(0.25 * w, 0.75 * w)
        by = rng.uniform(0.25 * h, 0.75 * h)
        add_block(V, (bx, by), rng.uniform(8, 16), rng.uniform(8, 16),
                  rng.uniform(6, 12))

    # decoys: alternate cones / pits (non-CP distinctive geometry)
    for di, c in enumerate(decoy_centers):
        if di % 2 == 0:
            add_cone(V, c, rng.uniform(9, 15), rng.uniform(7, 13))
        else:
            add_pit(V, c, rng.uniform(9, 15), rng.uniform(6, 11))

    # connection points: smooth bosses, added LAST so their apex z is clean
    cps = []
    for c_i, c in enumerate(cp_centers):
        vi, p = add_boss(V, c, radius, rng.uniform(7.0, 12.0))
        cps.append({
            "Index": c_i,
            "Name": f"X{idx}-{c_i}",
            "Point": {"X": float(p[0]), "Y": float(p[1]), "Z": float(p[2])},
            "InsertDirection": {"X": 0.0, "Y": 0.0, "Z": 1.0},
        })

    zmin, zmax = float(V[:, 2].min()), float(V[:, 2].max())
    return {
        "PartNr": f"SYNH.{idx:04d}",
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
    out = argv[0] if len(argv) > 0 else "synthetic_hard"
    n = int(argv[1]) if len(argv) > 1 else 400
    seed = int(argv[2]) if len(argv) > 2 else 0
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(seed)
    n_cps = n_neg = 0
    for i in range(n):
        obj = make_part(i, rng)
        if not obj["ConnectionPoints"]:
            n_neg += 1
        n_cps += len(obj["ConnectionPoints"])
        with open(os.path.join(out, f"part_{i:04d}.json"), "w", encoding="utf-8") as fh:
            json.dump(obj, fh)
    print(f"wrote {n} parts ({n_cps} CPs, {n_neg} zero-CP negatives) -> {os.path.abspath(out)}")


def _selftest():
    """A CP must be a boss apex; decoy cones/pits/blocks must NOT be CPs."""
    rng = np.random.default_rng(1)
    obj = make_part(3, rng)
    import json_dataset as jd
    part = jd.part_from_dict(obj)
    V = part.vertices
    for cp in part.cp_points:
        d = np.linalg.norm(V[:, :2] - cp[:2], axis=1)
        near = d < 8.0
        assert cp[2] >= V[near, 2].max() - 1e-6, "CP is not a local apex"
    print("make_samples_hard selftest OK:", part.part_nr, part.n_vertices,
          "verts", part.n_cps, "CP boss(es) + unlabelled decoys")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        _selftest()
    else:
        main()
