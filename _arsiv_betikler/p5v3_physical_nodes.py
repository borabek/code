# -*- coding: utf-8 -*-
"""Independent STEP physical-opening proposals for the high-ceiling P5-v3 branch.

The older P5 dictionaries only offered a B-rep mouth when it was already close
to a segmentation candidate.  That cannot recover an opening which the
segmentation candidate generator missed.  This module deliberately keeps the
two sources separate:

* segmentation candidates remain the thesis ``v_o`` proposals;
* cylinder mouths and planar inner loops become *independent* proposal nodes;
* co-located geometric descriptions are clustered into one physical resource,
  while all of their direction hypotheses are retained.

This file does not change the live product.  It is a deterministic proposal
builder used by ceiling audits and, after a selector passes manufacturer-out
validation, by the experimental P5-v3 ranker.
"""
from __future__ import annotations

from collections import Counter

import numpy as np


CYLINDER = "cylinder"
PLANAR = "planar"


def _unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else None


def _canonical_axis(v):
    """Return an unsigned axis in a deterministic sign convention."""
    u = _unit(v)
    if u is None:
        return None
    k = int(np.argmax(np.abs(u)))
    return -u if u[k] < 0 else u


def polygon_area_centroid(points, normal=None):
    """Area centroid of an ordered planar 3-D polygon.

    ``points.mean(0)`` is biased when STEP edges are sampled at different
    densities (straight-edge parameters and arc parameters have different
    units).  The projected shoelace centroid is invariant to that sampling.
    Degenerate inputs safely fall back to the arithmetic mean.
    """
    p = np.asarray(points, float)
    if len(p) < 3:
        return p.mean(0) if len(p) else np.zeros(3, float)
    c0 = p.mean(0)
    n = _unit(normal) if normal is not None else None
    if n is None:
        try:
            _u, _s, vt = np.linalg.svd(p - c0, full_matrices=False)
            n = _unit(vt[-1])
        except np.linalg.LinAlgError:
            n = None
    if n is None:
        return c0
    e1 = np.array([1.0, 0.0, 0.0])
    if abs(float(e1 @ n)) > 0.9:
        e1 = np.array([0.0, 1.0, 0.0])
    e1 -= float(e1 @ n) * n
    e1 = _unit(e1)
    if e1 is None:
        return c0
    e2 = np.cross(n, e1)
    q = p - c0
    x, y = q @ e1, q @ e2
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    a2 = float(cross.sum())
    if abs(a2) < 1e-10:
        return c0
    cx = float(((x + x1) * cross).sum() / (3.0 * a2))
    cy = float(((y + y1) * cross).sum() / (3.0 * a2))
    return c0 + cx * e1 + cy * e2


def _raw_nodes(cylinders, openings, radius=(0.5, 4.5), planar_radius=(0.8, 6.0)):
    """Build unclustered physical nodes from cached STEP features."""
    out = []
    rlo, rhi = map(float, radius)
    for ci, c in enumerate(cylinders or []):
        r = float(c.get("radius", 0.0))
        a = _canonical_axis(c.get("axis", [0, 0, 0]))
        if a is None or not (rlo <= r <= rhi):
            continue
        ma = np.asarray(c.get("mouth_a"), float)
        mb = np.asarray(c.get("mouth_b"), float)
        if ma.shape != (3,) or mb.shape != (3,) or not np.isfinite(ma).all() \
                or not np.isfinite(mb).all():
            continue
        length = float(np.linalg.norm(mb - ma))
        for side, p in enumerate((ma, mb)):
            out.append({"point": p, "axis": a, "kind": CYLINDER,
                        "source_id": (CYLINDER, int(ci), int(side)),
                        "radius": r, "length": length})

    plo, phi = map(float, planar_radius)
    for oi, o in enumerate(openings or []):
        er = float(o.get("esd_r", 0.0))
        n = _canonical_axis(o.get("normal", [0, 0, 0]))
        if n is None or not (plo <= er <= phi):
            continue
        center = np.asarray(o.get("center"), float)
        poly = np.asarray(o.get("poligon", []), float)
        if len(poly) >= 3:
            center = polygon_area_centroid(poly, n)
        if center.shape != (3,) or not np.isfinite(center).all():
            continue
        out.append({"point": center, "axis": n, "kind": PLANAR,
                    "source_id": (PLANAR, int(oi), 0),
                    "esd_r": er, "area": float(o.get("alan", 0.0)),
                    "perimeter": float(o.get("cevre", 0.0)),
                    "roundness": float(o.get("yuvarlaklik", 0.0))})
    return out


def _components(points, tolerance):
    """Single-linkage spatial components, deterministic and dependency-light."""
    from scipy.spatial import cKDTree

    n = len(points)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            if a > b:
                a, b = b, a
            parent[b] = a

    if n:
        for a, b in cKDTree(np.asarray(points, float)).query_pairs(float(tolerance)):
            union(int(a), int(b))
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [groups[k] for k in sorted(groups)]


def build_nodes(cylinders, openings, *, dedupe_mm=0.5,
                radius=(0.5, 4.5), planar_radius=(0.8, 6.0),
                direction_cos=0.999):
    """Return deduplicated physical-opening nodes.

    Each result owns one spatial point and one or more *unsigned* direction
    axes.  Cylinder and planar descriptions at the same mouth therefore compete
    as one resource instead of receiving duplicate oracle credit.
    """
    raw = _raw_nodes(cylinders, openings, radius=radius,
                     planar_radius=planar_radius)
    if not raw:
        return []
    groups = _components([r["point"] for r in raw], dedupe_mm)
    out = []
    for node_id, ii in enumerate(groups):
        members = [raw[i] for i in ii]
        points = np.asarray([m["point"] for m in members], float)
        # Medoid avoids moving a valid analytic centre between two descriptions.
        dist = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
        point = points[int(np.argmin(dist.sum(1)))]
        axes = []
        for m in members:
            a = m["axis"]
            if not any(abs(float(a @ b)) >= float(direction_cos) for b in axes):
                axes.append(a)
        kinds = Counter(m["kind"] for m in members)
        out.append({
            "node_id": int(node_id),
            "point": point,
            "axes": tuple(np.asarray(a, float) for a in axes),
            "kinds": dict(kinds),
            "n_members": int(len(members)),
            "radius_min": float(min((m.get("radius", np.inf) for m in members),
                                      default=np.inf)),
            "radius_max": float(max((m.get("radius", 0.0) for m in members),
                                      default=0.0)),
            "esd_min": float(min((m.get("esd_r", np.inf) for m in members),
                                   default=np.inf)),
            "esd_max": float(max((m.get("esd_r", 0.0) for m in members),
                                   default=0.0)),
            "sources": tuple(m["source_id"] for m in members),
        })
    return out


def signed_directions(node):
    """Both insertion signs for every unsigned node axis."""
    return tuple(s * a for a in node.get("axes", ()) for s in (1.0, -1.0))

