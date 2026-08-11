# -*- coding: utf-8 -*-
"""Read-only STEP hypotheses for non-cylindrical wire-entry mouths.

This module is deliberately isolated from the production chain.  It supplies two
missing physical-node families for oracle experiments:

* exact planar inner loops, excluding a face's real ``OuterWire`` rather than an
  area-ratio guess; and
* rectangular/slot pockets inferred from two opposing pairs of planar side walls.

The returned dictionaries are compatible with the pose-dictionary convention:
``center`` is a position hypothesis and ``normal`` is an *unsigned* axis (the caller
must add both signs).  No candidate is accepted here and no tolerance is changed.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


def _unit(value: Sequence[float]) -> np.ndarray | None:
    a = np.asarray(value, dtype=float)
    if a.shape != (3,) or not np.isfinite(a).all():
        return None
    n = float(np.linalg.norm(a))
    return a / n if n > 1e-10 else None


def _plane_basis(normal: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    n = _unit(normal)
    if n is None:
        raise ValueError("invalid plane normal")
    seed = np.eye(3)[int(np.argmin(np.abs(n)))]
    u = np.cross(n, seed)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return u, v


def polygon_metrics(
    points: Sequence[Sequence[float]], normal: Sequence[float]
) -> dict[str, object] | None:
    """Area centroid and shape statistics for an ordered planar boundary.

    The centroid uses the shoelace formula, not the mean of sampled boundary
    points.  The latter moves when long and short edges are sampled differently.
    """
    P = np.asarray(points, dtype=float)
    n = _unit(normal)
    if n is None or P.ndim != 2 or P.shape[1:] != (3,) or len(P) < 3:
        return None
    if not np.isfinite(P).all():
        return None
    # Adjacent duplicate vertices create zero edges and occasionally singular PCA.
    keep = np.ones(len(P), dtype=bool)
    if len(P) > 1:
        keep[1:] = np.linalg.norm(np.diff(P, axis=0), axis=1) > 1e-9
    P = P[keep]
    if len(P) > 2 and np.linalg.norm(P[0] - P[-1]) <= 1e-9:
        P = P[:-1]
    if len(P) < 3:
        return None

    origin = P.mean(axis=0)
    u, v = _plane_basis(n)
    Q = np.c_[(P - origin) @ u, (P - origin) @ v]
    Q1 = np.roll(Q, -1, axis=0)
    cross = Q[:, 0] * Q1[:, 1] - Q1[:, 0] * Q[:, 1]
    area2 = float(cross.sum())
    if abs(area2) <= 1e-9:
        return None
    center2 = ((Q + Q1) * cross[:, None]).sum(axis=0) / (3.0 * area2)
    center = origin + center2[0] * u + center2[1] * v
    area = abs(area2) * 0.5
    perimeter = float(
        np.linalg.norm(np.roll(P, -1, axis=0) - P, axis=1).sum()
    )

    cov = np.cov(Q.T) if len(Q) > 2 else np.eye(2)
    eigval, eigvec = np.linalg.eigh(cov)
    R = Q @ eigvec
    ext = np.ptp(R, axis=0)
    width, height = sorted(map(float, ext))
    box_area = width * height
    return {
        "center": center,
        "normal": n,
        "area": float(area),
        "perimeter": perimeter,
        "esd_r": float(math.sqrt(area / math.pi)),
        "circularity": float(4.0 * math.pi * area / max(perimeter * perimeter, 1e-12)),
        "rectangularity": float(min(area / max(box_area, 1e-12), 1.0)),
        "width": width,
        "height": height,
        "aspect": float(height / max(width, 1e-9)),
        "points": P,
    }


def _sample_wire(wire: object, sample_mm: float = 0.35) -> np.ndarray:
    """Sample an OCCT wire in traversal order with roughly uniform arc spacing."""
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.BRepTools import BRepTools_WireExplorer
    from OCP.GCPnts import GCPnts_AbscissaPoint, GCPnts_UniformAbscissa
    from OCP.TopAbs import TopAbs_REVERSED

    chunks: list[np.ndarray] = []
    explorer = BRepTools_WireExplorer(wire)
    while explorer.More():
        edge = explorer.Current()
        explorer.Next()
        try:
            curve = BRepAdaptor_Curve(edge)
            first, last = float(curve.FirstParameter()), float(curve.LastParameter())
            length = float(GCPnts_AbscissaPoint.Length_s(curve, first, last))
            count = int(np.clip(math.ceil(length / max(sample_mm, 0.05)) + 1, 2, 80))
            sampler = GCPnts_UniformAbscissa(curve, count, first, last)
            if sampler.IsDone() and sampler.NbPoints() >= 2:
                params = [float(sampler.Parameter(i)) for i in range(1, sampler.NbPoints() + 1)]
            else:
                params = np.linspace(first, last, count).tolist()
            if edge.Orientation() == TopAbs_REVERSED:
                params.reverse()
            pts = []
            for parameter in params:
                p = curve.Value(parameter)
                pts.append((p.X(), p.Y(), p.Z()))
            A = np.asarray(pts, dtype=float)
            if chunks and len(A) and np.linalg.norm(chunks[-1][-1] - A[0]) <= 1e-7:
                A = A[1:]
            if len(A):
                chunks.append(A)
        except Exception:
            continue
    if not chunks:
        return np.zeros((0, 3), dtype=float)
    P = np.vstack(chunks)
    if len(P) > 2 and np.linalg.norm(P[0] - P[-1]) <= 1e-7:
        P = P[:-1]
    return P


def _read_step(path: str | Path) -> object:
    from OCP.STEPControl import STEPControl_Reader

    reader = STEPControl_Reader()
    if reader.ReadFile(str(path)) != 1:
        raise ValueError(f"cannot read STEP: {path}")
    reader.TransferRoots()
    return reader.OneShape()


def extract_planar_geometry(
    step_path: str | Path,
    *,
    sample_mm: float = 0.35,
    esd_min: float = 0.5,
    esd_max: float = 8.0,
) -> tuple[list[dict[str, object]], list[dict[str, object]], np.ndarray]:
    """Return ``(inner_loops, planar_walls, all_sampled_points)`` from a STEP.

    ``BRepTools.OuterWire_s`` is the sole outer-loop decision.  In particular, a
    thin rim whose hole occupies >75% of the face is retained correctly.
    """
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepTools import BRepTools
    from OCP.GeomAbs import GeomAbs_Plane
    from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED, TopAbs_WIRE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    shape = _read_step(step_path)
    loops: list[dict[str, object]] = []
    walls: list[dict[str, object]] = []
    all_points: list[np.ndarray] = []
    faces = TopExp_Explorer(shape, TopAbs_FACE)
    face_index = 0
    while faces.More():
        face = TopoDS.Face_s(faces.Current())
        faces.Next()
        face_index += 1
        try:
            adaptor = BRepAdaptor_Surface(face)
            if adaptor.GetType() != GeomAbs_Plane:
                continue
            direction = adaptor.Plane().Axis().Direction()
            normal = np.array([direction.X(), direction.Y(), direction.Z()], dtype=float)
            if face.Orientation() == TopAbs_REVERSED:
                normal *= -1.0
            normal = _unit(normal)
            if normal is None:
                continue
            outer = BRepTools.OuterWire_s(face)
        except Exception:
            continue

        wire_index = 0
        outer_points = np.zeros((0, 3), dtype=float)
        wires = TopExp_Explorer(face, TopAbs_WIRE)
        while wires.More():
            wire = TopoDS.Wire_s(wires.Current())
            wires.Next()
            wire_index += 1
            P = _sample_wire(wire, sample_mm=sample_mm)
            if len(P) < 3:
                continue
            all_points.append(P)
            is_outer = not outer.IsNull() and wire.IsSame(outer)
            if is_outer:
                outer_points = P
                continue
            metrics = polygon_metrics(P, normal)
            if metrics is None or not (esd_min <= float(metrics["esd_r"]) <= esd_max):
                continue
            loops.append(
                {
                    "center": metrics["center"],
                    "normal": normal,
                    "esd_r": metrics["esd_r"],
                    "cevre": metrics["perimeter"],
                    "alan": metrics["area"],
                    "yuvarlaklik": metrics["circularity"],
                    "rectangularity": metrics["rectangularity"],
                    "aspect": metrics["aspect"],
                    "poligon": np.asarray(metrics["points"], dtype=np.float32),
                    "source": "planar_inner_exact",
                    "resource": f"face:{face_index}:wire:{wire_index}",
                }
            )

        outer_metrics = polygon_metrics(outer_points, normal) if len(outer_points) else None
        if outer_metrics is not None:
            walls.append(
                {
                    "center": outer_metrics["center"],
                    "normal": normal,
                    "points": outer_metrics["points"],
                    "area": outer_metrics["area"],
                    "width": outer_metrics["width"],
                    "height": outer_metrics["height"],
                    "resource": f"face:{face_index}",
                }
            )
    cloud = np.vstack(all_points) if all_points else np.zeros((0, 3), dtype=float)
    return loops, walls, cloud


def _wall_record(value: Mapping[str, object], index: int) -> dict[str, object] | None:
    normal = _unit(value.get("normal", ()))
    points = np.asarray(value.get("points", ()), dtype=float)
    if normal is None or points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 3:
        return None
    center = np.asarray(value.get("center", points.mean(axis=0)), dtype=float)
    if center.shape != (3,) or not np.isfinite(center).all() or not np.isfinite(points).all():
        return None
    return {
        "index": index,
        "center": center,
        "normal": normal,
        "points": points,
        "area": float(value.get("area", 0.0)),
        "resource": str(value.get("resource", f"wall:{index}")),
    }


def infer_rectangular_pockets(
    walls: Iterable[Mapping[str, object]],
    *,
    part_points: Sequence[Sequence[float]] | None = None,
    min_span: float = 0.8,
    max_span: float = 12.0,
    min_depth: float = 2.0,
    max_depth: float = 25.0,
    parallel_deg: float = 15.0,
    orthogonal_deg: float = 15.0,
    midpoint_tol: float = 3.0,
    require_opposed_orientation: bool = True,
) -> list[dict[str, object]]:
    """Infer four-wall rectangular pockets without assuming world-axis alignment.

    A valid hypothesis needs two disjoint opposing wall pairs, the pair normals must
    be orthogonal, and all four faces must overlap along a common depth interval.
    These guards reject a single lip and the large outer housing box.
    """
    W = [w for i, value in enumerate(walls) if (w := _wall_record(value, i)) is not None]
    if len(W) < 4:
        return []
    cos_parallel = math.cos(math.radians(parallel_deg))
    sin_orthogonal = math.sin(math.radians(orthogonal_deg))
    pairs: list[dict[str, object]] = []
    for i in range(len(W)):
        for j in range(i + 1, len(W)):
            ni, nj = W[i]["normal"], W[j]["normal"]
            dot = float(ni @ nj)
            if abs(dot) < cos_parallel:
                continue
            if require_opposed_orientation and dot > -cos_parallel:
                continue
            delta = W[j]["center"] - W[i]["center"]
            separation = abs(float(delta @ ni))
            tangential = float(np.linalg.norm(delta - (delta @ ni) * ni))
            if not (min_span <= separation <= max_span) or tangential > midpoint_tol:
                continue
            pair_normal = _unit(delta)
            if pair_normal is None:
                continue
            pairs.append(
                {
                    "members": (i, j),
                    "normal": pair_normal,
                    "midpoint": 0.5 * (W[i]["center"] + W[j]["center"]),
                    "span": separation,
                    "parallel": abs(dot),
                }
            )

    cloud = np.asarray(part_points, dtype=float) if part_points is not None else np.zeros((0, 3))
    out: list[dict[str, object]] = []
    for ai, first in enumerate(pairs):
        for second in pairs[ai + 1 :]:
            member_ids = set(first["members"]) | set(second["members"])
            if len(member_ids) != 4:
                continue
            na, nb = first["normal"], second["normal"]
            if abs(float(na @ nb)) > sin_orthogonal:
                continue
            axis = _unit(np.cross(na, nb))
            if axis is None:
                continue
            midpoint_delta = second["midpoint"] - first["midpoint"]
            transverse = midpoint_delta - (midpoint_delta @ axis) * axis
            if np.linalg.norm(transverse) > midpoint_tol:
                continue

            intervals = []
            for member in sorted(member_ids):
                projection = W[member]["points"] @ axis
                intervals.append((float(projection.min()), float(projection.max())))
            low = max(value[0] for value in intervals)
            high = min(value[1] for value in intervals)
            depth = high - low
            if not (min_depth <= depth <= max_depth):
                continue

            width, height = sorted((float(first["span"]), float(second["span"])))
            if width <= 0 or height / width > 8.0:
                continue
            # Pick the common-depth end nearer the exterior support of the part.
            mouth_t = high
            outward = axis
            exterior_gap = 0.0
            if cloud.ndim == 2 and cloud.shape[1:] == (3,) and len(cloud):
                part_t = cloud @ axis
                gap_low = abs(low - float(part_t.min()))
                gap_high = abs(float(part_t.max()) - high)
                if gap_low < gap_high:
                    mouth_t, outward, exterior_gap = low, -axis, gap_low
                else:
                    exterior_gap = gap_high

            A = np.vstack([na, nb, axis])
            rhs = np.array(
                [
                    0.5
                    * (
                        na @ W[first["members"][0]]["center"]
                        + na @ W[first["members"][1]]["center"]
                    ),
                    0.5
                    * (
                        nb @ W[second["members"][0]]["center"]
                        + nb @ W[second["members"][1]]["center"]
                    ),
                    mouth_t,
                ]
            )
            center = np.linalg.lstsq(A, rhs, rcond=None)[0]
            quality = float(
                min(first["parallel"], second["parallel"])
                * (1.0 - abs(float(na @ nb)))
                * math.exp(-float(np.linalg.norm(transverse)) / max(midpoint_tol, 1e-6))
            )
            out.append(
                {
                    "center": center,
                    "normal": outward,
                    "esd_r": float(math.sqrt(width * height / math.pi)),
                    "width": width,
                    "height": height,
                    "depth": float(depth),
                    "support_faces": 4,
                    "quality": quality,
                    "exterior_gap": float(exterior_gap),
                    "source": "rect_wall_pocket",
                    "resource": "pocket:" + "|".join(W[i]["resource"] for i in sorted(member_ids)),
                }
            )

    # The same four walls are found twice with swapped pair order.  Retain the
    # strongest physical node and do not let duplicates claim multiple GTs.
    deduped: list[dict[str, object]] = []
    for candidate in sorted(out, key=lambda value: float(value["quality"]), reverse=True):
        duplicate = False
        for kept in deduped:
            if (
                np.linalg.norm(candidate["center"] - kept["center"]) <= 0.75
                and abs(float(candidate["normal"] @ kept["normal"])) >= math.cos(math.radians(10))
            ):
                duplicate = True
                break
        if not duplicate:
            deduped.append(candidate)
    return deduped


def extract_pocket_nodes(
    step_path: str | Path,
    *,
    sample_mm: float = 0.35,
) -> dict[str, list[dict[str, object]]]:
    """Extract exact inner-loop and four-wall pocket physical nodes from one STEP."""
    loops, walls, cloud = extract_planar_geometry(step_path, sample_mm=sample_mm)
    pockets = infer_rectangular_pockets(walls, part_points=cloud)
    return {"inner_loops": loops, "rect_pockets": pockets}


def _jsonable(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items() if key != "poligon"}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("step")
    parser.add_argument("--sample-mm", type=float, default=0.35)
    args = parser.parse_args()
    print(json.dumps(_jsonable(extract_pocket_nodes(args.step, sample_mm=args.sample_mm)), indent=1))


if __name__ == "__main__":
    main()

