# -*- coding: utf-8 -*-
"""Isolated pose-dictionary option builders for the D7 oracle audit.

This module deliberately has no dependency on the production selector.  A branch is a
callable over a :class:`PoseCatalog`; add a new branch with ``register_branch`` without
changing the evaluator.

All dictionary options remain local to an existing segmentation candidate (8 mm by
default).  ``cartesian_all`` is intentionally an optimistic ceiling: it crosses every
nearby position hypothesis with every nearby signed direction hypothesis.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class PoseOption:
    point: np.ndarray
    direction: np.ndarray
    kind: str
    resource: str


@dataclass(frozen=True)
class PoseCatalog:
    current: tuple[PoseOption, ...]
    cylinder_endpoint: tuple[PoseOption, ...]
    cylinder_projection: tuple[PoseOption, ...]
    planar_center: tuple[PoseOption, ...]


BranchBuilder = Callable[[PoseCatalog], Iterable[PoseOption]]
BRANCH_BUILDERS: dict[str, BranchBuilder] = {}


def register_branch(name: str, builder: BranchBuilder) -> None:
    if not name or name in BRANCH_BUILDERS:
        raise ValueError(f"invalid/duplicate branch: {name!r}")
    BRANCH_BUILDERS[name] = builder


def _unit(v: Sequence[float]) -> np.ndarray | None:
    a = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(a))
    return a / n if np.isfinite(n) and n > 1e-9 else None


def _finite_point(v: Sequence[float]) -> np.ndarray | None:
    a = np.asarray(v, dtype=float)
    return a if a.shape == (3,) and np.isfinite(a).all() else None


def _dedupe(options: Iterable[PoseOption], decimals: int = 7) -> tuple[PoseOption, ...]:
    """Deduplicate numerically identical signed poses while preserving branch order."""
    out: list[PoseOption] = []
    seen: set[tuple[float, ...]] = set()
    for option in options:
        p = _finite_point(option.point)
        d = _unit(option.direction)
        if p is None or d is None:
            continue
        key = tuple(np.round(np.r_[p, d], decimals).tolist())
        if key in seen:
            continue
        seen.add(key)
        out.append(PoseOption(p, d, option.kind, option.resource))
    return tuple(out)


def build_catalog(
    point: Sequence[float],
    direction: Sequence[float],
    cylinders: Sequence[Mapping[str, object]] | None,
    planars: Sequence[Mapping[str, object]] | None,
    *,
    mm_max: float = 8.0,
    radius_min: float = 0.5,
    radius_max: float = 4.5,
) -> PoseCatalog:
    """Build primitive pose hypotheses near one segmentation candidate.

    ``cylinder_projection`` projects the candidate onto the *finite* cylinder-axis
    segment.  Infinite-line extensions are rejected because they are not supported by
    the STEP primitive.
    """
    p0 = _finite_point(point)
    d0 = _unit(direction)
    if p0 is None or d0 is None:
        return PoseCatalog((), (), (), ())

    current = (PoseOption(p0, d0, "current", "current"),)
    endpoints: list[PoseOption] = []
    projections: list[PoseOption] = []
    planar_options: list[PoseOption] = []

    for ci, cylinder in enumerate(cylinders or ()):
        try:
            radius = float(cylinder["radius"])
        except (KeyError, TypeError, ValueError):
            continue
        if not np.isfinite(radius) or not (radius_min <= radius <= radius_max):
            continue
        axis = _unit(cylinder.get("axis", ()))
        mouth_a = _finite_point(cylinder.get("mouth_a", ()))
        mouth_b = _finite_point(cylinder.get("mouth_b", ()))
        if axis is None or mouth_a is None or mouth_b is None:
            continue

        for mi, mouth in enumerate((mouth_a, mouth_b)):
            if float(np.linalg.norm(mouth - p0)) <= mm_max:
                resource = f"cyl:{ci}:mouth:{mi}"
                endpoints.extend(
                    PoseOption(mouth, sign * axis, "cylinder_endpoint", resource)
                    for sign in (1.0, -1.0)
                )

        # Candidate's axial coordinate on the finite mouth-to-mouth centerline.
        span = mouth_b - mouth_a
        span2 = float(span @ span)
        if span2 > 1e-12:
            t = float((p0 - mouth_a) @ span / span2)
            if -1e-9 <= t <= 1.0 + 1e-9:
                projected = mouth_a + np.clip(t, 0.0, 1.0) * span
                if float(np.linalg.norm(projected - p0)) <= mm_max:
                    resource = f"cyl:{ci}:projection"
                    projections.extend(
                        PoseOption(projected, sign * axis, "cylinder_projection", resource)
                        for sign in (1.0, -1.0)
                    )

    for pi, planar in enumerate(planars or ()):
        center = _finite_point(planar.get("center", ()))
        normal = _unit(planar.get("normal", ()))
        if center is None or normal is None:
            continue
        if float(np.linalg.norm(center - p0)) > mm_max:
            continue
        resource = f"planar:{pi}"
        planar_options.extend(
            PoseOption(center, sign * normal, "planar_center", resource)
            for sign in (1.0, -1.0)
        )

    return PoseCatalog(
        current=_dedupe(current),
        cylinder_endpoint=_dedupe(endpoints),
        cylinder_projection=_dedupe(projections),
        planar_center=_dedupe(planar_options),
    )


def _union(*groups: Iterable[PoseOption]) -> tuple[PoseOption, ...]:
    return _dedupe(option for group in groups for option in group)


def cartesian_components(
    catalog: PoseCatalog,
) -> tuple[
    tuple[tuple[np.ndarray, str, str], ...],
    tuple[tuple[np.ndarray, str, str], ...],
]:
    """Unique (position, kind, resource) and signed-direction components.

    The evaluator uses these components lazily: materialising their full Cartesian
    product can create thousands of redundant options on cylinder-dense parts.
    """
    primitives = _union(
        catalog.current,
        catalog.cylinder_endpoint,
        catalog.cylinder_projection,
        catalog.planar_center,
    )
    positions: list[tuple[np.ndarray, str, str]] = []
    directions: list[tuple[np.ndarray, str, str]] = []
    pos_seen: set[tuple[float, ...]] = set()
    dir_seen: set[tuple[float, ...]] = set()
    for option in primitives:
        pk = tuple(np.round(option.point, 7).tolist())
        dk = tuple(np.round(option.direction, 7).tolist())
        if pk not in pos_seen:
            pos_seen.add(pk)
            positions.append((option.point, option.kind, option.resource))
        if dk not in dir_seen:
            dir_seen.add(dk)
            directions.append((option.direction, option.kind, option.resource))
    return tuple(positions), tuple(directions)


def _cartesian_all(catalog: PoseCatalog) -> tuple[PoseOption, ...]:
    positions, directions = cartesian_components(catalog)
    return _dedupe(
        PoseOption(
            point,
            direction,
            f"cartesian:{point_kind}x{direction_kind}",
            f"cart:{point_resource}|{direction_resource}",
        )
        for point, point_kind, point_resource in positions
        for direction, direction_kind, direction_resource in directions
    )


register_branch("current", lambda c: c.current)
register_branch("cylinder_endpoint", lambda c: _union(c.current, c.cylinder_endpoint))
register_branch("cylinder_projection", lambda c: _union(c.current, c.cylinder_projection))
register_branch("planar_center", lambda c: _union(c.current, c.planar_center))
register_branch(
    "cylinder_all",
    lambda c: _union(c.current, c.cylinder_endpoint, c.cylinder_projection),
)
register_branch(
    "dictionary_all",
    lambda c: _union(
        c.current, c.cylinder_endpoint, c.cylinder_projection, c.planar_center
    ),
)
register_branch("cartesian_all", _cartesian_all)


def branch_options(name: str, catalog: PoseCatalog) -> tuple[PoseOption, ...]:
    try:
        builder = BRANCH_BUILDERS[name]
    except KeyError as exc:
        raise KeyError(f"unknown branch {name!r}; available={sorted(BRANCH_BUILDERS)}") from exc
    return _dedupe(builder(catalog))


__all__ = [
    "BRANCH_BUILDERS",
    "PoseCatalog",
    "PoseOption",
    "branch_options",
    "build_catalog",
    "cartesian_components",
    "register_branch",
]
