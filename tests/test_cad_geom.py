"""Regression tests for the CAD-direct geometric detector's 2026-07 additions.

These lock in behaviors that were tuned against REAL measured geometry (PXC
push-in terminal blocks) and were previously untested:
  - _pair_slot_halves : racetrack/oval slot half-cylinder fusion
  - _rect_openings    : rectangular spring-clamp openings from fillet clusters
  - validate_directions: mesh-clearance approach-vector adjudication
  - cad_eval.align_frames: STEP-frame -> corpus-frame recovery
All pure-numpy (no gmsh), so they run everywhere the suite runs.
"""
import numpy as np

import step_openings as so
import cad_eval as ce


def _hole(center, radius, axis_dim, n=1):
    """Minimal merged-hole dict as built inside openings_from_step."""
    c = np.asarray(center, float)
    return {"center": c, "members": [{"center": c.copy(), "radius": float(radius),
                                      "axis_dim": int(axis_dim), "kind": "cyl"}
                                     for _ in range(n)]}


# ---------------------------------------------------------------------------
# _pair_slot_halves
# ---------------------------------------------------------------------------

def test_slot_pair_fuses_racetrack_halves():
    # measured PXC geometry: two r=2.77 half-cylinders 5.7mm apart (2.06x r),
    # offset PERPENDICULAR to the shared z axis -> ONE opening at the midpoint
    a = _hole([3.7, 4.2, 36.5], 2.77, axis_dim=2)
    b = _hole([3.7, 9.9, 36.5], 2.77, axis_dim=2)
    out = so._pair_slot_halves([a, b], merge_tol=2.8)
    assert len(out) == 1, "racetrack halves must fuse into one opening"
    np.testing.assert_allclose(out[0]["center"], [3.7, 7.05, 36.5], atol=1e-9)
    assert out[0].get("slot") is True


def test_slot_pair_keeps_distinct_terminals_apart():
    # two SEPARATE terminals: same radius/axis but farther than 2.5x r
    a = _hole([0, 0, 0], 1.4, axis_dim=2)
    b = _hole([0, 5.4, 0], 1.4, axis_dim=2)   # 5.4mm pitch >> 2.5*1.4=3.5
    out = so._pair_slot_halves([a, b], merge_tol=2.0)
    assert len(out) == 2, "distinct terminals must NOT be fused"


def test_slot_pair_rejects_axial_offset():
    # coaxial segments of a through-channel: offset ALONG the axis -> not a slot
    a = _hole([2.3, 10, 10], 2.9, axis_dim=0)
    b = _hole([5.9, 10, 10], 2.9, axis_dim=0)   # offset in x == their axis
    out = so._pair_slot_halves([a, b], merge_tol=2.6)
    assert len(out) == 2, "coaxial through-channel ends are not a racetrack slot"


def test_slot_pair_rejects_radius_mismatch():
    a = _hole([0, 0, 0], 2.0, axis_dim=2)
    b = _hole([0, 4.0, 0], 2.6, axis_dim=2)     # radii differ ~26% > 5%
    out = so._pair_slot_halves([a, b], merge_tol=2.0)
    assert len(out) == 2


# ---------------------------------------------------------------------------
# _rect_openings
# ---------------------------------------------------------------------------

def _fillet(center, radius=0.5, axis_dim=2):
    return {"center": np.asarray(center, float), "radius": float(radius),
            "axis_dim": int(axis_dim), "ext": np.zeros(3), "kind": "cyl"}


def test_rect_cluster_detects_fillet_quad():
    # cluster-size floor was raised 2 -> 4 (2026-07-09, RESULTS 11c): 2-fillet
    # clusters were indistinguishable from cosmetic rounds and flooded FPs, so a
    # detection now needs a proper 4-corner fillet set (rect slot signature).
    bbox = np.array([0, 0, 0, 10, 50, 40], float)
    cyls = [_fillet([2.3, 5.0, 22.9]), _fillet([3.3, 5.0, 22.9]),
            _fillet([2.3, 6.2, 22.9]), _fillet([3.3, 6.2, 22.9])]
    ops = so._rect_openings(cyls, rmin=1.3, bbox=bbox, existing_ops=[],
                            merge_tol=2.8)
    assert len(ops) == 1
    assert ops[0]["kind"] == "rect"
    np.testing.assert_allclose(ops[0]["entry_point"][1], 5.6, atol=1e-9)


def test_rect_cluster_rejects_fillet_pair():
    # a mere PAIR of fillets (old floor) must NOT fire anymore -- measured on the
    # 3-part human-GT eval, pair-level clusters produced ~14 FPs (RESULTS 11c).
    bbox = np.array([0, 0, 0, 10, 50, 40], float)
    cyls = [_fillet([2.8, 5.0, 22.9]), _fillet([2.8, 6.2, 22.9])]
    ops = so._rect_openings(cyls, rmin=1.3, bbox=bbox, existing_ops=[],
                            merge_tol=2.8)
    assert ops == []


def test_rect_cluster_rejects_large_cosmetic_run():
    # a fillet run wider than a plausible wire slot (>8mm perpendicular extent)
    bbox = np.array([0, 0, 0, 10, 50, 40], float)
    cyls = [_fillet([2.8, y, 22.9]) for y in (0.0, 4.5, 9.0)]   # 9mm spread
    ops = so._rect_openings(cyls, rmin=1.3, bbox=bbox, existing_ops=[],
                            merge_tol=2.8)
    assert ops == []


def test_rect_cluster_suppressed_near_existing_opening():
    bbox = np.array([0, 0, 0, 10, 50, 40], float)
    cyls = [_fillet([2.3, 5.0, 22.9]), _fillet([3.3, 5.0, 22.9]),
            _fillet([2.3, 6.2, 22.9]), _fillet([3.3, 6.2, 22.9])]
    existing = [{"entry_point": [2.8, 5.6, 22.9], "approach_vector": [0, 0, 1],
                 "radius_mm": 2.0, "n_faces": 1, "kind": "hole"}]
    ops = so._rect_openings(cyls, rmin=1.3, bbox=bbox, existing_ops=existing,
                            merge_tol=2.8)
    assert ops == [], "must not duplicate an opening the bore path already has"


def test_rect_cluster_ignores_large_radius_fillets():
    # 0.7-1.2mm rounds are housing cosmetics, not slot corner fillets
    bbox = np.array([0, 0, 0, 10, 50, 40], float)
    cyls = [_fillet([2.3, 5.0, 22.9], radius=0.9),
            _fillet([3.3, 5.0, 22.9], radius=0.9),
            _fillet([2.3, 6.2, 22.9], radius=0.9),
            _fillet([3.3, 6.2, 22.9], radius=0.9)]
    ops = so._rect_openings(cyls, rmin=1.3, bbox=bbox, existing_ops=[],
                            merge_tol=2.8)
    assert ops == []


# ---------------------------------------------------------------------------
# validate_directions
# ---------------------------------------------------------------------------

def _plate_with_bore():
    """Flat plate at z in [-4, 0] with a vertical bore; opening at the top."""
    xs, ys = np.meshgrid(np.linspace(-15, 15, 31), np.linspace(-15, 15, 31))
    top = np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], axis=1)
    bot = top + [0, 0, -4.0]
    theta = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    wall = [np.stack([2.0 * np.cos(theta), 2.0 * np.sin(theta),
                      np.full_like(theta, z)], axis=1)
            for z in np.linspace(0, -4, 5)]
    return np.concatenate([top, bot] + wall, axis=0)


def test_validate_directions_flips_inward_vector():
    V = _plate_with_bore()
    ops = [{"entry_point": [0.0, 0.0, 0.0], "approach_vector": [0, 0, -1.0],
            "radius_mm": 2.0}]                     # WRONG: points into the bore
    flips = so.validate_directions(ops, V, part_nr="test")
    assert flips == 1
    np.testing.assert_allclose(ops[0]["approach_vector"], [0, 0, 1.0], atol=1e-9)


def test_validate_directions_keeps_correct_vector():
    V = _plate_with_bore()
    ops = [{"entry_point": [0.0, 0.0, 0.0], "approach_vector": [0, 0, 1.0],
            "radius_mm": 2.0}]                     # correct: out into free space
    flips = so.validate_directions(ops, V, part_nr="test")
    assert flips == 0
    np.testing.assert_allclose(ops[0]["approach_vector"], [0, 0, 1.0], atol=1e-9)


# ---------------------------------------------------------------------------
# cad_eval.align_frames
# ---------------------------------------------------------------------------

def test_align_frames_recovers_permutation_and_translation():
    rng = np.random.default_rng(7)
    V_gt = rng.uniform(0, 60, (500, 3))
    R_true = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float)  # proper rotation
    t_true = np.array([-12.0, 30.0, 5.0])
    V_step = (V_gt - t_true) @ R_true            # inverse mapping into STEP frame
    R, t, res = ce.align_frames(V_step, V_gt)
    assert res < 0.5, f"alignment residual too high: {res}"
    p_gt = V_gt[0]
    p_back = R @ ((p_gt - t_true) @ R_true) + t
    np.testing.assert_allclose(p_back, p_gt, atol=0.5)


def test_catalog_nr_extraction():
    assert ce.catalog_nr("wscaduniverse_3211813_2026-06-21-22-25-47") == "3211813"
    assert ce.catalog_nr("wscaduniverse_870-681_2026-01-01-00-00-00") == "870-681"
    assert ce.catalog_nr("PXC.3211813") is None


# --- mouth_coord: the A2 label-convention fix (RESULTS 11s -> 11u) -------------
# The blind `centre + depth/2` push overshot OUTSIDE the part on funnel+bore
# merges, because the merged centre is a MEAN of member centres while `depth` is
# the DEEPEST member's extent. mouth_coord takes each member face's own outer end
# instead -- a face ends AT the mouth, so its extreme IS the surface.

def _member(center, ext):
    return {"center": np.asarray(center, float), "ext": np.asarray(ext, float)}


def test_mouth_coord_single_bore_lands_on_face_not_mid_depth():
    # one bore, axis z, centre-of-mass at z=5, 10mm deep -> mouth at z=10.
    bbox = np.array([0., 0., 0., 20., 20., 10.])
    m = [_member([10, 10, 5], [6, 6, 10])]
    assert so.mouth_coord(m, 2, +1.0, bbox) == 10.0
    # opening the other way: mouth at z=0
    assert so.mouth_coord(m, 2, -1.0, bbox) == 0.0


def test_mouth_coord_funnel_bore_merge_does_not_overshoot():
    # THE 11s FAILURE: a shallow 2mm lead-in funnel (com z=9) merged with a deep
    # 10mm bore (com z=5). Blind push used mean-centre (7.0) + depth/2 (5.0) = 12.0
    # -- 2mm OUTSIDE a part that ends at z=10. The true mouth is the funnel's
    # outer end, z=10.
    bbox = np.array([0., 0., 0., 20., 20., 10.])
    members = [_member([10, 10, 5], [6, 6, 10]),    # deep bore
               _member([10, 10, 9], [8, 8, 2])]     # shallow funnel
    mouth = so.mouth_coord(members, 2, +1.0, bbox)
    assert mouth == 10.0
    blind = np.mean([m["center"][2] for m in members]) + 0.5 * 10.0
    assert blind > bbox[5]          # the old push really did leave the part
    assert mouth <= bbox[5]         # the new one cannot


def test_mouth_coord_never_leaves_the_part():
    # even a stray/oversized face extent stays clamped inside the bbox
    bbox = np.array([0., 0., 0., 20., 20., 10.])
    m = [_member([10, 10, 5], [6, 6, 40])]          # absurd 40mm extent
    assert so.mouth_coord(m, 2, +1.0, bbox) == 10.0
    assert so.mouth_coord(m, 2, -1.0, bbox) == 0.0


# --- plausible_openings: the physical-plausibility label filter ------------------
# Measured 2026-07-14 over corpus v8: 1119 of 2951 parts (37.9%) carry "connection
# points" on FOUR OR MORE faces and hold 53.6% of ALL ground truth. A wire enters a
# terminal block through one face (two on a feed-through), and a product accepts one
# or two wire gauges -- so openings scattered over six faces in six radius classes are
# mounting/screw/vent holes, not entries. These tests pin the rules AND the guard that
# stops this filter from repeating RESULTS 11s (which deleted 73% of all CPs and left
# 25 of 92 parts with ZERO -- a terminal block cannot have zero entries).

def _op(direction, radius, depth=None):
    o = {"entry_point": [0.0, 0.0, 0.0],
         "approach_vector": list(map(float, direction)),
         "radius_mm": float(radius), "n_faces": 1, "kind": "hole"}
    if depth is not None:
        o["depth_mm"] = float(depth)
    return o


def test_plausible_keeps_a_clean_single_face_block_untouched():
    """The control case: one face, one radius cluster -> nothing may be dropped."""
    ops = [_op([1, 0, 0], 1.6, 5.0) for _ in range(6)]
    out = so.plausible_openings(ops, part_nr="clean")
    assert len(out) == 6


def test_plausible_keeps_BOTH_faces_of_a_feed_through_block():
    """Why this is not --dir-consensus: that keeps only the SINGLE dominant group and
    would decapitate a feed-through block. top-2 must keep +X and -X."""
    ops = ([_op([1, 0, 0], 1.6) for _ in range(4)] +
           [_op([-1, 0, 0], 1.6) for _ in range(4)])
    out = so.plausible_openings(ops, part_nr="feedthrough")
    axes = {so._axis_key(o["approach_vector"]) for o in out}
    assert len(out) == 8
    assert axes == {(0, "+"), (0, "-")}


def test_plausible_drops_minor_face_holes():
    """6 entries on the front + 4 mounting/screw holes scattered over other faces."""
    ops = ([_op([1, 0, 0], 1.6) for _ in range(6)] +
           [_op([0, 0, -1], 1.6), _op([0, 1, 0], 1.6),
            _op([0, -1, 0], 1.6), _op([0, 0, 1], 1.6)])
    out = so.plausible_openings(ops, topk=1, part_nr="mounting")
    assert len(out) == 6
    assert all(so._axis_key(o["approach_vector"]) == (0, "+") for o in out)


def test_plausible_drops_the_wrong_radius_class():
    """A product takes one or two gauges; a 4.5mm hole among 1.5mm entries is a screw."""
    ops = ([_op([1, 0, 0], 1.5) for _ in range(8)] +
           [_op([1, 0, 0], 4.5) for _ in range(3)])
    out = so.plausible_openings(ops, radius_buckets=1, part_nr="gauge")
    assert len(out) == 8
    assert all(abs(o["radius_mm"] - 1.5) < 1e-9 for o in out)


def test_radius_rule_is_scoped_to_the_kept_faces():
    """The two rules must not fight. If the radius histogram were taken over ALL
    openings, a part with MANY identical screw holes on its minor faces would make
    SCREW the dominant gauge and the filter would delete the real entries. Scoping the
    radius rule to the surviving face makes the gauge come from the entries themselves.
    Here: 3 real 1.5mm entries on +X, and 4 screw holes at 4.5mm on other faces."""
    ops = ([_op([1, 0, 0], 1.5) for _ in range(3)] +
           [_op([0, 0, 1], 4.5) for _ in range(2)] +
           [_op([0, 1, 0], 4.5) for _ in range(2)])
    out = so.plausible_openings(ops, topk=1, radius_buckets=1, part_nr="screwy")
    assert len(out) == 3
    assert all(abs(o["radius_mm"] - 1.5) < 1e-9 for o in out)


def test_plausible_never_empties_a_part():
    """THE 11s GUARD: a terminal block with zero wire entries is always a labelling
    failure, and silently producing one is how RESULTS 11s destroyed 25 of 92 parts.
    Force the pathological case by monkeypatching the rules to select nothing."""
    ops = [_op([1, 0, 0], 1.5), _op([1, 0, 0], 1.5), _op([0, 0, 1], 4.5)]
    out = so.plausible_openings(ops, topk=0, radius_buckets=0, part_nr="degenerate")
    assert len(out) >= 1           # never zero, whatever the knobs say


def test_plausible_survives_missing_depth_mm():
    """The rect / rect_pocket paths never emit depth_mm -- the filter must not read it."""
    ops = [_op([1, 0, 0], 1.6) for _ in range(4)] + [_op([0, 0, 1], 1.6)]
    out = so.plausible_openings(ops, topk=1, part_nr="rect")
    assert len(out) == 4
    assert all("depth_mm" not in o for o in out)


def test_plausible_leaves_tiny_parts_alone():
    """Under 3 openings there is no pattern to infer -- do not guess."""
    ops = [_op([1, 0, 0], 1.5), _op([0, 0, 1], 4.5)]
    assert len(so.plausible_openings(ops, part_nr="tiny")) == 2
