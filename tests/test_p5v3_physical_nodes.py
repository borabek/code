import numpy as np

import p5v3_physical_nodes as PN


def _cyl(point, axis=(0, 0, 1), radius=2.0, length=4.0):
    p = np.asarray(point, float)
    a = np.asarray(axis, float); a /= np.linalg.norm(a)
    return {"center": p + 0.5 * length * a, "axis": a, "radius": radius,
            "mouth_a": p, "mouth_b": p + length * a}


def test_polygon_centroid_is_not_sampling_mean():
    # Rectangle with deliberately uneven samples along the bottom edge.
    p = np.array([[0., 0, 0], [1., 0, 0], [2., 0, 0], [4., 0, 0],
                  [4., 2, 0], [0., 2, 0]])
    c = PN.polygon_area_centroid(p, [0, 0, 1])
    assert np.allclose(c, [2., 1., 0.], atol=1e-8)


def test_colocated_sources_become_one_resource_and_keep_axes():
    cyl = [_cyl([0, 0, 0], [0, 0, 1])]
    planar = [{"center": np.array([0.1, 0, 0]), "normal": np.array([1., 0, 0]),
               "esd_r": 2.0, "alan": 12.0, "cevre": 14.0}]
    nodes = PN.build_nodes(cyl, planar, dedupe_mm=0.5)
    at_origin = min(nodes, key=lambda n: np.linalg.norm(n["point"]))
    assert at_origin["n_members"] == 2
    assert set(at_origin["kinds"]) == {PN.CYLINDER, PN.PLANAR}
    assert len(at_origin["axes"]) == 2


def test_neighbouring_real_mouths_are_not_merged():
    nodes = PN.build_nodes([_cyl([0, 0, 0]), _cyl([1.0, 0, 0])], [],
                           dedupe_mm=0.5)
    mouths = [n for n in nodes if abs(float(n["point"][2])) < 1e-8]
    assert len(mouths) == 2


def test_radius_filter_and_signed_directions():
    nodes = PN.build_nodes([_cyl([0, 0, 0], radius=0.2),
                            _cyl([2, 0, 0], radius=2.0)], [])
    assert len(nodes) == 2  # two mouths of only the valid cylinder
    d = PN.signed_directions(nodes[0])
    assert len(d) == 2 and np.allclose(d[0], -d[1])
