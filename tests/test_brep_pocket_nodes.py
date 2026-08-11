import numpy as np

from brep_pocket_nodes import infer_rectangular_pockets, polygon_metrics


def _wall(center, normal, points, name):
    return {
        "center": np.asarray(center, float),
        "normal": np.asarray(normal, float),
        "points": np.asarray(points, float),
        "area": 1.0,
        "resource": name,
    }


def _rectangular_pocket_walls():
    # 2 x 3 mm pocket, common depth z=0..8.  Normals face into the cavity.
    return [
        _wall([-1, 0, 4], [1, 0, 0], [[-1, -1.5, 0], [-1, 1.5, 0], [-1, 1.5, 8], [-1, -1.5, 8]], "x0"),
        _wall([1, 0, 4], [-1, 0, 0], [[1, -1.5, 0], [1, -1.5, 8], [1, 1.5, 8], [1, 1.5, 0]], "x1"),
        _wall([0, -1.5, 4], [0, 1, 0], [[-1, -1.5, 0], [-1, -1.5, 8], [1, -1.5, 8], [1, -1.5, 0]], "y0"),
        _wall([0, 1.5, 4], [0, -1, 0], [[-1, 1.5, 0], [1, 1.5, 0], [1, 1.5, 8], [-1, 1.5, 8]], "y1"),
    ]


def test_polygon_metrics_uses_area_centroid():
    # Deliberately uneven boundary sampling; the vertex mean is not the center.
    points = [[0, 0, 0], [1, 0, 0], [2, 0, 0], [4, 0, 0], [4, 2, 0], [0, 2, 0]]
    result = polygon_metrics(points, [0, 0, 1])
    assert result is not None
    assert np.allclose(result["center"], [2, 1, 0])
    assert np.isclose(result["area"], 8.0)


def test_four_opposing_walls_make_one_physical_pocket():
    part_points = np.array([[-5, -5, -10], [5, 5, 8]], float)
    nodes = infer_rectangular_pockets(_rectangular_pocket_walls(), part_points=part_points)
    assert len(nodes) == 1
    node = nodes[0]
    assert np.allclose(node["center"], [0, 0, 8], atol=1e-7)
    assert np.isclose(node["width"], 2.0)
    assert np.isclose(node["height"], 3.0)
    assert np.isclose(node["depth"], 8.0)
    assert node["support_faces"] == 4


def test_missing_wall_is_rejected():
    assert infer_rectangular_pockets(_rectangular_pocket_walls()[:3]) == []


def test_outer_housing_span_is_rejected():
    walls = _rectangular_pocket_walls()
    walls[0]["center"][0] = -20
    walls[1]["center"][0] = 20
    walls[0]["points"][:, 0] = -20
    walls[1]["points"][:, 0] = 20
    assert infer_rectangular_pockets(walls, max_span=12.0) == []
