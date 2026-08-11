import pickle

import numpy as np

import p5v3_selector_exp as P5


def test_part_cache_has_stable_importable_pickle_identity():
    part = P5.Part(
        pid="x", mfg="m", diag=1.0, gt_points=np.zeros((0, 3)),
        gt_directions=np.zeros((0, 3)), points=np.zeros((0, 3)),
        node_x=np.zeros((0, 1)), directions=[], direction_x=[],
        accept_y=np.zeros(0, np.int8), direction_y=[],
        source_counts=np.zeros((0, 3), int),
    )
    blob = pickle.dumps(part)
    assert b"__main__" not in blob
    assert isinstance(pickle.loads(blob), P5.Part)


def test_position_is_decomposed_on_gt_axis_not_proposed_axis():
    # Point is 3 mm lateral to the GT z axis, hence it must fail even though a
    # proposal-axis decomposition along x would incorrectly call it axial.
    point = np.array([[3.0, 0.0, 0.0]])
    directions = [np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])]
    y, dy = P5.assign_strict_labels(
        point, directions, np.array([[0.0, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 1.0]]),
    )
    assert y.tolist() == [0]
    assert dy[0].tolist() == [0, 0]


def test_signed_direction_and_explicit_null_labels():
    points = np.array([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
    directions = [
        np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]),
        np.array([[0.0, 0.0, 1.0]]),
    ]
    y, dy = P5.assign_strict_labels(
        points, directions, np.array([[0.0, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 1.0]]),
    )
    assert y.tolist() == [1, 0]       # second resource is explicit NULL
    assert dy[0].tolist() == [1, 0]   # signed, not abs(dot)
    assert dy[1].tolist() == [0]


def test_one_resource_capacity_for_two_nearby_gt_axes():
    # A single physical resource can satisfy both GT boxes but Hungarian may
    # assign it to only one; it must not create two positives.
    points = np.array([[0.0, 0.0, 0.0]])
    directions = [np.array([[0.0, 0.0, 1.0]])]
    y, dy = P5.assign_strict_labels(
        points, directions,
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
    )
    assert int(y.sum()) == 1
    assert int(dy[0].sum()) == 1


def test_equivalent_correct_directions_are_not_arbitrarily_negative():
    a = np.deg2rad(5.0)
    directions = [np.array([[0.0, 0.0, 1.0],
                            [np.sin(a), 0.0, np.cos(a)],
                            [1.0, 0.0, 0.0]])]
    y, dy = P5.assign_strict_labels(
        np.zeros((1, 3)), directions, np.zeros((1, 3)),
        np.array([[0.0, 0.0, 1.0]]),
    )
    assert y.tolist() == [1]
    assert dy[0].tolist() == [1, 1, 0]
