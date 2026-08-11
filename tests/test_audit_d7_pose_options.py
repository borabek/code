import numpy as np

import audit_d7_pose_options as options
from audit_d7_pose_dictionary_oracle import maximum_one_to_one


def _cylinder():
    return {
        "axis": np.array([0.0, 0.0, 1.0]),
        "radius": 1.0,
        "mouth_a": np.array([0.0, 0.0, 0.0]),
        "mouth_b": np.array([0.0, 0.0, 10.0]),
    }


def test_catalog_has_endpoint_projection_and_planar_branches():
    catalog = options.build_catalog(
        [1.0, 0.0, 5.0],
        [1.0, 0.0, 0.0],
        [_cylinder()],
        [{"center": [1.0, 0.0, 5.0], "normal": [0.0, 1.0, 0.0]}],
        mm_max=8.0,
    )
    assert catalog.current
    assert catalog.cylinder_endpoint
    assert catalog.cylinder_projection
    assert catalog.planar_center
    projected = {tuple(np.round(x.point, 6)) for x in catalog.cylinder_projection}
    assert (0.0, 0.0, 5.0) in projected


def test_cartesian_branch_crosses_position_and_direction():
    catalog = options.build_catalog(
        [1.0, 0.0, 5.0],
        [1.0, 0.0, 0.0],
        [_cylinder()],
        [{"center": [1.0, 0.0, 5.0], "normal": [0.0, 1.0, 0.0]}],
        mm_max=8.0,
    )
    dictionary = options.branch_options("dictionary_all", catalog)
    cartesian = options.branch_options("cartesian_all", catalog)
    assert len(cartesian) > len(dictionary)


def test_matching_is_one_to_one_and_uses_gt_axis_for_lateral():
    pose = options.PoseOption(
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        "current",
        "current",
    )
    # One candidate cannot claim both colocated GTs.
    tp, matches, _ = maximum_one_to_one(
        [[pose]],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
    )
    assert tp == 1
    assert len(matches) == 1

    # Lateral is around the GT Z axis: a 3 mm X offset must fail, regardless of
    # what an alternative predicted-axis decomposition might say.
    shifted = options.PoseOption(
        np.array([3.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        "current",
        "current",
    )
    tp, _, _ = maximum_one_to_one(
        [[shifted]], [[0.0, 0.0, 0.0]], [[0.0, 0.0, 1.0]]
    )
    assert tp == 0
