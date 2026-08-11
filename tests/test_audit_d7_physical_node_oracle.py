import numpy as np

import audit_d7_physical_node_oracle as audit


def _node(point, directions, member="n"):
    return audit.make_node(point, directions, "test", member)


def test_one_resource_node_can_claim_only_one_gt():
    # Both GT axes pass the same physical location, but a resource cluster is one
    # robot action and must not be counted twice by the oracle.
    nodes = audit.cluster_nodes([_node([0, 0, 0], [[1, 0, 0]], "only")])
    gt = np.array([[0, 0, 0], [0.5, 0, 0]], float)
    gd = np.array([[1, 0, 0], [1, 0, 0]], float)
    assert audit.match_nodes(nodes, gt, gd, 10.0, robot=False) == 1


def test_spatial_dedupe_keeps_one_candidate_row_capacity():
    nodes = audit.cluster_nodes(
        [
            _node([0, 0, 0], [[1, 0, 0]], "face-a"),
            _node([0.2, 0, 0], [[0, 1, 0]], "face-b"),
        ],
        tol=0.5,
    )
    assert len(nodes) == 1
    assert nodes[0]["member_ids"] == ["face-a", "face-b"]
    gt = np.array([[0, 0, 0], [0.3, 0, 0]], float)
    gd = np.array([[1, 0, 0], [1, 0, 0]], float)
    assert audit.match_nodes(nodes, gt, gd, 10.0, robot=False) == 1


def test_local_direction_factor_obeys_radius_without_creating_pose_rows():
    # The origin node has a wrong own axis.  A useful X axis exists 7.9 mm away,
    # but its own position is >2 mm lateral to the GT and cannot score directly.
    nodes = audit.cluster_nodes(
        [
            _node([0, 0, 0], [[0, 0, 1]], "position"),
            _node([0, 7.9, 0], [[1, 0, 0]], "direction-source"),
        ],
        tol=0.5,
    )
    gt = np.array([[0, 0, 0]], float)
    gd = np.array([[1, 0, 0]], float)
    assert len(nodes) == 2  # factorisation must not expand/collapse candidate rows
    assert audit.match_nodes_direction_factor(
        nodes, gt, gd, 20.0, mode="local", local_mm=8.0
    ) == 1
    assert audit.match_nodes_direction_factor(
        nodes, gt, gd, 20.0, mode="local", local_mm=7.0
    ) == 0


def test_obb_axis_is_an_explicit_signed_factor():
    nodes = audit.cluster_nodes([_node([0, 0, 0], [[0, 0, 1]])])
    gt = np.array([[0, 0, 0]], float)
    # -X verifies that the OBB axis is explicitly factorised into both signs.
    gd = np.array([[-1, 0, 0]], float)
    assert audit.match_nodes_direction_factor(
        nodes, gt, gd, 10.0, mode="obb", part_obb_axes=[[1, 0, 0]]
    ) == 1
    assert audit.match_nodes_direction_factor(
        nodes, gt, gd, 10.0, mode="obb", part_obb_axes=[[0, 1, 0], [0, 0, 1]]
    ) == 0


def test_own_direction_branch_remains_strictly_signed():
    gt = np.array([[0, 0, 0]], float)
    gd = np.array([[-1, 0, 0]], float)
    one_sign = audit.cluster_nodes([_node([0, 0, 0], [[1, 0, 0]])])
    both_signs = audit.cluster_nodes([_node([0, 0, 0], [[1, 0, 0], [-1, 0, 0]])])
    assert audit.match_nodes(one_sign, gt, gd, 10.0, robot=True) == 0
    assert audit.match_nodes(both_signs, gt, gd, 10.0, robot=True) == 1
