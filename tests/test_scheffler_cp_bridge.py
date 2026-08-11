import numpy as np

from derive_scheffler_cp_candidates import align_exact_step_to_obj


def _asymmetric_closed_mesh():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [0.4, 3.0, 0.0],
            [0.2, 0.6, 2.0],
        ],
        dtype=float,
    )
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=int)
    return vertices, faces


def test_recovers_but_quarantines_nonidentity_exact_frame():
    step_vertices, faces = _asymmetric_closed_mesh()
    rotation = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    translation = np.array([11.0, -4.0, 7.0])
    obj_vertices = step_vertices @ rotation.T + translation

    result = align_exact_step_to_obj(
        step_vertices,
        faces,
        obj_vertices,
        faces,
        seed=7,
        sample_count=12_000,
        mean_tolerance_mm=0.2,
        p95_tolerance_mm=0.5,
        bbox_absolute_tolerance_mm=0.01,
        center_tolerance_mm=0.01,
    )
    recovered_rotation = np.asarray(result["rotation_step_to_obj"])
    recovered_translation = np.asarray(result["translation_step_to_obj"])
    assert not result["accepted"]
    assert result["status"] == "review_required_alignment"
    assert np.array_equal(recovered_rotation, rotation)
    assert np.allclose(recovered_translation, translation)

    point_step = np.array([1.2, 0.7, 0.4])
    point_obj = rotation @ point_step + translation
    recovered_step = (point_obj - recovered_translation) @ recovered_rotation
    assert np.allclose(recovered_step, point_step)


def test_exact_identity_frame_is_preferred():
    step_vertices, faces = _asymmetric_closed_mesh()
    result = align_exact_step_to_obj(
        step_vertices,
        faces,
        step_vertices.copy(),
        faces,
        seed=9,
        sample_count=8_000,
        mean_tolerance_mm=0.2,
        p95_tolerance_mm=0.5,
        bbox_absolute_tolerance_mm=0.01,
        center_tolerance_mm=0.01,
    )
    assert result["accepted"]
    assert result["identity_selected"]
    assert np.allclose(result["translation_step_to_obj"], np.zeros(3))
