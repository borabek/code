"""Derive reviewable CP candidates from human CableEntry semantic regions.

This command is deliberately development-only (71 train + 20 validation).  It
never opens ``test_locked``.  Outputs are named ``human-region-derived`` because
the humans labelled regions, while point positions and directions are geometric
derivations that still require QA before they become CP ground truth.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import connector3d
from connector_constants import CABLE_ENTRY, CONTACT
import scheffler_dataset as dataset
import step_openings
import step_to_json


HERE = Path(__file__).resolve().parent


def _proper_signed_permutations() -> list[np.ndarray]:
    matrices = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            matrix = np.zeros((3, 3), dtype=float)
            for row, (column, sign) in enumerate(zip(permutation, signs)):
                matrix[row, column] = sign
            if np.linalg.det(matrix) > 0.5:
                matrices.append(matrix)
    return matrices


ROTATIONS = _proper_signed_permutations()


def _sample_vertices(
    vertices: np.ndarray, count: int, rng: np.random.Generator
) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=float)
    if len(vertices) <= count:
        return vertices
    return vertices[rng.choice(len(vertices), count, replace=False)]


def _score_alignment(
    step_vertices: np.ndarray,
    obj_vertices: np.ndarray,
    step_sample: np.ndarray,
    obj_sample: np.ndarray,
    step_tree: cKDTree,
    obj_tree: cKDTree,
    rotation: np.ndarray,
) -> dict:
    step_center = 0.5 * (step_vertices.min(axis=0) + step_vertices.max(axis=0))
    obj_center = 0.5 * (obj_vertices.min(axis=0) + obj_vertices.max(axis=0))
    translation = obj_center - rotation @ step_center

    # Row-vector forms of x_obj = R @ x_step + t and its inverse.
    obj_in_step = (obj_sample - translation) @ rotation
    step_in_obj = step_sample @ rotation.T + translation
    d_obj_to_step = step_tree.query(obj_in_step, k=1)[0]
    d_step_to_obj = obj_tree.query(step_in_obj, k=1)[0]
    distances = np.concatenate([d_obj_to_step, d_step_to_obj])

    obj_span = obj_vertices.max(axis=0) - obj_vertices.min(axis=0)
    transformed_step_span = np.abs(rotation) @ (
        step_vertices.max(axis=0) - step_vertices.min(axis=0)
    )
    bbox_relative_error = float(
        np.max(np.abs(transformed_step_span - obj_span) / np.maximum(obj_span, 0.1))
    )
    bbox_span_delta = obj_span - transformed_step_span
    return {
        "rotation": rotation,
        "translation": translation,
        # Primary registration residual follows sparse labelled OBJ -> denser
        # STEP. The reverse vertex distance is density-biased, so it is retained
        # as a diagnostic rather than used as the acceptance gate.
        "mean_obj_to_step_nn_mm": float(d_obj_to_step.mean()),
        "p95_obj_to_step_nn_mm": float(np.percentile(d_obj_to_step, 95)),
        "max_obj_to_step_nn_mm": float(d_obj_to_step.max()),
        "mean_step_to_obj_nn_mm_density_biased": float(d_step_to_obj.mean()),
        "mean_symmetric_vertex_nn_mm_diagnostic": float(distances.mean()),
        "bbox_relative_error_max": bbox_relative_error,
        "bbox_span_delta_mm": bbox_span_delta,
        "bbox_span_delta_abs_max_mm": float(np.max(np.abs(bbox_span_delta))),
        "translation_norm_mm": float(np.linalg.norm(translation)),
    }


def align_exact_step_to_obj(
    step_vertices: np.ndarray,
    step_faces: np.ndarray,
    obj_vertices: np.ndarray,
    obj_faces: np.ndarray,
    *,
    seed: int,
    sample_count: int = 1500,
    mean_tolerance_mm: float = 0.80,
    p95_tolerance_mm: float = 1.10,
    bbox_absolute_tolerance_mm: float = 2.0,
    center_tolerance_mm: float = 0.10,
    identity_slack_mm: float = 0.05,
) -> dict:
    """Find a conservative rigid frame map and return full QA diagnostics."""
    step_vertices = np.asarray(step_vertices, dtype=float)
    obj_vertices = np.asarray(obj_vertices, dtype=float)
    rng = np.random.default_rng(seed)
    step_sample = _sample_vertices(step_vertices, sample_count, rng)
    obj_sample = _sample_vertices(obj_vertices, sample_count, rng)
    step_tree = cKDTree(step_vertices)
    obj_tree = cKDTree(obj_vertices)
    scored = [
        _score_alignment(
            step_vertices,
            obj_vertices,
            step_sample,
            obj_sample,
            step_tree,
            obj_tree,
            rotation,
        )
        for rotation in ROTATIONS
    ]
    scored.sort(key=lambda item: item["mean_obj_to_step_nn_mm"])
    numerical_best = scored[0]
    identity = next(item for item in scored if np.allclose(item["rotation"], np.eye(3)))
    if (
        identity["mean_obj_to_step_nn_mm"]
        <= numerical_best["mean_obj_to_step_nn_mm"] + identity_slack_mm
    ):
        selected = identity
        selection_reason = "exact_source_identity_frame_within_best_slack"
    else:
        selected = numerical_best
        selection_reason = "best_proper_signed_axis_permutation"

    alternatives = [item for item in scored if not np.array_equal(item["rotation"], selected["rotation"])]
    second = alternatives[0]
    margin_mm = (
        second["mean_obj_to_step_nn_mm"]
        - selected["mean_obj_to_step_nn_mm"]
    )
    margin_ratio = margin_mm / max(selected["mean_obj_to_step_nn_mm"], 1e-9)
    identity_selected = bool(np.allclose(selected["rotation"], np.eye(3)))
    thresholds_ok = (
        selected["mean_obj_to_step_nn_mm"] <= mean_tolerance_mm
        and selected["p95_obj_to_step_nn_mm"] <= p95_tolerance_mm
        and selected["bbox_span_delta_abs_max_mm"] <= bbox_absolute_tolerance_mm
        and selected["translation_norm_mm"] <= center_tolerance_mm
    )
    # Exact-hash development audit: 15/15 publisher conversions preserve the
    # STEP axes. A non-identity result is therefore quarantined, never accepted.
    orientation_ok = identity_selected
    accepted = bool(thresholds_ok and orientation_ok)
    return {
        "accepted": accepted,
        "status": "accepted_for_candidate_mapping" if accepted else "review_required_alignment",
        "selection_reason": selection_reason,
        "rotation_step_to_obj": selected["rotation"].tolist(),
        "translation_step_to_obj": selected["translation"].tolist(),
        "mean_obj_to_step_nn_mm": selected["mean_obj_to_step_nn_mm"],
        "p95_obj_to_step_nn_mm": selected["p95_obj_to_step_nn_mm"],
        "max_obj_to_step_nn_mm": selected["max_obj_to_step_nn_mm"],
        "mean_step_to_obj_nn_mm_density_biased": selected[
            "mean_step_to_obj_nn_mm_density_biased"
        ],
        "mean_symmetric_vertex_nn_mm_diagnostic": selected[
            "mean_symmetric_vertex_nn_mm_diagnostic"
        ],
        "bbox_relative_error_max": selected["bbox_relative_error_max"],
        "bbox_span_delta_mm": selected["bbox_span_delta_mm"].tolist(),
        "bbox_span_delta_abs_max_mm": selected["bbox_span_delta_abs_max_mm"],
        "translation_norm_mm": selected["translation_norm_mm"],
        "best_alternative_mean_mm": second["mean_obj_to_step_nn_mm"],
        "best_alternative_margin_mm": margin_mm,
        "best_alternative_margin_ratio": margin_ratio,
        "identity_selected": identity_selected,
        "thresholds": {
            "mean_obj_to_step_nn_mm_max": mean_tolerance_mm,
            "p95_obj_to_step_nn_mm_max": p95_tolerance_mm,
            "bbox_span_delta_abs_max_mm": bbox_absolute_tolerance_mm,
            "translation_norm_mm_max": center_tolerance_mm,
            "identity_rotation_required": True,
            "identity_best_slack_mm": identity_slack_mm,
        },
    }


def _seed_for(part_id: str) -> int:
    return int(hashlib.sha256(part_id.encode("utf-8")).hexdigest()[:8], 16)


def _component_candidates(sample: dict) -> tuple[list[connector3d.ConnectionPoint], int]:
    fragments = connector3d.build_fragments(
        sample["verts"], sample["faces"], sample["labels"], min_vertices=1
    )
    cable_fragments = [fragment for fragment in fragments if int(fragment.label) == int(CABLE_ENTRY)]
    contact_count = sum(int(fragment.label) == int(CONTACT) for fragment in fragments)
    body_center = np.asarray(sample["verts"], dtype=float).mean(axis=0)
    candidates = []
    for fragment in cable_fragments:
        candidate = connector3d.ConnectionPoint([fragment])
        candidate.compute_direction(
            sample["verts"], sample["faces"], smooth_subdiv=0, body_center=body_center
        )
        candidates.append(candidate)
    return candidates, contact_count


def _vector(value: np.ndarray | None) -> list[float] | None:
    if value is None:
        return None
    return [float(component) for component in np.asarray(value, dtype=float)]


def _map_candidates(candidates: list[connector3d.ConnectionPoint], alignment: dict, step_vertices: np.ndarray) -> list[dict]:
    rotation = np.asarray(alignment["rotation_step_to_obj"], dtype=float)
    translation = np.asarray(alignment["translation_step_to_obj"], dtype=float)
    step_tree = cKDTree(np.asarray(step_vertices, dtype=float))
    output = []
    for index, candidate in enumerate(candidates):
        point_obj = np.asarray(candidate.entry_point, dtype=float)
        direction_obj = np.asarray(candidate.approach_vector, dtype=float)
        point_step = (point_obj - translation) @ rotation
        direction_step = direction_obj @ rotation
        norm = np.linalg.norm(direction_step)
        if norm > 1e-12:
            direction_step = direction_step / norm

        probe = {
            "entry_point": point_step.tolist(),
            "approach_vector": direction_step.tolist(),
            "radius_mm": max(float(candidate.fragments[0].radius), 0.5),
        }
        flipped = False
        if alignment["accepted"]:
            flipped = bool(
                step_openings.validate_directions([probe], step_vertices, part_nr="bridge")
            )
        direction_step_checked = np.asarray(probe["approach_vector"], dtype=float)
        if flipped:
            direction_obj_checked = direction_step_checked @ rotation.T
        else:
            direction_obj_checked = direction_obj

        output.append(
            {
                "candidate_index": index,
                "provenance_label": "human-region-derived",
                "semantic_source": "human CableEntry vertex connected component",
                "human_clicked_point": False,
                "human_labelled_direction": False,
                "obj_frame": {
                    "entry_point": _vector(point_obj),
                    "approach_vector_after_step_probe": _vector(direction_obj_checked),
                    "surface_centroid_v_s": _vector(candidate.v_s),
                    "opening_boundary_centroid_v_o": _vector(candidate.v_o),
                },
                "step_frame": {
                    "entry_point": _vector(point_step),
                    "approach_vector": _vector(direction_step_checked),
                    "nearest_STEP_vertex_distance_mm_diagnostic": float(
                        step_tree.query(point_step, k=1)[0]
                    ),
                    "position_snapped_to_mesh_vertex": False,
                } if alignment["accepted"] else None,
                "direction_source": "thesis_boundary_geometry_plus_STEP_free_space_probe",
                "direction_flipped_by_step_probe": flipped,
                "component_qa": {
                    "vertices": candidate.n_vertices,
                    "surface_area_mm2": float(candidate.size),
                    "component_radius_mm": float(candidate.fragments[0].radius),
                    "thesis_depth_mm": float(candidate.insertion_depth_mm),
                    "axis_span_mm": float(candidate.insertion_depth_axis_mm),
                    "normal_stability": float(candidate.normal_stability),
                },
                "acceptance": "review_required_human_region_derived_CP",
            }
        )
    return output


def derive_part(sample: dict, args: argparse.Namespace) -> dict:
    candidates, contact_components = _component_candidates(sample)
    base = {
        "part_id": sample["part_id"],
        "split": sample["split"],
        "provenance_label": "human-region-derived",
        "semantic_training_eligible": True,
        "cable_entry_components": len(candidates),
        "contact_components_auxiliary_only": contact_components,
        "contact_components_used_as_CP_count": False,
        "source_obj": sample["obj_path"],
        "source_step": sample["step_path"],
    }
    if not candidates:
        return {
            **base,
            "status": "review_required_missing_cable_entry_region",
            "zero_connection_ground_truth": False,
            "alignment": None,
            "candidates": [],
        }

    step_vertices, step_faces = step_to_json.load_any_mesh(
        sample["step_path"], deflection=args.deflection
    )
    step_vertices = np.asarray(step_vertices, dtype=float)
    step_faces = np.asarray(step_faces, dtype=np.int64)
    if len(step_vertices) == 0 or len(step_faces) == 0:
        raise RuntimeError(f"{sample['part_id']}: empty STEP tessellation")
    alignment = align_exact_step_to_obj(
        step_vertices,
        step_faces,
        sample["verts"],
        sample["faces"],
        seed=_seed_for(sample["part_id"]),
        sample_count=args.alignment_samples,
        mean_tolerance_mm=args.align_mean_tol,
        p95_tolerance_mm=args.align_p95_tol,
        bbox_absolute_tolerance_mm=args.align_bbox_abs_tol,
        center_tolerance_mm=args.align_center_tol,
    )
    mapped = _map_candidates(candidates, alignment, step_vertices)
    return {
        **base,
        "status": (
            "review_required_human_region_derived_CP"
            if alignment["accepted"]
            else "review_required_alignment_and_CP"
        ),
        "zero_connection_ground_truth": False,
        "step_tessellation": {
            "vertices": len(step_vertices),
            "faces": len(step_faces),
            "deflection_mm": args.deflection,
        },
        "alignment": alignment,
        "candidates": mapped,
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--corpus", type=Path, default=HERE / "wscad_corpus_scheffler_exact")
    parser.add_argument("--out", type=Path, default=HERE / "results" / "scheffler_cp_bridge" / "development_candidates.json")
    parser.add_argument("--split", choices=("train", "val", "both"), default="both")
    parser.add_argument("--deflection", type=float, default=0.3)
    parser.add_argument("--alignment-samples", type=int, default=8000)
    parser.add_argument("--align-mean-tol", type=float, default=0.80)
    parser.add_argument("--align-p95-tol", type=float, default=1.10)
    parser.add_argument("--align-bbox-abs-tol", type=float, default=2.0)
    parser.add_argument("--align-center-tol", type=float, default=0.10)
    parser.add_argument("--workers", type=int, default=1, help="separate gmsh worker processes")
    parser.add_argument("--limit", type=int, default=0, help="development smoke-test cap; 0 means all")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.out.resolve()
    if output.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite {output}; pass --force")
    splits = ("train", "val") if args.split == "both" else (args.split,)
    samples = []
    for split in splits:
        samples.extend(dataset.load_split(args.corpus, split, verify_hashes=True))
    if args.limit:
        samples = samples[: args.limit]

    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    parts_by_id = {}
    if args.workers == 1:
        for index, sample in enumerate(samples, 1):
            part = derive_part(sample, args)
            parts_by_id[part["part_id"]] = part
            print(
                f"{index}/{len(samples)} {sample['part_id']}: {part['status']} "
                f"cable_components={part['cable_entry_components']}",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(derive_part, sample, args): sample["part_id"] for sample in samples}
            for index, future in enumerate(as_completed(futures), 1):
                part_id = futures[future]
                part = future.result()
                parts_by_id[part_id] = part
                print(
                    f"{index}/{len(samples)} {part_id}: {part['status']} "
                    f"cable_components={part['cable_entry_components']}",
                    flush=True,
                )
    parts = [parts_by_id[sample["part_id"]] for sample in samples]

    summary = {
        "parts": len(parts),
        "candidate_parts": sum(part["cable_entry_components"] > 0 for part in parts),
        "missing_cable_entry_review_required": sum(
            part["cable_entry_components"] == 0 for part in parts
        ),
        "alignment_accepted_parts": sum(
            bool((part.get("alignment") or {}).get("accepted")) for part in parts
        ),
        "alignment_review_required_parts": sum(
            part.get("alignment") is not None and not part["alignment"]["accepted"]
            for part in parts
        ),
        "derived_candidate_points": sum(len(part["candidates"]) for part in parts),
        "human_approved_CP_points": 0,
    }
    payload = {
        "schema_version": "scheffler-human-region-derived-cp/1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "corpus_manifest_sha256": dataset.manifest_sha256(args.corpus),
        "scope": "development_only_train_and_validation; test_locked_not_opened",
        "ground_truth_warning": (
            "Candidates come from human semantic regions, but point/direction are geometric "
            "derivations. Human approval is required before CP training or CP accuracy claims."
        ),
        "summary": summary,
        "parameters": {
            "deflection_mm": args.deflection,
            "alignment_samples": args.alignment_samples,
            "align_mean_tol_mm": args.align_mean_tol,
            "align_p95_tol_mm": args.align_p95_tol,
            "align_bbox_abs_tol_mm": args.align_bbox_abs_tol,
            "align_center_tol_mm": args.align_center_tol,
            "workers": args.workers,
        },
        "test_labels_opened_by_this_command": False,
        "test_labels_used_for_CP_tuning": False,
        "parts": parts,
    }
    _write_json(output, payload)
    print(json.dumps(summary, indent=2))
    print(f"output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
