"""Prepare the exact-WSCAD subset of the Scheffler-Brundl human labels.

The public dataset stores human vertex labels on normalized OBJ meshes.  Its
companion ``parts.json`` stores the SHA-256 of each original STEP.  This script
matches those hashes against the local WSCAD corpus, preserves the published
train/val/test split, and downloads only the OBJ/label files needed for exact
terminal matches.

The published test split is written as ``test_locked``.  Its majority-vote
labels are independently recomputed from the three rater files and must match.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
from typing import Any

import numpy as np
import requests

import meshio


DOI = "doi:10.7910/DVN/D3ODGT"
DATAVERSE_API = "https://dataverse.harvard.edu/api/datasets/:persistentId/"
DATAVERSE_FILE = "https://dataverse.harvard.edu/api/access/datafile/{file_id}"
GITHUB_COMMIT_API = "https://api.github.com/repos/bensch98/eec-analysis/commits/main"
GITHUB_RAW = "https://raw.githubusercontent.com/bensch98/eec-analysis/{commit}/parts.json"
DEFAULT_OUT = "wscad_corpus_scheffler_exact"
CLASS_NAMES = ["Housing", "Contact", "SnapPoint", "CableEntry", "LabelSurface"]


def _request_json(url: str, *, params: dict | None = None, attempts: int = 4) -> dict:
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            response = requests.get(url, params=params, timeout=90)
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # noqa: BLE001
            last = exc
            if attempt + 1 < attempts:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"request failed after {attempts} attempts: {url}: {last}")


def _request_bytes(url: str, *, attempts: int = 4) -> bytes:
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            return response.content
        except Exception as exc:  # noqa: BLE001
            last = exc
            if attempt + 1 < attempts:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"download failed after {attempts} attempts: {url}: {last}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _md5_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()  # noqa: S324 - verifying publisher checksum


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _normalise(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def _discover_steps(repo: Path, output: Path) -> list[Path]:
    skip_names = {".git", ".venv", "__pycache__", ".pytest_cache"}
    output_resolved = output.resolve()
    found: list[Path] = []
    for current, directories, files in os.walk(repo):
        current_path = Path(current)
        directories[:] = [
            name
            for name in directories
            if name not in skip_names
            and not (current_path / name).resolve().is_relative_to(output_resolved)
        ]
        for filename in files:
            if Path(filename).suffix.lower() in {".stp", ".step"}:
                found.append(current_path / filename)
    return sorted(found)


def _file_index(version: dict) -> dict[tuple[str, str], dict]:
    return {
        (entry.get("directoryLabel", ""), entry["dataFile"]["filename"]): entry["dataFile"]
        for entry in version["files"]
    }


def _dataverse_bytes(data_file: dict) -> bytes:
    payload = _request_bytes(DATAVERSE_FILE.format(file_id=data_file["id"]))
    checksum = data_file.get("checksum", {})
    if checksum.get("type", "").upper() == "MD5":
        actual = _md5_bytes(payload)
        if actual.lower() != checksum["value"].lower():
            raise ValueError(
                f"Dataverse MD5 mismatch for {data_file['filename']}: "
                f"{actual} != {checksum['value']}"
            )
    return payload


def _split_ids(index: dict[tuple[str, str], dict]) -> tuple[dict[str, list[str]], dict[str, dict]]:
    splits: dict[str, list[str]] = {}
    split_sources: dict[str, dict] = {}
    for split in ("train", "val", "test"):
        data_file = index[("split", f"{split}.txt")]
        payload = _dataverse_bytes(data_file)
        ids = [Path(line.strip()).stem for line in payload.decode("utf-8-sig").splitlines() if line.strip()]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate IDs in published {split} split")
        splits[split] = ids
        split_sources[split] = {
            "datafile_id": data_file["id"],
            "md5": data_file["checksum"]["value"],
            "raw_text": payload.decode("utf-8-sig"),
        }
    all_ids = [part_id for values in splits.values() for part_id in values]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("published train/val/test splits overlap")
    return splits, split_sources


def _select_exact_parts(
    parts: dict,
    splits: dict[str, list[str]],
    steps: list[Path],
) -> tuple[list[dict], list[dict]]:
    split_by_id = {part_id: split for split, values in splits.items() for part_id in values}
    step_names = [(path, _normalise(path.stem)) for path in steps]
    hash_cache: dict[Path, str] = {}
    selected: list[dict] = []
    catalog_only: list[dict] = []
    for part_id, metadata in parts.items():
        if metadata.get("Category") != "Terminals":
            continue
        token = _normalise(part_id)
        candidates = [path for path, stem in step_names if len(token) >= 5 and token in stem]
        if not candidates:
            continue
        expected = str(metadata.get("sha256", "")).lower()
        exact: list[Path] = []
        candidate_hashes = []
        for candidate in candidates:
            actual = hash_cache.setdefault(candidate, _sha256(candidate))
            candidate_hashes.append({"path": str(candidate.resolve()), "sha256": actual})
            if actual.lower() == expected:
                exact.append(candidate)
        if not exact:
            catalog_only.append(
                {
                    "part_id": part_id,
                    "expected_step_sha256": expected,
                    "local_candidates": candidate_hashes,
                }
            )
            continue
        exact.sort(
            key=lambda path: (
                0 if "all_wscad_stp" in {piece.lower() for piece in path.parts} else 1,
                len(path.parts),
                str(path).lower(),
            )
        )
        selected.append(
            {
                "part_id": part_id,
                "split": split_by_id[part_id],
                "source_step": exact[0].resolve(),
                "all_exact_local_steps": [path.resolve() for path in exact],
                "step_sha256": expected,
                "metadata": metadata,
            }
        )
    selected.sort(key=lambda item: (item["split"], item["part_id"]))
    return selected, catalog_only


def _download_to(data_file: dict, destination: Path) -> dict:
    expected_md5 = data_file["checksum"]["value"].lower()
    if destination.is_file():
        existing = hashlib.md5(destination.read_bytes()).hexdigest()  # noqa: S324
        if existing == expected_md5:
            return {"path": str(destination), "bytes": destination.stat().st_size, "md5": existing}
    payload = _dataverse_bytes(data_file)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".download")
    temporary.write_bytes(payload)
    os.replace(temporary, destination)
    return {"path": str(destination), "bytes": len(payload), "md5": expected_md5}


def _copy_step(source: Path, destination: Path, expected_sha256: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists() or _sha256(destination).lower() != expected_sha256.lower():
        temporary = destination.with_suffix(destination.suffix + ".copy")
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    actual = _sha256(destination)
    if actual.lower() != expected_sha256.lower():
        raise ValueError(f"copied STEP SHA-256 mismatch: {destination}")
    data = destination.read_bytes()
    upper = data.upper()
    if b"ISO-10303-21" not in upper[:4096] or b"END-ISO-10303-21" not in upper[-4096:]:
        raise ValueError(f"bad STEP signature: {destination}")


def _label_array(path: Path) -> np.ndarray:
    labels = np.loadtxt(path, dtype=np.int64).reshape(-1)
    if len(labels) == 0 or labels.min() < 0 or labels.max() >= len(CLASS_NAMES):
        raise ValueError(f"invalid vertex labels: {path}")
    return labels


def _majority_vote(raters: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(raters, axis=0)
    if stack.shape[0] != 3:
        raise ValueError("the locked test requires exactly three raters")
    counts = np.stack([(stack == class_id).sum(axis=0) for class_id in range(len(CLASS_NAMES))])
    if np.any(counts.max(axis=0) < 2):
        raise ValueError("three-rater labels contain a position without a majority")
    return counts.argmax(axis=0).astype(np.int64)


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _validate_part(part: dict, output: Path) -> dict:
    part_dir = output / part["output_split"] / part["part_id"]
    obj_path = part_dir / f"{part['part_id']}.obj"
    label_path = part_dir / f"{part['part_id']}.labels.txt"
    step_path = part_dir / f"{part['part_id']}.stp"
    vertices, faces = meshio.load_obj(str(obj_path))
    labels = _label_array(label_path)
    if len(vertices) != len(labels):
        raise ValueError(
            f"{part['part_id']}: OBJ vertices {len(vertices)} != labels {len(labels)}"
        )
    if len(vertices) == 0 or len(faces) == 0 or not np.isfinite(vertices).all():
        raise ValueError(f"{part['part_id']}: unusable OBJ geometry")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError(f"{part['part_id']}: OBJ face index out of range")
    if _sha256(step_path).lower() != part["step_sha256"].lower():
        raise ValueError(f"{part['part_id']}: final STEP hash mismatch")

    rater_agreement = None
    if part["split"] == "test":
        raters = [
            _label_array(part_dir / f"{part['part_id']}.rater_{index}.labels.txt")
            for index in range(3)
        ]
        if any(len(rater) != len(labels) for rater in raters):
            raise ValueError(f"{part['part_id']}: rater label length mismatch")
        recomputed = _majority_vote(raters)
        if not np.array_equal(recomputed, labels):
            differing = int(np.count_nonzero(recomputed != labels))
            raise ValueError(
                f"{part['part_id']}: published majority differs at {differing} vertices"
            )
        pairwise = {}
        for left, right in ((0, 1), (0, 2), (1, 2)):
            pairwise[f"{left}_{right}"] = float((raters[left] == raters[right]).mean())
        rater_agreement = {
            "published_majority_recomputed": True,
            "pairwise_vertex_agreement": pairwise,
        }

    label_counts = {
        CLASS_NAMES[class_id]: int(np.count_nonzero(labels == class_id))
        for class_id in range(len(CLASS_NAMES))
    }
    has_cable_entry_region = label_counts["CableEntry"] > 0
    return {
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "bbox_min": vertices.min(axis=0).tolist(),
        "bbox_max": vertices.max(axis=0).tolist(),
        "label_counts": label_counts,
        "all_five_classes_present": set(np.unique(labels).tolist()) == set(range(5)),
        "semantic_training_eligible": True,
        "has_cable_entry_region": has_cable_entry_region,
        "human_region_cp_status": (
            "candidate_requires_bridge_qa"
            if has_cable_entry_region
            else "review_required_missing_cable_entry_region"
        ),
        "step_sha256": _sha256(step_path),
        "obj_sha256": _sha256(obj_path),
        "labels_sha256": _sha256(label_path),
        "rater_qa": rater_agreement,
    }


def _dataset_readme(summary: dict) -> str:
    split_counts = summary["split_counts"]
    return f"""# Scheffler-Brundl exact WSCAD terminal subset

This corpus joins the public human-labelled Electrical and Electronic Components
Dataset to byte-identical local WSCAD STEP files through the publisher's original
STEP SHA-256 values.

## Frozen scope

- Published train subset: {split_counts['train']} parts
- Published validation subset: {split_counts['val']} parts
- Locked three-rater test subset: {split_counts['test_locked']} parts
- Development total (train + validation): {split_counts['train'] + split_counts['val']} parts
- Total exact terminal parts: {summary['selected_exact_parts']} parts

The published split is preserved.  `test_locked` must never be used for training,
threshold selection, model selection, preprocessing selection, or CP conversion
parameter tuning.  Its `*.labels.txt` is the published majority vote; the three
individual rater files are retained and the majority is independently verified.
The preparation step reads those test labels only for checksum, shape, and
majority-vote integrity QA; it does not derive any model or bridge parameter.

## Label IDs

0 Housing, 1 Contact, 2 SnapPoint, 3 CableEntry, 4 LabelSurface.

These are human semantic region labels, not directly clicked CP XYZ labels.  Any
point/direction derived from Contact/CableEntry regions must be reported as
`human-region-derived` until a human signs off the final CP positions.

`CableEntry` is present in {summary['cp_region_counts']['development_candidate']} of the
91 development parts and {summary['cp_region_counts']['test_candidate']} of the 11
locked-test parts.  A missing CableEntry region does **not** mean that a part has
zero physical connection points.  Those parts are marked `review_required` and
must not be used as negative CP ground truth.

## Files per part

- `.stp`: byte-identical local WSCAD source geometry
- `.obj`: publisher's remeshed/downsampled labelled mesh (vertex order preserved)
- `.labels.txt`: one integer label per OBJ vertex
- test only: `.rater_0/1/2.labels.txt`
- `.provenance.json`: source IDs, checksums, split, and metadata

Dataset DOI: https://doi.org/10.7910/DVN/D3ODGT
Article: https://doi.org/10.1038/s41597-024-03155-w
Analysis code/metadata: https://github.com/bensch98/eec-analysis

The Dataverse snapshot reports CC0-1.0 for the public labelled dataset.  The
copied WSCAD STEP geometry remains subject to its own source terms; do not
redistribute the combined corpus without checking those rights.
"""


def prepare(args: argparse.Namespace) -> dict:
    repo = args.repo.resolve()
    output = args.out.resolve()
    print("Fetching pinned publisher metadata...", flush=True)
    dataset_response = _request_json(DATAVERSE_API, params={"persistentId": DOI})
    version = dataset_response["data"]["latestVersion"]
    license_info = version.get("license") or {}
    if license_info.get("rightsIdentifier") != "CC0-1.0":
        raise ValueError(f"unexpected dataset license: {license_info}")
    index = _file_index(version)
    splits, split_sources = _split_ids(index)

    commit_response = _request_json(GITHUB_COMMIT_API)
    commit = commit_response["sha"]
    parts_url = GITHUB_RAW.format(commit=commit)
    parts_payload = _request_bytes(parts_url)
    parts = json.loads(parts_payload.decode("utf-8"))
    if len(parts) != 234:
        raise ValueError(f"expected 234 publisher metadata records, got {len(parts)}")

    print("Scanning local STEP files and checking publisher SHA-256 values...", flush=True)
    steps = _discover_steps(repo, output)
    selected, catalog_only = _select_exact_parts(parts, splits, steps)
    split_counts_raw = Counter(item["split"] for item in selected)
    split_counts = {
        "train": split_counts_raw["train"],
        "val": split_counts_raw["val"],
        "test_locked": split_counts_raw["test"],
    }
    print(
        f"Selected exact terminal STEP files: {len(selected)} "
        f"(train={split_counts['train']}, val={split_counts['val']}, "
        f"test_locked={split_counts['test_locked']}); catalog-only mismatch={len(catalog_only)}",
        flush=True,
    )
    if not args.allow_count_drift:
        if len(selected) != args.expected_total or split_counts["test_locked"] != args.expected_test:
            raise ValueError(
                f"selection drift: expected total/test {args.expected_total}/{args.expected_test}, "
                f"got {len(selected)}/{split_counts['test_locked']}; inspect before proceeding"
            )
    if args.dry_run:
        return {
            "selected_exact_parts": len(selected),
            "split_counts": split_counts,
            "catalog_only_mismatches": catalog_only,
            "parts": [
                {
                    "part_id": item["part_id"],
                    "split": item["split"],
                    "source_step": str(item["source_step"]),
                    "step_sha256": item["step_sha256"],
                }
                for item in selected
            ],
        }

    output.mkdir(parents=True, exist_ok=True)
    source_dir = output / "source_snapshot"
    source_dir.mkdir(exist_ok=True)
    _write_json(source_dir / "dataverse_dataset.json", dataset_response)
    (source_dir / "parts.json").write_bytes(parts_payload)
    _write_text(source_dir / "parts_commit.txt", commit + "\n")
    for split, source in split_sources.items():
        _write_text(source_dir / f"published_{split}.txt", source["raw_text"])

    jobs: list[tuple[dict, Path, str, str]] = []
    prepared: dict[str, dict] = {}
    for item in selected:
        part_id = item["part_id"]
        split = item["split"]
        output_split = "test_locked" if split == "test" else split
        part_dir = output / output_split / part_id
        part_dir.mkdir(parents=True, exist_ok=True)
        step_destination = part_dir / f"{part_id}.stp"
        _copy_step(item["source_step"], step_destination, item["step_sha256"])

        obj_file = index[(f"{split}/obj", f"{part_id}.obj")]
        label_file = index[(f"{split}/txt", f"{part_id}.txt")]
        jobs.append((obj_file, part_dir / f"{part_id}.obj", part_id, "obj"))
        jobs.append((label_file, part_dir / f"{part_id}.labels.txt", part_id, "labels"))
        rater_files = []
        if split == "test":
            for rater in range(3):
                rater_file = index[(f"test_raters/{rater}", f"{part_id}.txt")]
                jobs.append(
                    (
                        rater_file,
                        part_dir / f"{part_id}.rater_{rater}.labels.txt",
                        part_id,
                        f"rater_{rater}",
                    )
                )
                rater_files.append(rater_file)
        prepared[part_id] = {
            **item,
            "output_split": output_split,
            "part_dir": part_dir,
            "step_destination": step_destination,
            "obj_datafile": obj_file,
            "label_datafile": label_file,
            "rater_datafiles": rater_files,
            "downloads": {},
        }

    print(f"Downloading/verifying {len(jobs)} Dataverse files...", flush=True)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(_download_to, data_file, destination): (part_id, role)
            for data_file, destination, part_id, role in jobs
        }
        completed = 0
        for future in as_completed(futures):
            part_id, role = futures[future]
            prepared[part_id]["downloads"][role] = future.result()
            completed += 1
            if completed % 25 == 0 or completed == len(futures):
                print(f"  {completed}/{len(futures)} files", flush=True)

    part_entries = []
    print("Validating geometry, vertex labels, and three-rater majorities...", flush=True)
    for part_id in sorted(prepared):
        item = prepared[part_id]
        qa = _validate_part(item, output)
        part_dir = item["part_dir"]
        provenance = {
            "part_id": part_id,
            "category": item["metadata"].get("Category"),
            "manufacturer": item["metadata"].get("Manufacturer"),
            "product_name": item["metadata"].get("Name"),
            "published_split": item["split"],
            "output_split": item["output_split"],
            "locked_final_test": item["split"] == "test",
            "source_step": str(item["source_step"]),
            "source_step_sha256": item["step_sha256"],
            "all_exact_local_steps": [str(path) for path in item["all_exact_local_steps"]],
            "dataset_doi": DOI,
            "dataset_version": f"{version['versionNumber']}.{version['versionMinorNumber']}",
            "parts_metadata_commit": commit,
            "dataverse_files": {
                "obj": {
                    "id": item["obj_datafile"]["id"],
                    "md5": item["obj_datafile"]["checksum"]["value"],
                },
                "labels": {
                    "id": item["label_datafile"]["id"],
                    "md5": item["label_datafile"]["checksum"]["value"],
                    "kind": "published_three_rater_majority" if item["split"] == "test" else "human_vertex_labels",
                },
                "raters": [
                    {"id": data_file["id"], "md5": data_file["checksum"]["value"]}
                    for data_file in item["rater_datafiles"]
                ],
            },
            "label_classes": {str(index): name for index, name in enumerate(CLASS_NAMES)},
            "qa": qa,
        }
        provenance_path = part_dir / f"{part_id}.provenance.json"
        _write_json(provenance_path, provenance)
        part_entries.append(
            {
                "part_id": part_id,
                "published_split": item["split"],
                "output_split": item["output_split"],
                "locked_final_test": item["split"] == "test",
                "step": _relative(item["step_destination"], output),
                "obj": _relative(part_dir / f"{part_id}.obj", output),
                "labels": _relative(part_dir / f"{part_id}.labels.txt", output),
                "raters": [
                    _relative(part_dir / f"{part_id}.rater_{rater}.labels.txt", output)
                    for rater in range(3)
                ] if item["split"] == "test" else [],
                "provenance": _relative(provenance_path, output),
                "human_region_cp_status": qa["human_region_cp_status"],
                "qa": qa,
            }
        )

    ids_by_output_split = {
        split: sorted(entry["part_id"] for entry in part_entries if entry["output_split"] == split)
        for split in ("train", "val", "test_locked")
    }
    split_dir = output / "splits"
    for split, ids in ids_by_output_split.items():
        _write_text(split_dir / f"{split}.txt", "\n".join(ids) + "\n")
    _write_text(
        output / "LOCKED_FINAL_TEST.md",
        "# LOCKED FINAL TEST\n\n"
        "The `test_locked` directory is the untouched published test subset.\n"
        "Do not use it for training, threshold selection, model selection,\n"
        "preprocessing selection, or CP-conversion parameter tuning.\n",
    )

    all_step_hashes = [entry["qa"]["step_sha256"] for entry in part_entries]
    if len(all_step_hashes) != len(set(all_step_hashes)):
        raise ValueError("selected corpus contains duplicate original STEP hashes")
    cp_region_counts = {
        "train_candidate": sum(
            entry["output_split"] == "train" and entry["qa"]["has_cable_entry_region"]
            for entry in part_entries
        ),
        "train_review_required": sum(
            entry["output_split"] == "train" and not entry["qa"]["has_cable_entry_region"]
            for entry in part_entries
        ),
        "val_candidate": sum(
            entry["output_split"] == "val" and entry["qa"]["has_cable_entry_region"]
            for entry in part_entries
        ),
        "val_review_required": sum(
            entry["output_split"] == "val" and not entry["qa"]["has_cable_entry_region"]
            for entry in part_entries
        ),
        "test_candidate": sum(
            entry["output_split"] == "test_locked" and entry["qa"]["has_cable_entry_region"]
            for entry in part_entries
        ),
        "test_review_required": sum(
            entry["output_split"] == "test_locked" and not entry["qa"]["has_cable_entry_region"]
            for entry in part_entries
        ),
    }
    cp_region_counts["development_candidate"] = (
        cp_region_counts["train_candidate"] + cp_region_counts["val_candidate"]
    )
    cp_region_counts["development_review_required"] = (
        cp_region_counts["train_review_required"] + cp_region_counts["val_review_required"]
    )
    summary = {
        "schema_version": "scheffler-brundl-exact-wscad/1.0",
        "status": "complete_verified",
        "dataset": {
            "doi": DOI,
            "version": f"{version['versionNumber']}.{version['versionMinorNumber']}",
            "release_time": version.get("releaseTime"),
            "license": license_info,
            "parts_metadata_commit": commit,
        },
        "selection_rule": "Category=Terminals and local STEP SHA-256 equals publisher original STEP SHA-256",
        "published_dataset_parts": len(parts),
        "published_terminal_parts": sum(meta.get("Category") == "Terminals" for meta in parts.values()),
        "selected_exact_parts": len(part_entries),
        "split_counts": split_counts,
        "development_parts": split_counts["train"] + split_counts["val"],
        "locked_three_rater_test_parts": split_counts["test_locked"],
        "cp_region_counts": cp_region_counts,
        "catalog_match_but_hash_mismatch": catalog_only,
        "label_classes": {str(index): name for index, name in enumerate(CLASS_NAMES)},
        "global_qa": {
            "published_splits_disjoint": True,
            "all_step_hashes_exact": True,
            "all_selected_step_hashes_unique": True,
            "all_obj_label_lengths_match": True,
            "all_test_majorities_recomputed": True,
            "test_locked": True,
        },
        "parts": sorted(part_entries, key=lambda entry: (entry["output_split"], entry["part_id"])),
    }
    _write_json(output / "manifest.json", summary)
    _write_text(output / "README.md", _dataset_readme(summary))
    _write_text(
        output / "CITATION.bib",
        "@data{scheffler_bruendl_2023,\n"
        "  author = {Scheffler, Benedikt and Br{\\\"u}ndl, Patrick},\n"
        "  title = {Electrical and Electronic Components Dataset},\n"
        "  publisher = {Harvard Dataverse},\n"
        "  year = {2023},\n"
        "  doi = {10.7910/DVN/D3ODGT},\n"
        "  version = {V1}\n"
        "}\n",
    )
    print(
        f"COMPLETE: {len(part_entries)} exact parts; "
        f"development={summary['development_parts']}; "
        f"locked_test={summary['locked_three_rater_test_parts']} -> {output}",
        flush=True,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--out", type=Path, default=Path(DEFAULT_OUT))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--expected-total", type=int, default=102)
    parser.add_argument("--expected-test", type=int, default=11)
    parser.add_argument("--allow-count-drift", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
