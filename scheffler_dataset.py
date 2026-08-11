"""Leakage-safe loader for the exact Scheffler-Brundl/WSCAD corpus.

The human labels belong to the publisher OBJ vertex order.  The byte-identical
STEP beside each OBJ is a provenance/deployment twin; it is deliberately not
used as the semantic training mesh because re-tessellation changes vertices.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np

import meshio
from connector_constants import CLASS_NAMES


SCHEMA_VERSION = "scheffler-brundl-exact-wscad/1.0"
SPLITS = ("train", "val", "test_locked")
EXPECTED_COUNTS = {"train": 71, "val": 20, "test_locked": 11}


class CorpusContractError(RuntimeError):
    """Raised when corpus files no longer match their frozen manifest."""


class LockedTestAccessError(PermissionError):
    """Raised when code tries to read final-test labels without an explicit gate."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _root(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _inside(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root):
        raise CorpusContractError(f"manifest path escapes corpus root: {relative!r}")
    if not candidate.is_file():
        raise CorpusContractError(f"missing corpus file: {candidate}")
    return candidate


def load_manifest(corpus_root: str | Path) -> dict:
    root = _root(corpus_root)
    path = root / "manifest.json"
    if not path.is_file():
        raise CorpusContractError(
            f"missing {path}; run prepare_scheffler_wscad_exact.py first"
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise CorpusContractError(
            f"unexpected corpus schema {manifest.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION!r}"
        )
    if manifest.get("status") != "complete_verified":
        raise CorpusContractError(f"corpus is not complete_verified: {path}")
    if manifest.get("global_qa", {}).get("test_locked") is not True:
        raise CorpusContractError("manifest does not mark the final test as locked")
    return manifest


def manifest_sha256(corpus_root: str | Path) -> str:
    return _sha256(_root(corpus_root) / "manifest.json")


def validate_manifest_contract(corpus_root: str | Path, manifest: dict | None = None) -> dict:
    """Validate split identities without opening any locked label file."""
    root = _root(corpus_root)
    manifest = load_manifest(root) if manifest is None else manifest
    parts = manifest.get("parts", [])
    by_split = {
        split: [entry for entry in parts if entry.get("output_split") == split]
        for split in SPLITS
    }
    counts = {split: len(entries) for split, entries in by_split.items()}
    declared = manifest.get("split_counts", {})
    if counts != EXPECTED_COUNTS or declared != EXPECTED_COUNTS:
        raise CorpusContractError(
            f"split-count drift: entries={counts}, manifest={declared}, "
            f"expected={EXPECTED_COUNTS}"
        )

    all_ids: list[str] = []
    step_hashes: list[str] = []
    for split, entries in by_split.items():
        for entry in entries:
            part_id = str(entry.get("part_id", ""))
            if not part_id:
                raise CorpusContractError(f"part without id in {split}")
            if bool(entry.get("locked_final_test")) != (split == "test_locked"):
                raise CorpusContractError(f"bad locked flag for {part_id}")
            all_ids.append(part_id)
            step_hashes.append(str(entry.get("qa", {}).get("step_sha256", "")))
            # Path containment and existence only. Do not read locked contents here.
            for field in ("step", "obj", "labels", "provenance"):
                _inside(root, entry[field])
            for rater_path in entry.get("raters", []):
                _inside(root, rater_path)

    if len(all_ids) != len(set(all_ids)):
        raise CorpusContractError("part IDs overlap across frozen splits")
    if "" in step_hashes or len(step_hashes) != len(set(step_hashes)):
        raise CorpusContractError("missing or duplicate original STEP SHA-256")
    if manifest.get("development_parts") != 91:
        raise CorpusContractError("development pool must remain 71 train + 20 validation")
    if manifest.get("locked_three_rater_test_parts") != 11:
        raise CorpusContractError("locked final test must remain 11 parts")
    return {"counts": counts, "ids": {k: sorted(e["part_id"] for e in v) for k, v in by_split.items()}}


def split_entries(corpus_root: str | Path, split: str) -> list[dict]:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    manifest = load_manifest(corpus_root)
    validate_manifest_contract(corpus_root, manifest)
    return sorted(
        (entry for entry in manifest["parts"] if entry["output_split"] == split),
        key=lambda entry: entry["part_id"],
    )


def load_split(
    corpus_root: str | Path,
    split: str,
    *,
    allow_locked: bool = False,
    verify_hashes: bool = True,
    require_cable_entry: bool = False,
) -> list[dict]:
    """Load one semantic split as DiffusionNet sample dictionaries.

    ``test_locked`` is rejected unless ``allow_locked=True``.  Training code
    must never set it.  ``require_cable_entry`` is only for CP-bridge candidate
    selection; it must not be interpreted as a zero-CP negative filter.
    """
    if split == "test_locked" and not allow_locked:
        raise LockedTestAccessError(
            "test_locked is the one-shot final benchmark; use the dedicated "
            "eval_scheffler_semantic.py confirmation gate"
        )
    root = _root(corpus_root)
    samples: list[dict] = []
    for entry in split_entries(root, split):
        if require_cable_entry and not entry["qa"].get("has_cable_entry_region", False):
            continue
        obj_path = _inside(root, entry["obj"])
        label_path = _inside(root, entry["labels"])
        if verify_hashes:
            expected_obj = entry["qa"].get("obj_sha256")
            expected_labels = entry["qa"].get("labels_sha256")
            if _sha256(obj_path) != expected_obj:
                raise CorpusContractError(f"OBJ SHA-256 drift for {entry['part_id']}")
            if _sha256(label_path) != expected_labels:
                raise CorpusContractError(f"label SHA-256 drift for {entry['part_id']}")

        vertices, faces = meshio.load_obj(str(obj_path))
        labels = meshio.load_labels(str(label_path))
        if len(vertices) != len(labels):
            raise CorpusContractError(
                f"{entry['part_id']}: {len(vertices)} vertices != {len(labels)} labels"
            )
        if len(vertices) == 0 or len(faces) == 0 or not np.isfinite(vertices).all():
            raise CorpusContractError(f"{entry['part_id']}: unusable OBJ geometry")
        if faces.min(initial=0) < 0 or faces.max(initial=-1) >= len(vertices):
            raise CorpusContractError(f"{entry['part_id']}: face index outside vertex array")
        unique = set(np.unique(labels).tolist())
        if not unique.issubset(set(range(len(CLASS_NAMES)))):
            raise CorpusContractError(f"{entry['part_id']}: labels outside 0..4: {unique}")
        samples.append(
            {
                "part_id": entry["part_id"],
                "split": split,
                "verts": np.asarray(vertices, dtype=np.float64),
                "faces": np.asarray(faces, dtype=np.int64),
                "labels": np.asarray(labels, dtype=np.int64),
                "obj_path": str(obj_path),
                "step_path": str(_inside(root, entry["step"])),
                "human_region_cp_status": entry.get("human_region_cp_status"),
            }
        )
    return samples


def load_rater_labels(
    corpus_root: str | Path,
    part_id: str,
    *,
    allow_locked: bool = False,
) -> list[np.ndarray]:
    """Read the three individual test raters behind the explicit final-test gate."""
    if not allow_locked:
        raise LockedTestAccessError("individual rater labels are part of test_locked")
    root = _root(corpus_root)
    entries = {entry["part_id"]: entry for entry in split_entries(root, "test_locked")}
    if part_id not in entries:
        raise KeyError(f"{part_id!r} is not in test_locked")
    paths = entries[part_id].get("raters", [])
    if len(paths) != 3:
        raise CorpusContractError(f"{part_id}: expected exactly three rater files")
    raters = [meshio.load_labels(str(_inside(root, path))) for path in paths]
    expected_len = entries[part_id]["qa"]["vertices"]
    if any(len(labels) != expected_len for labels in raters):
        raise CorpusContractError(f"{part_id}: rater/vertex length mismatch")
    return raters


def class_counts(samples: Iterable[dict]) -> dict[str, int]:
    counts: Counter[int] = Counter()
    for sample in samples:
        values, value_counts = np.unique(sample["labels"], return_counts=True)
        counts.update({int(value): int(count) for value, count in zip(values, value_counts)})
    return {CLASS_NAMES[class_id]: counts[class_id] for class_id in range(len(CLASS_NAMES))}


def split_receipt(corpus_root: str | Path, split: str) -> dict:
    entries = split_entries(corpus_root, split)
    return {
        "split": split,
        "count": len(entries),
        "part_ids": [entry["part_id"] for entry in entries],
        "step_sha256": {entry["part_id"]: entry["qa"]["step_sha256"] for entry in entries},
    }

