import hashlib
import json
from pathlib import Path

import pytest

import scheffler_dataset as sd


OBJ = """v 0 0 0
v 1 0 0
v 0 1 0
v 0 0 1
f 1 2 3
f 1 2 4
f 1 3 4
f 2 3 4
"""
LABELS = "0\n1\n2\n3\n"


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_corpus(root: Path) -> Path:
    counts = {"train": 71, "val": 20, "test_locked": 11}
    parts = []
    serial = 0
    for split, count in counts.items():
        for _ in range(count):
            part_id = f"p{serial:03d}"
            part_dir = root / split / part_id
            part_dir.mkdir(parents=True)
            obj = part_dir / f"{part_id}.obj"
            labels = part_dir / f"{part_id}.labels.txt"
            step = part_dir / f"{part_id}.stp"
            provenance = part_dir / f"{part_id}.provenance.json"
            obj.write_text(OBJ, encoding="utf-8")
            labels.write_text(LABELS, encoding="utf-8")
            step.write_text("ISO-10303-21;\nEND-ISO-10303-21;\n", encoding="ascii")
            provenance.write_text("{}\n", encoding="utf-8")
            raters = []
            if split == "test_locked":
                for rater in range(3):
                    path = part_dir / f"{part_id}.rater_{rater}.labels.txt"
                    path.write_text(LABELS, encoding="utf-8")
                    raters.append(path.relative_to(root).as_posix())
            parts.append(
                {
                    "part_id": part_id,
                    "output_split": split,
                    "locked_final_test": split == "test_locked",
                    "step": step.relative_to(root).as_posix(),
                    "obj": obj.relative_to(root).as_posix(),
                    "labels": labels.relative_to(root).as_posix(),
                    "raters": raters,
                    "provenance": provenance.relative_to(root).as_posix(),
                    "human_region_cp_status": "candidate_requires_bridge_qa",
                    "qa": {
                        "vertices": 4,
                        "obj_sha256": _sha_file(obj),
                        "labels_sha256": _sha_file(labels),
                        "step_sha256": f"{serial:064x}",
                        "has_cable_entry_region": True,
                    },
                }
            )
            serial += 1
    manifest = {
        "schema_version": sd.SCHEMA_VERSION,
        "status": "complete_verified",
        "split_counts": counts,
        "development_parts": 91,
        "locked_three_rater_test_parts": 11,
        "global_qa": {"test_locked": True},
        "parts": parts,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_official_counts_and_locked_gate(tmp_path):
    root = _make_corpus(tmp_path)
    assert sd.validate_manifest_contract(root)["counts"] == sd.EXPECTED_COUNTS
    assert len(sd.load_split(root, "train")) == 71
    assert len(sd.load_split(root, "val")) == 20
    with pytest.raises(sd.LockedTestAccessError):
        sd.load_split(root, "test_locked")
    assert len(sd.load_split(root, "test_locked", allow_locked=True)) == 11


def test_hash_drift_fails_fast(tmp_path):
    root = _make_corpus(tmp_path)
    first = sd.split_entries(root, "train")[0]
    (root / first["labels"]).write_text("0\n0\n0\n0\n", encoding="utf-8")
    with pytest.raises(sd.CorpusContractError, match="label SHA-256 drift"):
        sd.load_split(root, "train")


def test_manifest_path_cannot_escape_root(tmp_path):
    root = _make_corpus(tmp_path / "corpus")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["parts"][0]["obj"] = "../outside.obj"
    (root.parent / "outside.obj").write_text(OBJ, encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(sd.CorpusContractError, match="escapes corpus root"):
        sd.validate_manifest_contract(root)
