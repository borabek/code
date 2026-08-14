# -*- coding: utf-8 -*-
"""Freeze ONE split manifest: which part is in which split, its role, and SHA-256 of its exact
STEP / OBJ / labels. The role separation the audit demands:
  train (71), val (20)  -> development (may tune on these)
  test_locked (11)      -> HISTORICAL SEMANTIC benchmark + CP regression/dev. BURNED for any CP
                           final claim: CP rules were diagnosed on these parts, and v31 saw all
                           11 in training / v28 saw 6. A real CP final needs a NEW untouched set.
Output: split_manifest.json
"""
import json, os, hashlib
import scheffler_dataset as dataset

ROOT = "wscad_corpus_scheffler_exact"
ROLE = {"train": "development", "val": "development",
        "test_locked": "historical_semantic_benchmark + cp_regression_dev (BURNED for CP-final)"}


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest() if os.path.exists(path) else None


def main():
    man = {"corpus": ROOT, "frozen": "2026-07-18", "cp_def_version": "cp-v2-cableentry-primary",
           "roles": ROLE, "splits": {}}
    for sp in ("train", "val", "test_locked"):
        parts = []
        for s in dataset.load_split(ROOT, sp, allow_locked=True, verify_hashes=False):
            pid = s["part_id"]; d = os.path.join(ROOT, sp, pid)
            parts.append({"part_id": pid,
                          "obj_sha256": sha(os.path.join(d, f"{pid}.obj")),
                          "labels_sha256": sha(os.path.join(d, f"{pid}.labels.txt")),
                          "step_sha256": sha(os.path.join(d, f"{pid}.stp"))})
        man["splits"][sp] = {"role": ROLE[sp], "n": len(parts), "parts": parts}
    json.dump(man, open("split_manifest.json", "w"), indent=1)
    print("split_manifest.json:", {k: v["n"] for k, v in man["splits"].items()})
    print("test_locked role:", ROLE["test_locked"])


if __name__ == "__main__":
    main()
