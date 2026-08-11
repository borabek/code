# Convert a raw 3D model (STEP/STL/OBJ/OFF) into the ABB connection-point JSON
# format so the trained detector can run on it -- WITHOUT any ground-truth labels.
#
# The CP detector (predict.py / cp_regressor) reads the ABB JSON schema
# (json_dataset.part_from_dict): Graphic3d.Points + Graphic3d.Indices for the
# mesh, and ConnectionPoints for the labels. A STEP file has only geometry and
# no connection points, so this writer emits the mesh with an EMPTY
# ConnectionPoints list -- exactly what inference needs (the model PREDICTS the
# points; json_dataset accepts cps=[] and yields a Part with n_cps==0).
#
# STEP -> mesh reuses the existing CAD pipeline (dataset_convert.convert_file,
# gmsh backend); STL/OBJ/OFF load straight through meshio. So this module adds
# only the thin "mesh (V,F) -> ABB dict -> JSON" step on top.
#
# Usage:
#   python step_to_json.py part.stp --out part.json
#   python step_to_json.py ./stp_dir --out ./json_dir          # whole folder
#   python step_to_json.py part.stp --out part.json --deflection 0.3   # coarser mesh
#
# Then detect connection points on it:
#   python predict.py model_cp_knn_demo.pt part.json --out preds.json

import os
import json
import glob
import shutil
import logging
import tempfile
import argparse

import numpy as np

from meshio import load_mesh
import dataset_convert as dc

logger = logging.getLogger(__name__)

# extensions meshio can load directly (no CAD kernel needed)
_MESH_EXT = {".obj", ".off", ".stl"}


def mesh_to_abb_dict(part_nr, V, F):
    """(V, F) numpy arrays -> a dict in the ABB training/inference JSON schema.

    ConnectionPoints is intentionally empty: this is for INFERENCE on an
    unlabelled part, so the detector supplies the points. Coordinates are passed
    through unchanged -- the pipeline assumes millimetres (STEP is usually mm).
    """
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    return {
        "PartNr": str(part_nr),
        "Graphic3d": {
            "Points": [{"X": float(x), "Y": float(y), "Z": float(z)}
                       for x, y, z in V],
            "Indices": [int(i) for i in F.reshape(-1)],
        },
        "BoundingBox": {
            "Dimension": {"X": float(V[:, 0].max() - V[:, 0].min()) if len(V) else 0.0,
                          "Y": float(V[:, 1].max() - V[:, 1].min()) if len(V) else 0.0,
                          "Z": float(V[:, 2].max() - V[:, 2].min()) if len(V) else 0.0},
            "Location": {"X": float(V[:, 0].min()) if len(V) else 0.0,
                         "Y": float(V[:, 1].min()) if len(V) else 0.0,
                         "Z": float(V[:, 2].min()) if len(V) else 0.0},
        },
        "ConnectionPoints": [],
    }


def load_any_mesh(src, tmp_dir=None, deflection=0.1):
    """Load (V, F) from STEP/STL/OBJ/OFF. STEP is tessellated via gmsh into a
    temporary OBJ (reusing dataset_convert) and then loaded; mesh formats load
    directly. Returns (V, F).

    The STEP intermediate goes to a PRIVATE temp dir (not the source folder,
    which may be read-only or shared) and the whole dir is removed afterwards."""
    ext = os.path.splitext(src)[1].lower()
    if ext in dc.STEP_EXT:                       # .step / .stp -> needs CAD kernel
        own_tmp = tmp_dir is None
        work = tmp_dir or tempfile.mkdtemp(prefix="step2json_")
        try:
            obj = dc.convert_file(src, work, linear_deflection=deflection)
            return load_mesh(obj)
        finally:
            if own_tmp:
                shutil.rmtree(work, ignore_errors=True)
    if ext in _MESH_EXT:
        return load_mesh(src)
    raise ValueError(f"unsupported extension {ext!r} "
                     f"(STEP {sorted(dc.STEP_EXT)} or mesh {sorted(_MESH_EXT)})")


def convert_one(src, out_json, deflection=0.1):
    """Convert one 3D model file to an ABB JSON file. Returns (n_vertices, n_faces)."""
    V, F = load_any_mesh(src, deflection=deflection)
    part_nr = os.path.splitext(os.path.basename(src))[0]
    obj = mesh_to_abb_dict(part_nr, V, F)
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(obj, fh)
    logger.info("  %s -> %s  (%d verts, %d faces)",
                os.path.basename(src), out_json, len(V), len(F))
    return len(V), len(F)


def convert_path(src, out, deflection=0.1):
    """Convert a single file or a whole directory of 3D models to ABB JSON.

    If `src` is a directory, every STEP/STL/OBJ/OFF inside is converted into
    `out` (treated as a directory). If `src` is a file, `out` is the JSON path
    (or a directory, in which case the name is derived from the source).
    Returns the list of written JSON paths.
    """
    written = []
    if os.path.isdir(src):
        os.makedirs(out, exist_ok=True)
        exts = dc.STEP_EXT | _MESH_EXT
        files = sorted(f for f in glob.glob(os.path.join(src, "**", "*"), recursive=True)
                       if os.path.splitext(f)[1].lower() in exts)
        if not files:
            raise SystemExit(f"no STEP/STL/OBJ/OFF files under {src}")
        logger.info("converting %d file(s) from %s -> %s", len(files), src, out)
        for f in files:
            stem = os.path.splitext(os.path.basename(f))[0]
            out_json = os.path.join(out, stem + ".json")
            try:
                convert_one(f, out_json, deflection=deflection)
                written.append(out_json)
            except Exception as exc:                          # noqa: BLE001
                logger.warning("  SKIPPED %s: %s", os.path.basename(f), exc)
        return written

    # single file
    out_json = out
    if os.path.isdir(out) or out.endswith(("/", "\\")):
        stem = os.path.splitext(os.path.basename(src))[0]
        out_json = os.path.join(out, stem + ".json")
    convert_one(src, out_json, deflection=deflection)
    return [out_json]


def part_from_mesh(src, deflection=0.1):
    """Load a 3D model file (STEP/STL/OBJ/OFF) -> a json_dataset.Part with EMPTY
    ConnectionPoints, ready for inference. No temp JSON is written -- the ABB
    dict is built in memory and parsed straight into a Part."""
    import json_dataset as jd
    V, F = load_any_mesh(src, deflection=deflection)
    part_nr = os.path.splitext(os.path.basename(src))[0]
    return jd.part_from_dict(mesh_to_abb_dict(part_nr, V, F))


def iter_parts_any(source, deflection=0.1):
    """Yield json_dataset.Part objects from a source that may be ABB JSON OR raw
    3D models (STEP/STL/OBJ/OFF), or a directory mixing them. JSON is read by
    json_dataset unchanged; mesh/CAD files are tessellated (gmsh for STEP) and
    wrapped with empty ConnectionPoints. Lets predict.py accept `.stp` directly,
    so `.stp -> connection points` is one command (no separate conversion step)."""
    import json_dataset as jd
    mesh_exts = dc.STEP_EXT | _MESH_EXT
    if os.path.isdir(source):
        for f in sorted(glob.glob(os.path.join(source, "**", "*"), recursive=True)):
            ext = os.path.splitext(f)[1].lower()
            if ext == ".json":
                try:
                    yield jd.load_part_file(f)
                except ValueError as exc:
                    logger.warning("skipping %s: %s", f, exc)
            elif ext in mesh_exts:
                try:
                    yield part_from_mesh(f, deflection=deflection)
                except Exception as exc:                      # noqa: BLE001
                    logger.warning("skipping %s: %s", f, exc)
        return
    if os.path.splitext(source)[1].lower() in mesh_exts:
        yield part_from_mesh(source, deflection=deflection)
    else:
        yield from jd.iter_parts(source)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Convert STEP/STL/OBJ/OFF 3D models to ABB connection-point "
                    "JSON (empty ConnectionPoints) for inference with predict.py")
    ap.add_argument("src", help="a 3D model file (.stp/.step/.stl/.obj/.off) or a directory of them")
    ap.add_argument("--out", required=True,
                    help="output JSON path (single file) or directory (batch/dir input)")
    ap.add_argument("--deflection", type=float, default=0.1,
                    help="gmsh max mesh size in mm for STEP tessellation (default 0.1; "
                         "raise it, e.g. 0.3-1.0, for a coarser/faster mesh on big parts)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    written = convert_path(args.src, args.out, deflection=args.deflection)
    print(f"\nwrote {len(written)} JSON file(s).")
    if written:
        print("detect connection points with:")
        target = args.out if os.path.isdir(args.out) else written[0]
        print(f"  python predict.py <model.pt> {target} --out preds.json")


def _selftest():
    """Round-trip a tiny tetrahedron mesh through the ABB dict (no CAD kernel)."""
    V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    F = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int64)
    d = mesh_to_abb_dict("SELFTEST", V, F)
    import json_dataset as jd
    part = jd.part_from_dict(d)
    assert part.n_vertices == 4 and part.faces.shape == (4, 3)
    assert part.n_cps == 0, "inference JSON must carry no ground-truth CPs"
    print("step_to_json selftest OK:", part.part_nr, part.n_vertices, "verts",
          len(part.faces), "faces", part.n_cps, "cps")


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        _selftest()
    else:
        main()
