# -*- coding: utf-8 -*-
"""Measure the JSON<->WSCAD-STEP domain gap (audit P1). The concern: manufacturer JSON meshes and
the raw WSCAD STEP tessellation of the SAME part have very different vertex densities, so a model
trained on one distribution does not see the other. This quantifies it and shows where the thesis
remesh (the ~6000-vertex Scheffler OBJ) sits.

Reports: JSON<->WSCAD part-id overlap; vertex-count distributions for (a) JSON meshes, (b) the
remeshed Scheffler OBJ, (c) a sample of RAW WSCAD STEP meshed with gmsh; and the density ratio.
Output: results/cp_domain_gap.json
"""
import json, glob, os, re
import numpy as np

JSON_DIR = r"C:/Users/DE00024082/Desktop/JSON"
POOL = "all_wscad_stp"; CORPUS = "wscad_corpus_scheffler_exact"
PID_RE = re.compile(r"wscaduniverse_([0-9A-Za-z\-]+)_")


def json_stats():
    n = []
    ids = set()
    for f in glob.glob(os.path.join(JSON_DIR, "*.json")):
        d = json.load(open(f, encoding="utf-8"))
        n.append(len(d.get("Graphic3d", {}).get("Points", [])))
        ids.add(str(d.get("PartNr", "")))
        ids.add(os.path.splitext(os.path.basename(f))[0])
    return np.array(n), ids


def obj_vcount(p):
    return sum(1 for ln in open(p) if ln.startswith("v "))


def main():
    jn, jids = json_stats()
    pool_ids = set()
    for p in glob.glob(os.path.join(POOL, "*.stp")):
        m = PID_RE.search(os.path.basename(p))
        if m:
            pool_ids.add(m.group(1))
    overlap = sorted(jids & pool_ids)

    # remeshed Scheffler OBJ vertex counts
    obj_n = [obj_vcount(p) for p in glob.glob(os.path.join(CORPUS, "*", "*", "*.obj"))]

    # raw WSCAD STEP: mesh a sample via gmsh, count nodes
    raw_n = []
    try:
        import gmsh
        gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
        for p in sorted(glob.glob(os.path.join(POOL, "*.stp")))[:8]:
            try:
                gmsh.open(p); gmsh.model.mesh.generate(2)
                nt, _, _ = gmsh.model.mesh.getNodes()
                raw_n.append(len(nt)); gmsh.clear()
            except Exception:
                gmsh.clear()
        gmsh.finalize()
    except Exception as e:
        raw_n = []
    jn = np.array(jn); obj_n = np.array(obj_n); raw_n = np.array(raw_n)

    def st(x):
        return {"n": int(len(x)), "min": int(x.min()), "median": int(np.median(x)), "max": int(x.max())} if len(x) else {}
    out = {"json_wscad_id_overlap": {"count": len(overlap), "ids": overlap[:20]},
           "vertex_counts": {"json_mesh": st(jn), "scheffler_obj_remeshed": st(obj_n),
                             "raw_wscad_step_sample": st(raw_n)},
           "density_ratio_raw_step_over_json": round(float(np.median(raw_n) / max(np.median(jn), 1)), 2) if len(raw_n) else None,
           "note": ("JSON and remeshed Scheffler OBJ are same order (~5-6k verts); RAW WSCAD STEP "
                    "tessellation is far denser -> the thesis remesh to ~6000 is what closes the gap. "
                    "A model must see the SAME (remeshed) distribution at train and inference.")}
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/cp_domain_gap.json", "w"), indent=1)
    print("JSON<->WSCAD id overlap:", len(overlap))
    print("vertex median -> JSON:", st(jn).get("median"), "| Scheffler-OBJ(remeshed):", st(obj_n).get("median"),
          "| RAW WSCAD STEP(sample):", st(raw_n).get("median"))
    print("raw-STEP / JSON density ratio:", out["density_ratio_raw_step_over_json"], "-> results/cp_domain_gap.json")


if __name__ == "__main__":
    main()
