# -*- coding: utf-8 -*-
"""Prepare a HUMAN-LABELLING package for the families the model misses (working-product lever 1a).
From the generalisation report, take the noisy + no_cableentry parts (the model's real gaps),
run STEP -> thesis remesh -> save a ready-to-label OBJ + a template labels.txt (all Housing) +
provenance, so an annotator can paint the 5 classes and drop the part straight into the corpus.

Output: _label_targets/<pid>/{<pid>.obj, <pid>.labels.template.txt, <pid>.provenance.json} + README.
Usage: PYTHONPATH=_diffusion_net_repo/src .venv/Scripts/python.exe prepare_labeling.py
"""
import os, json, hashlib
import numpy as np
import thesis_remesh
from infer_step_cp import step_to_mesh

OUT = "_label_targets"
CLASSES = "0=Housing 1=Contact 2=SnapPoint 3=CableEntry 4=LabelSurface"


def save_obj(path, V, F):
    with open(path, "w") as f:
        for v in V:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for t in F:
            f.write(f"f {t[0]+1} {t[1]+1} {t[2]+1}\n")


def main():
    os.makedirs(OUT, exist_ok=True)
    rep = json.load(open("results/generalization_report.json"))
    cand = {c["part_id"]: c for c in json.load(open("benchmark_candidates.json"))["candidates"]}
    targets = rep["noisy_ids"] + rep["no_cableentry_ids"]
    done = []
    for pid in targets:
        c = cand.get(pid)
        if not c:
            continue
        try:
            Vr, Fr = step_to_mesh(c["step_file"])
            V, F = thesis_remesh.remesh_uniform(Vr, Fr, target=6000)
            V = np.ascontiguousarray(V, float); F = np.ascontiguousarray(F, np.int64)
        except Exception as e:
            print(f"  {pid}: ERR {str(e)[:50]}"); continue
        d = os.path.join(OUT, pid); os.makedirs(d, exist_ok=True)
        save_obj(os.path.join(d, f"{pid}.obj"), V, F)
        np.savetxt(os.path.join(d, f"{pid}.labels.template.txt"), np.zeros(len(V), int), fmt="%d")
        json.dump({"part_id": pid, "source_step": c["step_file"], "step_sha256": c["sha256"],
                   "n_verts": len(V), "reason": "generalisation gap (model missed CableEntry / noisy)",
                   "classes": CLASSES, "frame": "STEP frame (remesh is identity)"},
                  open(os.path.join(d, f"{pid}.provenance.json"), "w"), indent=1)
        done.append(pid); print(f"  {pid}: {len(V)}v -> {d}/", flush=True)
    readme = f"""# Labelling package -- working-product generalisation gaps ({len(done)} parts)

These are UNSEEN WSCAD terminal blocks the 91-part model handles poorly (missed CableEntry or noisy
segmentation), per results/generalization_report.json. Label them to close the gap.

## Task (thesis 5-class, per vertex)
{CLASSES}
Each part: <pid>/<pid>.obj (the mesh) + <pid>.labels.template.txt (all 0=Housing -> edit to the
real class per vertex, same vertex order as the OBJ).

## Workflow
1. Paint the 5 classes on <pid>.obj (any mesh-vertex labelling tool; keep vertex order).
2. Save as <pid>.labels.txt (one int per line, len == #vertices).
3. Add <pid>/ into wscad_corpus_scheffler_exact/train/ and append a split-manifest row.
4. Retrain (train_scheffler_semantic.py) or finetune, then re-run measure_generalization.py --
   the family should move noisy/no_cableentry -> clean. Frame is STEP-identity so the CPs are
   directly usable.

Priority: the parts where the model predicts Contact/LabelSurface on the wire openings but 0
CableEntry (e.g. 0294380000) -- teaching CableEntry there is the highest-value fix.
"""
    open(os.path.join(OUT, "README.md"), "w").write(readme)
    print(f"\n{len(done)} parts prepared -> {OUT}/ (+ README.md). Labelling is the human step.")


if __name__ == "__main__":
    main()
