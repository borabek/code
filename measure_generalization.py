# -*- coding: utf-8 -*-
"""Measure how well the 91-part model GENERALISES to the 40 unseen benchmark_candidates
(no labels -> geometric/structural sanity, not F1). Per part run the thesis pipeline
(STEP->remesh->seg->CP) and record: class vertex counts, CableEntry-CP count, and a NOISE proxy
= #CableEntry connected components + fraction of CableEntry verts in the largest component
(clean segmentation = few large components; noisy = many scattered small ones). Classify each
part clean / noisy / no_cableentry so we know which families need labels.

Output: results/generalization_report.json
Usage: PYTHONPATH=_diffusion_net_repo/src .venv/Scripts/python.exe measure_generalization.py [--limit N]
"""
import argparse, os, json
import numpy as np
import diffusionnet, connector3d, cp_openings
from infer_step_cp import step_to_mesh
import thesis_remesh

CKPT = "results/scheffler_semantic/refit91.pt"; OP = "results/step_infer/ops"
CE = int(connector3d.CABLE_ENTRY); CT = int(connector3d.CONTACT)
NAMES = {int(getattr(connector3d, n)): n for n in ("HOUSING", "CONTACT", "SNAP_POINT", "CABLE_ENTRY", "LABEL_SURFACE")}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
    cand = json.load(open("benchmark_candidates.json"))["candidates"]
    if a.limit:
        cand = cand[:a.limit]
    model, meta, _ = diffusionnet.load_checkpoint(CKPT, device="cuda")
    rows = []
    for c in cand:
        pid = c["part_id"]
        try:
            Vr, Fr = step_to_mesh(c["step_file"])
            V, F = thesis_remesh.remesh_uniform(Vr, Fr, target=6000)
            V = np.ascontiguousarray(V, float); F = np.ascontiguousarray(F, np.int64)
            plab = np.asarray(diffusionnet.predict(model, meta, V, F, device="cuda", op_cache_dir=OP))
        except Exception as e:
            rows.append({"part_id": pid, "error": str(e)[:80]}); print(f"  {pid}: ERR {str(e)[:60]}"); continue
        counts = {NAMES[k]: int((plab == k).sum()) for k in NAMES}
        ce_frags = [f for f in connector3d.build_fragments(V, F, plab, min_vertices=5) if int(f.label) == CE]
        n_comp = len(ce_frags)
        big = max((len(f.idx) for f in ce_frags), default=0)
        ce_tot = counts["CABLE_ENTRY"]
        largest_frac = round(big / max(ce_tot, 1), 2)
        cps = cp_openings.connection_points(V, F, plab, min_v=20, classes=(CE,))
        # verdict
        if ce_tot == 0:
            verdict = "no_cableentry"
        elif n_comp <= 8 and largest_frac >= 0.25 and 1 <= len(cps) <= 12:
            verdict = "clean"
        else:
            verdict = "noisy"
        rows.append({"part_id": pid, "verts": len(V), "class_counts": counts, "cableentry_components": n_comp,
                     "largest_component_frac": largest_frac, "cp_count": len(cps), "verdict": verdict})
        print(f"  {pid}: CE={ce_tot}v/{n_comp}comp cp={len(cps)} -> {verdict}", flush=True)
    ok = [r for r in rows if r.get("verdict") == "clean"]
    noisy = [r for r in rows if r.get("verdict") == "noisy"]
    none = [r for r in rows if r.get("verdict") == "no_cableentry"]
    err = [r for r in rows if "error" in r]
    summary = {"n": len(rows), "clean": len(ok), "noisy": len(noisy), "no_cableentry": len(none), "error": len(err),
               "clean_ids": [r["part_id"] for r in ok], "noisy_ids": [r["part_id"] for r in noisy],
               "no_cableentry_ids": [r["part_id"] for r in none], "rows": rows}
    os.makedirs("results", exist_ok=True)
    json.dump(summary, open("results/generalization_report.json", "w"), indent=1)
    print(f"\nGENERALISATION: clean {len(ok)} | noisy {len(noisy)} | no_cableentry {len(none)} | err {len(err)}")
    print("-> results/generalization_report.json")


if __name__ == "__main__":
    main()
