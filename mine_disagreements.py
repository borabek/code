# -*- coding: utf-8 -*-
"""Mine MODEL-vs-HUMAN disagreements on the TRAINING parts, for targeted adjudication.

WHY THIS AND NOT "LABEL MORE PARTS": the label learning curve is FLAT (103 labels did not beat 77).
More random parts add nothing. But the model's remaining failure is DISCRIMINATION, not capacity --
raising capacity (9k verts, k_eig 96) buys recall and loses precision every time, because the masked
loss only ever says "these vertices ARE connection" and NEVER "this opening-like thing is NOT one".
Negative examples are the one supervision signal this project has never given.

So: find every region the model calls Connection that the annotator did NOT mark, on the parts we
already train on. Each one is worth adjudicating and is useful either way:
  - "real opening I missed"  -> a POSITIVE label correction (helps recall)
  - "not a CP" (screw head, test slot, mounting hole) -> a NEGATIVE example (helps precision)

NEVER mine batch-4 (_label_targets_4): it is the held-out measurement set and must stay unseen.

Usage: PYTHONPATH=_diffusion_net_repo/src .venv/Scripts/python.exe mine_disagreements.py [--limit N]
"""
import os, sys, glob, json, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diffusionnet as D, connector3d, cp_openings
from region_label_helper import load_obj
from infer_step_cp import load_any

CE = int(connector3d.CABLE_ENTRY); CT = int(connector3d.CONTACT)
OP = "results/step_infer/ops"
TRAIN_DIRS = ["_label_targets", "_label_targets_2", "_label_targets_3"]   # batch-4 EXCLUDED on purpose


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ckpts", nargs="+", default=[f"results/seg_extra/human77c_s{i}.pt" for i in (0, 1, 2)],
                    help="which model to mine with. Round 2 uses the adjudication-corrected models "
                         "(adj_s*): a better model disagrees in DIFFERENT places, so each round "
                         "surfaces new label errors. Round 1 scored 87/91 real openings (96%).")
    ap.add_argument("--out", default="results/disagreements.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    models = [load_any(c, dev=a.device) for c in a.ckpts]
    print(f"  model: {[os.path.basename(c) for c in a.ckpts]}", flush=True)
    rows = []; n_parts = 0
    for root in TRAIN_DIRS:
        for d in sorted(glob.glob(os.path.join(root, "*"))):
            if not os.path.isdir(d): continue
            pid = os.path.basename(os.path.normpath(d))
            of, lf = os.path.join(d, f"{pid}.obj"), os.path.join(d, f"{pid}.labels.txt")
            if not (os.path.exists(of) and os.path.exists(lf)): continue
            L = np.array([int(x) for x in open(lf).read().split()], np.int64)
            V, F = load_obj(of)
            if len(L) != len(V) or not (L == CE).any(): continue
            V = np.ascontiguousarray(V, np.float64); F = np.ascontiguousarray(F, np.int64)
            acc = None
            for model, meta, _ in models:
                _, pb = D.predict(model, meta, V, F, device=a.device, op_cache_dir=OP, return_probs=True)
                pb = np.asarray(pb, float); acc = pb if acc is None else acc + pb
            probs = acc / len(models); lab = probs.argmax(-1)
            # the product's own derivation settings, so we mine exactly what the PRODUCT would emit
            cps = cp_openings.connection_points(V, F, lab, min_v=45, classes=(CE, CT), dedupe_mm=10.0,
                                                probs=probs, vertex_conf=0.7, ct_depth_min_mm=1.0,
                                                cluster_mm=5.0)
            hum = [f for f in connector3d.build_fragments(V, F, np.where(L == CE, CE, 0), min_vertices=20)
                   if int(f.label) == CE]
            hc = np.array([V[np.asarray(f.idx, int)].mean(0) for f in hum]) if hum else np.zeros((0, 3))
            diag = float(np.linalg.norm(V.max(0) - V.min(0))); tol = max(3.0, 0.06 * diag)
            n_parts += 1
            for c in cps:
                p = np.asarray(c["point"], float)
                dmin = float(np.linalg.norm(hc - p, axis=1).min()) if len(hc) else 1e9
                if dmin <= tol:
                    continue                     # matches a human mark -> agreed, nothing to adjudicate
                rows.append({"part_id": pid, "dir": root, "point": p.tolist(),
                             "direction": np.asarray(c["direction"], float).tolist(),
                             "confidence": float(c.get("confidence", 0)),
                             "n_verts": int(c.get("n_verts", 0)),
                             "source": "CableEntry" if int(c.get("source_label", CE)) == CE else "Contact",
                             "nearest_human_mm": round(dmin, 1), "tol_mm": round(tol, 1)})
            if a.limit and n_parts >= a.limit: break
        if a.limit and n_parts >= a.limit: break

    per_part = {}
    for r in rows: per_part[r["part_id"]] = per_part.get(r["part_id"], 0) + 1
    os.makedirs("results", exist_ok=True)
    json.dump({"n_parts": n_parts, "n_disagreements": len(rows),
               "per_part": per_part, "items": rows}, open(a.out, "w"), indent=1)
    print(f"\n{n_parts} egitim parcasi tarandi -> {len(rows)} anlasmazlik bolgesi")
    print(f"  parca basina ortalama {len(rows)/max(n_parts,1):.1f}")
    top = sorted(per_part.items(), key=lambda kv: -kv[1])[:8]
    print("  en cok anlasmazlik: " + ", ".join(f"{k}({v})" for k, v in top))
    print(f"  -> {a.out}")


if __name__ == "__main__":
    main()
