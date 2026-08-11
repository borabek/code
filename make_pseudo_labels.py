# -*- coding: utf-8 -*-
"""SELF-TRAINING pseudo-labels: the one untested autonomous lever (2026-07-20).

Human labelling is blocked (user has no electrical domain knowledge; my own blind attempts scored
Connection IoU 0.408 / 0.276 vs the model's 0.672 -> my labels would REGRESS the model). Annotation
tools (CVAT/SAM/LabelCloud) accelerate CLICKING, but the bottleneck is DECIDING the class.

What IS still available: 3571 UNLABELLED WSCAD parts. The model fails on the 22 hard queue parts, but
is confident on many easy ones. Standard semi-supervised recipe: keep only the most CONFIDENT parts,
use the model's own prediction as a pseudo-label, add to training, and MEASURE on human-GT val.

Honest risk (this project has been burned before): pseudo-labels can entrench the model's own errors.
That is exactly why the only thing that counts is the val Connection IoU vs the 0.622 baseline --
if it does not beat it, we discard this and say so.

Usage: PYTHONPATH=_diffusion_net_repo/src .venv/Scripts/python.exe make_pseudo_labels.py --n 250 --keep 120
"""
import os, json, glob, argparse, hashlib
import numpy as np
import torch
import diffusionnet as D
import connector3d, thesis_remesh
from infer_step_cp import step_to_mesh, load_any

OUT = "_pseudo_extra"; OP = "results/step_infer/ops"
CKPT = "results/scheffler_semantic/refit91.pt"
CE = int(connector3d.CABLE_ENTRY); CT = int(connector3d.CONTACT)


def save_obj(path, V, F):
    with open(path, "w") as f:
        for v in V: f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for t in F: f.write(f"f {t[0]+1} {t[1]+1} {t[2]+1}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=250, help="how many unlabelled parts to try")
    ap.add_argument("--keep", type=int, default=120, help="keep the K most confident")
    ap.add_argument("--min-conn", type=int, default=60, help="require at least this many connection-class vertices (a usable part)")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, meta, _ = load_any(CKPT, dev=dev)

    # exclude everything already used: labelled corpus, gen_dev, holdout, labelling queue
    excl = set()
    for f, key in [("gen_dev_40.json", "candidates"), ("final_holdout_40.json", "candidates"),
                   ("benchmark_candidates.json", "candidates")]:
        try: excl |= {c["part_id"] for c in json.load(open(f))[key]}
        except Exception: pass
    try: excl |= {os.path.basename(d.rstrip("\\/")) for d in glob.glob("_label_targets/*/")}
    except Exception: pass
    try:
        import scheffler_dataset as ds
        for sp in ("train", "val"):
            excl |= {s["part_id"] for s in ds.load_split("wscad_corpus_scheffler_exact", sp, verify_hashes=False)}
    except Exception as e:
        print("split exclude failed:", e)
    print(f"excluding {len(excl)} already-used part ids")

    steps = sorted(glob.glob("all_wscad_stp/*.stp"))
    cand = []
    for s in steps:
        pid = os.path.basename(s).split("_")[1]
        if pid not in excl: cand.append((pid, s))
    print(f"{len(cand)} unlabelled candidates; trying {a.n}")

    rows = []
    for pid, stp in cand[:a.n]:
        try:
            Vr, Fr = step_to_mesh(stp)
            V, F = thesis_remesh.remesh_uniform(Vr, Fr, target=6000)
            V = np.ascontiguousarray(V, np.float64); F = np.ascontiguousarray(F, np.int64)
            lab, probs = D.predict(model, meta, V, F, device=dev, op_cache_dir=OP, return_probs=True)
            lab = np.asarray(lab); probs = np.asarray(probs)
        except Exception as e:
            continue
        conf = float(probs.max(-1).mean())                       # overall confidence
        nconn = int(np.isin(lab, (CE, CT)).sum())                # does it even have connection features
        conn_conf = float(probs[np.isin(lab, (CE, CT))].max(-1).mean()) if nconn else 0.0
        rows.append({"part_id": pid, "conf": conf, "conn_conf": conn_conf, "n_conn": nconn,
                     "V": V, "F": F, "lab": lab})
        print(f"  {pid}: conf={conf:.3f} conn_conf={conn_conf:.3f} n_conn={nconn}", flush=True)

    ok = [r for r in rows if r["n_conn"] >= a.min_conn]
    ok.sort(key=lambda r: -(r["conf"] + r["conn_conf"]))
    keep = ok[:a.keep]
    print(f"\nusable {len(ok)}/{len(rows)} (>= {a.min_conn} connection verts); keeping top {len(keep)} by confidence")
    man = []
    for r in keep:
        d = os.path.join(OUT, r["part_id"]); os.makedirs(d, exist_ok=True)
        save_obj(os.path.join(d, f"{r['part_id']}.obj"), r["V"], r["F"])
        open(os.path.join(d, f"{r['part_id']}.labels.txt"), "w").write("\n".join(map(str, r["lab"].tolist())) + "\n")
        man.append({k: r[k] for k in ("part_id", "conf", "conn_conf", "n_conn")})
    json.dump({"source": "SELF-TRAINING pseudo-labels from " + CKPT,
               "warning": "MODEL PREDICTIONS, NOT human labels. Only valid if val Connection IoU beats the 0.622 baseline.",
               "n_kept": len(keep), "parts": man}, open("pseudo_manifest.json", "w"), indent=1)
    print(f"wrote {len(keep)} pseudo-labelled parts -> {OUT}/  (manifest: pseudo_manifest.json)")


if __name__ == "__main__":
    main()
