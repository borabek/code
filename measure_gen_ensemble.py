# -*- coding: utf-8 -*-
"""Ensemble generalisation measure: average the softmax of several seed checkpoints, argmax, then
apply the SAME verdict logic as measure_gen_newmodel.py. Tests whether the 3-seed ensemble is more
robust (steadier no_cableentry) than any single seed (s0=0, s1=3, s2=2 -> seed-luck).
Usage: PYTHONPATH=_diffusion_net_repo/src .venv/Scripts/python.exe measure_gen_ensemble.py \
         --ckpts results/seg_extra/rec_augcew_s0.pt results/seg_extra/rec_augcew_s1.pt results/seg_extra/rec_augcew_s2.pt
"""
import argparse, os, json, hashlib
import numpy as np
import torch
import diffusionnet as D
import connector3d, cp_openings, thesis_remesh
from infer_step_cp import step_to_mesh

CE = int(connector3d.CABLE_ENTRY); OP = "results/step_infer/ops"
NAMES = {int(getattr(connector3d, n)): n for n in ("HOUSING", "CONTACT", "SNAP_POINT", "CABLE_ENTRY", "LABEL_SURFACE")}


def load(ckpt, dev):
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    meta = ck["meta"]; meta.setdefault("n_eig", ck["cfg"].get("n_eig", 48))
    m, _ = D.build_diffusionnet(ck["cfg"], n_classes=5); m.load_state_dict(ck["state"]); m.to(dev).eval()
    return m, meta


def ens_probs(models, metas, V, F, dev):
    """Average softmax across models. Operators are cached per-part; n_eig shared across seeds here."""
    V = np.ascontiguousarray(V, np.float64); F = np.ascontiguousarray(F, np.int64)
    acc = None
    for (m, meta) in zip(models, metas):
        ops = D.precompute_operators(V, F, meta.get("n_eig", 48), op_cache_dir=OP)
        ops = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in ops.items()}
        with torch.no_grad():
            out = D._forward(m, ops, D._model_input(ops, meta))
            p = torch.softmax(out, dim=-1).cpu().numpy()
        acc = p if acc is None else acc + p
    return (acc / len(models)).argmax(-1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpts", nargs="+", required=True); a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    models, metas = [], []
    for ck in a.ckpts:
        m, meta = load(ck, dev); models.append(m); metas.append(meta)
    setname = "gen_dev_40.json" if os.path.exists("gen_dev_40.json") else "benchmark_candidates.json"
    cand = json.load(open(setname))["candidates"]
    def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16] if os.path.exists(p) else None
    clean = noisy = none = err = 0; rows = []
    for c in cand:
        pid = c["part_id"]
        try:
            Vr, Fr = step_to_mesh(c["step_file"]); V, F = thesis_remesh.remesh_uniform(Vr, Fr, target=6000)
            plab = ens_probs(models, metas, V, F, dev)
        except Exception as e:
            err += 1; rows.append({"part_id": pid, "error": str(e)[:60]}); continue
        cef = [f for f in connector3d.build_fragments(V, F, plab, min_vertices=5) if int(f.label) == CE]
        ce_tot = int((plab == CE).sum()); n_comp = len(cef)
        big = max((len(f.idx) for f in cef), default=0); frac = big / max(ce_tot, 1)
        cps = cp_openings.connection_points(V, F, plab, min_v=20, classes=(CE,))
        v = "no_cableentry" if ce_tot == 0 else ("clean" if (n_comp <= 8 and frac >= 0.25 and 1 <= len(cps) <= 12) else "noisy")
        clean += v == "clean"; noisy += v == "noisy"; none += v == "no_cableentry"
        rows.append({"part_id": pid, "verdict": v, "cp_count": len(cps)})
        print(f"  {pid}: cp={len(cps)} -> {v}", flush=True)
    rep = {"metric": "generalisation_heuristic ENSEMBLE (avg softmax; NOT a scored benchmark)",
           "candidate_set": setname, "candidate_set_role": "gen_dev (model selection -> NOT final benchmark)",
           "ckpts": a.ckpts, "ckpt_shas": [sha(c) for c in a.ckpts], "candidate_set_sha16": sha(setname),
           "clean": clean, "noisy": noisy, "no_cableentry": none, "err": err, "n": len(cand),
           "single_seed_no_cableentry": {"s0": 0, "s1": 3, "s2": 2}, "per_part": rows}
    os.makedirs("results/gen_eval", exist_ok=True)
    out = "results/gen_eval/rec_augcew_ENSEMBLE3_gen_dev_40.json"
    json.dump(rep, open(out, "w"), indent=1)
    print(f"\nENSEMBLE(3): clean {clean} | noisy {noisy} | no_cableentry {none} | err {err}  -> {out}")
    print("SINGLE SEEDS were: clean 34/33/29 | no_cableentry 0/3/2")


if __name__ == "__main__":
    main()
