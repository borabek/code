# -*- coding: utf-8 -*-
"""Did the generalisation recipe (rec_augcew) segment the CONNECTION classes BETTER or WORSE than
the in-domain refit91 model, measured on HUMAN-GT val (leakage-safe: locked test untouched)?

REVISED 2026-07-20 (cp-v3 correction): CP placement is no longer driven by CableEntry alone --
the arbiter (measure_cp_defs.py, results/cp_def_eval.json) showed real CPs live on Contact
components too (square/clamp entries), gated by thesis insertion depth. So this now reports THREE
numbers: CableEntry IoU, Contact IoU, and "Connection IoU" (CableEntry+Contact merged into one
foreground class vs the rest) -- the last one is the real cp-v3 CP-placement driver, since a CP can
come from either class. Ranking below uses Connection IoU as the primary sort key.

Usage: PYTHONPATH=_diffusion_net_repo/src .venv/Scripts/python.exe measure_val_cableentry.py
"""
import os, numpy as np, torch
import diffusionnet as D
import scheffler_dataset as ds
import connector3d
CE = int(connector3d.CABLE_ENTRY); CT = int(connector3d.CONTACT); NCLS = 5
NAMES = {int(getattr(connector3d, n)): n for n in ("HOUSING","CONTACT","SNAP_POINT","CABLE_ENTRY","LABEL_SURFACE")}
OPS = "results/seg_extra/ops"


def load_any(ckpt, dev):
    if os.path.basename(ckpt).startswith("refit"):
        m, meta, _ = D.load_checkpoint(ckpt, device=dev); return m, meta
    d = torch.load(ckpt, map_location=dev, weights_only=False)
    meta = d["meta"]; meta.setdefault("n_eig", d["cfg"].get("n_eig", 48))
    m, _ = D.build_diffusionnet(d["cfg"], n_classes=5); m.load_state_dict(d["state"]); m.to(dev).eval()
    return m, meta


def per_class_iou(model, meta, data, dev):
    inter = np.zeros(NCLS); union = np.zeros(NCLS); acc = tot = 0
    conn_inter = conn_union = 0   # CableEntry+Contact merged into one "connection" foreground class
    for s in data:
        V = np.ascontiguousarray(s["verts"], np.float64); F = np.ascontiguousarray(s["faces"], np.int64)
        gt = np.asarray(s["labels"])
        ops = D.precompute_operators(V, F, meta.get("n_eig", 48), op_cache_dir=OPS)
        ops = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in ops.items()}
        with torch.no_grad():
            pr = D._forward(model, ops, D._model_input(ops, meta)).argmax(-1).cpu().numpy()
        acc += (pr == gt).sum(); tot += len(gt)
        for c in range(NCLS):
            inter[c] += ((pr == c) & (gt == c)).sum(); union[c] += ((pr == c) | (gt == c)).sum()
        pr_conn = np.isin(pr, (CE, CT)); gt_conn = np.isin(gt, (CE, CT))
        conn_inter += (pr_conn & gt_conn).sum(); conn_union += (pr_conn | gt_conn).sum()
    iou = inter / np.maximum(union, 1)
    conn_iou = conn_inter / max(conn_union, 1)
    return iou, acc / max(tot, 1), conn_iou


def main():
    import argparse, glob
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="*", default=None,
                    help="checkpoints to measure; default = refit91 + rec_augcew + all sweep sw_*.pt + lc_*.pt")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    va = ds.load_split("wscad_corpus_scheffler_exact", "val", verify_hashes=False)
    print(f"val human-GT parts: {len(va)} (locked test NOT touched)")
    print("cp-v3: CP driver = CableEntry OR depth-gated Contact -> 'Connection IoU' (merged) is the honest selector\n")
    if a.ckpts:
        items = [(os.path.basename(c), c) for c in a.ckpts]
    else:
        items = [("refit91 (in-domain benchmark)", "results/scheffler_semantic/refit91.pt"),
                 ("rec_augcew_s0 (gen product; heuristic-tuned)", "results/seg_extra/rec_augcew_s0.pt")]
        items += [(os.path.basename(c), c) for c in sorted(glob.glob("results/seg_extra/sw_*.pt"))]
        items += [(os.path.basename(c), c) for c in sorted(glob.glob("results/seg_extra/lc_*.pt"))]
    results = []
    for name, ckpt in items:
        if not os.path.exists(ckpt):
            print(f"{name}: MISSING {ckpt}\n"); continue
        try:
            m, meta = load_any(ckpt, dev)
        except Exception as e:
            print(f"{name}: LOAD FAILED {type(e).__name__}: {e}\n"); continue
        iou, acc, conn_iou = per_class_iou(m, meta, va, dev)
        results.append((name, conn_iou, iou[CE], iou[CT], iou.mean(), acc))
        print(f"{name}")
        print(f"   acc={acc:.4f}  mIoU={iou.mean():.4f}")
        print("   " + "  ".join(f"{NAMES[c]}={iou[c]:.3f}" for c in range(NCLS)))
        print(f"   >>> CableEntry IoU={iou[CE]:.4f}  Contact IoU={iou[CT]:.4f}  Connection IoU (merged, cp-v3 driver)={conn_iou:.4f}\n")
    if results:
        print("=== RANKED by Connection IoU (cp-v3 honest CP selector: CableEntry+Contact merged) ===")
        for name, conn, ce, ct, mi, ac in sorted(results, key=lambda r: -r[1]):
            print(f"   conn={conn:.4f}  (CE={ce:.3f} CT={ct:.3f})  mIoU {mi:.3f} acc {ac:.3f}  <- {name}")
        print("\n=== for reference, RANKED by CableEntry IoU alone (cp-v2 metric, superseded) ===")
        for name, conn, ce, ct, mi, ac in sorted(results, key=lambda r: -r[2]):
            print(f"   CE={ce:.4f}  (conn={conn:.3f} CT={ct:.3f})  <- {name}")


if __name__ == "__main__":
    main()
