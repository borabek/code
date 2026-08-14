# Two failure types remain. Type 2 is suspicious: parts where the model predicts the RIGHT NUMBER of
# CPs yet matches NONE (3249017: 2 human / 2 pred / tp0). If those predictions sit just OUTSIDE the
# tolerance, it is a systematic v_o OFFSET (the prediction's region is eroded by vertex_conf while the
# human GT region is not -> their opening-midpoints shift differently), which would be fixable.
# If they sit far away, they are genuine wrong-place detections.
import os, glob
import numpy as np, torch
import diffusionnet as D, connector3d, cp_openings
from region_label_helper import load_obj
from infer_step_cp import load_any
CE = int(connector3d.CABLE_ENTRY); CT = int(connector3d.CONTACT); OP = "results/step_infer/ops"
dev = "cuda" if torch.cuda.is_available() else "cpu"
CK = [f"results/seg_extra/human77c_s{i}.pt" for i in (0, 1, 2)]
models = [load_any(c, dev=dev) for c in CK]
pids = ["3249017", "3280117", "3076031", "3212095", "3031047", "3046703"]
print(f"{'part':10s} {'tol':>5s}  each tahminin en yakin insan CP'sine mesafesi (mm)", flush=True)
allnear = []
for pid in pids:
    V, F = load_obj(f"_label_targets_4/{pid}/{pid}.obj")
    L = np.array([int(x) for x in open(f"_label_targets_4/{pid}/{pid}.labels.txt").read().split()])
    Vc = np.ascontiguousarray(V, np.float64); Fc = np.ascontiguousarray(F, np.int64)
    G_cps = cp_openings.connection_points(Vc, Fc, L, min_v=20, classes=(CE,), dedupe_mm=0.0, cluster_mm=0.0)
    G = np.array([np.asarray(c["point"]) for c in G_cps], float)
    acc = None
    for model, meta, _ in models:
        _, pb = D.predict(model, meta, Vc, Fc, device=dev, op_cache_dir=OP, return_probs=True)
        pb = np.asarray(pb, float); acc = pb if acc is None else acc + pb
    probs = acc / len(models); lab = probs.argmax(-1)
    P_cps = cp_openings.connection_points(Vc, Fc, lab, min_v=45, classes=(CE, CT), dedupe_mm=10.0,
                                          probs=probs, vertex_conf=0.7, ct_depth_min_mm=1.0, cluster_mm=5.0)
    P = np.array([np.asarray(c["point"]) for c in P_cps], float) if P_cps else np.zeros((0, 3))
    tol = max(3.0, 0.06 * float(np.linalg.norm(V.max(0) - V.min(0))))
    if not len(P) or not len(G):
        print(f"{pid:10s} {tol:5.1f}  (pred {len(P)} / human {len(G)})", flush=True); continue
    dm = np.linalg.norm(P[:, None, :] - G[None, :, :], axis=2)
    near = dm.min(1)
    allnear += [d for d in near]
    print(f"{pid:10s} {tol:5.1f}  " + "  ".join(f"{d:.1f}" for d in np.sort(near)), flush=True)
if allnear:
    a = np.array(allnear)
    print(f"\nTUM tahminler: medyan {np.median(a):.1f}mm | tolerans-ici {(a<=5).sum()}/{len(a)}", flush=True)
    print(f"  kil payi kacan (tol..2xtol): {((a>5)&(a<=10)).sum()} | tamamen uzak (>10mm): {(a>10).sum()}", flush=True)
