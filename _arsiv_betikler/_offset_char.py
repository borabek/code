# TODO A, step 1: characterise the ~5mm offset between the model's v_o and the human-derived v_o
# BEFORE trying to fix it. If the offset is SYSTEMATIC (consistent direction, e.g. along the insertion
# axis or outward) a correction is possible. If it is random, it is noise and no correction helps.
# For each near-matched pair we decompose the offset into: along the CP's insertion direction
# (normal), and tangential (in the opening plane).
import os
import numpy as np, torch
import diffusionnet as D, connector3d, cp_openings
from region_label_helper import load_obj
from infer_step_cp import load_any
CE = int(connector3d.CABLE_ENTRY); CT = int(connector3d.CONTACT); OP = "results/step_infer/ops"
dev = "cuda" if torch.cuda.is_available() else "cpu"
models = [load_any(f"results/seg_extra/human77c_s{i}.pt", dev=dev) for i in (0, 1, 2)]
import glob, json
rows = []
for d in sorted(glob.glob("_label_targets_4/*/")):
    pid = os.path.basename(os.path.normpath(d))
    of, lf = f"{d}{pid}.obj", f"{d}{pid}.labels.txt"
    if not (os.path.exists(of) and os.path.exists(lf)): continue
    L = np.array([int(x) for x in open(lf).read().split()])
    V, F = load_obj(of)
    if len(L) != len(V) or not (L == CE).any(): continue
    Vc = np.ascontiguousarray(V, np.float64); Fc = np.ascontiguousarray(F, np.int64)
    G_cps = cp_openings.connection_points(Vc, Fc, L, min_v=20, classes=(CE,), dedupe_mm=0.0, cluster_mm=0.0)
    if not G_cps: continue
    acc = None
    for model, meta, _ in models:
        _, pb = D.predict(model, meta, Vc, Fc, device=dev, op_cache_dir=OP, return_probs=True)
        pb = np.asarray(pb, float); acc = pb if acc is None else acc + pb
    probs = acc / len(models); lab = probs.argmax(-1)
    P_cps = cp_openings.connection_points(Vc, Fc, lab, min_v=45, classes=(CE, CT), dedupe_mm=10.0,
                                          probs=probs, vertex_conf=0.7, ct_depth_min_mm=1.0, cluster_mm=5.0)
    if not P_cps: continue
    G = np.array([np.asarray(c["point"]) for c in G_cps], float)
    P = np.array([np.asarray(c["point"]) for c in P_cps], float)
    dm = np.linalg.norm(P[:, None, :] - G[None, :, :], axis=2)
    for i in range(len(P)):
        j = int(dm[i].argmin()); dist = float(dm[i, j])
        if dist > 15: continue                      # genuinely different location, not an offset
        off = P[i] - G[j]                           # offset vector, prediction minus human
        nrm = np.asarray(P_cps[i]["direction"], float)
        nrm = nrm / (np.linalg.norm(nrm) + 1e-9)
        along = float(off @ nrm)                    # + = prediction sits FURTHER OUT along the CP axis
        tang = float(np.linalg.norm(off - along * nrm))
        rows.append((pid, dist, along, tang))
a = np.array([[r[1], r[2], r[3]] for r in rows])
print(f"{len(rows)} eslesme-adayi (mesafe<15mm)")
print(f"  toplam mesafe : medyan {np.median(a[:,0]):.2f}mm  ort {a[:,0].mean():.2f}")
print(f"  EKSEN boyunca : medyan {np.median(a[:,1]):+.2f}mm  ort {a[:,1].mean():+.2f}  |ort| {np.abs(a[:,1]).mean():.2f}")
print(f"  TEGET (duzlem): medyan {np.median(a[:,2]):.2f}mm  ort {a[:,2].mean():.2f}")
pos = (a[:, 1] > 0).sum()
print(f"  axis yonu: {pos}/{len(rows)} tahmin DISARIDA (+), {len(rows)-pos} iceride (-)")
print("\n  -> axis ortalamasi |buyuk| ve tek yonlu ise SISTEMATIK (duzeltilebilir);")
print("     teget baskin and direction dagilmissa RASTGELE (duzeltilemez).")
