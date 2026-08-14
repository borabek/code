# -*- coding: utf-8 -*-
"""DURUST + SIZINTISIZ per-mfg gate esik analizi. f1_sweep_data.npz (urunun tam feature seti: WEI-held+tum
PXC, union CP, wire-gate feat, axis-aware TP) -> OOF gate skoru (GroupKFold, sizinti yok) -> per-mfg esik.
- global 0.35 (mevcut urun) OOF F1 -> f1_sweep ile dogrula (~WEI 0.641 / PXC 0.703 / ALL 0.693)
- global CV-optimal esik (metadata GEREKMEZ, deployable)
- per-mfg NESTED-CV esik (esik train'de sec, test'te olc; metadata gerekir = upper-bound)
GPU YOK."""
import numpy as np, json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

d = np.load("results/f1_sweep_data.npz", allow_pickle=True)
X, v, y, grp, mfg = d["X"], d["votes"], d["y"], d["groups"], d["mfg"]
ngt_of = dict(zip(d["grp_ids"].tolist(), d["ngt"].tolist()))
THRS = np.round(np.arange(0.0, 0.71, 0.02), 3)


def oof_scores(Xa, ya, ga, seed=0):
    """GroupKFold OOF RF olasilik -- sizinti yok (parca bazli fold)."""
    oof = np.zeros(len(ya))
    for tr, te in GroupKFold(n_splits=5).split(Xa, ya, ga):
        clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=3, n_jobs=-1, random_state=seed)
        clf.fit(Xa[tr], ya[tr]); oof[te] = clf.predict_proba(Xa[te])[:, 1]
    return oof


def prf(tp, nk, gt):
    p = tp / max(nk, 1); r = tp / max(gt, 1); return p, r, 2 * p * r / max(p + r, 1e-9)


def metrics(mask, oof, thr):
    k = mask & (v >= 1) & (oof >= thr)
    tp = int((y[k] == 1).sum()); nk = int(k.sum())
    gt = sum(ngt_of[g] for g in np.unique(grp[mask]))
    return prf(tp, nk, gt)


OOF = oof_scores(X, y, grp)                       # tek OOF (urun gate, sizintisiz)
masks = {"ALL": np.ones(len(y), bool), "WEI": mfg == 1, "PXC": mfg == 0}

print("=== (1) MEVCUT URUN: global gate 0.35, OOF (f1_sweep ile ayni olmali) ===")
for nm, m in masks.items():
    p, r, f = metrics(m, OOF, 0.35); print(f"  {nm:4s}: P={p:.3f} R={r:.3f} F1={f:.3f}")


def insample_best(m, oof):
    return max(THRS, key=lambda t: metrics(m, oof, t)[2])


print("\n=== (2) IN-SAMPLE optimal esik (ust-sinir referans, deploy EDILEMEZ) ===")
for nm, m in masks.items():
    bt = insample_best(m, OOF); p, r, f = metrics(m, OOF, bt)
    print(f"  {nm:4s}: thr={bt:.2f} -> P={p:.3f} R={r:.3f} F1={f:.3f}")


def nested_cv(m, K=5):
    """Dis GroupKFold: esik outer-train'de sec (kendi ic-OOF skorunda), outer-test'te uygula.
    Hem gate hem esik test-parcalarindan bagimsiz -> tam durust. TP/FP/FN test-fold'lar boyunca toplanir."""
    idx = np.where(m)[0]; pids = np.unique(grp[idx])
    rng = np.random.RandomState(0); pids = pids[rng.permutation(len(pids))]
    folds = np.array_split(pids, K)
    TP = NK = GT = 0; chosen = []
    for f in folds:
        te_p = set(f.tolist())
        tr_mask = m & np.array([g not in te_p for g in grp])
        te_mask = m & np.array([g in te_p for g in grp])
        # esigi outer-train'in KENDI OOF'unda sec (outer-test gate egitimine hic girmez)
        tr_idx = np.where(tr_mask)[0]
        oof_tr = oof_scores(X[tr_idx], y[tr_idx], grp[tr_idx])
        best_t, best_f = 0.35, -1
        for t in THRS:
            k = (v[tr_idx] >= 1) & (oof_tr >= t); tp = int((y[tr_idx][k] == 1).sum())
            gt = sum(ngt_of[g] for g in np.unique(grp[tr_idx]))
            _, _, ff = prf(tp, int(k.sum()), gt)
            if ff > best_f: best_f, best_t = ff, t
        chosen.append(best_t)
        # outer-test: tam-veri gate'in OOF skorunu kullan (bu parcalar o fold'da zaten dislanmisti)
        k = te_mask & (v >= 1) & (OOF >= best_t); TP += int((y[k] == 1).sum()); NK += int(k.sum())
        GT += sum(ngt_of[g] for g in np.unique(grp[te_mask]))
    p, r, ff = prf(TP, NK, GT)
    return p, r, ff, float(np.median(chosen)), [round(c, 2) for c in chosen]


print("\n=== (3) NESTED-CV per-mfg esik (DURUST, deployable eger mfg biliniyorsa) ===")
report = {}
for nm, m in masks.items():
    p, r, f, mt, ch = nested_cv(m)
    g = metrics(m, OOF, 0.35)[2]
    print(f"  {nm:4s}: CV thr(med)={mt:.2f} -> P={p:.3f} R={r:.3f} F1={f:.3f}  | vs 0.35 F1={g:.3f} = {f-g:+.3f}  folds={ch}")
    report[nm] = {"cv_f1": f, "cv_thr_med": mt, "global035_f1": g, "gain": f - g}
json.dump(report, open("results/gate_thr_oof_report.json", "w"), indent=2)
print("\n-> results/gate_thr_oof_report.json")
print("KARAR: (3) kazanc >0 ise gercek+deployable. ALL kazanci = metadata'siz global esik degisimi.")
