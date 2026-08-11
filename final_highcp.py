# -*- coding: utf-8 -*-
"""Op 3+4+5 FINAL: gate GENISLETILMIS havuzda (9k adaylari dahil) egit -> secim darbogazini kapat.
High-CP parca p icin: gate = standart-non-highCP + TUM high-CP genisletilmis (p HARIC) -> p'yi skorla ->
lattice re-rank -> top-N. Non-high-CP: standart OOF top-N. Overall metadata F1 -> 0.80 gecti mi?"""
import json, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

pool = json.load(open("results/highcp_pool.json"))
d = np.load("results/f1_sweep_data.npz", allow_pickle=True)
Xs, ys, grp = d["X"], d["y"], d["groups"]; ngt_of = dict(zip(d["grp_ids"].tolist(), d["ngt"].tolist()))
HICP = 11; hicp = {g for g in ngt_of if ngt_of[g] >= HICP}
std_keep = ~np.isin(grp, list(hicp))              # standart non-high-CP havuz
Xstd, ystd = Xs[std_keep], ys[std_keep]


def face_axes(P):
    Q = P - P.mean(0); _, _, Vt = np.linalg.svd(Q, full_matrices=False); return Vt[0], Vt[1]

def consistency(P, ws, tf=0.04):
    u, v = face_axes(P); pu = P @ u; pv = P @ v; span = max(pu.max()-pu.min(), pv.max()-pv.min(), 1.0); tol = tf*span
    c = np.zeros(len(P))
    for i in range(len(P)):
        al = (np.abs(pv-pv[i]) < tol) | (np.abs(pu-pu[i]) < tol); al[i] = False; c[i] = float(ws[al].sum())
    return c/max(c.max(), 1e-9)

pids = list(pool)
Xexp = {p: np.array(pool[p]["X"]) for p in pids}
yexp = {p: np.array(pool[p]["y"]) for p in pids}
Pexp = {p: np.array(pool[p]["P"]) for p in pids}
Nexp = {p: pool[p]["N"] for p in pids}

# non-high-CP standart OOF (RF)
oof = np.zeros(len(ys))
for tr, te in GroupKFold(5).split(Xs, ys, grp):
    c = RandomForestClassifier(n_estimators=400, min_samples_leaf=3, n_jobs=-1, random_state=0).fit(Xs[tr], ys[tr]); oof[te] = c.predict_proba(Xs[te])[:, 1]
nh_tp = nh_nk = nh_gt = 0
for g in np.unique(grp):
    if g in hicp: continue
    idx = np.where(grp == g)[0]; nh_gt += ngt_of[g]
    if len(idx): keep = idx[np.argsort(-oof[idx])[:ngt_of[g]]]; nh_tp += int((ys[keep] == 1).sum()); nh_nk += len(keep)

def f1(tp, nk, gt): p = tp/max(nk, 1); r = tp/max(gt, 1); return p, r, 2*p*r/max(p+r, 1e-9)

for lam in [0.0, 1.5]:
    hc_tp = hc_nk = hc_gt = 0
    for p in pids:
        # gate: standart-non-highCP + digger high-CP genisletilmis (p HARIC)
        Xtr = [Xstd] + [Xexp[q] for q in pids if q != p]
        ytr = [ystd] + [yexp[q] for q in pids if q != p]
        clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=3, n_jobs=-1, random_state=0)
        clf.fit(np.vstack(Xtr), np.concatenate(ytr))
        ws = clf.predict_proba(Xexp[p])[:, 1]
        sc = ws + lam * consistency(Pexp[p], ws)
        N = Nexp[p]; keep = np.argsort(-sc)[:N]
        hc_tp += int((yexp[p][keep] == 1).sum()); hc_nk += len(keep); hc_gt += N
    hr = hc_tp/max(hc_gt, 1)
    o = f1(nh_tp+hc_tp, nh_nk+hc_nk, nh_gt+hc_gt)
    tag = "sadece expanded-gate" if lam == 0 else f"expanded-gate + lattice(lam{lam})"
    print(f"[{tag}] high-CP recall {hr:.3f} F1 {f1(hc_tp,hc_nk,hc_gt)[2]:.3f} | OVERALL metadata P{o[0]:.3f} R{o[1]:.3f} F1 {o[2]:.3f} | 0.80 {'GECILDI' if o[2]>=0.80 else 'kaldi '+str(round(0.80-o[2],3))}")
