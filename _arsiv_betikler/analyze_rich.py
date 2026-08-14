# -*- coding: utf-8 -*-
"""ZENGIN TEMSIL ANALIZI (build_rich_feats.py ciktisi uzerinde, CPU).
Sorular: (1) hangi feature BLOGU ne katiyor, (2) AILE-out'ta ayakta mi, (3) urun merdivenine
(base 0.750 / metadata 0.775) cevrildiginde ne oluyor, (4) WEI/PXC ayri.
Bloklar: A konum(9) | B cok-yaricap sinif-profili(24) | C taper(5) | D egrilik(3).
Iki metrik: (i) esik-modu F1 (deployed base urun), (ii) top-N F1 (metadata-assisted mod)."""
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

d = np.load("results/rich_feats.npz", allow_pickle=True)
X13, XR, Y, G, MF, POS = d["X13"], d["XR"], d["y"], d["groups"], d["mfg"], d["pos"]
ngt_of = dict(zip(d["grp_ids"].tolist(), d["ngt"].tolist()))
BLK = {"A_konum": slice(0, 9), "B_yaricap": slice(9, 33), "C_taper": slice(33, 38), "D_egrilik": slice(38, 41)}

# aile anahtari (etiketsiz): aday-bulutu bbox + log2(aday sayisi)
FAM = {}
for g in np.unique(G):
    m = G == g; P = POS[m]
    FAM[g] = f"g:{np.round(P.max(0)-P.min(0),0).tolist()}|c{int(np.log2(max(m.sum(),1)))}"
uf = {f: i for i, f in enumerate(sorted(set(FAM.values())))}
FG = np.array([uf[FAM[g]] for g in G])
THRS = np.round(np.arange(0.0, 0.71, 0.02), 3)


def oof(Xa, groups):
    o = np.zeros(len(Y))
    for tr, te in GroupKFold(5).split(Xa, Y, groups):
        c = RandomForestClassifier(400, min_samples_leaf=3, n_jobs=-1, random_state=0).fit(Xa[tr], Y[tr])
        o[te] = c.predict_proba(Xa[te])[:, 1]
    return o


def f1_thr(sc, mask, thr):
    k = mask & (sc >= thr); tp = int(Y[k].sum())
    gt = sum(ngt_of[g] for g in np.unique(G[mask]))
    p = tp / max(int(k.sum()), 1); r = tp / max(gt, 1); return 2 * p * r / max(p + r, 1e-9)


def best_thr_f1(sc, mask):
    return max(f1_thr(sc, mask, t) for t in THRS)


def topn_f1(sc, mask):
    TP = NK = GT = 0
    for g in np.unique(G[mask]):
        m = np.where(mask & (G == g))[0]; n = ngt_of[g]
        idx = m[np.argsort(-sc[m])[:n]]
        TP += int(Y[idx].sum()); NK += len(idx); GT += n
    p = TP / max(NK, 1); r = TP / max(GT, 1); return 2 * p * r / max(p + r, 1e-9)


masks = {"ALL": np.ones(len(Y), bool), "WEI": MF == 1, "PXC": MF == 0}
print(f"{len(Y)} aday | {int(Y.sum())} TP | {len(ngt_of)} parca | {len(uf)} aile | rich-dim {XR.shape[1]}")
print(f"  WEI {int((MF==1).sum())} aday / PXC {int((MF==0).sum())} aday\n")

sets = [("13 BAZ", X13)]
for nm, sl in BLK.items():
    sets.append((f"13+{nm}", np.hstack([X13, XR[:, sl]])))
sets.append(("13+HEPSI(41)", np.hstack([X13, XR])))

for split_nm, groups in (("PARCA-out", G), ("AILE-out (KATI)", FG)):
    print(f"================ {split_nm} ================")
    print(f"{'feature seti':16s} {'AUC':>7s} {'ALL thr-F1':>11s} {'ALL topN':>9s} {'WEI topN':>9s} {'PXC topN':>9s}")
    base = None
    for nm, Xa in sets:
        o = oof(Xa, groups)
        row = (roc_auc_score(Y, o), best_thr_f1(o, masks["ALL"]), topn_f1(o, masks["ALL"]),
               topn_f1(o, masks["WEI"]), topn_f1(o, masks["PXC"]))
        if base is None: base = row
        dz = "" if nm == "13 BAZ" else f"  ({row[2]-base[2]:+.4f} topN)"
        print(f"{nm:16s} {row[0]:7.4f} {row[1]:11.4f} {row[2]:9.4f} {row[3]:9.4f} {row[4]:9.4f}{dz}")
    print()
print("KIYAS (canonical receipt): base urun ALL 0.750 (thr) | metadata-assisted ALL 0.775 (top-N)")
print("KARAR: 13+HEPSI, AILE-out'ta bile bazi materyal geciyorsa -> gate'i zengin feature'la yeniden egit (deploy adayi).")
