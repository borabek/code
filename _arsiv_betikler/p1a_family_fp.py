# -*- coding: utf-8 -*-
"""P1-a: surviving-FP'ler AILE-lerde yogunlasiyor mu? (CPU, mevcut veri)
Mekanizma: ayni govde-platformu kardesleri AYNI aciklik tiplerini tasir; bir acikligin kimligi ailede
SABITTIR. Eger FP kutlesi az sayida aile-tipinde toplaniyorsa, aile basina TEK karar ('bu aciklik-tipi
wire mi?') buyuk kutleyi temizler -> P1-c'nin hedef listesi = bu cikti.
GO: FP kutlesinin >=%60'i <=40 aile-tipinde. Ayrica: aile-ici FP TUTARLILIGI (ayni ailede hep ayni mi)."""
import json, sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

POOL = sys.argv[1] if len(sys.argv) > 1 else "results/wei_aggr_pool.json"
d = json.load(open(POOL))
rows = []
Xs, Ys, Gs = [], [], []
meta = []
gi = 0
for pid, v in d.items():
    y = np.array(v["y"], int); X = np.array(v["X"], float); P = np.array(v["P"], float)
    if len(y) < 4: continue
    Xs.append(X); Ys.append(y); Gs.append(np.full(len(y), gi)); gi += 1
    dims = np.round(P.max(0) - P.min(0), 0)
    meta.append((pid, f"g:{dims.tolist()}|c{int(np.log2(max(len(y),1)))}", len(y)))
X = np.vstack(Xs); Y = np.concatenate(Ys); G = np.concatenate(Gs)
fam = np.array([meta[g][1] for g in G])

# OOF gate -> surviving FP (esik 0.35)
oof = np.zeros(len(Y))
for tr, te in GroupKFold(5).split(X, Y, G):
    c = RandomForestClassifier(400, min_samples_leaf=3, n_jobs=-1, random_state=0).fit(X[tr], Y[tr])
    oof[te] = c.predict_proba(X[te])[:, 1]
surv = (Y == 0) & (oof >= 0.35)
print(f"{len(Y)} aday | surviving-FP (gate 0.35 gecen sahte): {int(surv.sum())}")

fams, cnts = np.unique(fam[surv], return_counts=True)
order = np.argsort(-cnts); fams, cnts = fams[order], cnts[order]
tot = cnts.sum(); cum = np.cumsum(cnts) / tot
print(f"\n=== FP kutlesinin AILE yogunlasmasi ({len(fams)} farkli aile) ===")
for k in (10, 20, 40, 60):
    if k <= len(cnts):
        print(f"  en buyuk {k:3d} aile -> FP kutlesinin %{100*cum[k-1]:.0f}'i")
n40 = cum[min(39, len(cum) - 1)]
print(f"\n  GO-kriteri (<=40 aile, >=%60 kutle): {'GECTI' if n40 >= 0.60 else 'KALDI'} (%{100*n40:.0f})")

# aile-ici tutarlilik: ayni ailede FP orani ne kadar sabit
print(f"\n=== AILE-ICI TUTARLILIK (ayni ailede FP davranisi sabit mi) ===")
cons = []
for f in np.unique(fam):
    m = fam == f
    if m.sum() < 6: continue
    parts_in = np.unique(G[m])
    if len(parts_in) < 2: continue
    rates = [float(surv[m & (G == p)].mean()) for p in parts_in]
    cons.append(float(np.std(rates)))
if cons:
    print(f"  cok-parcali aile sayisi {len(cons)} | aile-ici FP-orani std ort {np.mean(cons):.3f}")
    print("  (dusuk std = aile-ici tutarli = tek karar ailenin tamamini temizler)")
else:
    print("  cok-parcali aile YOK (bu havuzda aileler tek-parcali) -> aile kaldiraci bu veride ZAYIF")
print("\n-> Bu cikti P1-c'nin (FAZ B) hedef listesidir; simdilik SADECE olcum.")
