# -*- coding: utf-8 -*-
"""P1-x: TEMSIL-ZENGINLESTIRME testi (GPU YOK, mevcut havuzdan).
13 feature'in KOR NOKTALARI: (a) parcada NEREDE oldugu yok -- vida USTTE, tel ONDE (kullanicinin
'vidadan ok cikiyor' bulgusu tam bu), (b) parca-ici GORECELIK yok (benim capim digerlerine gore buyuk mu),
(c) tek olcek.
Bu testte, sadece havuzdaki P (pozisyon) + X (13 feat) ile turetilebilen zengin feature'lari deneriz:
  A) normalize konum (parca bbox icinde x,y,z)      B) bbox yuzeylerine uzaklik (6)
  C) her feature'in parca-ici RANK'i (13)           D) parca medyanina ORAN (13)
KARAR: AUC/top-N F1 materyal olarak artiyorsa -> temsil (feature) darbogazi GERCEK, GPU'lu zengin
cikarim (cok-yaricap olasilik histogrami + taper profili) HAKLI. Artmiyorsa temsil tavani teyit."""
import json, sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

POOL = sys.argv[1] if len(sys.argv) > 1 else "results/wei_aggr_pool.json"
d = json.load(open(POOL))
Xs, Ys, Gs, Ps, Ns = [], [], [], [], {}
for gi, (pid, v) in enumerate(d.items()):
    y = np.array(v["y"], int)
    if len(y) < 4: continue
    Xs.append(np.array(v["X"], float)); Ys.append(y); Gs.append(np.full(len(y), gi))
    Ps.append(np.array(v["P"], float)); Ns[gi] = v["N"]
X = np.vstack(Xs); Y = np.concatenate(Ys); G = np.concatenate(Gs); P = np.vstack(Ps)

A = np.zeros((len(Y), 3)); B = np.zeros((len(Y), 6))
C = np.zeros_like(X); Dr = np.zeros_like(X)
for gi in np.unique(G):
    m = np.where(G == gi)[0]
    pp = P[m]; lo, hi = pp.min(0), pp.max(0); rng_ = np.maximum(hi - lo, 1e-6)
    A[m] = (pp - lo) / rng_                                   # normalize konum
    B[m] = np.column_stack([pp - lo, hi - pp])                # bbox yuzeylerine uzaklik
    xx = X[m]
    C[m] = np.argsort(np.argsort(xx, axis=0), axis=0) / max(len(m) - 1, 1)   # parca-ici rank
    med = np.median(xx, axis=0); Dr[m] = xx / (np.abs(med) + 1e-6)           # medyana oran


def oof(Xa):
    o = np.zeros(len(Y))
    for tr, te in GroupKFold(5).split(Xa, Y, G):
        c = RandomForestClassifier(400, min_samples_leaf=3, n_jobs=-1, random_state=0).fit(Xa[tr], Y[tr])
        o[te] = c.predict_proba(Xa[te])[:, 1]
    return o


def topn(sc):
    TP = NK = GT = 0
    for gi in np.unique(G):
        m = np.where(G == gi)[0]; n = Ns[gi]
        idx = m[np.argsort(-sc[m])[:n]]
        TP += int(Y[idx].sum()); NK += len(idx); GT += n
    p = TP / max(NK, 1); r = TP / max(GT, 1); return 2 * p * r / max(p + r, 1e-9)


print(f"{len(Y)} aday / {int(Y.sum())} TP / {len(Ns)} parca | POOL {POOL}")
o0 = oof(X); a0, f0 = roc_auc_score(Y, o0), topn(o0)
print(f"\n  BAZ 13 feature            AUC {a0:.4f}  top-N F1 {f0:.4f}")
sets = [("+ konum(A)", np.hstack([X, A])),
        ("+ konum+bbox(A,B)", np.hstack([X, A, B])),
        ("+ parca-ici rank(C)", np.hstack([X, C])),
        ("+ medyan-oran(D)", np.hstack([X, Dr])),
        ("+ HEPSI (A,B,C,D)", np.hstack([X, A, B, C, Dr]))]
best = (a0, f0, "baz")
for nm, Xa in sets:
    o = oof(Xa); a, f = roc_auc_score(Y, o), topn(o)
    print(f"  {nm:26s} AUC {a:.4f} ({a-a0:+.4f})  top-N F1 {f:.4f} ({f-f0:+.4f})")
    if f > best[1]: best = (a, f, nm)
print(f"\n  EN IYI: {best[2]}  AUC {best[0]:.4f}  top-N F1 {best[1]:.4f}  (baz F1 {f0:.4f}, kazanc {best[1]-f0:+.4f})")
print("  -> top-N F1 kazanci >+0.02 ise TEMSIL DARBOGAZI GERCEK: GPU'lu zengin cikarim HAKLI.")
print("     <+0.02 ise ucuz zenginlestirme yetmiyor; karar GPU-zengin cikarim (cok-yaricap/taper) ya da temsil-tavani.")
