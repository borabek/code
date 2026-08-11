# -*- coding: utf-8 -*-
"""Wire/tool kapisi UCTAN-UCA dogrulama (results/wire_discr_data.npz'den, GPU yok).

CAPRAZ-URETICI (sizinti yok): ayirici PXC CP'lerinde egitilir, WEI CP'lerine uygulanir (ve tersi).
Kapi esigi supurulur: skor<esik olan CP atilir. Olcum:
  precision = tutulan-tel / tutulan-toplam   (yukselmeli = tool agizlari atiliyor)
  recall    = tutulan-tel / uretici-CP-sayisi (dusmemeli = gercek teller korunuyor)
Baz (kapi yok) ile karsilastir. Amac: precision belirgin artsin, recall ~sabit kalsin.
"""
import json, numpy as np
from sklearn.ensemble import GradientBoostingClassifier

d = np.load("results/wire_discr_data.npz", allow_pickle=True)
X, y, mfg, groups = d["X"], d["y"], d["mfg"], d["groups"]
ngt, grp_ids = d["ngt"], d["grp_ids"]
feat = list(d["feat_names"])
ngt_of = dict(zip(grp_ids.tolist(), ngt.tolist()))


def train_predict(train_mask, test_mask):
    clf = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05)
    clf.fit(X[train_mask], y[train_mask])
    s = np.full(len(y), np.nan); s[test_mask] = clf.predict_proba(X[test_mask])[:, 1]
    return s


def report(name, test_mask, score):
    yy = y[test_mask]; sc = score[test_mask]; gg = groups[test_mask]
    n_gt = sum(ngt_of.get(int(g), 0) for g in np.unique(gg))
    base_tp = int(yy.sum()); base_n = len(yy)
    bp = base_tp / max(base_n, 1); br = base_tp / max(n_gt, 1)
    print(f"\n=== {name} (uretici CP {n_gt}) ===")
    print(f"  BAZ (kapi yok):        P {bp:.3f}  R {br:.3f}  ({base_tp} tel / {base_n} CP)")
    print(f"  {'esik':>5s} | {'P':>6s} {'R':>6s} {'atilan':>7s}")
    best = (bp, br, 0.0)
    for th in [0.3, 0.4, 0.5, 0.6, 0.7]:
        keep = sc >= th
        tp = int((yy[keep] == 1).sum()); nk = int(keep.sum())
        p = tp / max(nk, 1); r = tp / max(n_gt, 1); drop = base_n - nk
        star = ""
        # iyi calisma noktasi: recall'i >=0.95*baz tutup precision'i maksimize et
        if r >= 0.95 * br and p > best[0]:
            best = (p, r, th); star = " <--"
        print(f"  {th:5.2f} | {p:6.3f} {r:6.3f} {drop:7d}{star}")
    print(f"  --> onerilen esik {best[2]:.2f}: P {bp:.3f}->{best[0]:.3f}  R {br:.3f}->{best[1]:.3f}")


wei = mfg == 1; pxc = mfg == 0
print(f"veri: {len(y)} CP ({int(wei.sum())} WEI / {int(pxc.sum())} PXC), {int(y.sum())} tel")
# PXC'de egit -> WEI'de test
report("WEI (ayirici PXC'de egitildi)", wei, train_predict(pxc, wei))
# WEI'de egit -> PXC'de test
report("PXC (ayirici WEI'de egitildi)", pxc, train_predict(wei, pxc))
