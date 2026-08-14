# -*- coding: utf-8 -*-
"""FAZ 2 -- FINAL MODEL SECIMI (SADECE WORK setinde; kilitli holdout'a DOKUNULMAZ).
Kaydedilmis OOF skorlarindan (results/oof_scores.npz) tum kombinasyonlari ayni protokolde olcer:
nested-CV esik (family-out), havuzlanmis TEK agregat F1. Kazanan konfigurasyon + esik KILITLENIR,
sonra P7 kilitli holdout'u TEK KEZ acar."""
import json
import numpy as np

d = np.load("results/oof_scores.npz", allow_pickle=True)
s_rf, s_mlp, s_set = d["rf"], d["mlp"], d["setnet"]
WORK, Y, G, MF, FG = d["work"], d["y"], d["groups"], d["mfg"], d["fam"]
r = np.load("results/rich_feats.npz", allow_pickle=True)
ngt_of = dict(zip(r["grp_ids"].tolist(), r["ngt"].tolist()))
THRS = np.round(np.arange(0.05, 0.96, 0.01), 3)


def gt_of(m): return int(sum(ngt_of[int(g)] for g in np.unique(G[m])))
def f1(TP, NK, GT):
    p = TP / max(NK, 1); rr = TP / max(GT, 1); return 2 * p * rr / max(p + rr, 1e-9)


def nested(sc, m, K=5):
    gp = np.unique(FG[np.where(m)[0]]); rs = np.random.RandomState(0)
    gp = gp[rs.permutation(len(gp))]; TP = NK = GT = 0; chosen = []
    for f in np.array_split(gp, K):
        tg = set(f.tolist())
        trm = m & np.array([g not in tg for g in FG]); tem = m & np.array([g in tg for g in FG])
        if not trm.any() or not tem.any(): continue
        bt, bf = 0.35, -1
        for t in THRS:
            k = trm & (sc >= t)
            ff = f1(int(Y[k].sum()), int(k.sum()), gt_of(trm))
            if ff > bf: bf, bt = ff, t
        chosen.append(bt)
        k = tem & (sc >= bt); TP += int(Y[k].sum()); NK += int(k.sum()); GT += gt_of(tem)
    return f1(TP, NK, GT), float(np.median(chosen))


def rank01(s):
    o = np.zeros(len(s)); i = np.where(WORK)[0]
    o[i] = np.argsort(np.argsort(s[i])) / max(len(i) - 1, 1); return o


# DEPLOY EDILEBILIRLIK KURALI: robot TEK PARCA isler -> skor parca-bagimsiz olmali.
# rank-normalizasyon TUM KORPUS uzerinde hesaplanir (esik aslinda bir yuzdelik) -> DEPLOY EDILEMEZ.
# Bu yuzden SADECE olasilik uzayinda birlesen konfigurasyonlar aday.
cands = {
    "RF(rich)": s_rf, "MLP(rich+emb)": s_mlp, "SetNet(3seed)": s_set,
    "prob: RF+MLP": (s_rf + s_mlp) / 2, "prob: RF+SetNet": (s_rf + s_set) / 2,
    "prob: MLP+SetNet": (s_mlp + s_set) / 2, "prob: RF+MLP+SetNet": (s_rf + s_mlp + s_set) / 3,
    "prob: SetNet*2+RF+MLP": (2 * s_set + s_rf + s_mlp) / 4,
}
R_, M_, S_ = rank01(s_rf), rank01(s_mlp), rank01(s_set)
ref_rank = {"(referans, DEPLOY EDILEMEZ) rank: RF+SetNet": (R_ + S_) / 2}
W_, P_ = WORK & (MF == 1), WORK & (MF == 0)
print(f"{'konfigurasyon':24s} {'ALL':>8s} {'WEI':>8s} {'PXC':>8s}  {'esik':>6s}")
best = None
for nm, sc in cands.items():
    a, th = nested(sc, WORK); w, _ = nested(sc, W_); p, _ = nested(sc, P_)
    print(f"{nm:24s} {a:8.4f} {w:8.4f} {p:8.4f}  {th:6.2f}")
    if best is None or a > best[1]: best = (nm, a, w, p, th)
for nm, sc in ref_rank.items():
    a, th = nested(sc, WORK)
    print(f"{nm:44s} {a:8.4f}  (sadece kiyas icin; parca-bagimsiz DEGIL)")
print(f"\n>>> KAZANAN (deploy edilebilir): {best[0]}  ALL {best[1]:.4f} "
      f"(WEI {best[2]:.4f} / PXC {best[3]:.4f}), esik {best[4]:.2f}")
json.dump({"winner": best[0], "WORK_ALL": best[1], "WORK_WEI": best[2], "WORK_PXC": best[3],
           "locked_threshold": best[4],
           "deployability_rule": "rank-normalizasyonlu ensemble'lar ELENDI: rank tum korpustan hesaplanir, "
                                 "esik yuzdelige donusur, robot tek parca islerken TANIMSIZ. Sadece "
                                 "olasilik-uzayi birlesimleri aday alindi.",
           "note": "SADECE WORK (541 parca) uzerinde secildi; kilitli holdout'a dokunulmadi."},
          open("results/final_config.json", "w"), indent=1)
print("-> results/final_config.json (KILITLENDI; P7 bunu tek kez holdout'ta calistiracak)")
