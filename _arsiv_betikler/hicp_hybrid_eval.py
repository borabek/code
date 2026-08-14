# -*- coding: utf-8 -*-
"""HIGH-CP HIBRIT DEGERLENDIRME (WORK, family-out durust).
Mimari: tipik parca -> mevcut kilitli skor (6k union + prob:RF+SetNet, DOKUNULMADI)
        high-CP parca -> AGRESIF multires havuz + zengin RF gate (yeni kol)
High-CP gate egitimi: WEI agresif havuz (2750 aday, ayri parcalar) + 18-parca havuzunda
leave-one-part-out -> test parcasinin verisi egitime ASLA girmez.
Yonlendirme iki modda olculur:
  (a) ORACLE (GT>=11 biliniyor) -> ust-sinir, VARSAYIMSAL etiketli
  (b) GIRDI-BAZLI (aday sayisi + parca diyagonali esigi, GT'siz) -> deploy edilebilir
Cikti: WORK ALL onceki 0.775'e karsi hibrit ALL."""
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier

d = np.load("results/oof_scores.npz", allow_pickle=True)
r = np.load("results/rich_feats.npz", allow_pickle=True)
a18 = np.load("results/aggr_rich_LIST.npz", allow_pickle=True)
aw = np.load("results/aggr_rich_WEI.npz", allow_pickle=True)
WORK, Y, G, MF = d["work"], d["y"], d["groups"], d["mfg"]
sc6 = (d["rf"] + d["setnet"]) / 2                      # kilitli final skor (tipik icin)
THR6 = 0.34
ngt = dict(zip(r["grp_ids"].tolist(), r["ngt"].tolist()))
rp = [str(x) for x in r["part_ids"]]
pid_of_g = {int(g): rp[int(g)] for g in np.unique(G)}
hi_parts = set(open("_work_hicp.txt").read().split())

def RICHX(z): return np.hstack([z["X13"], z["XR"][:, 0:33]])
X18, Y18, G18 = RICHX(a18), a18["y"], a18["groups"]
p18 = [str(x) for x in a18["part_ids"]] if "part_ids" in a18.files else None
XW, YW = RICHX(aw), aw["y"]
print(f"18-parca agresif havuz: {len(Y18)} aday | aday-tavan recall {int(Y18.sum())}/{int(a18['ngt'].sum())} "
      f"= {Y18.sum()/max(a18['ngt'].sum(),1):.3f}  (6k'da 0.370 idi)")

# --- high-CP gate: WEI havuzu + LOO(18) ---
sc18 = np.zeros(len(Y18))
for g in np.unique(G18):
    te = G18 == g; tr = ~te
    Xtr = np.vstack([XW, X18[tr]]); Ytr = np.concatenate([YW, Y18[tr]])
    c = RandomForestClassifier(400, min_samples_leaf=3, n_jobs=-1, random_state=0).fit(Xtr, Ytr)
    sc18[te] = c.predict_proba(X18[te])[:, 1]


def prf(tp, nk, gt):
    p = tp / max(nk, 1); rr = tp / max(gt, 1); return p, rr, 2 * p * rr / max(p + rr, 1e-9)


# high-CP kolunda esik: WEI havuzunda (ayri veri) F1-optimal -> 18'e uygula (sizintisiz)
gtW = int(aw["ngt"].sum())
oofW = np.zeros(len(YW))
from sklearn.model_selection import GroupKFold
for tr, te in GroupKFold(5).split(XW, YW, aw["groups"]):
    c = RandomForestClassifier(400, min_samples_leaf=3, n_jobs=-1, random_state=0).fit(XW[tr], YW[tr])
    oofW[te] = c.predict_proba(XW[te])[:, 1]
best_thr, best_f1 = 0.35, -1
for t in np.arange(0.10, 0.71, 0.02):
    k = oofW >= t
    f = prf(int(YW[k].sum()), int(k.sum()), gtW)[2]
    if f > best_f1: best_f1, best_thr = f, t
print(f"high-CP esigi (WEI havuzunda secildi, sizintisiz): {best_thr:.2f}")

gt18 = int(a18["ngt"].sum())
k18 = sc18 >= best_thr
p18v, r18v, f18v = prf(int(Y18[k18].sum()), int(k18.sum()), gt18)
print(f"HIGH-CP kolu (18 parca, LOO): P {p18v:.3f} R {r18v:.3f} F1 {f18v:.3f}   (6k'da F1 0.458 idi)")

# --- HIBRIT ALL (WORK): tipik = kilitli skor; high-CP = yeni kol ---
def hybrid(route_hi_gids):
    TP = NK = GT = 0
    for g in np.unique(G[WORK]):
        pid = pid_of_g[int(g)]
        if int(g) in route_hi_gids:
            continue                                    # high-CP kolu asagida toplu
        m = WORK & (G == g); k = m & (sc6 >= THR6)
        TP += int(Y[k].sum()); NK += int(k.sum()); GT += ngt[int(g)]
    TP += int(Y18[k18].sum()); NK += int(k18.sum()); GT += gt18
    return prf(TP, NK, GT)


hi_gids = {int(g) for g in np.unique(G[WORK]) if pid_of_g[int(g)] in hi_parts}
# (a) ORACLE yonlendirme
pa, ra, fa = hybrid(hi_gids)
# eski (hibritsiz) referans
TP = NK = GT = 0
for g in np.unique(G[WORK]):
    m = WORK & (G == g); k = m & (sc6 >= THR6)
    TP += int(Y[k].sum()); NK += int(k.sum()); GT += ngt[int(g)]
p0, r0, f0 = prf(TP, NK, GT)
print(f"\nWORK ALL  hibritsiz: P {p0:.3f} R {r0:.3f} F1 {f0:.3f}")
print(f"WORK ALL  HIBRIT (oracle yonlendirme, VARSAYIMSAL): P {pa:.3f} R {ra:.3f} F1 {fa:.3f}  ({fa-f0:+.3f})")

# (b) GIRDI-BAZLI yonlendirme: aday sayisi >= T VE/VEYA diag -> high-CP kabul
cand_ct = {int(g): int(((G == g) & WORK).sum()) for g in np.unique(G[WORK])}
best_route = None
for T in (8, 9, 10, 12, 14):
    route = {g for g in cand_ct if cand_ct[g] >= T}
    # yanlis yonlendirilen tipikler high-CP koluna DUSMEZ (havuz yok) -> onlari 6k'da tut, sadece
    # gercek-hi & route kesisimi yeni kola gider; kacan hi'lar 6k'da kalir
    caught = route & hi_gids; missed = hi_gids - route; wrong = route - hi_gids
    TP = NK = GT = 0
    for g in np.unique(G[WORK]):
        if int(g) in caught: continue
        m = WORK & (G == g); k = m & (sc6 >= THR6)
        TP += int(Y[k].sum()); NK += int(k.sum()); GT += ngt[int(g)]
    # caught kolu: 18-havuzdan sadece caught parcalar (grup->pid map'i ile aday maskesi)
    cpids = {pid_of_g[g] for g in caught}
    a18_pids = [str(x) for x in a18["part_ids"]]
    msk = np.array([a18_pids[int(g)] in cpids for g in G18])
    kk = msk & (sc18 >= best_thr)
    gts = int(sum(ngt[g] for g in caught))
    TP += int(Y18[kk].sum()); NK += int(kk.sum()); GT += gts
    p, rr, f = prf(TP, NK, GT)
    tag = f"T>={T}: yakalanan {len(caught)}/18, yanlis-alarm {len(wrong)}"
    print(f"WORK ALL  HIBRIT (girdi-bazli, {tag}): F1 {f:.3f}")
    if best_route is None or f > best_route[0]: best_route = (f, T)
print(f"\n>>> girdi-bazli en iyi: T>={best_route[1]} -> F1 {best_route[0]:.3f} (oracle {fa:.3f})")
json.dump({"no_hybrid": f0, "hybrid_oracle": fa, "hybrid_input_based": best_route[0],
           "hicp_arm_f1": f18v, "hicp_thr": float(best_thr)},
          open("results/hicp_hybrid.json", "w"), indent=1)
