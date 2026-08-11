# -*- coding: utf-8 -*-
"""HIGH-CP SECICI (deploy modulu, yan-etkisiz). high-CP(>=11) parcalarda genisletilmis 6k+9k aday havuzunu
lattice-augmented RF ile top-N sirala. Egitim etiketi = uretici-CP hakemi (certain source: Desktop\\JSON).
Bu, wire_gate'in USTUNE gelen 2. kademe secici: sadece high-CP parcalarda devreye girer.

feats:  13 wire_gate (feats_for) + 8 lattice (lattice_feats) + 4 ranking (rank_feats) = 25.
train:  results/highcp_pool.json (24 parca, 6k+9k, sizintisiz base-gate ws) -> full-train RF -> pkl.
apply:  cikarimda cps + probs + point'lerden 25 feat kur -> skor -> top-N.
OOF/family-out skorlari AYRI (leave_family_out.py / receipt); bu artefact FULL-TRAIN deploy modeli."""
import os, json, numpy as np

MODEL_PATH = "results/highcp_selector.pkl"
LATTICE_NAMES = ["lat_cons", "lat_gfu", "lat_gfv", "lat_dens", "lat_nrow", "lat_ncol", "lat_upos", "lat_vpos"]
RANK_NAMES = ["rk_prank", "rk_dpen", "rk_cellbest", "rk_nratio"]  # #12 ranking-mode: leave-FAMILY-out 3/3 win (+0.007)


def lattice_feats(P, ws):
    """her aday icin lattice sinyalleri: consistency, grid-fit(u,v), yogunluk, satir/kolon, u/v-pozisyon.
    Frame-invariant (SVD ile yuzey eksenleri + goreli pozisyon) -> STEP ya da JSON frame ayni sonuc."""
    P = np.asarray(P, float); ws = np.asarray(ws, float)
    Q = P - P.mean(0); _, _, Vt = np.linalg.svd(Q, full_matrices=False); u, v = Vt[0], Vt[1]
    pu = P @ u; pv = P @ v
    span = max(pu.max()-pu.min(), pv.max()-pv.min(), 1.0); tol = 0.04*span
    n = len(P); cons = np.zeros(n); nrow = np.zeros(n); ncol = np.zeros(n)
    for i in range(n):
        sr = np.abs(pv-pv[i]) < tol; sc = np.abs(pu-pu[i]) < tol
        al = (sr | sc); al[i] = False
        cons[i] = float(ws[al].sum()); nrow[i] = float(sr.sum()-1); ncol[i] = float(sc.sum()-1)
    cons = cons/max(cons.max(), 1e-9)
    hi = ws >= np.percentile(ws, 60)
    def gridfit(pp):
        g = np.zeros(n)
        if hi.sum() >= 3:
            pos = np.sort(pp[hi]); gaps = np.diff(pos); gaps = gaps[gaps > 0.02*span]
            if len(gaps):
                pitch = np.median(gaps); r = np.abs((pp - pos[0]) % pitch); r = np.minimum(r, pitch-r)
                g = 1.0 - r/(pitch/2 + 1e-9)
        return g
    gfu = gridfit(pu); gfv = gridfit(pv)
    dens = np.zeros(n)
    for i in range(n):
        near = (np.abs(pu-pu[i]) < 0.08*span) & (np.abs(pv-pv[i]) < 0.08*span); near[i] = False
        dens[i] = float(ws[near].sum())
    dens = dens/max(dens.max(), 1e-9)
    return np.stack([cons, gfu, gfv, dens, nrow/max(nrow.max(), 1), ncol/max(ncol.max(), 1),
                     (pu-pu.min())/span, (pv-pv.min())/span], 1)


def rank_feats(P, ws, N):
    """4 ranking-mode sinyali (kullanici #12): within-part ws-yuzdelik, duplicate-penalty (daha yuksek-ws
    yakin komsu sayisi), grid-cell-best (lattice hucresinde en yuksek-ws mi), N-orani. leave-FAMILY-out 3/3
    win (part-out'ta hafif zarar; yeni-aile genellemesi = deployment metrigi kazaniyor)."""
    P = np.asarray(P, float); ws = np.asarray(ws, float); n = len(ws)
    prank = np.argsort(np.argsort(ws)) / max(n-1, 1)
    dpen = np.zeros(n)
    for i in range(n):
        dd = np.linalg.norm(P - P[i], axis=1); dpen[i] = float(((dd < 8.0) & (ws > ws[i])).sum())
    dpen = dpen / max(dpen.max(), 1.0)
    Q = P - P.mean(0); _, _, Vt = np.linalg.svd(Q, full_matrices=False); u, v = Vt[0], Vt[1]
    pu = P @ u; pv = P @ v; span = max(pu.max()-pu.min(), pv.max()-pv.min(), 1.0); cell = 0.10*span
    cu = np.round(pu/cell); cv = np.round(pv/cell); best = np.zeros(n)
    for i in range(n):
        same = (cu == cu[i]) & (cv == cv[i]); best[i] = float(ws[i] >= ws[same].max() - 1e-9)
    nratio = np.full(n, N/max(n, 1))
    return np.stack([prank, dpen, best, nratio], 1)


def augment(X13, P, ws, N):
    """13 wire_gate feat + 8 lattice + 4 ranking (#12) -> 25."""
    return np.hstack([np.asarray(X13, float), lattice_feats(P, ws), rank_feats(P, ws, N)])


def train_and_save(pool="results/highcp_pool.json", out=MODEL_PATH):
    """full-train RF (deploy). OOF/family-out DEGERLENDIRME leave_family_out.py'de -- bu SADECE deploy artefact."""
    from sklearn.ensemble import RandomForestClassifier
    from wire_gate import FEAT_NAMES
    import pickle
    pd = json.load(open(pool)); pids = list(pd)
    X = np.vstack([augment(pd[p]["X"], pd[p]["P"], pd[p]["ws"], pd[p]["N"]) for p in pids])
    y = np.concatenate([np.array(pd[p]["y"]) for p in pids])
    clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=3, n_jobs=-1, random_state=0).fit(X, y)
    names = FEAT_NAMES + LATTICE_NAMES + RANK_NAMES
    pickle.dump({"clf": clf, "feat_names": names, "n_parts": len(pids),
                 "trained_on": pids}, open(out, "wb"))
    print(f"high-CP selector RF ({len(y)} aday, {len(pids)} parca, {len(names)} feat) -> {out}")
    return out


_CACHE = {}
def _load(path=MODEL_PATH):
    if path not in _CACHE:
        import pickle
        _CACHE[path] = pickle.load(open(path, "rb")) if os.path.exists(path) else None
    return _CACHE[path]


def apply(cps, V, probs, CE, CT, N, base_ws=None, model_path=MODEL_PATH):
    """high-CP top-N secici. cps=union aday listesi (6k+9k), V/probs=6k mesh+avg olasilik, N=uretici CP sayisi.
    base_ws verilmezse wire_gate base-gate ile hesaplanir (lattice consistency icin gerekli). top-N cps dondurur."""
    import wire_gate
    m = _load(model_path)
    if m is None or len(cps) <= N:
        return sorted(cps, key=lambda c: -c.get("wire_score", c.get("confidence", 0.0)))[:max(int(N), 0)] if len(cps) > N else cps
    X13 = wire_gate.feats_for(V, None, np.asarray(probs, float), cps, CE, CT)
    if base_ws is None:
        wm = wire_gate._load()
        base_ws = wm["clf"].predict_proba(X13)[:, 1] if wm is not None else np.ones(len(cps))
    P = np.array([np.asarray(c["point"], float) for c in cps])
    Xa = augment(X13, P, base_ws, int(N))
    s = m["clf"].predict_proba(Xa)[:, 1]
    for c, sc in zip(cps, s):
        c["wire_score"] = float(sc); c["_highcp_selected"] = True
    return sorted(cps, key=lambda c: -c["wire_score"])[:max(int(N), 0)]


if __name__ == "__main__":
    train_and_save()
