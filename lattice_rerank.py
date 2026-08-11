# -*- coding: utf-8 -*-
"""Op 4 (offline): high-CP havuzda LATTICE/row-consistency re-ranking. Darbogaz secim: candidate recall
0.800 ama top-N recall 0.570 -- gercek teller havuzda ama gate top-N alti siraliyor. High-CP terminaller
periyodik grid; on-grid adaylari yukari siralamak top-N recall'i artirabilir.
Sinyal: her aday icin ROW/COLUMN-consistency = yuzey-eksenlerinde (u,v) hizali yuksek-skor komsu sayisi.
Re-rank = ws + lambda * consistency. lambda taranir, top-N F1 karsilastirilir."""
import json, numpy as np

pool = json.load(open("results/highcp_pool.json"))


def face_axes(P):
    Q = P - P.mean(0)
    _, _, Vt = np.linalg.svd(Q, full_matrices=False)
    return Vt[0], Vt[1]   # iki en buyuk varyans yonu = yuzey eksenleri


def consistency(P, ws, tol_frac=0.04):
    """her aday icin: u ya da v ekseninde hizali (ayni satir/kolon) yuksek-skor komsu agirligi."""
    u, v = face_axes(P)
    pu = P @ u; pv = P @ v
    span = max(pu.max()-pu.min(), pv.max()-pv.min(), 1.0); tol = tol_frac * span
    n = len(P); c = np.zeros(n)
    for i in range(n):
        same_row = np.abs(pv - pv[i]) < tol   # ayni v -> satir
        same_col = np.abs(pu - pu[i]) < tol   # ayni u -> kolon
        aligned = (same_row | same_col); aligned[i] = False
        c[i] = float((ws[aligned]).sum())     # hizali komsularin ws toplami
    return c / max(c.max(), 1e-9)


def f1_topN(rank_scores, mode="all"):
    tp = nk = gt = 0
    for pid, d in pool.items():
        ws = np.array(d["ws"]); y = np.array(d["y"]); N = d["N"]; gt += N
        sc = rank_scores(pid, d)
        keep = np.argsort(-sc)[:N]; tp += int((y[keep] == 1).sum()); nk += len(keep)
    p = tp/max(nk, 1); r = tp/max(gt, 1); return p, r, 2*p*r/max(p+r, 1e-9)


print(f"high-CP pool: {len(pool)} parca, {sum(d['N'] for d in pool.values())} uretici CP")
# baseline: ws
ps = f1_topN(lambda pid, d: np.array(d["ws"]))
print(f"\nbaseline (ws top-N):        P{ps[0]:.3f} R{ps[1]:.3f} F1 {ps[2]:.3f}")
# lattice: ws + lambda*consistency
print("lambda | P     R     F1")
best = (ps[2], 0.0)
cons_cache = {pid: consistency(np.array(d["P"]), np.array(d["ws"])) for pid, d in pool.items()}
for lam in [0.1, 0.2, 0.3, 0.5, 0.8, 1.2]:
    def rk(pid, d, lam=lam): return np.array(d["ws"]) + lam * cons_cache[pid]
    p, r, f = f1_topN(rk)
    mark = " <--" if f > best[0] else ""
    if f > best[0]: best = (f, lam)
    print(f"{lam:6.1f} | {p:.3f} {r:.3f} {f:.3f}{mark}")
print(f"\nEN IYI: lambda {best[1]} -> F1 {best[0]:.3f} (baseline ws {ps[2]:.3f})")
print(f"  -> lattice re-rank high-CP F1'i {'YUKSELTTI' if best[0]>ps[2] else 'YUKSELTMEDI'} (+{best[0]-ps[2]:.3f})")
