# -*- coding: utf-8 -*-
"""GOREV 2a offline: tol_audit_pool'da farkli MATCH toleranslariyla base F1 (ws>=0.35). Amac: mevcut
tolerans FIZIKSELDEN sikiysa, dogru toleransla ALL 0.756 kalkar mi (metrik duzeltmesi, model degil).
Kill: +0.01 altiysa kapat. Modlar: current / opening_radius / size_scaled / wire_fit."""
import json, numpy as np

pool = json.load(open("results/tol_audit_pool.json"))
THR = 0.35   # base wire-gate threshold
ALONG = 40.0


def tol_for(mode, diag, size_a):
    if mode == "current": return max(3.0, 0.06*diag)
    if mode == "opening_radius": return max(size_a/2.0, 1.0)          # aday acikligin yaricapi
    if mode == "size_scaled": return max(size_a, 2.0)                 # tam cap
    if mode == "wire_fit": return max(size_a/2.0, 2.5)                # yaricap ya da min tel-yaricapi 2.5mm
    return max(3.0, 0.06*diag)


def f1_mode(mode):
    res = {"ALL": [0, 0, 0], "WEI": [0, 0, 0], "PXC": [0, 0, 0]}   # tp, pred, gt
    for pid, d in pool.items():
        ws = np.array(d["ws"]); size = np.array(d["size"]); N = d["N"]; diag = d["diag"]; mfg = d["mfg"]
        perp = np.array(d["perp"]); along = np.array(d["along"])    # (n_cand x n_gt)
        if perp.ndim != 2 or perp.shape[1] == 0: continue
        pred = ws >= THR                                            # base tahmin
        # her aday icin tolerans (size'a bagli olabilir)
        tol = np.array([tol_for(mode, diag, size[a]) for a in range(len(ws))])
        ok = (perp <= tol[:, None]) & (np.abs(along) <= ALONG)
        # greedy 1-1 eslesme (sadece pred adaylar, perp'e gore)
        order = sorted((perp[a, b], a, b) for a in range(len(ws)) for b in range(N) if pred[a] and ok[a, b])
        ua, ub = set(), set(); tp = 0
        for dd, a, b in order:
            if a in ua or b in ub: continue
            ua.add(a); ub.add(b); tp += 1
        npred = int(pred.sum())
        for k in ("ALL", mfg):
            res[k][0] += tp; res[k][1] += npred; res[k][2] += N
    out = {}
    for k, (tp, pr, gt) in res.items():
        p = tp/max(pr, 1); r = tp/max(gt, 1); out[k] = 2*p*r/max(p+r, 1e-9)
    return out


print(f"{len(pool)} parca | base F1 (ws>=0.35) farkli MATCH toleransi")
print(f"{'mode':16} {'ALL':>7} {'WEI':>7} {'PXC':>7}")
base = None
for mode in ["current", "opening_radius", "size_scaled", "wire_fit"]:
    r = f1_mode(mode)
    if base is None: base = r["ALL"]
    d = r["ALL"]-base
    mark = " <-- +{:.3f}".format(d) if d > 0.01 else (" (kill <0.01)" if mode != "current" and d <= 0.01 else "")
    print(f"{mode:16} {r['ALL']:>7.4f} {r['WEI']:>7.4f} {r['PXC']:>7.4f}{mark}")
print("\nKARAR: fiziksel tolerans ALL'i +0.01+ kaldiriyorsa metrik-duzeltmesi (durust), degilse mevcut kalir")
