# -*- coding: utf-8 -*-
"""Gate v7 REFIT: g10 ile yeniden turetilmis korpus uzerinde.

`parca_ici_dagit.py` ile AYNI recete (parca_ici PARCA PARCA + RF 400/leaf3), ama
girdi/cikti PARAMETRIK -- dagitilan gate'in uzerine YAZMAZ, A/B yapilabilsin.

NEDEN ZORUNLU (gate-refit-minv4 dersi): "iki gate AYNI dagilimda egitilmeli".
g10 farkli adaylar uretiyor; eski gate onlari gormedi.
"""
import argparse, pickle, numpy as np
from sklearn.ensemble import RandomForestClassifier
import wire_gate

ap = argparse.ArgumentParser()
ap.add_argument("--korpus", default="results/zengin_parite_v4_g10.npz")
ap.add_argument("--cikti", default="results/wire_gate_v7.pkl")
ap.add_argument("--donusum", default="zskor")
a = ap.parse_args()

d = np.load(a.korpus, allow_pickle=True)
X = np.hstack([d["X22"], d["XR"]]).astype(float)
y = np.asarray(d["y"]).astype(int)
pids = np.array([str(p) for p in d["pids"]])
M = np.zeros((len(X), X.shape[1] * 2))
for u in np.unique(pids):
    i = np.where(pids == u)[0]
    M[i] = wire_gate.parca_ici(X[i], a.donusum)
print(f"egitim: {M.shape[0]} aday x {M.shape[1]} sutun | "
      f"{len(np.unique(pids))} parca | pozitif {y.mean():.4f}", flush=True)
clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=3, n_jobs=-1,
                             random_state=0).fit(M, y)
with open(a.cikti, "wb") as f:
    pickle.dump({"clf": clf,
                 "feat_names": list(wire_gate.FEAT_NAMES) +
                               [n + "_z" for n in wire_gate.FEAT_NAMES],
                 "cols": None, "n_feat": M.shape[1], "donusum": a.donusum,
                 "topo_r": float(wire_gate.TOPO_R),
                 "korpus": a.korpus,
                 "note": "gate v7: g10 ile yeniden turetilmis korpus (44675 aday / "
                         "2596 parca). parca_ici_dagit.py ile ayni recete."}, f)
print(f"-> {a.cikti}")
