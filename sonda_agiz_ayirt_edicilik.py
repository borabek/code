# -*- coding: utf-8 -*-
"""AGIZ TANIMLAYICILARI GERCEKTEN AYIRT EDIYOR MU? (egitim yok, saf AUC)

Dun soyledigim "bu bilgi elimde yok" ERKEN bir yargiydi: yalnizca ELIMDE HAZIR
olan 58 ozniteligi denemistim. `agiz_tanimlayici.py` o bilgiyi URETIYOR. Once
bir model kurmadan sunu olcuyoruz: her tanimlayici, GT'ye dusen B-rep onerisini
dusmeyenden ayirabiliyor mu?

Olcut AUC (0.5 = bilgi yok). Ayrica mevcut gate'in ayni adaylardaki AUC'u de
raporlanir -- yeni bilginin ESKISINDEN iyi olup olmadigi ancak boyle gorulur.

D7 marka-disi. Etiket: aday GT'ye TESPIT toleransinda mi.
"""
import json
import os
import pickle
import sys

import numpy as np
import trimesh

import makbuz_hash

os.environ.setdefault("BA_ALLOW_SEEN", "1")
os.environ["WG_FIZ_FEATS"] = "1"
os.environ["WG_TOPO"] = "1"
os.environ["WG_ZENGIN"] = "1"
sys.path.insert(0, ".")
import agiz_tanimlayici as AT   # noqa: E402
import brep_havuz               # noqa: E402
import kanonik_d7 as K          # noqa: E402

OB = "results/_p1_olasilik_d7"


def auc(skor, y):
    """Mann-Whitney AUC. Sabit sutun 0.5 doner (bilgi yok)."""
    y = np.asarray(y).astype(bool)
    if y.all() or not y.any():
        return float("nan")
    s = np.asarray(skor, float)
    if not np.isfinite(s).all() or s.std() < 1e-12:
        return 0.5
    r = np.argsort(np.argsort(s)) + 1.0
    n1 = int(y.sum()); n0 = len(y) - n1
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def main():
    cy = pickle.load(open("results/_d7_silindirler.pkl", "rb"))
    ac = pickle.load(open("results/_d7_acikliklar.pkl", "rb"))
    pids = json.load(open("results/d7_sinav_kumesi.json"))["pidler"]
    kay = K.yukle(pids)

    X, Y, PID = [], [], []
    n_atlanan = 0
    for i, (pid, r) in enumerate(sorted(kay.items())):
        G = np.asarray(r.get("G", []), float)
        f = f"{OB}/{pid}.npz"
        if not len(G) or not os.path.exists(f):
            n_atlanan += 1
            continue
        P, D, met = brep_havuz.brep_adaylari(cy.get(pid), ac.get(pid), meta=True)
        if len(P) < 2:
            continue
        z = np.load(f)
        mesh = trimesh.Trimesh(np.asarray(z["V"], float),
                               np.asarray(z["F"], np.int64), process=False)
        T = AT.tanimla(P, D, met, mesh, float(r["diag"]))
        tol = max(3.0, 0.06 * float(r["diag"]))
        y = (np.linalg.norm(P[:, None] - G[None], axis=-1).min(1) <= tol)
        X.append(T); Y.append(y); PID += [pid] * len(P)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(kay)} parca", flush=True)
    X = np.vstack(X); Y = np.concatenate(Y)
    print(f"\n{len(X)} B-rep onerisi | GT'ye dusen {Y.mean():.4f} | "
          f"atlanan parca {n_atlanan}\n", flush=True)

    print(f"{'tanimlayici':<16} {'AUC':>7}  (0.5 = bilgi yok)")
    sonuc = {}
    for j, ad in enumerate(AT.AD):
        a = auc(X[:, j], Y)
        sonuc[ad] = a
        yon = "" if not np.isfinite(a) else ("  <-- AYIRT EDIYOR"
                                             if abs(a - 0.5) >= 0.10 else "")
        print(f"{ad:<16} {a:>7.4f}{yon}")

    # PARCA-ICI AUC: mutlak degerler markadan markaya kayar; asil soru bir parca
    # ICINDE dogru agzi yanlisindan ayirabiliyor muyuz.
    print(f"\n{'tanimlayici':<16} {'parca-ici AUC':>14}")
    PID = np.asarray(PID)
    ici = {}
    for j, ad in enumerate(AT.AD):
        v = []
        for u in np.unique(PID):
            k = PID == u
            if Y[k].any() and not Y[k].all():
                a = auc(X[k, j], Y[k])
                if np.isfinite(a):
                    v.append(a)
        ici[ad] = float(np.mean(v)) if v else float("nan")
        print(f"{ad:<16} {ici[ad]:>14.4f}"
              + ("  <-- AYIRT EDIYOR" if abs(ici[ad] - 0.5) >= 0.10 else ""))

    json.dump({"damga": makbuz_hash.damga(), "auc": sonuc, "parca_ici_auc": ici,
               "n_aday": int(len(X)), "pozitif_oran": float(Y.mean()),
               "not": "EGITIM YOK, saf ayrilabilirlik. D7 marka-disi. Etiket: "
                      "B-rep onerisi GT'ye tespit toleransinda mi."},
              open("results/agiz_ayirt_edicilik.json", "w"), indent=1)
    print("\nmakbuz -> results/agiz_ayirt_edicilik.json")


if __name__ == "__main__":
    main()
