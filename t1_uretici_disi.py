# -*- coding: utf-8 -*-
"""T1 TESHIS: gate URETICI-DISI bolmede coker mi? (ezberleme riskinin gercek testi)

RISK (olculmus, ama ESKI gate icin): ExtraTrees aile-disi bolmede +0.012 verirken uretici-disi
bolmede -0.061 kaybetmisti, bir ureticide -0.128. O yuzden urunde RandomForest kullaniyoruz.
Ama o olcum 13 sutunlu donemdendi; gate o gunden beri 18 sutuna cikti.

SORU 1: mevcut 18 sutunlu gate uretici-disi bolmede ne kaybediyor?
SORU 2: B-rep FIZIKSEL sutunlari (brep_r, esesenli, r_orani, bos_derinlik, gecen) daha iyi
        transfer ediyor mu? Hipotez: onlar FIZIKSEL buyukluk (mm), digerleri korpus istatistigi
        (komsu sayisi, kose sayisi, olasilik ortalamasi) -- fizik ureticiden ureticiye degismez,
        istatistik degisir.

Bu betik CIKARIM YAPMAZ: yalniz gate verisi uzerinde egitim/deger. Ucuz ve hizli.
"""
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold


def auc_mw(x, y):
    x = np.asarray(x, float); y = np.asarray(y, bool)
    if y.all() or not y.any():
        return float("nan")
    r = np.empty(len(x), float); o = np.argsort(x, kind="mergesort"); xs = x[o]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n1 = int(y.sum()); n0 = len(y) - n1
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def f1_at(p, y, thr):
    s = p >= thr
    tp = int((y & s).sum()); fp = int((~y & s).sum()); fn = int((y & ~s).sum())
    pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
    return 2 * pr * rc / max(pr + rc, 1e-9), pr, rc


def main():
    d = np.load("results/gate_regrow_data_fiz.npz", allow_pickle=True)
    X = d["X"]; y = d["y"].astype(bool)
    mfg = np.array([str(x) for x in d["mfg"]])
    pids = np.array([str(x) for x in d["pids"]])
    gk = json.load(open("results/_strict_geometry_keys.json"))
    grp = np.array([gk.get(p, "yok:" + p) for p in pids])
    THR = float(json.load(open("cp_config.json", encoding="utf-8"))["robot_wire_gate_threshold"])
    U = sorted(set(mfg))
    print(f"{len(y)} aday | ureticiler {U} | esik {THR}\n")

    def egit(Xtr, ytr, Xte):
        return RandomForestClassifier(n_estimators=400, min_samples_leaf=3, n_jobs=-1,
                                      random_state=0).fit(Xtr, ytr).predict_proba(Xte)[:, 1]

    print(f"{'bolme':<26}{'sutun':>6}{'AUC':>8}{'F1':>8}{'kesin':>8}{'recall':>8}")
    res = {}
    for nc, ad in ((13, "13 (fiziksel YOK)"), (18, "18 (fiziksel VAR)")):
        # A) GEOMETRI-disi (mevcut durust protokol) -- referans
        o = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=5).split(X[:, :nc], y, grp):
            o[te] = egit(X[tr][:, :nc], y[tr], X[te][:, :nc])
        f, p_, r_ = f1_at(o, y, THR)
        print(f"{'geometri-disi (referans)':<26}{nc:>6}{auc_mw(o, y):>8.4f}{f:>8.4f}{p_:>8.3f}{r_:>8.3f}")
        res[f"geo|{nc}"] = {"auc": auc_mw(o, y), "f1": f}
        # B) URETICI-disi: birinde egit, otekinde dene
        for u in U:
            te = mfg == u
            s = egit(X[~te][:, :nc], y[~te], X[te][:, :nc])
            f2, p2, r2 = f1_at(s, y[te], THR)
            print(f"{'  -> uretici ' + u + ' disarida':<26}{nc:>6}{auc_mw(s, y[te]):>8.4f}"
                  f"{f2:>8.4f}{p2:>8.3f}{r2:>8.3f}")
            res[f"mfg{u}|{nc}"] = {"auc": auc_mw(s, y[te]), "f1": f2}
        print()

    print("KAYIP (uretici-disi - geometri-disi), F1:")
    for nc in (13, 18):
        ref = res[f"geo|{nc}"]["f1"]
        dl = [res[f"mfg{u}|{nc}"]["f1"] - ref for u in U]
        print(f"  {nc} sutun: " + " | ".join(f"{u}: {x:+.4f}" for u, x in zip(U, dl))
              + f"   ORTALAMA {np.mean(dl):+.4f}")
    d13 = np.mean([res[f"mfg{u}|13"]["f1"] - res["geo|13"]["f1"] for u in U])
    d18 = np.mean([res[f"mfg{u}|18"]["f1"] - res["geo|18"]["f1"] for u in U])
    print(f"\nFIZIKSEL SUTUNLAR TRANSFERI IYILESTIRDI MI: {d18 - d13:+.4f} "
          f"({'EVET' if d18 > d13 else 'HAYIR'})")
    json.dump(res, open("results/t1_uretici_disi.json", "w"), indent=1)
    print("makbuz -> results/t1_uretici_disi.json")


if __name__ == "__main__":
    main()
