# -*- coding: utf-8 -*-
"""L2'yi GERI AL: gate yeniden 13 ozellikle egitilir.

NEDEN: L2 (en cok ezberleyen 3 ozelligi atmak) DEV kumesinde +0.0066/+0.0068 kazanmisti; ama
o kume L2'nin SECILDIGI kumeydi. Geometri olarak AYRIK VAL kumesinde ayni kol -0.0111/-0.0104
KAYBETTI (results/sinav_val.json). Onceden yazilan kural: "DEV'de kazanip VAL'de kaybeden kol
asiri-uydurmadir, geri alinir."

Ders: bir ozellik "ezberliyor" damgasi yiyor diye atilamaz -- AUC dususu urun metrigi degil.
13 ozellikli gate hem VAL tespitinde (0.6530 vs 0.6418) hem kesinlikte (0.695 vs 0.640) onde.

Eski model results/wire_gate.pkl.l2_2026_07_31 olarak saklanir (tek adimda geri donulebilir).
"""
import os, sys, json, pickle, shutil
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
PKL = "results/wire_gate.pkl"
BAK = "results/wire_gate.pkl.l2_2026_07_31"


def main():
    from sklearn.ensemble import RandomForestClassifier
    import wire_gate

    eski = pickle.load(open(PKL, "rb"))
    if eski.get("cols") is None:
        print("gate zaten 13 ozellikli, yapacak is yok"); return
    if not os.path.exists(BAK):
        shutil.copy(PKL, BAK)
        print(f"yedek -> {BAK}")

    d = np.load("results/gate_regrow_data_rt2.npz", allow_pickle=True)
    X = d["X"]; y = d["y"]
    print(f"egitim: {X.shape[0]} aday x {X.shape[1]} ozellik "
          f"(eski gate {len(eski['cols'])} sutun kullaniyordu)")
    clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=3, n_jobs=-1,
                                 random_state=0).fit(X, y)
    pickle.dump({
        "clf": clf,
        "feat_names": list(wire_gate.FEAT_NAMES_13),
        "cols": None,
        "note": ("L2 GERI ALINDI 2026-07-31: 13 ozellik. L2 (depth/size/aspect atma) DEV'de "
                 "+0.0066 kazanmisti ama SECILDIGI kumeydi; ayrik VAL kumesinde -0.0111 tespit "
                 "/ -0.0104 robot KAYBETTI (results/sinav_val.json). Onceden yazili kill kurali "
                 "uygulandi. Eski model: results/wire_gate.pkl.l2_2026_07_31"),
    }, open(PKL, "wb"))
    print("gate 13 ozellikle yeniden egitildi ve yazildi")

    m = pickle.load(open(PKL, "rb"))
    assert m["cols"] is None and len(m["feat_names"]) == 13
    Xs = X[:5]
    assert m["clf"].predict_proba(Xs).shape == (5, 2), "13 sutunlu tahmin calismiyor"
    print("dogrulama: cols=None, 13 ad, tahmin 13 sutunla calisiyor")


if __name__ == "__main__":
    main()
