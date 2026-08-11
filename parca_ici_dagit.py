# -*- coding: utf-8 -*-
"""PARCA-ICI z-skor baglamini URUNE AL (gate 22 -> 44 sutun).

GEREKCE -- UCTAN UCA olculdu (results/u4_parca_ici_uctan_uca.json, 200 parca, eslestirilmis
bootstrap %95 GA):

    bolme                    22 sutun   44 sutun     fark   %95 GA
    gorulmemis uretici WEI     0.4832     0.5702   +0.0869  [+0.048, +0.126]   <- HEDEF
    gorulmemis uretici PXC     0.7203     0.6828   -0.0375  [-0.065, -0.009]   <- GERCEK BEDEL
    tanidik (DEV+VAL)          0.7410     0.7390   -0.0023  [-0.023, +0.019]   (gurultu)

Onceden yazili kill: tanidik >= -0.01 VE gorulmemis uretici EN KOTU >= +0.01 -> GECTI.

BU BEDAVA BIR KAZANC DEGIL, RISK TAKASIDIR. Dogru okumasi:
    ureticiler arasi ORTALAMA   0.6018 -> 0.6265  (+0.0247)
    ureticiler arasi EN KOTU    0.4832 -> 0.5702  (+0.0869)
    ureticiler arasi YAYILIM    0.2370 -> 0.1126  (-0.1244)
Yani model, "kolay" ureticiden bir miktar verip "zor" ureticiden cok daha fazlasini aliyor.
Bu gece ExtraTrees'i tam da bu sekle benzedigi icin OLDURMUSTUM; fark olculdu ve YAPISAL:
ExtraTrees ortalamayi +0.007 ile yerinde birakip riski TASIYORDU, bu kol ortalamayi da
yukseltiyor ve yayilimi YARIYA indiriyor. Uc bagimsiz deneme (yalniz-ezber sutunlar,
yalniz-fiziksel sutunlar, topluluk) takasi kaldiramadi -- kazanc ve bedel ayni mekanizmadan
geliyor (results/u5_takas.json).

TEZ CIZGISI: ag, remesh, sinif tanimlari ve CP turetmesi DEGISMEDI. Degisen tek sey, tezde
zaten bulunmayan wire-gate katmaninin ozellik gosterimi.

GERI ALMA TEK ADIM: results/wire_gate.pkl.pre_parca_ici -> results/wire_gate.pkl
(model donusumu KENDI tasidigi icin baska hicbir sey degistirilmez.)
"""
import json
import os
import pickle
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
PKL = "results/wire_gate.pkl"
BAK = "results/wire_gate.pkl.pre_parca_ici"
NPZ = "results/gate_regrow_data_topo.npz"
DONUSUM = "zskor"


def main():
    os.environ["WG_FIZ_FEATS"] = "1"; os.environ["WG_TOPO"] = "1"
    from sklearn.ensemble import RandomForestClassifier
    import wire_gate

    assert len(wire_gate.FEAT_NAMES) == 22, len(wire_gate.FEAT_NAMES)
    if not os.path.exists(BAK):
        shutil.copy(PKL, BAK)
        print(f"yedek -> {BAK}")

    d = np.load(NPZ, allow_pickle=True)
    X = np.asarray(d["X"], float); y = np.asarray(d["y"])
    pids = np.array([str(p) for p in d["pids"]])
    # EGITIM donusumu, CIKARIM donusumuyle AYNI FONKSIYON olmali: burada da parca parca
    # uygulaniyor (calisma aninda bir parcanin adaylari ne ise, egitimde de o).
    M = np.zeros((len(X), X.shape[1] * 2))
    for u in np.unique(pids):
        i = np.where(pids == u)[0]
        M[i] = wire_gate.parca_ici(X[i], DONUSUM)
    print(f"egitim: {M.shape[0]} aday x {M.shape[1]} sutun | {len(np.unique(pids))} parca | "
          f"pozitif {y.mean():.4f}")

    clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=3, n_jobs=-1,
                                 random_state=0).fit(M, y)
    with open(PKL, "wb") as f:
        pickle.dump({
            "clf": clf, "feat_names": list(wire_gate.FEAT_NAMES) +
                        [n + "_z" for n in wire_gate.FEAT_NAMES],
            "cols": None, "n_feat": M.shape[1], "donusum": DONUSUM,
            # AYARI MODEL TASIR: `n_feat` yaricap uyusmazligini YAKALAYAMAZ (sutun SAYISI ayni,
            # ANLAMI farkli). wire_gate._dogrula_uyum bunu okuyup patlar.
            "topo_r": float(wire_gate.TOPO_R),
            "note": ("PARCA-ICI 2026-08-01: 22 ham sutun + 22 parca-ici z-skor. Gorulmemis "
                     "ureticide EN KOTU durum 0.4832 -> 0.5702 (GA [+0.048,+0.126]), tanidik "
                     "-0.0023 (gurultu), DIGER uretici-disi bolme -0.0375 (GERCEK BEDEL). "
                     "Ureticiler arasi yayilim 0.2370 -> 0.1126. Takasi kaldirmak icin 3 kol "
                     "denendi ve basarisiz (u5_takas.json). Onceki model: "
                     "results/wire_gate.pkl.pre_parca_ici"),
        }, f)
    print(f"model yazildi -> {PKL} ({M.shape[1]} sutun, donusum={DONUSUM})")

    with open("cp_config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["gate_parca_ici"] = DONUSUM
    cfg["gate_parca_ici_not"] = (
        "BILGI AMACLI: donusumu MODELIN KENDISI tasir (wire_gate.pkl['donusum']), bu alan "
        "degistirilerek acilip kapatilamaz. Geri alma: results/wire_gate.pkl.pre_parca_ici -> "
        "results/wire_gate.pkl. Olcum: results/u4_parca_ici_uctan_uca.json")
    with open("cp_config.json", "w", encoding="ascii") as f:
        json.dump(cfg, f, indent=1, ensure_ascii=True)
    print("cp_config.gate_parca_ici = " + DONUSUM)

    # DOGRULAMA: kaydedilen modeli DISKTEN geri okuyup gercek bir parcada calistir.
    import importlib
    importlib.reload(wire_gate)
    m = wire_gate._load(PKL)
    assert m["n_feat"] == 44 and m["donusum"] == DONUSUM, m.get("n_feat")
    Xd = wire_gate.parca_ici(X[:5], DONUSUM)
    assert Xd.shape == (5, 44), Xd.shape
    s = m["clf"].predict_proba(Xd)[:, 1]
    print(f"diskten geri okundu ve calisti | 5 aday skoru: {np.round(s, 3).tolist()}")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except BaseException:
        traceback.print_exc(); raise
