# -*- coding: utf-8 -*-
"""G2/G3/G4: EGIM KOLLARI -- ayni veriden daha cok GENELLEME cikar.

G1 tavan hesabi ([[aday-yok-temsil-darbogazi]]): gate MUKEMMEL olsa (sifir FP dahil) bile
gorulmemis-uretici tavani **0.5964**. Yani bu uc kol 0.70'i TEK BASINA GETIREMEZ; amaclari
o tavanin altinda ek pay almak. Asil kol G5 (temsil).

UC KOL, UCU DE AYRI MODEL (birlestirilmez -- yoksa hangisinin ne getirdigi atfedilemez):
  G2 URETICI-DENGELI : egitim satirlarinin %69'u WEI+PXC. Uretici basina esit toplam agirlik.
  G3 IKIZ-AGIRLIKLI  : geometri grubu basina ~2.5 kopya; her GRUP esit toplam agirlik.
  G4 BUDAMA          : 13 ozniteligin yalniz `votes`'u transfer ediyor
                       ([[gate-memorizes-not-learns]]). Dusuk kapasite + parca-ici z-skor
                       sutunlari (mutlak buyuklukler ureticiye ozgudur, goreli olanlar degil).

HER KOL IKI KUMEDE RAPORLANIR -- 194'luk kume ve gorulmemis-uretici sinavi. Yalniz birinde
bakmak yaniltir: [[gorulmemis-uretici-sinavi-ilk-olcum]] veri kolunda +0.0058 vs +0.0535
gostermisti.

G4 TUZAGI: dejenere "hepsini kabul et" modeli de mukemmel transfer eder. Bu yuzden
KESINLIK TABANI sart -- kesinlik 0.60'in altina duserse kol GECMEZ.

TEZ DEGISMEZ: ag egitilmez, remesh degismez, `v_o` degismez. Yalniz gate'in EGITIM
AGIRLIKLARI ve OZNITELIK ALT KUMESI degisir.
"""
import collections
import io
import json
import os
import pickle
import sys

import numpy as np

os.environ.setdefault("BA_ALLOW_SEEN", "1")
os.environ["WG_FIZ_FEATS"] = "1"; os.environ["WG_TOPO"] = "1"; os.environ["WG_ZENGIN"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import wire_gate
from f2_12_veri_kolu import olc

V3 = "results/zengin_parite_v3.npz"
SINAV = "results/_der_sinav_yeni.pkl"
MAKBUZ = "results/g234_egim_kollari.json"


def _z(X, pid):
    """Parca-ici z-skor -- calisma anindaki ile AYNI (parca bazinda)."""
    Z = np.zeros((len(X), X.shape[1] * 2), float)
    for p in np.unique(pid):
        m = pid == p
        Z[m] = wire_gate.parca_ici(X[m], "zskor")
    return Z


def egit(npz, agirlik=None, cols=None, derinlik=None, yaprak=3, tohum=0):
    """agirlik: None | 'uretici' | 'grup'  ·  cols: kullanilacak HAM sutun indeksleri."""
    from sklearn.ensemble import RandomForestClassifier
    d = np.load(npz, allow_pickle=True)
    X = np.hstack([d["X22"], d["XR"]]).astype(float)
    pid = np.asarray(d["pids"], str); mfg = np.asarray(d["mfg"], str)
    Z = _z(X, pid)
    # SUTUN SECIMI KALDIRILDI -- URUNUN KARAR YOLUYLA UYUMSUZDU (2026-08-04):
    # `wire_gate.karar_skoru` once TUM 58 sutuna parca-ici z-skor uygular (->116), sonra
    # `Xd[:, :n_feat]` ile ONEK alir. Yani "13 ham + onlarin z'si" gibi bir SECIM cikarimda
    # okunamaz; model 26 sutunla egitilir ama cikarimda ham 0..25 okunur -> F1 TAM 0.0000.
    # (Ilk kosuda aynen bu oldu; sonuc sanilmasin diye kayda geciyor.)
    # Ayni hipotez -- "gate EZBERLIYOR, kapasite dusurulunce transfer artar" -- sutun
    # budamadan da sinanabilir: KAPASITE kisitlamasi (derinlik/yaprak). O yol urunun
    # karar yoluyla TAM UYUMLU cunku oznitelik duzeni degismiyor.
    w = None
    if agirlik == "uretici":
        c = collections.Counter(mfg.tolist())
        w = np.array([1.0 / c[m] for m in mfg]); w *= len(w) / w.sum()
    elif agirlik == "grup":
        import olcum_kumesi as OK
        gk = OK.geo_anahtarlari()
        g = np.array([gk.get(p, "yok:" + p) for p in pid])
        c = collections.Counter(g.tolist())
        w = np.array([1.0 / c[x] for x in g]); w *= len(w) / w.sum()
    clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=yaprak,
                                 max_depth=derinlik, n_jobs=-1, random_state=tohum)
    clf.fit(Z, d["y"], sample_weight=w)
    return {"clf": clf, "n_feat": Z.shape[1], "donusum": "zskor",
            "cols": (list(cols) if cols is not None else None), "feat_names": None}


def main():
    import protokol
    protokol.tez_dogrula()
    import olcum_kumesi as OK
    D194, _ = OK.kume("results/_der_tam.pkl")
    with open(SINAV, "rb") as f:
        DSIN = pickle.load(f)
    print(f"olcum: 194'luk {len(D194)} parca | sinav {len(DSIN)} parca\n")

    # G4 sutun secimi: `votes` (11) + parca-ici z-skoru transfer eden tek sey oldugu icin
    # taban 13 sutunun GORELI hali. Mutlak boyut/derinlik sutunlari ureticiye ozgudur.
    TABAN13 = list(range(13))

    KOL = [
        ("v3 TABAN (degismemis)", dict()),
        ("G2 uretici-dengeli", dict(agirlik="uretici")),
        ("G3 ikiz-agirlikli", dict(agirlik="grup")),
        ("G4a kapasite dusuk (derinlik 8, yaprak 10)", dict(derinlik=8, yaprak=10)),
        ("G4b kapasite cok dusuk (derinlik 5, yaprak 25)", dict(derinlik=5, yaprak=25)),
    ]
    S = {}
    print(f"{'kol':<38}{'194 F1':>9}{'SINAV F1':>10}{'sinav kesinlik':>16}")
    for ad, kw in KOL:
        m = egit(V3, **kw)
        r1 = olc(m, D194); r2 = olc(m, DSIN)
        kes = r2["TP"] / max(r2["TP"] + r2["FP"], 1)
        S[ad] = {"m194": r1["F1"], "sinav": r2["F1"], "kesinlik": kes,
                 "TP": r2["TP"], "FP": r2["FP"], "FN": r2["FN"]}
        print(f"{ad:<38}{r1['F1']:>9.4f}{r2['F1']:>10.4f}{kes:>16.4f}")

    t = S["v3 TABAN (degismemis)"]
    print(f"\n{'kol':<38}{'194 fark':>10}{'SINAV fark':>12}   GO")
    GO = {"G2 uretici-dengeli": (0.020, -0.010), "G3 ikiz-agirlikli": (0.015, -1.0),
          "G4a kapasite dusuk (derinlik 8, yaprak 10)": (0.025, -1.0),
          "G4b kapasite cok dusuk (derinlik 5, yaprak 25)": (0.025, -1.0)}
    for ad, (esik_s, esik_p) in GO.items():
        d1 = S[ad]["m194"] - t["m194"]; d2 = S[ad]["sinav"] - t["sinav"]
        ok = d2 >= esik_s and d1 >= esik_p
        if ad.startswith("G4"):
            ok = ok and S[ad]["kesinlik"] >= 0.60
        print(f"{ad:<38}{d1:>+10.4f}{d2:>+12.4f}   {'GECTI' if ok else 'gecmedi'}")

    en = max(S, key=lambda k: S[k]["sinav"])
    print(f"\nEN IYI SINAV F1: {en} -> {S[en]['sinav']:.4f}")
    print(f"GUN 1 KAPISI (>= 0.47): {'GECTI' if S[en]['sinav'] >= 0.47 else 'GECMEDI'}")
    print(f"  NOT: G1 tavan hesabi zaten 0.5964 diyordu -- bu kollar 0.70'i tek basina")
    print(f"       getiremez. Asil kol G5 (temsil), ADAY_YOK 563 FN kovasi.")
    with io.open(MAKBUZ, "w", encoding="utf-8") as f:
        json.dump(S, f, indent=1, ensure_ascii=False)
    print(f"makbuz -> {MAKBUZ}")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except BaseException:
        traceback.print_exc(); raise
