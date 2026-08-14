# -*- coding: utf-8 -*-
"""P3-POZ-B: PARCA-ICI EKSEN UZLASISI -- 90 derecelik sistematik hatayi kir.

BULGU (temiz sinav, uretici kirilimi): medyan aci UPUN 0.0 / ONV 0.5 iken
NIT 90.0 / MOR 90.0 / S+S 90.0 / SE 90.0. Yani bu ureticilerde urettigimiz eksen
uretici yonune TAM DIK. 10 derecelik oturtma kapisi bunu TANIM GEREGI duzeltemez.

KAHIN OLCUMU: dogru (konum, eksen) cifti B-rep adaylari arasindan secilebilseydi
robot 0.2060 -> **0.3880** (+0.1820). Yani bilgi VAR, secim YOK.

FIZIKSEL SEZGI (uretici kimligi KULLANMAZ, dolayisiyla genellenir):
bir klemens blogunda TUM tel girisleri BIRBIRINE PARALELDIR. O halde parcadaki
aday silindir eksenlerinin UZLASISI (en cok tekrar eden yon) muhtemelen gercek
giris yonudur. Uzlasi, tek tek adaylarin gurultusunu yutar.

ALGORITMA
  1. Parcadaki tum uygun silindirlerin eksenlerini topla (isaret duyarsiz: +a ~ -a)
  2. Yonleri `uzlasi_tol` derece icinde kumele, EN KALABALIK kumenin ortalamasi = u
  3. Her aday icin: agzi `mm_max` icinde olan ve ekseni u'ya `uzlasi_tol` icinde
     olan silindirlerden EN YAKININA oturt; yoksa adayi OLDUGU GIBI birak
  4. Isaret: mevcut yonle ayni tarafa cevrilir (isaret kolu ayri ve OLCUYLE OLU)

TEZ DEGISMEZ: `v_o` turetmesi ve 5 sinif aynen kalir; bu bir SON ISLEM adimidir ve
ham `v_o` yan yana raporlanabilir.

SECIM YANLILIGI: ayarlar DEV yarisinda (SUPU/NIT/S+S/SE) secilir, SINAV yarisinda
(UPUN/MOR/UTL/ONV) TEK ATIS olculur.
"""
import argparse
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

import d6_kayit

MAKBUZ = "results/p3b_eksen_uzlasi.json"
DEV_MFG = {"SUPU", "NIT", "S+S", "SE"}
ROBOT_YANAL, ROBOT_ACI = 2.0, 10.0


def uzlasi_ekseni(cyls, tol_der):
    """Parcadaki silindir eksenlerinin EN KALABALIK yon kumesinin ortalamasi."""
    import brep_snap
    A = [c["axis"] for c in (cyls or [])
         if brep_snap.R_MIN <= c["radius"] <= brep_snap.R_MAX]
    if not A:
        return None
    A = np.asarray(A, float)
    kos = np.cos(np.radians(tol_der))
    en, en_n = None, 0
    for i in range(len(A)):
        # isaret duyarsiz benzerlik: |a.b|
        m = np.abs(A @ A[i]) >= kos
        if int(m.sum()) > en_n:
            # kume ortalamasi -- hepsini A[i] tarafina cevir
            B = A[m] * np.sign(A[m] @ A[i])[:, None]
            v = B.mean(0); v /= (np.linalg.norm(v) + 1e-12)
            en, en_n = v, int(m.sum())
    return None if en is None else (en, en_n)


def uzlasiya_oturt(cyls, P, D, tol_der, mm_max, min_uye=2):
    """Uzlasi eksenine uyan silindirlere oturt. Doner: (P2, D2, kac_oturdu)."""
    import brep_snap
    if not cyls or not len(P):
        return P, D, 0
    u = uzlasi_ekseni(cyls, tol_der)
    if u is None or u[1] < min_uye:
        return P, D, 0
    uz, _n = u
    kos = np.cos(np.radians(tol_der))
    uygun = [c for c in cyls
             if brep_snap.R_MIN <= c["radius"] <= brep_snap.R_MAX
             and abs(float(c["axis"] @ uz)) >= kos]
    if not uygun:
        return P, D, 0
    P2 = P.copy(); D2 = D.copy(); n = 0
    for i in range(len(P)):
        p, d = P[i], D[i]
        en = None
        for c in uygun:
            for m in (c["mouth_a"], c["mouth_b"]):
                dist = float(np.linalg.norm(m - p))
                if dist > mm_max:
                    continue
                if en is None or dist < en[0]:
                    a = c["axis"]
                    en = (dist, np.asarray(m, float), a if float(a @ d) >= 0 else -a)
        if en is not None:
            P2[i] = en[1]; D2[i] = en[2]; n += 1
    return P2, D2, n


def puanla(kayit, model, ob, tol, mm, esle_detay, f1w, oran=0.40, taban=0.30,
           mfgler=None, sayac=None):
    import wire_gate
    from p1c_esik import maske
    T, R = [], []
    for pid, r in kayit.items():
        if mfgler is not None and r["mfg"] not in mfgler:
            continue
        G = np.asarray(r["G"], float); Gd = np.asarray(r["Gd"], float)
        rj = "cok" if r["n"] >= 8 else "dusuk"
        P = np.zeros((0, 3)); D = np.zeros((0, 3))
        M = d6_kayit.x58(r)
        if M is not None and r.get("P") is not None and len(r["P"]) \
                and M.shape[1] * 2 == model["n_feat"]:
            k = maske(np.asarray(wire_gate.karar_skoru(model, M), float), oran, taban)
            if k.any():
                P = np.asarray(r["P"], float)[k]; D = np.asarray(r["Pd"], float)[k]
        if tol is not None and len(P):
            P, D, n = uzlasiya_oturt(ob.get(pid) or [], P, D, tol, mm)
            if sayac is not None:
                sayac["oturan"] += n; sayac["aday"] += len(P)
        T.append((rj,) + esle_detay(P, D, G, Gd, r["diag"], 0.0, 180.0, True)[:3])
        R.append((rj,) + esle_detay(P, D, G, Gd, r["diag"], ROBOT_YANAL, ROBOT_ACI,
                                    False, isaretli=True)[:3])
    return f1w(T), f1w(R)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="results/wire_gate_v5.pkl")
    a = ap.parse_args()
    import protokol
    protokol.tez_dogrula()
    from sina_kume import esle_detay, f1w

    sv = d6_kayit.sinav()
    kayit = d6_kayit.yukle(set(sv["pidler"]))
    with open(a.model, "rb") as f:
        model = pickle.load(f)
    with open("results/_d6_silindirler.pkl", "rb") as f:
        ob = pickle.load(f)
    dev = {p: r for p, r in kayit.items() if r["mfg"] in DEV_MFG}
    sin = {p: r for p, r in kayit.items() if r["mfg"] not in DEV_MFG}

    d0t, d0r = puanla(dev, model, ob, None, 0, esle_detay, f1w)
    print(f"DEV oturtmasiz: tespit {d0t:.4f} robot {d0r:.4f}\n")
    print(f"{'tol\\mm':<9}" + "".join(f"{m:>10.0f}" for m in (4, 6, 8, 12)))
    en_iyi, en_iyi_r, izgara = None, d0r, {}
    for tol in (5, 10, 15, 25):
        satir = []
        for mm in (4, 6, 8, 12):
            tf, rf = puanla(dev, model, ob, tol, mm, esle_detay, f1w)
            izgara[f"{tol}/{mm}"] = {"tespit": tf, "robot": rf}
            satir.append(rf)
            if rf > en_iyi_r:
                en_iyi_r, en_iyi = rf, (tol, mm)
        print(f"{tol:<9}" + "".join(f"{v:>10.4f}" for v in satir))

    if en_iyi is None:
        print("\nDEV'de hicbir ayar tabani gecmedi -> KOL KAPANDI")
        with io.open(MAKBUZ, "w", encoding="utf-8") as f:
            json.dump({"karar": "KAPANDI", "dev_taban_robot": d0r, "izgara": izgara}, f,
                      indent=1)
        return
    tol, mm = en_iyi
    print(f"\nDEV'de secilen: tol {tol} derece / mesafe {mm}mm -> robot {en_iyi_r:.4f} "
          f"({en_iyi_r-d0r:+.4f})")
    sc = collections.Counter()
    s0t, s0r = puanla(sin, model, ob, None, 0, esle_detay, f1w)
    s1t, s1r = puanla(sin, model, ob, tol, mm, esle_detay, f1w, sayac=sc)
    print(f"\n--- SINAV YARISI (TEK ATIS) ---")
    print(f"{'ayar':<20}{'TESPIT':>9}{'ROBOT':>9}")
    print(f"{'uzlasisiz':<20}{s0t:>9.4f}{s0r:>9.4f}")
    print(f"{f'uzlasi {tol}/{mm}':<20}{s1t:>9.4f}{s1r:>9.4f}")
    print(f"{'FARK':<20}{s1t-s0t:>+9.4f}{s1r-s0r:>+9.4f}")
    print(f"  oturan aday: {sc['oturan']}/{sc['aday']}")
    karar = "DAGIT" if s1r > s0r and s1t >= s0t - 0.01 else "GERI AL"
    print(f"\nKARAR: {karar}")
    with io.open(MAKBUZ, "w", encoding="utf-8") as f:
        json.dump({"izgara": izgara, "dev_secim": {"tol": tol, "mm": mm},
                   "sinav_uzlasisiz": {"tespit": s0t, "robot": s0r},
                   "sinav_uzlasili": {"tespit": s1t, "robot": s1r},
                   "oturan": sc["oturan"], "aday": sc["aday"], "karar": karar},
                  f, indent=1, ensure_ascii=False)
    print(f"makbuz -> {MAKBUZ}")


if __name__ == "__main__":
    main()
