# -*- coding: utf-8 -*-
"""KAZANAN yapilandirmanin HATA BANKASI: kayip tam olarak nerede?

Kazanan (2026-08-11 gece): B-rep havuzu + agiz tanimlayicilari + HGB-derin,
D7 marka-disi TAM ZINCIR robot **0.2773** / tespit 0.4813.

`hata-bankasi-fn-taksonomisi` kaydindaki kovalar ESKI yigin icin olculmustu
(ADAY_YOK %57.0 / POZ %27.2 / KALABALIK %10.2 / GATE_REDDI %5.6). Havuz, etiket
ve secici DEGISTI -- o dagilim artik gecerli olmayabilir. 0.40'a giden kolu
secmeden once yeniden olculur.

KOVALAR (her kacirilan GT icin, SIRAYLA sorulur):
  ADAY_YOK      havuzda 2mm yanal + 40mm eksenel icinde HIC aday yok
  YON_YOK       konum var ama hicbir adayin yonu 10 derece icinde degil
  GATE_REDDI    robot-uygun aday VARDI ama gate esigi altinda kaldi
  NMS_YEDI      gate'i gecti ama kalabalik bastirma sildi
  POZ_BOZDU     secildi, ama poz kafasi ciktiyi tolerans disina tasidi
  ESLESME       hepsi tamam ama bire-bir eslesmede baska GT'ye gitti
Ayrica FP'ler kaynagina gore (segmentasyon / B-rep) ayrilir.
"""
import collections
import json
import os
import pickle
import sys

import numpy as np

import makbuz_hash

os.environ.setdefault("BA_ALLOW_SEEN", "1")
os.environ["WG_FIZ_FEATS"] = "1"
os.environ["WG_TOPO"] = "1"
os.environ["WG_ZENGIN"] = "1"
sys.path.insert(0, ".")
import kanonik_d7 as K          # noqa: E402
import urun_zinciri             # noqa: E402
import wire_gate                # noqa: E402
from sina_kume import esle_macar  # noqa: E402

OZ = "results/_tam_oz"
TAN = "results/_tan_hizali"
OB = "results/_p1_olasilik_d7"
KAYNAKLAR = (0, 1)
ESIK = 0.05                      # HGB-derin icin D6'da secilmisti
GIRME, ERISIM = 3 + 58, 5 + 58   # agiz_tanimlayici sutunlari X'in SONUNDA
ISARET = os.environ.get("ISARET", "1") not in ("0", "")
YANAL, ACI, EKSENEL = 2.0, 10.0, 40.0


def uygun(P, D, g, gd):
    """Bu havuzda `g` icin ROBOT olcutunu saglayan adaylarin maskesi."""
    n = np.linalg.norm(gd)
    if n < 1e-9 or not len(P):
        return np.zeros(len(P), bool), np.zeros(len(P), bool)
    u = gd / n
    w = P - g
    e = w @ u
    yan = np.linalg.norm(w - e[:, None] * u[None], axis=1)
    kon = (yan <= YANAL) & (np.abs(e) <= EKSENEL)
    Dn = D / np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-12)
    aci = np.degrees(np.arccos(np.clip(Dn @ u, -1.0, 1.0)))
    return kon, kon & (aci <= ACI)


def main():
    model = pickle.load(open("results/kazanan_hgb_derin.pkl", "rb"))["HGB-derin"]
    S = K.step_haritasi()
    kova = collections.Counter()
    fp_kaynak = collections.Counter()
    n_gt = n_fp = 0
    parcalar = [f[3:-4] for f in sorted(os.listdir(OZ))
                if f.startswith("d7_") and f.endswith(".npz")]
    kay = K.yukle(parcalar)
    for pid in parcalar:
        r = kay.get(pid)
        if r is None:
            continue
        G = np.asarray(r.get("G", []), float)
        if not len(G):
            continue
        Gd = np.asarray(r["Gd"], float)
        dg = float(r["diag"])
        z = np.load(f"{OZ}/d7_{pid}.npz")
        kayn = np.asarray(z["kaynak"], int)
        m = np.isin(kayn, KAYNAKLAR)
        T = np.asarray(np.load(f"{TAN}/d7_{pid}.npz")["T"], float)
        X = np.hstack([np.asarray(z["X"], float), T])[m]
        P0, D0 = np.asarray(z["P"], float)[m], np.asarray(z["D"], float)[m]
        kay0 = kayn[m]
        if len(X) < 2:
            n_gt += len(G)
            kova["ADAY_YOK"] += len(G)
            continue
        s = np.asarray(model.predict_proba(
            wire_gate.parca_ici(X, "zskor"))[:, 1], float)
        gk = s >= ESIK
        P1, D1, k1 = P0[gk], D0[gk], kay0[gk]
        if len(P1) > 1:
            nm = wire_gate.kalabalik_maskesi(P1, s[gk])
            P2, D2, k2 = P1[nm], D1[nm], k1[nm]
        else:
            P2, D2, k2 = P1, D1, k1
        if ISARET and len(P2):
            # B2a ISARET DUZELTMESI (+0.0316) artik TABANIN parcasi.
            # Tanimlayici sutunlari X'in sonunda: 58 gate ozniteligi + 9 tanim.
            Tn = X[gk][nm] if len(P1) > 1 else X[gk]
            D2 = np.where((Tn[:, ERISIM] < Tn[:, GIRME])[:, None], -D2, D2)
        # poz kafasi
        P3, D3 = P2, D2
        f = f"{OB}/{pid}.npz"
        if len(P2) and os.path.exists(f):
            zz = np.load(f)
            P3, D3 = urun_zinciri.tam_poz(
                np.ascontiguousarray(zz["V"], np.float64),
                np.ascontiguousarray(zz["F"], np.int64),
                np.asarray(zz["pbs"], float).mean(0), P2, D2,
                step_path=S.get(pid))
        tp, fp, fn = esle_macar(P3, D3, G, Gd, dg, YANAL, ACI, False,
                                isaretli=True)[:3]
        n_gt += len(G)
        n_fp += fp
        # FP kaynagi (poz kafasi sirayi korur)
        if fp and len(k2) == len(P3):
            _kon = np.zeros(len(P3), bool)
            for j in range(len(G)):
                _kon |= uygun(P3, D3, G[j], Gd[j])[1]
            for i in np.where(~_kon)[0]:
                fp_kaynak["seg" if k2[i] == 0 else "B-rep"] += 1
        # kacirilanlari kovala
        eslesen = set()
        for j in range(len(G)):
            _k, ok3 = uygun(P3, D3, G[j], Gd[j])
            if ok3.any():
                eslesen.add(j)
        for j in range(len(G)):
            if j in eslesen:
                continue
            _k0, ok0 = uygun(P0, D0, G[j], Gd[j])
            if not _k0.any():
                kova["ADAY_YOK"] += 1
            elif not ok0.any():
                kova["YON_YOK"] += 1
            elif not uygun(P1, D1, G[j], Gd[j])[1].any():
                kova["GATE_REDDI"] += 1
            elif not uygun(P2, D2, G[j], Gd[j])[1].any():
                kova["NMS_YEDI"] += 1
            else:
                kova["POZ_BOZDU"] += 1
        # eslesen ama Macar'da kaybedilenler
        kova["ESLESME"] += max(len(eslesen) - tp, 0)
    print(f"D7 {n_gt} GT | FP {n_fp}\n")
    print(f"{'kova':<14}{'sayi':>7}{'GT payi':>10}")
    for k, v in kova.most_common():
        print(f"{k:<14}{v:>7}{v/max(n_gt,1):>10.3f}")
    print(f"\nFP kaynagi: {dict(fp_kaynak)}")
    json.dump({"damga": makbuz_hash.damga(), "n_gt": n_gt, "n_fp": int(n_fp),
               "kova": dict(kova), "fp_kaynak": dict(fp_kaynak),
               "yigin": "B-rep havuzu + agiz tanimlayicilari + HGB-derin, esik 0.05",
               "not": "KAZANAN yapilandirmanin hata bankasi. D7 marka-disi, "
                      "TAM ZINCIR. Kovalar SIRAYLA sorulur, ilk eslesende durur."},
              open(os.environ.get("HB_CIKTI", "results/kazanan_hata_bankasi.json"), "w"), indent=1)
    print("makbuz -> results/kazanan_hata_bankasi.json")


if __name__ == "__main__":
    main()
