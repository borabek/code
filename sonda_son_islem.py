# -*- coding: utf-8 -*-
"""SON-ISLEM KOLLARI — cevrimdisi, tahmin dokumunden

NEDEN BURASI. Sunum sayilarindaki EN BUYUK tek kayip burada:

    tespit F1            0.7878     (deligi buldu)
    robot, eksen olcutu  0.5764     -> -0.2114  (eksen 10 dereceden sapmis)
    robot, ISARETLI      0.4839     -> -0.0925  (180 derece TERS)

Ikisi de SAF GEOMETRI ve ikisi de SON-ISLEM ile duzeltilebilir OLABILIR.
Egitim gerektirmez; dokum sayesinde saniyeler icinde olculur.

KOLLAR:
  taban        : dokunma
  disari       : yonu govdeden DISARI bakacak sekilde cevir
                 (GT sozlesmesi %100 disari -- `isaretli-yon-secici`)
  eksen_snap   : yonu parcanin ANA EKSENLERINDEN en yakinina oturt
  snap_disari  : ikisi birden
  gt_isaret    : isaret GT'den (UST SINIR -- isaretten ne kadar kaybettigimizi
                 olcer, DAGITILAMAZ)

GOVDE MERKEZI: dokumde mesh yok; tahmin edilen CP'lerin agirlik merkezi
vekil olarak kullanilir. Klemenste girisler karsit yuzlerde oldugu icin bu
merkez govde merkezine yakindir. Vekil ZAYIFSA kol haksiz yere duser --
bu yuzden `gt_isaret` UST SINIRI de olculur: aradaki fark vekilin
kusurunu gosterir.
"""
import collections
import json
import os
import sys

import numpy as np

os.environ.setdefault("BA_ALLOW_SEEN", "1")
sys.path.insert(0, ".")
import kanonik_d7 as K             # noqa: E402
from sina_kume import esle_macar   # noqa: E402

DOKUM = os.environ.get("SI_DOKUM", "results/_tahmin_dokumu.json")
YOL = os.environ.get("SI_YOL", "saha")


def _birim(v):
    v = np.asarray(v, float)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def ana_eksenler(P):
    """parcanin ana eksenleri (PCA) + eksen hizali uc yon."""
    if len(P) < 3:
        return np.eye(3)
    Q = P - P.mean(0)
    _, _, Vt = np.linalg.svd(Q, full_matrices=False)
    return np.vstack([Vt, np.eye(3)])


def uygula(P, D, kol, G=None, Gd=None, mesh_merkez=None, yerel_n=None,
           halka_n=None):
    if not len(P):
        return P, D
    D = _birim(D)
    merkez = P.mean(0)
    if kol in ("eksen_snap", "snap_disari"):
        E = _birim(ana_eksenler(P))
        # her yonu, |cos| en buyuk olan ana eksene oturt (ISARETSIZ eslesme)
        c = np.abs(D @ E.T)
        j = np.argmax(c, axis=1)
        yeni = E[j] * np.sign(np.sum(D * E[j], axis=1))[:, None]
        D = _birim(yeni)
    if kol == "disari_mesh" and mesh_merkez is not None:
        r = P - np.asarray(mesh_merkez, float)[None, :]
        isaret = np.sign(np.sum(D * r, axis=1))
        isaret[isaret == 0] = 1.0
        D = D * isaret[:, None]
    if kol == "disari_normal" and yerel_n is not None and len(yerel_n):
        N = _birim(np.asarray(yerel_n, float))
        isaret = np.sign(np.sum(D * N, axis=1))
        isaret[isaret == 0] = 1.0
        D = D * isaret[:, None]
    if kol in ("disari", "snap_disari"):
        r = P - merkez[None, :]
        isaret = np.sign(np.sum(D * r, axis=1))
        isaret[isaret == 0] = 1.0
        D = D * isaret[:, None]
    if kol == "halka_disari" and halka_n is not None and len(halka_n):
        H = _birim(np.asarray(halka_n, float))
        if len(H) == len(D):
            isaret = np.sign(np.sum(D * H, axis=1))
            isaret[isaret == 0] = 1.0
            D = D * isaret[:, None]
    if kol == "halka_kesin" and halka_n is not None and len(halka_n):
        H = _birim(np.asarray(halka_n, float))
        if len(H) == len(D):
            aci = np.degrees(np.arccos(np.clip(np.sum(D * H, axis=1), -1, 1)))
            _e = float(os.environ.get("SI_ESIK", "150"))
            D = np.where((aci >= _e)[:, None], -D, D)
    if kol in ("normal_kesin", "mesh_kesin"):
        ref = (np.asarray(yerel_n, float) if kol == "normal_kesin"
               and yerel_n is not None and len(yerel_n) else None)
        if ref is None and kol == "mesh_kesin" and mesh_merkez is not None:
            ref = P - np.asarray(mesh_merkez, float)[None, :]
        if ref is not None and len(ref) == len(D):
            R = _birim(ref)
            cos = np.clip(np.sum(D * R, axis=1), -1, 1)
            aci = np.degrees(np.arccos(cos))
            # ESIK CEVREDEN: `normal_kesin` 150 derecede HIC tetiklenmedi
            # (tam +0.0000). Esik taranabilir olmali.
            _esik = float(os.environ.get("SI_ESIK", "150"))
            cevir = aci >= _esik
            D = np.where(cevir[:, None], -D, D)
    if kol == "parca_modal" and len(D) >= 3:
        # parcadaki cogunluk yonune uy: en buyuk isaretli kumeyi bul,
        # ona 90 dereceden fazla ters olanlari cevir
        c = np.clip(D @ D.T, -1, 1)
        oy = (np.degrees(np.arccos(c)) <= 90.0).sum(1)
        ana = D[int(np.argmax(oy))]
        # yalniz ana yone DIK OLMAYANLARI (yani ayni eksende olanlari) cevir
        hiz = np.abs(np.clip(D @ ana, -1, 1))
        ters = (D @ ana < 0) & (hiz > 0.5)
        D = np.where(ters[:, None], -D, D)
    if kol == "gt_isaret" and G is not None and len(G):
        # UST SINIR: her tahmini, EN YAKIN GT'nin isaretine cevir
        d = np.linalg.norm(P[:, None, :] - G[None, :, :], axis=-1)
        j = np.argmin(d, axis=1)
        isaret = np.sign(np.sum(D * _birim(Gd)[j], axis=1))
        isaret[isaret == 0] = 1.0
        D = D * isaret[:, None]
    return P, D


def main():
    d = [r for r in json.load(open(DOKUM)) if r["yol"] == YOL]
    if not d:
        sys.exit(f"{DOKUM} icinde '{YOL}' yok")
    # YENI KOLLAR (2026-08-13): dokum artik GERCEK govde bilgisi tasiyor.
    # `disari_mesh`  : mesh MERKEZINDEN disari (tahmin ortalamasi degil)
    # `disari_normal`: YEREL YUZEY NORMALIYLE ayni yone (en dogru vekil)
    # SECICI ISARET KURALLARI (2026-08-13). "HEP disari cevir" ZARAR verdi
    # (-0.027..-0.113), yani modelin isaretleri COGUNLUKLA DOGRU ve hata
    # AZINLIKTA. O halde dogru kural "hep cevir" degil "EMIN OLUNCA cevir".
    #   normal_kesin : yalniz yerel normalle 150 dereceden fazla celisiyorsa
    #   mesh_kesin   : yalniz mesh merkeziyle 150 dereceden fazla celisiyorsa
    #   parca_modal  : parcadaki COGUNLUGUN isaretine uy (tutarlilik)
    # HALKA NORMALI (2026-08-13). Teshis: en yakin tepenin normali deligin
    # DUVAR normalidir ve eksene DIKTIR (ortanca 88.9 derece) -- o yuzden
    # `disari_normal` −0.0266 verdi. Dogru referans, agiz CEVRESINDEKI
    # halkadan (3-8mm) alinan yuz normalidir.
    KOLLAR = ("taban", "disari_normal", "halka_disari", "halka_kesin",
              "mesh_kesin", "gt_isaret")
    agg = {k: collections.Counter() for k in KOLLAR}
    for r in d:
        P0 = np.asarray(r["P"], float).reshape(-1, 3)
        D0 = np.asarray(r["D"], float).reshape(-1, 3)
        G = np.asarray(r["G"], float)
        Gd = np.asarray(r["Gd"], float)
        dg = float(r["diag"])
        for kol in KOLLAR:
            P, D = uygula(P0.copy(), D0.copy(), kol, G, Gd,
                          r.get("mesh_merkez"), r.get("yerel_normal"),
                          r.get("halka_normal"))
            c = agg[kol]
            for ad, im in (("isaretli", True), ("isaretsiz", False)):
                tp, fp, fn = esle_macar(P, D, G, Gd, dg, K.YANAL, K.ACI,
                                        False, isaretli=im)[:3]
                c[ad + "_tp"] += tp; c[ad + "_fp"] += fp; c[ad + "_fn"] += fn
            tp, fp, fn = esle_macar(P, D, G, Gd, dg, 0.0, 180.0, True)[:3]
            c["tespit_tp"] += tp; c["tespit_fp"] += fp; c["tespit_fn"] += fn

    def f1(c, on):
        return (2 * c[on + "_tp"] /
                max(2 * c[on + "_tp"] + c[on + "_fp"] + c[on + "_fn"], 1))

    print(f"{len(d)} parca | yol={YOL}\n")
    print(f"{'kol':<14}{'tespit':>9}{'isaretsiz':>11}{'ISARETLI':>10}"
          f"{'fark':>9}")
    tab = f1(agg["taban"], "isaretli")
    out = {}
    for kol in KOLLAR:
        c = agg[kol]
        r = {"tespit": f1(c, "tespit"), "isaretsiz": f1(c, "isaretsiz"),
             "isaretli": f1(c, "isaretli")}
        out[kol] = r
        et = ""
        if kol == "gt_isaret":
            et = "  (UST SINIR)"
        elif r["isaretli"] > tab + 0.01:
            et = "  <- KAPI GECTI"
        print(f"{kol:<14}{r['tespit']:>9.4f}{r['isaretsiz']:>11.4f}"
              f"{r['isaretli']:>10.4f}{r['isaretli'] - tab:>+9.4f}{et}")
    json.dump({"yol": YOL, "n_parca": len(d), "sonuc": out,
               "not": "Son-islem kollari, cevrimdisi. gt_isaret UST "
                      "SINIRDIR (dagitilamaz). D7'ye BAKILMADI."},
              open(f"results/son_islem_{YOL}.json", "w"), indent=1)
    print(f"\nmakbuz -> results/son_islem_{YOL}.json")


if __name__ == "__main__":
    main()
