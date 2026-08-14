# -*- coding: utf-8 -*-
"""POSE HEAD'i BUYUK ARTIKLAR ICIN YENIDEN EGIT — cevrimdisi secme turu.

TESHIS (Bolum 21.53): dagitilan pose head'in onerdigi en buyuk yer
degistirme **2.87 mm**; kirpma sinirini (3 mm) HIC zorlamiyor. Yani
`maks_mm` atil, sinirlayan modelin kendisi.

IKI OLASI MEKANIZMA -- ikisi de burada test ediliyor:

 (1) SECIM YANLILIGI: egitim verisi yalnizca `tt = max(3, 0.06*diag)`
     icinde ESLESMIS adaylardan kuruluyordu. -> `Q3_TOL=15` ile yeniden
     kuruldu: buyuk artik orani %10.2 -> **%18.4**, maks 12.1 -> 14.9 mm.

 (2) REGRESYON BUZULMESI: ormanin yaprak ortalamasi ucdegerleri iceri
     ceker. Taban veride ZATEN %10.2 buyuk hedef vardi ama model 2.87mm'yi
     asmiyor -- yani buzulme tek basina yeterli aciklama olabilir.
     -> `min_samples_leaf` kucultulerek test ediliyor.

Bu betik EGITIP OLCMEZ; adaylari kendi arasinda **grup-disi (OOF)**
karsilastirir ve yalniz kazanani zincire sokmaya deger kilar. Olcut,
metrigin kendisi degil ama metrikle DOGRUDAN baglantili: kabul kutusuna
(2 mm) giren aday orani.
"""
import json
import os
import pickle
import sys

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

N_FEAT = 58            # dagitilan wire_gate.feats_for genisligi
KATLAR = 4


def yukle(fp):
    d = np.load(fp, allow_pickle=True)
    X = np.asarray(d["X"], float)[:, :N_FEAT]
    Y = np.asarray(d["Y"], float)
    g = np.array([str(x) for x in d["geo"]])
    p = np.array([str(x) for x in d["pid"]])
    return X, Y, g, p


def val_gruplari():
    """VAL parcalarinin GEOMETRI GRUPLARI -- egitimden cikarilacak."""
    try:
        gk = json.load(open("results/_strict_geometry_keys.json",
                            encoding="utf-8"))
        s3 = json.load(open("results/split3.json", encoding="utf-8"))
    except Exception as e:                              # noqa: BLE001
        print(f"  UYARI: sizinti kapisi kurulamadi ({type(e).__name__})")
        return set()
    # DIKKAT: split3.json'da `val` bir SOZLUK ({"n":..., "parts":[...]}).
    # Dogrudan uzerinde donmek anahtarlari ('n','parts') verir ve sizinti
    # kapisi SESSIZCE hicbir seyi elemez -- ilk kosuda tam boyle oldu
    # (0 grup cikarildi).
    v = s3.get("val") or s3.get("VAL") or []
    if isinstance(v, dict):
        v = v.get("parts") or v.get("pids") or []
    val = [str(x) for x in v]
    grup = {gk[p] for p in val if p in gk}
    assert val, "VAL parca listesi BOS -- sizinti kapisi kurulamadi"
    assert grup, "VAL parcalarinin hicbiri geometri haritasinda YOK"
    print(f"  VAL {len(val)} parca -> {len(grup)} geometri grubu")
    return grup


def oof_degerlendir(X, Y, g, ad, Xd=None, Yd=None, gd=None, **kw):
    """grup-disi tahmin -> yanal artigin kabul kutusuna girme orani.

    ORTAK DEGERLENDIRME POPULASYONU (2026-08-14 duzeltmesi): egitim verisi
    (X,Y,g) ile DEGERLENDIRME verisi (Xd,Yd,gd) ayrilir. Ilk kosuda iki
    aday KENDI veri kumesinde puanlandi ve genis veri "daha kotu" gorundu
    -- oysa genis kume DAHA ZOR satirlar iceriyor. Ayni satirlarda
    olculmeyen iki oran KIYASLANAMAZ.
    """
    if Xd is None:
        Xd, Yd, gd = X, Y, g
    tah = np.zeros_like(Yd)
    gkf = GroupKFold(n_splits=KATLAR)
    # Degerlendirme kumesini katlara bol; her kat icin EGITIM kumesinden
    # o katin gruplarini CIKARARAK egit (grup-disi, sizintisiz).
    for _, te in gkf.split(Xd, Yd, groups=gd):
        tut = ~np.isin(g, np.unique(gd[te]))
        m = RandomForestRegressor(n_jobs=-1, random_state=0, **kw).fit(
            X[tut], Y[tut])
        tah[te] = m.predict(Xd[te])
    # yanal artik: duzeltmeden ONCE |Y|, duzeltmeden SONRA |Y - tahmin|
    onc = np.linalg.norm(Yd[:, :2], axis=1)
    son = np.linalg.norm(Yd[:, :2] - tah[:, :2], axis=1)
    oner = np.linalg.norm(tah[:, :2], axis=1)
    print(f"{ad:34s} kutuda(<=2mm) {100*(onc <= 2).mean():5.1f}% -> "
          f"{100*(son <= 2).mean():5.1f}%  | artik ortanca "
          f"{np.median(onc):.2f} -> {np.median(son):.2f} mm | "
          f"oneri maks {oner.max():5.2f} >3mm {100*(oner > 3).mean():4.1f}%")
    return {"kutu_once": float((onc <= 2).mean()),
            "kutu_sonra": float((son <= 2).mean()),
            "artik_ortanca": float(np.median(son)),
            "oneri_maks": float(oner.max()),
            "oneri_buyuk_oran": float((oner > 3).mean())}


def main():
    VG = val_gruplari()
    print(f"sizinti kapisi: {len(VG)} VAL geometri grubu egitimden CIKARILDI\n")
    kaynak = [("TABAN veri (tol ~3-6mm)", "results/pose_veri_graf.npz"),
              ("GENIS veri (tol 15mm)", "results/pose_veri_tol15.npz")]
    ayar = [("orman yaprak>=5 (MEVCUT)", dict(n_estimators=400,
                                              min_samples_leaf=5)),
            ("orman yaprak>=2", dict(n_estimators=400, min_samples_leaf=2)),
            ("orman yaprak>=1", dict(n_estimators=400, min_samples_leaf=1))]
    # ORTAK DEGERLENDIRME KUMESI: her aday AYNI satirlarda puanlanir.
    # Taban veri secildi cunku dagitilan modelin gordugu populasyon odur.
    Xd, Yd, gd, _ = yukle(kaynak[0][1])
    _t = ~np.isin(gd, list(VG))
    Xd, Yd, gd = Xd[_t], Yd[_t], gd[_t]
    print(f"ORTAK DEGERLENDIRME KUMESI: {len(Yd)} satir / "
          f"{len(set(gd))} grup\n")

    rapor = {}
    for vad, vfp in kaynak:
        if not os.path.exists(vfp):
            print(f"EKSIK: {vfp}")
            continue
        X, Y, g, p = yukle(vfp)
        tut = ~np.isin(g, list(VG))
        X, Y, g = X[tut], Y[tut], g[tut]
        print(f"--- EGITIM: {vad} -> {len(Y)} satir / {len(set(g))} grup ---")
        for aad, kw in ayar:
            rapor[f"{vad} | {aad}"] = oof_degerlendir(
                X, Y, g, "  " + aad, Xd=Xd, Yd=Yd, gd=gd, **kw)
        print()

    json.dump(rapor, open("results/pose_yeniden.json", "w"), indent=1)
    print("-> results/pose_yeniden.json")

    # EN IYI ADAYI EGIT VE KAYDET (zincire sokulmak uzere)
    en = max(rapor, key=lambda k: rapor[k]["kutu_sonra"])
    print(f"\nEN IYI: {en}  (kutuda {100*rapor[en]['kutu_sonra']:.1f}%)")
    vad, aad = [x.strip() for x in en.split("|")]
    vfp = dict(kaynak)[vad]
    kw = dict(ayar)[aad]
    X, Y, g, p = yukle(vfp)
    tut = ~np.isin(g, list(VG))
    m = RandomForestRegressor(n_jobs=-1, random_state=0, **kw).fit(
        X[tut], Y[tut])
    cik = {"model": m, "n_feat": N_FEAT, "maks_mm": 10.0, "yon": False,
           "hedef": "[w_perp.u, w_perp.v, g.u, g.v] -- YEREL cercevede",
           "note": (f"2026-08-14 YENIDEN EGITIM. Kaynak={vad}, ayar={aad}. "
                    f"VAL geometri gruplari ({len(VG)}) egitimden CIKARILDI. "
                    f"maks_mm 3->10 (Bolum 21.53: eski model 2.87mm'yi "
                    f"asmiyordu). OOF kutuda-oran "
                    f"{100*rapor[en]['kutu_sonra']:.1f}%.")}
    with open("results/pose_head_yeni.pkl", "wb") as fh:
        pickle.dump(cik, fh)
    print("-> results/pose_head_yeni.pkl (DAGITILMADI, olcum icin)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
