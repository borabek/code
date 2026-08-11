# -*- coding: utf-8 -*-
"""MAX_SEC TAVANI BAGLIYOR MU? Aday basina secenek tavani yonlu recall'u kirpiyor mu?

BULGU (2026-08-12). Yeni korpusta aday basina secenek ortalamalari
seg 11.80 / B-rep 10.79 / mesh 11.17 -- hepsi `yon_bankasi.MAX_SEC = 12`
tavaninin DIBINDE. Yani 256 isinlik yelpaze yeni yon EKLEYEMIYOR, yalnizca
mevcut yonleri ITIYOR. Oyleyse yelpaze cozunurlugunu artirmak KAPI A'yi
gecirmeye yetmez; tavani artirmak gerekir.

BU SONDA. Ayni parcalarda secenek tavanini degistirip YONLU RECALL'u olcer.
Kabul kutusu urun metrigiyle ayni: yanal <= K.YANAL, ISARETLI aci <= K.ACI.
Secici YOK -- bu bir TAVAN olcumu (mukemmel secici varsayimi).

Maliyet dogrusaldir: tavan iki katina cikarsa kademe-2 satir sayisi da yaklasik
iki katina cikar. Bu yuzden karar "recall ne kadar artiyor" ile verilir.

Kullanim:  MS_N=40 python sonda_max_sec.py
"""
import json
import os
import sys
import time

import numpy as np

os.environ.setdefault("BA_ALLOW_SEEN", "1")
sys.path.insert(0, ".")

OZ = "results/_tam_oz"
KUME = {"d6": ("results/_d6_silindirler.pkl", "results/_p1_olasilik"),
        "tam": ("results/_brepegit_silindirler.pkl",
                "results/_p1_olasilik_brepegit")}
TAVANLAR = tuple(int(x) for x in
                 os.environ.get("MS_TAVANLAR", "12,24,40").split(","))


def main():
    import pickle

    import trimesh

    import connector3d
    import havuz_seyrelt
    import kanonik_d7 as K
    import yon_bankasi as YB
    from kos_p6_ortak import kayitlar

    on = os.environ.get("MS_ON", "d6")
    n_parca = int(os.environ.get("MS_N", "40"))
    sil_y, ob = KUME[on]
    cy = pickle.load(open(sil_y, "rb"))
    # MARKA SUZGECI SART: dosyalar pid'e gore sirali oldugu icin ilk N dosya
    # TEK MARKADAN gelir. Ilk kosuda orneklem UPUN/SUPU idi -- oysa tavani
    # asagi ceken marka NIT (D6 GT'sinin %45.7'si, yonlu recall 0.5254 ve
    # kaybi tam olarak YON kaybi: konum 0.7807 -> yonlu 0.5254). Yanlis
    # kumede olculen bir sonda, kolu haksiz yere "olu" ilan ettirir.
    hepsi = sorted(f for f in os.listdir(OZ)
                   if f.startswith(on + "_") and f.endswith(".npz"))
    tum_pid = [f[len(on) + 1:-4] for f in hepsi]
    kay_hepsi = kayitlar(tum_pid)
    hedef = os.environ.get("MS_MARKA", "").strip()
    if hedef:
        cift_ = [(f, p) for f, p in zip(hepsi, tum_pid)
                 if (kay_hepsi.get(p) or {}).get("mfg") == hedef]
        print(f"MARKA SUZGECI '{hedef}': {len(cift_)} parca bulundu",
              flush=True)
    else:
        cift_ = list(zip(hepsi, tum_pid))
    cift_ = cift_[:n_parca]
    fs = [a for a, _ in cift_]
    pidler = [b for _, b in cift_]
    kay_gt = kay_hepsi

    os.environ["YB_FAN"] = os.environ.get("YB_FAN", "256")
    YB.FAN_N = int(os.environ["YB_FAN"])
    print(f"{on}: {len(fs)} parca | yelpaze {YB.FAN_N} | tavanlar {TAVANLAR}",
          flush=True)

    top = {t: [0, 0, 0] for t in TAVANLAR}   # [gt, yakalanan, secenek]
    t0 = time.time()
    for i, (f, pid) in enumerate(zip(fs, pidler), 1):
        r = kay_gt.get(pid)
        if not r or not len(r.get("G", [])):
            continue
        mf = f"{ob}/{pid}.npz"
        if not os.path.exists(mf):
            continue
        z = np.load(f"{OZ}/{f}")
        kay = np.asarray(z["kaynak"], int)
        m = np.isin(kay, (0, 1, 2))
        if int(m.sum()) < 2:
            continue
        zz = np.load(mf)
        V = np.ascontiguousarray(zz["V"], np.float64)
        Fc = np.ascontiguousarray(zz["F"], np.int64)
        # mesh adaylarini urunle AYNI sekilde seyrelt
        if int((kay == 2).sum()):
            pb = np.mean([np.asarray(q, float) for q in zz["pbs"]], axis=0)
            pp = havuz_seyrelt.ppos(pb, connector3d.CABLE_ENTRY,
                                    connector3d.CONTACT)
            i2 = np.where(kay == 2)[0]
            P2 = np.asarray(z["P"], float)[i2]
            s2 = (pp[np.argmin(np.linalg.norm(
                P2[:, None, :] - V[None, :, :], axis=-1), axis=1)]
                if len(P2) * len(V) < 6e7 else np.zeros(len(P2)))
            n01 = int(np.isin(kay, (0, 1)).sum())
            tut = np.zeros(len(i2), bool)
            tut[havuz_seyrelt.seyrelt(P2, s2, n01, 2.5, 350, 5)] = True
            m2 = m.copy()
            m2[i2] = tut
            m = m2
        P = np.asarray(z["P"], float)[m]
        D = np.asarray(z["D"], float)[m]
        diag = float(np.linalg.norm(V.max(0) - V.min(0)))
        mesh = trimesh.Trimesh(V, Fc, process=False)
        fmask = (kay[m] != 2)
        G = np.asarray(r["G"], float)
        Gd = np.asarray(r["Gd"], float)
        dg = float(r["diag"])
        for t in TAVANLAR:
            idx, YD, _C = YB.secenekler(P, D, cy.get(str(pid)), V, mesh=mesh,
                                        diag=diag, fan_maske=fmask, max_sec=t)
            if not len(idx):
                top[t][0] += len(G)
                continue
            PO = P[idx]
            # YONLU TAVAN: her GT icin kabul kutusuna giren BIR secenek var mi
            d_ = PO[:, None, :] - G[None, :, :]
            al = (d_ * Gd[None, :, :]).sum(-1)
            pe = np.linalg.norm(d_ - al[..., None] * Gd[None, :, :], axis=-1)
            cc = YD @ Gd.T
            an = np.degrees(np.arccos(np.clip(cc, -1, 1)))     # ISARETLI
            kabul = (pe <= K.YANAL) & (an <= K.ACI) & (np.abs(al) <= 40.0)
            top[t][0] += len(G)
            top[t][1] += int(kabul.any(0).sum())
            top[t][2] += len(idx)
        if i % 10 == 0:
            print(f"  {i}/{len(fs)} ({time.time() - t0:.0f} s)", flush=True)

    print(f"\n{'tavan':>7}{'GT':>7}{'yakalanan':>11}{'yonlu recall':>14}"
          f"{'secenek':>10}{'maliyet x':>11}")
    taban = None
    out = {}
    for t in TAVANLAR:
        gt, ya, se = top[t]
        rec = ya / max(gt, 1)
        taban = se if taban is None else taban
        out[str(t)] = {"gt": gt, "yakalanan": ya, "yonlu_recall": rec,
                       "secenek": se, "maliyet_kat": se / max(taban, 1)}
        print(f"{t:>7d}{gt:>7d}{ya:>11d}{rec:>14.4f}{se:>10d}"
              f"{se / max(taban, 1):>11.2f}")
    # MAKBUZ ADI ORNEKLEMI TASIR: sabit ad kullanilinca iki farkli orneklem
    # (UPUN/SUPU ve NIT) ayni dosyayi ezip birbirinin sonucu sanildi.
    _yol = (f"results/max_sec_sondasi_{hedef or 'karisik'}"
            f"{len(fs)}.json")
    json.dump({"on": on, "n_parca": len(fs), "marka": hedef or "karisik",
               "yelpaze": YB.FAN_N,
               "sonuc": out,
               "not": "Yonlu TAVAN (mukemmel secici). Kabul kutusu urun "
                      "metrigiyle ayni. D7'ye BAKILMADI."},
              open(_yol, "w"), indent=1)
    print(f"\nmakbuz -> {_yol}")


if __name__ == "__main__":
    main()
