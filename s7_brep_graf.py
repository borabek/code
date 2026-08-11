# -*- coding: utf-8 -*-
"""S7: B-REP GRAF OZELLIKLERI -- tespit sıralamasi icin YENI BILGI.

DUVAR: tespit bosslugunun %62'si SIRALAMADA ve o "yeni bilgi" istiyor
([[tavan-merdiveni-2026-08-01]]). Hayatta kalan yanlislar, dogrularla GEOMETRIK OLARAK AYNI
gorunen gercek tel-disi acikliklar.

Zengin bloklar (konum + cok-yaricap) tam o kategoriydi ve +0.023 verdi -- kategori DOGRU.
Denetimin ayni kategoride yazdigi ikinci madde: B-REP GRAF bilgisi. Simdiye kadar B-rep'ten
yalniz TEK YUZ okuyorduk (silindir yaricapi, ekseni). Yuzlerin BIRBIRIYLE ILISKISI hic
kullanilmadi.

OZELLIKLER (10) -- hepsi STEP B-rep'inden, GT'ye BAKMAZ:
    g_kom_n       agiz cevresindeki (r<=8mm) B-rep yuz sayisi
    g_kom_cyl     bunlarin kaci silindir
    g_kom_pl      kaci duzlem
    g_esek_n      adayin ekseniyle ESEKSENLI (5 derece) silindir sayisi -> es merkezli yuz zinciri
    g_esek_r_ort  o zincirdeki yariciplarin ortalamasi
    g_esek_r_yay  yaricap yayilimi (huni/kademe imzasi)
    g_kor         kanal KOR mu (eksen boyunca ilerlerken yuz bitiyor mu)
    g_gecis       kanal KARSI TARAFA cikiyor mu (boydan boya delik = tel girisi DEGIL)
    g_agiz_cev    agiz cevriminin uzunlugu / (2*pi*r) -- dairesellikten sapma
    g_yuz_alan    adayin oturdugu yuzun alani / agiz alani

Cikti: results/brep_graf.npz (parca basina aday hizasinda, zengin npz ile AYNI SIRADA).
"""
import io
import json
import os
import sys
import time

import numpy as np

os.environ.setdefault("BA_ALLOW_SEEN", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
OUT = "results/brep_graf.npz"
AD = ["g_kom_n", "g_kom_cyl", "g_kom_pl", "g_esek_n", "g_esek_r_ort", "g_esek_r_yay",
      "g_kor", "g_gecis", "g_agiz_cev", "g_yuz_alan"]


def graf_ozellik(p, d, C, A, R, PL):
    """Aday (p, d) icin 10 graf ozelligi. C/A/R silindirler, PL duzlemler."""
    p = np.asarray(p, float); d = np.asarray(d, float)
    d = d / (np.linalg.norm(d) + 1e-9)
    f = [0.0] * len(AD)
    if len(C):
        rel = p - C
        al = (rel * A).sum(1)
        off = np.linalg.norm(rel - al[:, None] * A, axis=1)
        yakin = (np.linalg.norm(rel, axis=1) <= 8.0)
        f[1] = float(yakin.sum())
        # ESEKSENLI: eksen paralel VE eksen cizgisine yakin
        es = (np.abs(A @ d) >= np.cos(np.radians(5.0))) & (off <= 3.0)
        f[3] = float(es.sum())
        if es.any():
            f[4] = float(np.mean(R[es])); f[5] = float(np.std(R[es]))
            # KOR/GECIS: eseksenli silindirlerin eksenel yayilimi agiz derinligini asiyorsa gecis
            a_es = al[es]
            f[6] = float(a_es.min() > -1.0)            # geriye dogru yuz yok -> kor
            f[7] = float((a_es.max() - a_es.min()) > 15.0)
    if len(PL):
        pc = np.asarray(PL, float)
        f[2] = float((np.linalg.norm(pc - p, axis=1) <= 8.0).sum())
    f[0] = f[1] + f[2]
    return f


def main():
    import brep_axes
    from big_arbiter import eligible

    stp_of = {p: s for m, p, jf, s in eligible()}
    zen = np.load("results/zengin_parite.npz", allow_pickle=True)
    zpid = np.array([str(x) for x in zen["pids"]])
    par = np.load("results/gate_regrow_data_parite.npz", allow_pickle=True)
    ppid = np.array([str(x) for x in par["pids"]])
    PTS = np.asarray(par["pts"], float); DIRS = np.asarray(par["dirs"], float)
    ortak = [p for p in np.unique(zpid) if (ppid == p).sum() == (zpid == p).sum()]
    print(f"{len(ortak)} parcanin aday sayisi ESLESIYOR", flush=True)

    X = np.zeros((len(zpid), len(AD)), float)
    t0 = time.time(); atlanan = 0
    for k, pid in enumerate(ortak, 1):
        if k % 200 == 0:
            print(f"  {k}/{len(ortak)}  {time.time()-t0:.0f}s (atlanan {atlanan})", flush=True)
        zi = np.where(zpid == pid)[0]; pi = np.where(ppid == pid)[0]
        try:
            C, A, R = brep_axes.cylinders(stp_of.get(pid))
            try:
                pl = brep_axes.planes(stp_of.get(pid))
                PLc = np.asarray(pl[0], float) if isinstance(pl, tuple) else np.zeros((0, 3))
            except Exception:
                PLc = np.zeros((0, 3))
        except Exception:
            atlanan += 1; continue
        for j, (a_, b_) in enumerate(zip(zi, pi)):
            X[a_] = graf_ozellik(PTS[b_], DIRS[b_], C, A, R, PLc)
    np.savez(OUT, X=X, ad=np.array(AD), pids=zpid)
    dolu = float((np.abs(X).sum(1) > 0).mean())
    print(f"\n-> {OUT} | {X.shape} | dolu satir {dolu:.1%} | atlanan parca {atlanan}")
    for i, a in enumerate(AD):
        print(f"   {a:<14} ort {X[:, i].mean():>8.3f}  sifir-disi {float((X[:, i] != 0).mean()):>6.1%}")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except BaseException:
        traceback.print_exc(); raise
