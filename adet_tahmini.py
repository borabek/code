# -*- coding: utf-8 -*-
"""CP ADEDINI GEOMETRIDEN TAHMIN ET — metadata YOK.

NEDEN. `robot_cp.extract_highcp` yogun parcada taban yoldan cok daha iyi
(OOF F1 0.807) ama **`cp_count` ISTER** -- ureticinin CP sayisi, yani
METADATA. Kod okundu: `cp_count` yalnizca IKI yerde kullaniliyor:
  1. `highcp_selector.rank_feats` icinde `nratio = N/n` (tek oznitelik)
  2. son `[:N]` kirpmasi
Havuz uretimi ve lattice/gate oznitelikleri **N'den bagimsiz**. Yani
metadata bagimliligini kaldirmak icin tek gereken bir **N tahmini**.

FIKIR. Klemensin CP'leri DUZENLI BIR IZGARADA durur. Izgara adimini
(pitch) ve yayilimi olcersen site sayisini sayabilirsin -- bu tamamen
GEOMETRIKTIR, hicbir kunye bilgisi gerektirmez. Gereken makine zaten
`highcp_selector.lattice_feats` icinde var (SVD ile yuzey eksenleri +
`gridfit` pitch tahmini); burada adet tahminine uyarlaniyor.

ILK KAPI (bu dosyanin `__main__`'i). Tahminci GT NOKTALARINDAN adedi geri
bulabiliyor mu? Bulamiyorsa gurultulu adaydan hic bulamaz. Bu kontrol
CIKARIM GEREKTIRMEZ, saniyeler surer ve kolu bosuna kosmaktan kurtarir.
"""
import os
import sys

import numpy as np


def _eksenler(P):
    """noktalarin yayildigi iki ana eksen (SVD) + izdusumler."""
    Q = P - P.mean(0)
    _, _, Vt = np.linalg.svd(Q, full_matrices=False)
    u, v = Vt[0], Vt[1]
    return P @ u, P @ v


def _pitch(pp, span, min_oran=0.02):
    """bir eksende IZGARA ADIMI: pozitif bosluklarin ortancasi."""
    pos = np.sort(pp)
    g = np.diff(pos)
    g = g[g > min_oran * span]
    if not len(g):
        return None
    return float(np.median(g))


def tahmin_et(P, ws=None, min_n=1):
    """(n,3) aday/nokta kumesinden CP ADEDINI tahmin et. metadata YOK.

    Yontem: iki ana eksende izgara adimi bulunur, yayilim/adim ile her
    eksendeki site sayisi cikarilir, ikisinin CARPIMI site sayisidir.
    Tek sirali parcalarda ikinci eksende adim bulunamaz -> 1 alinir.
    Doner: (N_tahmin, teshis_sozlugu)
    """
    P = np.asarray(P, float).reshape(-1, 3)
    n = len(P)
    if n < 2:
        return max(n, min_n), {"sebep": "cok az nokta"}
    pu, pv = _eksenler(P)
    span_u = float(pu.max() - pu.min())
    span_v = float(pv.max() - pv.min())
    span = max(span_u, span_v, 1.0)
    hu = _pitch(pu, span)
    hv = _pitch(pv, span)
    # her eksende site sayisi = yayilim / adim + 1
    nu = int(round(span_u / hu)) + 1 if hu else 1
    nv = int(round(span_v / hv)) + 1 if hv else 1
    # DEJENERE DURUM: bir eksende yayilim adimdan kucukse tek sira demektir
    if hu and span_u < 0.5 * hu:
        nu = 1
    if hv and span_v < 0.5 * hv:
        nv = 1
    N = max(int(nu * nv), min_n)
    return N, {"nu": nu, "nv": nv, "pitch_u": hu, "pitch_v": hv,
               "span_u": span_u, "span_v": span_v, "n_nokta": n}


def _kapi():
    """ILK KAPI: GT noktalarindan adedi geri bulabiliyor muyuz?"""
    os.environ.setdefault("BA_ALLOW_SEEN", "1")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import kanonik_d7 as K
    import d6_kayit
    kay = K.yukle()
    kay.update({str(p): r for p, r in d6_kayit.yukle().items()
                if str(p) not in kay})
    ger, tah, yogun_ger, yogun_tah = [], [], [], []
    for pid, r in kay.items():
        G = np.asarray(r["G"], float).reshape(-1, 3)
        if len(G) < 1:
            continue
        N, _ = tahmin_et(G)
        ger.append(len(G))
        tah.append(N)
        if len(G) >= 11:                       # yogun rejim (hicp_threshold)
            yogun_ger.append(len(G))
            yogun_tah.append(N)
    ger = np.array(ger, float)
    tah = np.array(tah, float)
    yg = np.array(yogun_ger, float)
    yt = np.array(yogun_tah, float)
    print(f"GT'den adet tahmini -- {len(ger)} parca")
    print(f"  TAM isabet        : {100*(tah == ger).mean():5.1f}%")
    print(f"  +/-1 icinde       : {100*(np.abs(tah-ger) <= 1).mean():5.1f}%")
    print(f"  +/-2 icinde       : {100*(np.abs(tah-ger) <= 2).mean():5.1f}%")
    print(f"  ortanca mutlak hata: {np.median(np.abs(tah-ger)):.1f} CP")
    if len(yg):
        print(f"\nYOGUN parcalar (GT>=11) -- {len(yg)} parca  <- ASIL HEDEF")
        print(f"  TAM isabet        : {100*(yt == yg).mean():5.1f}%")
        print(f"  +/-1 icinde       : {100*(np.abs(yt-yg) <= 1).mean():5.1f}%")
        print(f"  +/-2 icinde       : {100*(np.abs(yt-yg) <= 2).mean():5.1f}%")
        print(f"  ortanca mutlak hata: {np.median(np.abs(yt-yg)):.1f} CP")
        print(f"  ortanca goreli hata: "
              f"{100*np.median(np.abs(yt-yg)/yg):.1f}%")


if __name__ == "__main__":
    _kapi()
