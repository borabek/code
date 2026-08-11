# -*- coding: utf-8 -*-
"""MESH ADAYI SEYRELTME -- egitim de urun de BU fonksiyonu cagirir.

OLCULDU (D6, 468 parca; olcut YALNIZ KONUM recall'u / aday-parca):
    en yuksek olasilik 60      0.6362 / 160
    en yuksek olasilik 150     0.7163 / 214
    uzamsal 4.0mm, 2x120       0.8091 / 178
    uzamsal 2.5mm, 4x250       0.8713 / 251   <- SECILEN (maliyet dizi)
    uzamsal 2.0mm, sinirsiz    0.9768 / 458

KAPSAMA GUVENI YENIYOR: segmentasyon olasiligi en yuksek tepeler AYNI agiz
cevresinde kumeleniyor; "en iyi 60"i almak parcanin yarisini bos birakiyor.
Uzamsal seyreltme yuksek olasilikli tepeden baslar ve `r` yaricapinda bastirir.

Bu kural iki yerde yazilirsa olcum urunden ayrisir (bu projede iki kez oldu).
Tek kaynak burasidir.
"""
import numpy as np

R = 2.5            # seyreltme yaricapi (mm)
TABAN = 250        # en az bu kadar mesh adayi
KAT = 4            # kaynak 0+1 sayisinin kati kadar da olabilir


def cap(n01, taban=TABAN, kat=KAT):
    """Parca karmasikligiyla olceklenen ust sinir."""
    return max(int(taban), int(kat) * int(n01))


def seyrelt(P2, s2, n01, r=R, taban=TABAN, kat=KAT):
    """Mesh adaylarindan tutulacaklarin INDEKSLERI.

    `P2` mesh aday konumlari, `s2` her birinin segmentasyon olasiligi,
    `n01` mesh DISI aday sayisi (seg + B-rep).
    """
    P2 = np.asarray(P2, float).reshape(-1, 3)
    if not len(P2):
        return np.zeros(0, int)
    s2 = np.asarray(s2, float).reshape(-1)
    ust = cap(n01, taban, kat)
    sec = []
    for j in np.argsort(-s2):
        if len(sec) >= ust:
            break
        if sec and float(np.min(np.linalg.norm(P2[sec] - P2[j], axis=1))) < r:
            continue
        sec.append(int(j))
    return np.asarray(sec, int)


def ppos(avg_probs, CE, CT):
    """Tepe basina 'CP olma' olasiligi -- havuzun kendi olcutuyle AYNI."""
    a = np.asarray(avg_probs, float)
    return a[:, int(CE)] + a[:, int(CT)]
