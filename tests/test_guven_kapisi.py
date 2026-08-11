# -*- coding: utf-8 -*-
"""Guven kapisinin KESINLIK GARANTISI tutmali.

Saha sozu "robotun kullandigi isaretler >=0.90 kesinliktedir" bu fonksiyona
dayaniyor; hesap yanlissa soz de yanlis olur.
"""
import numpy as np

import kos_guven_kapisi as G


def _makbuz(skor, dogru):
    return {"sonuc": {"parca_kirilim": {
        f"p{i}": {"mfg": "X", "rob": [int(d), 1 - int(d), 0], "tes": [0, 0, 0],
                  "skor": [float(s)], "dogru": [int(d)]}
        for i, (s, d) in enumerate(zip(skor, dogru))}}}


def test_secilen_esik_hedef_kesinligi_TUTAR():
    rng = np.random.default_rng(0)
    s = rng.uniform(0, 1, 2000)
    d = (rng.uniform(0, 1, 2000) < s ** 1.5).astype(int)
    S, Y, n = G.cift(_makbuz(s, d))
    r = G.esik_sec(S, Y, n, 0.90)
    assert r is not None
    k = S >= r[0]
    assert Y[k].mean() >= 0.90 - 1e-9


def test_hedef_imkansizsa_None_doner():
    s = np.linspace(0, 1, 100)
    d = np.zeros(100, int)          # hicbiri dogru degil
    S, Y, n = G.cift(_makbuz(s, d))
    assert G.esik_sec(S, Y, n, 0.90) is None


def test_daha_yuksek_hedef_daha_dar_kapsama():
    rng = np.random.default_rng(1)
    s = rng.uniform(0, 1, 3000)
    d = (rng.uniform(0, 1, 3000) < s).astype(int)
    S, Y, n = G.cift(_makbuz(s, d))
    a = G.esik_sec(S, Y, n, 0.70)
    b = G.esik_sec(S, Y, n, 0.90)
    assert a is not None and b is not None
    assert (S >= b[0]).mean() <= (S >= a[0]).mean()
