# -*- coding: utf-8 -*-
"""`sec_ayrintili`nin BUTUN cikis dallari AYNI sayida deger dondurmeli.

NEDEN VAR: skor alani eklendiginde normal dal 4'e cikti, erken cikis dali 3'te
kaldi ve D7 sinav okumasinin P6 kolu "expected 4, got 3" ile coktu. Cikis
dallari imza degisikliginde gozden kaciyor.
"""
import numpy as np

import p6_karar


def _girdi():
    P = np.array([[0.0, 0, 0], [10, 0, 0], [20, 0, 0]])
    idx = np.array([0, 1, 2])
    YD = np.tile([0.0, 0, 1], (3, 1))
    return P, idx, YD


def test_erken_cikis_ve_normal_dal_AYNI_sayida_deger_dondurur():
    P, idx, YD = _girdi()
    bos = p6_karar.sec_ayrintili(P, idx, YD, np.zeros(3), 0.9)
    dolu = p6_karar.sec_ayrintili(P, idx, YD, np.ones(3), 0.1)
    assert len(bos) == len(dolu) == 4


def test_sec_govdesizdir_ve_ayrintiliyla_AYNI_secimi_verir():
    rng = np.random.default_rng(0)
    P = rng.uniform(-20, 20, (30, 3))
    idx = rng.integers(0, 30, 200)
    YD = rng.normal(size=(200, 3))
    YD /= np.linalg.norm(YD, axis=1, keepdims=True)
    s = rng.uniform(0, 1, 200)
    for kural in (0.3, ("mutlak", 0.5), ("goreli", 0.7, 0.2)):
        a = p6_karar.sec(P, idx, YD, s, kural)
        b = p6_karar.sec_ayrintili(P, idx, YD, s, kural)
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_skor_alani_secilen_secenegin_skorunu_tasir():
    P, idx, YD = _girdi()
    s = np.array([0.2, 0.9, 0.5])
    _, _, ai, sc = p6_karar.sec_ayrintili(P, idx, YD, s, 0.1, nms_mm=0.0)
    assert len(sc) == len(ai)
    assert np.isclose(sc[0], s.max())          # en yuksek skor ONCE secilir
