# -*- coding: utf-8 -*-
"""Oznitelik genisligi tuzagi -- F1'i TAM 0.0000 yapan sessiz eleme.

`d5_birlestir.py` birlesik dosyada X'i 58 -> 22 kirpar; shard'lar 58 kalir. Glob
ikisini de getirir ve `_der_yeni.pkl` siralamada ONCE gelir ('.' < '_'), dolayisiyla
`setdefault` 22'lik surumu tutar. Sonra 22*2=44 != 116 diye HER parca elenir.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import d6_kayit


def test_58lik_kayit_aynen_doner():
    r = {"X": np.zeros((3, 58)), "XR": np.zeros((3, 36))}
    assert d6_kayit.x58(r).shape == (3, 58)


def test_22lik_kayit_XR_ile_58e_tamamlanir():
    X = np.arange(3 * 22, dtype=float).reshape(3, 22)
    XR = np.arange(3 * 36, dtype=float).reshape(3, 36) + 100
    out = d6_kayit.x58({"X": X, "XR": XR})
    assert out.shape == (3, 58)
    assert np.array_equal(out[:, :22], X)
    assert np.array_equal(out[:, 22:], XR), "kuyruk XR olmali -- gate onu bekliyor"


def test_gate_genisligiyle_UYUSUR():
    """Dagitilan gate n_feat=116 = 58*2 (parca-ici z-skor genisligi ikiye katlar)."""
    r = {"X": np.zeros((5, 22)), "XR": np.zeros((5, 36))}
    assert d6_kayit.x58(r).shape[1] * 2 == 116


def test_XR_yoksa_None_doner_SESSIZ_YANLIS_YOK():
    assert d6_kayit.x58({"X": np.zeros((3, 22))}) is None
    assert d6_kayit.x58({"X": None}) is None


def test_beklenmeyen_genislik_None():
    assert d6_kayit.x58({"X": np.zeros((3, 30)), "XR": np.zeros((3, 36))}) is None


def test_yukle_SHARDLARI_ONCE_okur():
    """Birlesik dosya yalniz eksigi tamamlamali -- yoksa kirpilmis surum kazanir."""
    import inspect
    src = inspect.getsource(d6_kayit.yukle)
    assert src.index("shard = ") < src.index("birlesik = ")
    assert "shard + birlesik" in src, "shard'lar ONCE gezilmeli"
