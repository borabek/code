# -*- coding: utf-8 -*-
"""kume_baglami: elle hesaplanabilir kucuk sahnede dogrulama."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import kume_baglami as KB  # noqa: E402

A = {a: i for i, a in enumerate(KB.OZ_AD)}


def _sahne():
    """3 aday: 0 -> 3 secenek, 1 -> 2 secenek, 2 -> 1 secenek."""
    idx = np.array([0, 0, 0, 1, 1, 2])
    P = np.array([[0., 0, 0]] * 3 + [[2., 0, 0]] * 2 + [[50., 0, 0]])
    D = np.array([[0., 0, 1], [0, 1., 0], [1., 0, 0],
                  [0., 0, 1], [0, 0, -1.],
                  [0., 0, 1]])
    s = np.array([0.9, 0.4, 0.1, 0.7, 0.2, 0.3])
    return P, D, idx, s


def test_aday_ici_sira_ve_fark():
    P, D, idx, s = _sahne()
    X = KB.oznitelik(P, D, idx, s, diag=100.0)
    # aday 0'in secenekleri: skorlar 0.9/0.4/0.1 -> sira 0/1/2
    assert list(X[:3, A["aday_sira"]]) == [0, 1, 2], X[:3, A["aday_sira"]]
    # fark = skor - adayin en iyisi
    assert np.allclose(X[:3, A["aday_fark"]], [0.0, -0.5, -0.8])
    # aday 1: 0.7/0.2
    assert list(X[3:5, A["aday_sira"]]) == [0, 1]
    assert np.allclose(X[3:5, A["aday_fark"]], [0.0, -0.5])


def test_aday_secenek_sayisi():
    P, D, idx, s = _sahne()
    X = KB.oznitelik(P, D, idx, s, diag=100.0)
    assert list(X[:, A["aday_secenek"]]) == [3, 3, 3, 2, 2, 1]


def test_en_iyi_ile_iliski():
    P, D, idx, s = _sahne()
    X = KB.oznitelik(P, D, idx, s, diag=100.0)
    # en iyi secenek 0. satir (skor 0.9): kendine mesafe 0, aci 0
    assert X[0, A["en_iyi_uzak"]] == 0.0
    assert X[0, A["en_iyi_aci"]] == 0.0
    # 5. satir 50mm otede, kosegen 100 -> 0.5
    assert abs(X[5, A["en_iyi_uzak"]] - 0.5) < 1e-9
    # 1. satir yonu +y, en iyi +z -> 90 derece -> 0.5
    assert abs(X[1, A["en_iyi_aci"]] - 0.5) < 1e-9


def test_yerel_rakip():
    P, D, idx, s = _sahne()
    X = KB.oznitelik(P, D, idx, s, diag=100.0)
    # aday konumlari: 0 -> x=0 (en iyi 0.9), 1 -> x=2 (0.7), 2 -> x=50 (0.3)
    # aday 0'in 5mm icinde daha iyi rakibi YOK
    assert X[0, A["rakip_5mm"]] == 0
    # aday 1'in 5mm icinde aday 0 var ve DAHA IYI -> 1
    assert X[3, A["rakip_5mm"]] == 1
    # aday 2 uzakta -> 0
    assert X[5, A["rakip_5mm"]] == 0


def test_ardisik_olmayan_aday_indeksi():
    """`idx` 0..k-1 olmak zorunda degil; yeniden etiketleme calismali."""
    P, D, idx, s = _sahne()
    X1 = KB.oznitelik(P, D, idx, s, diag=100.0)
    X2 = KB.oznitelik(P, D, idx * 7 + 13, s, diag=100.0)
    assert np.allclose(X1, X2)


def test_bos_girdi():
    X = KB.oznitelik(np.zeros((0, 3)), np.zeros((0, 3)), np.array([], int),
                     np.array([]), 1.0)
    assert X.shape == (0, len(KB.OZ_AD))


if __name__ == "__main__":
    for ad, f in sorted(globals().items()):
        if ad.startswith("test_"):
            f()
            print(f"  GECTI  {ad}")
    print("hepsi gecti")
