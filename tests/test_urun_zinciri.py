# -*- coding: utf-8 -*-
"""Olcum betikleri urunun TAM zincirini kosmali.

2026-08-07'de bulundu: butun D6/D7 olcumleri `adaylari_uret` + gate ile duruyordu;
urunun gercek yolu gate'ten sonra `pose_duzelt` + `yon_sozluk_sec` de kosuyor.
D6'da olculdu: 0.2103 -> 0.2251 -> 0.2469.
"""
import io
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import urun_zinciri


def test_bos_girdi_bos_doner():
    P = np.zeros((0, 3)); D = np.zeros((0, 3))
    P2, D2 = urun_zinciri.tam_poz(None, None, None, P, D)
    assert len(P2) == 0 and len(D2) == 0


def test_adim_duserse_GIRDI_aynen_doner():
    """Uydurma YOK: bir adim patlarsa girdi degismeden geri gelmeli."""
    P = np.array([[1., 2, 3]]); D = np.array([[0., 0, 1]])
    P2, D2 = urun_zinciri.tam_poz(None, None, None, P, D)   # feats_for patlar
    assert np.allclose(P2, P) and np.allclose(D2, D)


def test_zincir_sirasi_pose_sonra_yon():
    import inspect
    src = inspect.getsource(urun_zinciri.tam_poz)
    assert src.index("pose_duzelt") < src.index("yon_sozluk_sec"), \
        "yon sozlugu, poz duzeltmesinin CIKTISI uzerine calismali"
