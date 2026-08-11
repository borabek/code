# -*- coding: utf-8 -*-
"""F0-1 PROTOKOL BEKCISI TESTLERI -- 'sessiz gecis' bir daha olmasin.

Bu testler iki seyi PIN'ler:
  1. SIZINTI: LOCKED geometri gruplarindaki bir parca egitim kumesine girerse
     `protokol.dogrula()` HATA FIRLATMALI. (2026-08-03'te dagitilan gate LOCKED'in
     77/95'ini gormustu ve hicbir sey uyarmadi.)
  2. TEZ DEGISMEZLERI: 34 maddelik liste boyunca ag/remesh/CP-tanimi degismemeli.
     Degisirse bu test kirmizi yanar -- soz degil, olcum.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("BA_ALLOW_SEEN", "1")


@pytest.fixture(scope="module")
def P():
    import protokol
    return protokol


def test_locked_gruplari_bos_degil(P):
    assert len(P.yasak_gruplar()) > 0, "LOCKED grubu bos -- sinav korumasiz"


def test_bekci_kirli_parcayi_yakalar(P):
    import olcum_kumesi as OK
    gk = OK.geo_anahtarlari()
    lg = sorted(P.yasak_gruplar())
    kirli = [p for p, g in gk.items() if g == lg[0]]
    assert kirli, "LOCKED grubuna ait parca yok"
    assert not P.egitim_maskesi(kirli).any()
    with pytest.raises(AssertionError):
        P.dogrula(kirli, ad="_pytest_kirli", sert=True)


def test_bekci_temiz_parcayi_elemez(P):
    import olcum_kumesi as OK
    yg = P.yasak_gruplar()
    temiz = [p for p, g in OK.geo_anahtarlari().items() if g not in yg][:20]
    assert temiz
    assert P.egitim_maskesi(temiz).all()


def test_olcum_da_yasak_daha_genis(P):
    a = P.yasak_gruplar(olcum_da=False)
    b = P.yasak_gruplar(olcum_da=True)
    assert a <= b and len(b) > len(a), "olcum_da=True yasak kumeyi genisletmeli"


def test_tez_degismezleri_yerinde(P):
    """AG (4 ckpt) / ORGU (~6000 uniform) / CP TANIMI (v_o) degismedi mi?"""
    sapma = P.tez_dogrula(sert=False)
    assert not sapma, f"TEZDEN SAPMA: {sapma}"


def test_selftest(P):
    assert P._selftest() is True
