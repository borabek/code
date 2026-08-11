# -*- coding: utf-8 -*-
"""H1 KIMLIK AYRISTIRMA HATASI -- duzeltildi, bir daha geri gelmesin.

BULUNDU (2026-08-04, dataset-4 entegrasyonunda): `big_arbiter` parca kimligini
`basename.split("_")[0]` ile cikariyordu. Iki yerde kiriliyordu:
  * URETICI kodunda alt cizgi (`A-B_N.1492-H4_...`) -> head "A-B", nokta yok, pid BOS
    -> 291 dosya korpusa ASLA giremezdi (STEP'i gelse bile)
  * PARCA numarasinda alt cizgi (`ELMEX.KUT16_GY_...`) -> "KUT16", "_GY" duser
    -> 35 kimlik CAKISIYORDU (14 farkli parca tek kimlige dusuyordu)

ETKI OLCULDU (duzeltmeden ONCE): cakisan 35 kimligin HICBIRINDE STEP yoktu, yani
yanlis JSON<->STEP eslesmesi OLMAMISTI; mevcut olcumler zarar gormemisti. Duzeltme
sonrasi eslesme 1926 -> 1926, olcum kumesi 194/194 korundu.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("BA_ALLOW_SEEN", "1")


@pytest.fixture(scope="module")
def K():
    import korpus_kimlik
    return korpus_kimlik


@pytest.mark.parametrize("dosya,bek_mfg,bek_pid", [
    ("A-B_N.1492-H4_ElectricalTerminal_ElectricalTerminal.json", "A-B_N", "1492-H4"),
    ("ELMEX.KUT16_GY_ElectricalTerminal_ElectricalTerminal.json", "ELMEX", "KUT16_GY"),
    ("PXC.3271055_ElectricalTerminal_ElectricalTerminal.json", "PXC", "3271055"),
    ("WEI.2847370000_ElectricalTerminal_ElectricalTerminal.json", "WEI", "2847370000"),
    ("CWT.CX2.5_4BU_ElectricalTerminal_ElectricalTerminal.json", "CWT", "CX2.5_4BU"),
])
def test_kimlik_alt_cizgiyi_bozmaz(K, dosya, bek_mfg, bek_pid):
    assert K.kimlik(dosya) == (bek_mfg, bek_pid)


def test_eski_ayristirma_gercekten_bozuktu(K):
    """Regresyon nobetcisi: eski kural bu iki durumu KAYBEDIYORDU."""
    assert K.kimlik_eski("A-B_N.1492-H4_ElectricalTerminal_x.json")[1] == "", \
        "eski kural artik bozuk degilse bu test guncellenmeli"
    assert K.kimlik_eski("ELMEX.KUT16_GY_ElectricalTerminal_x.json")[1] == "KUT16"


@pytest.mark.parametrize("yol,bek", [
    ("all_wscad_stp/wscaduniverse_3271055_2026-07-13-15-40-44.stp", "3271055"),
    ("all_wscad_stp/wscaduniverse_KUDD4D1_GY_2026-07-30-15-44-00.stp", "KUDD4D1_GY"),
    ("all_wscad_stp/wscaduniverse_2206-1671_1000-849_2026-07-13-15-42-03.stp",
     "2206-1671_1000-849"),
])
def test_step_kimlik(K, yol, bek):
    assert K.step_kimlik(yol) == bek


def test_olcum_kumesi_korunuyor():
    """Kimlik degisimi OLCUM KUMESINI oynatmamali -- manset buna bagli."""
    from big_arbiter import eligible
    import olcum_kumesi as OK
    uy = {p for m, p, j, s in eligible()}
    D, _ = OK.kume("results/_der_tam.pkl")
    olcum = {r["pid"] for r in D}
    assert olcum <= uy, f"olcum kumesinden {len(olcum - uy)} parca kayboldu: {sorted(olcum - uy)[:5]}"
