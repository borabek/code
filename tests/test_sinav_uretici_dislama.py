# -*- coding: utf-8 -*-
"""SINAV URETICISI EGITIME GIREMEZ -- 2026-08-05'te bulunan kirlilige karsi bekci.

BULGU: sinav kumesi "GORULMEMIS URETICI sinavi" olarak tanimlanmisti, ama dislama yalniz
PARCA + IKIZ duzeyindeydi. Korpus buyudukce ayni ureticilerin BASKA parcalari egitime
girdi. Olculdu:

  * gate korpusu (`zengin_parite_v3`): sinav ureticilerinden **1502 tekil parca**
    (CWT 691, A-B 358, WIE 226, WEG 73, KLM 57, CCD 44, DIN 39, ELMEX 9, DEG 5)
  * seg oto-etiket korpusu (`_label_auto_obj`): sinav ureticilerinden **958 parca**
    VE sinav kumesinin 250 parcasindan **169'u DOGRUDAN** (%68)

Yani kumenin ADI yalan olmustu ve o kumede olculen her sayi sisikti.
[[locked-sinav-kirliligi-yakalandi]] ile birebir ayni desen.

Bu dosya iki seyi kilitler:
 1. gate korpusu kurucusu uretici duzeyinde de eliyor mu (kod duzeyi)
 2. uretilmis korpus dosyasinda sinav ureticisi VAR MI (veri duzeyi)
"""
import io
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SINAV = "results/d5_4_sinav_kumesi.json"


def _sinav():
    if not os.path.exists(SINAV):
        pytest.skip("sinav kumesi yok")
    return json.load(io.open(SINAV, encoding="utf-8"))


def test_gate_korpusu_uretici_duzeyinde_de_eliyor():
    """KOD duzeyi: parca kimligi YETMEZ, uretici de elenmelidir."""
    src = io.open("d5_gate_korpus.py", encoding="utf-8").read()
    assert "SINAV_MFG" in src, "uretici duzeyi dislama YOK"
    assert 'atlanan["sinav_ureticisi"]' in src, "uretici elemesi sayilmali (sessiz olmasin)"
    i_pid = src.index('atlanan["sinav_kumesi"]')
    i_mfg = src.index('atlanan["sinav_ureticisi"]')
    assert abs(i_mfg - i_pid) < 400, "iki eleme AYNI kapida olmali"


def test_sinav_kumesinin_ureticileri_kayitli():
    sv = _sinav()
    assert sv.get("uretici"), "kume kendi ureticilerini KAYDETMELI (dislama buna dayaniyor)"


@pytest.mark.parametrize("korpus", ["results/zengin_parite_v5.npz"])
def test_uretilmis_korpusta_sinav_ureticisi_YOK(korpus):
    """VERI duzeyi: bekci calissa bile ciktiyi DOGRULA (kod dogru, dosya bayat olabilir)."""
    import numpy as np
    if not os.path.exists(korpus):
        pytest.skip(f"{korpus} henuz uretilmedi")
    sv = _sinav()
    SU = set(sv["uretici"])
    d = np.load(korpus, allow_pickle=True)
    kirli = sorted({str(m) for m in np.asarray(d["mfg"]) if str(m) in SU})
    assert not kirli, f"SINAV URETICISI egitim korpusunda: {kirli}"
    SP = set(sv["pidler"])
    ort = SP & {str(p) for p in np.asarray(d["pids"])}
    assert not ort, f"SINAV PARCASI egitim korpusunda: {sorted(ort)[:5]}"


def test_IKI_sinav_kumesi_de_dislaniyor():
    """d6 kumesi eklendiginde eski blok SINAV_MFG'yi EZIYORDU (atama vs birlesim).

    Belirti sessizdi: gate korpusunda SE 74 / UTL 36 / SUPU 24 aday belirdi, yani
    yeni sinavin ureticileri egitime girdi. Testin isi: atama bir daha yazilmasin.
    """
    src = io.open("d5_gate_korpus.py", encoding="utf-8").read()
    assert "d6_sinav_kumesi.json" in src, "yeni sinav kumesi dislanmiyor"
    import re
    atamalar = re.findall(r"^\s*SINAV_MFG\s*(\|?=)", src, re.M)
    assert atamalar, "SINAV_MFG hic kurulmuyor"
    # ilk kurulum `= set()` olabilir; SONRAKI her dokunus BIRLESIM olmali
    assert all(a == "|=" for a in atamalar[1:]), (
        f"SINAV_MFG uzerine YAZILIYOR ({atamalar}) -- birlesim olmali")


def test_oto_etiket_uretici_SINAVI_disliyor():
    """KIRLILIGIN KAYNAGI: g5_agiz_etiket.py sinav kumelerini HIC dislamiyordu.

    `_label_auto_obj` eski sinav kumesinin 250 parcasindan 169'unu iceriyordu; seg agi
    sinavin ucte ikisini gormustu. Bu test o kapinin kaldirilmasini engeller.
    """
    src = io.open("g5_agiz_etiket.py", encoding="utf-8").read()
    assert "d6_sinav_kumesi.json" in src and "d5_4_sinav_kumesi.json" in src, \
        "iki sinav kumesi de dislanmali"
    assert "t[0] not in _SM" in src, "URETICI duzeyi dislama yok"
    assert "t[1] not in _SP" in src, "PARCA duzeyi dislama yok"
