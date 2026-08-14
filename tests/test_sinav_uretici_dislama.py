# -*- coding: utf-8 -*-
"""SINAV URETICISI EGITIME GIREMEZ -- 2026-08-05'te bulunan kirlilige karsi guard.

FINDING: exam kumesi "GORULMEMIS URETICI sinavi" as tanimlanmisti, but dislama only
PARCA + IKIZ duzeyindeydi. Korpus buyudukce same ureticilerin BASKA parcalari egitime
input. Olculdu:

  * gate korpusu (`zengin_parite_v3`): exam ureticilerinden **1502 tekil part**
    (CWT 691, A-B 358, WIE 226, WEG 73, KLM 57, CCD 44, DIN 39, ELMEX 9, DEG 5)
  * seg oto-label korpusu (`_label_auto_obj`): exam ureticilerinden **958 part**
    VE exam kumesinin 250 parcasindan **169'u DOGRUDAN** (%68)

Yani kumenin ADI yalan olmustu and that kumede olculen each number sisikti.
[[locked-exam-kirliligi-yakalandi]] with birebir same desen.

Bu file two seyi kilitler:
 1. gate korpusu kurucusu manufacturer duzeyinde de eliyor mu (kod duzeyi)
 2. uretilmis corpus dosyasinda exam ureticisi VAR MI (data duzeyi)
"""
import io 
import json 
import os 
import sys 

import pytest 

sys .path .insert (0 ,os .path .dirname (os .path .dirname (os .path .abspath (__file__ ))))

SINAV ="results/d5_4_sinav_kumesi.json"


def _sinav ():
    if not os .path .exists (SINAV ):
        pytest .skip ("exam kumesi yok")
    return json .load (io .open (SINAV ,encoding ="utf-8"))


def test_gate_korpusu_uretici_duzeyinde_de_eliyor ():
    """KOD duzeyi: part kimligi YETMEZ, manufacturer de elenmelidir."""
    src =io .open ("d5_gate_korpus.py",encoding ="utf-8").read ()
    assert "SINAV_MFG"in src ,"manufacturer duzeyi dislama YOK"
    assert 'atlanan["sinav_ureticisi"]'in src ,"manufacturer elemesi sayilmali (sessiz olmasin)"
    i_pid =src .index ('atlanan["sinav_kumesi"]')
    i_mfg =src .index ('atlanan["sinav_ureticisi"]')
    assert abs (i_mfg -i_pid )<400 ,"iki eleme AYNI kapida olmali"


def test_sinav_kumesinin_ureticileri_kayitli ():
    sv =_sinav ()
    assert sv .get ("manufacturer"),"cluster kendi ureticilerini KAYDETMELI (dislama buna dayaniyor)"


@pytest .mark .parametrize ("corpus",["results/zengin_parite_v5.npz"])
def test_uretilmis_korpusta_sinav_ureticisi_YOK (corpus ):
    """VERI duzeyi: guard calissa bile ciktiyi DOGRULA (kod correct, file bayat may be)."""
    import numpy as np 
    if not os .path .exists (corpus ):
        pytest .skip (f"{corpus } henuz uretilmedi")
    sv =_sinav ()
    SU =set (sv ["manufacturer"])
    d =np .load (corpus ,allow_pickle =True )
    kirli =sorted ({str (m )for m in np .asarray (d ["mfg"])if str (m )in SU })
    assert not kirli ,f"SINAV URETICISI training korpusunda: {kirli }"
    SP =set (sv ["pidler"])
    ort =SP &{str (p )for p in np .asarray (d ["pids"])}
    assert not ort ,f"SINAV PARCASI training korpusunda: {sorted (ort )[:5 ]}"


def test_IKI_sinav_kumesi_de_dislaniyor ():
    """d6 kumesi eklendiginde old blok SINAV_MFG'yi EZIYORDU (atama vs birlesim).

    Belirti sessizdi: gate korpusunda SE 74 / UTL 36 / SUPU 24 candidate belirdi, i.e.
    new sinavin ureticileri egitime input. Testin isi: atama a more yazilmasin.
    """
    src =io .open ("d5_gate_korpus.py",encoding ="utf-8").read ()
    assert "d6_sinav_kumesi.json"in src ,"yeni exam kumesi dislanmiyor"
    import re 
    atamalar =re .findall (r"^\s*SINAV_MFG\s*(\|?=)",src ,re .M )
    assert atamalar ,"SINAV_MFG hic kurulmuyor"
    # first kurulum `= set()` may be; SONRAKI each dokunus BIRLESIM must be
    assert all (a =="|="for a in atamalar [1 :]),(
    f"SINAV_MFG uzerine YAZILIYOR ({atamalar }) -- birlesim olmali")


def test_oto_etiket_uretici_SINAVI_disliyor ():
    """KIRLILIGIN KAYNAGI: g5_agiz_etiket.py exam kumelerini HIC dislamiyordu.

    `_label_auto_obj` old exam kumesinin 250 parcasindan 169'unu iceriyordu; seg agi
    sinavin ucte ikisini gormustu. Bu test that kapinin kaldirilmasini engeller.
    """
    src =io .open ("g5_agiz_etiket.py",encoding ="utf-8").read ()
    assert "d6_sinav_kumesi.json"in src and "d5_4_sinav_kumesi.json"in src ,"iki exam kumesi de dislanmali"
    assert "t[0] not in _SM"in src ,"URETICI duzeyi dislama yok"
    assert "t[1] not in _SP"in src ,"PARCA duzeyi dislama yok"
