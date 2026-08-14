# -*- coding: utf-8 -*-
"""Gate sonrasi kalabalik bastirma (NMS) testleri. See. results/nms_tarama.json."""
import os 
import sys 

import pytest 

sys .path .insert (0 ,os .path .dirname (os .path .dirname (os .path .abspath (__file__ ))))
import wire_gate # noqa: E402


def cp (x ,s ):
    return {"point":[float (x ),0.0 ,0.0 ],"wire_score":float (s )}


def test_yakin_ciftte_yuksek_skorlu_kalir ():
    out =wire_gate .suppress_crowd ([cp (0 ,0.3 ),cp (2 ,0.9 )],r_mm =6.0 )
    assert len (out )==1 and out [0 ]["wire_score"]==pytest .approx (0.9 )


def test_uzak_cift_ikisi_de_kalir ():
    out =wire_gate .suppress_crowd ([cp (0 ,0.3 ),cp (20 ,0.9 )],r_mm =6.0 )
    assert len (out )==2 


def test_sinir_tam_yaricapta_BASTIRILMAZ ():
    """Kural `< r`; full r mesafesi ayri mouth sayilir."""
    assert len (wire_gate .suppress_crowd ([cp (0 ,0.3 ),cp (6 ,0.9 )],r_mm =6.0 ))==2 
    assert len (wire_gate .suppress_crowd ([cp (0 ,0.3 ),cp (5.99 ,0.9 )],r_mm =6.0 ))==1 


def test_olcum_ve_urun_yolu_AYNI_maskeyi_alir ():
    """`crowd_mask` TEK KAYNAK; `suppress_crowd` onu cagirir."""
    inp_ =[cp (0 ,0.5 ),cp (4 ,0.9 ),cp (40 ,0.6 )]
    m =wire_gate .crowd_mask ([c ["point"]for c in inp_ ],
    [c ["wire_score"]for c in inp_ ],6.0 )
    assert [c ["point"][0 ]for c in wire_gate .suppress_crowd (inp_ ,r_mm =6.0 )]==[c ["point"][0 ]for c ,k in zip (inp_ ,m )if k ]


def test_sifir_yaricap_KAPATIR ():
    inp_ =[cp (0 ,0.3 ),cp (1 ,0.9 )]
    assert wire_gate .suppress_crowd (inp_ ,r_mm =0.0 )==inp_ 


def test_girdi_sirasi_korunur ():
    out =wire_gate .suppress_crowd ([cp (0 ,0.9 ),cp (30 ,0.5 ),cp (60 ,0.7 )],r_mm =6.0 )
    assert [c ["point"][0 ]for c in out ]==[0.0 ,30.0 ,60.0 ]


def test_zincir_ortadaki_en_yuksekten_bastirilir ():
    """0-4-8: 4 most high. 0 and 8 ona 6mm'den yakin -> ikisi de duser."""
    out =wire_gate .suppress_crowd ([cp (0 ,0.5 ),cp (4 ,0.9 ),cp (8 ,0.6 )],r_mm =6.0 )
    assert [c ["point"][0 ]for c in out ]==[4.0 ]


def test_tek_cp_ve_bos_liste ():
    assert len (wire_gate .suppress_crowd ([cp (0 ,0.5 )],r_mm =6.0 ))==1 
    assert wire_gate .suppress_crowd ([],r_mm =6.0 )==[]


def test_varsayilan_yaricap_konfigden_5mm ():
    """r=5.0 DAGITILAN value. Kural: no markayi yikmayan most large radius.
    r=6 D7'de more high robot gives but CEM markasini 0.0164 -> 0.0000 yikar.
    See. results/urun_nms_uctan_uca*.json."""
    assert wire_gate .NMS_MM ==pytest .approx (5.0 )
