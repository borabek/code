# -*- coding: utf-8 -*-
"""P1: birlesme yaricaplari CONFIG'ten okunmali, sabit kodlu OLMAMALI.

`dedupe_mm` 10.0 sabit kodluydu and oy havuzu `_vote2`'nin varsayilanini (5.0)
kullaniyordu. P1 taramasi bunlarin KOMSU GERCEK GIRISLERI single adaya yuttugunu
olctu (very-CP kahini +0.0331). Bu test sabit kodun geri gelmesini engeller.
"""
import io 
import json 
import os 
import re 
import sys 

sys .path .insert (0 ,os .path .dirname (os .path .dirname (os .path .abspath (__file__ ))))


def _src ():
    return io .open ("robot_cp.py",encoding ="utf-8").read ()


def _govde (ad ):
    """Fonksiyon govdesini UST DUZEY def'e according to kes.

    Ilk version `s.index("def ", i+10)` kullaniyordu and IC ICE `_turet` tanimi govdeyi
    erken bitirip testi yaniltiyordu -- i.e. test kodu not KENDISI hataliydi.
    """
    s =_src ()
    i =s .index ("def "+ad )
    j =s .find (chr (10 )+"def ",i )
    return s [i :]if j <0 else s [i :j ]


def test_dedupe_config_ten ():
    body =_govde ("derive_candidates")
    assert "dedupe_mm=dd"in body ,"dedupe yaricapi sabit kodlu"
    assert '_pp.get("dedupe_mm"'in body 


def test_oy_havuzu_config_ten ():
    body =_govde ("derive_candidates")
    assert '_pp.get("vote_pool_mm"'in body ,"oy havuzu yaricapi config'ten gelmiyor"
    assert body .count ("cluster_mm=oy")>=2 ,"_vote2 cagrilarinin ikisi de oy almali"


def test_varsayilanlar_ESKI_davranis ():
    """Config'i olmayan kurulum AYNEN eskisi like calismali."""
    s =_src ()
    assert '_pp.get("dedupe_mm", 10.0)'in s 
    assert '_pp.get("vote_pool_mm", 5.0)'in s 


def test_cp_config_P1_kazananini_tasiyor ():
    c =json .load (io .open ("cp_config.json",encoding ="utf-8"))["prediction_postproc"]
    assert c ["cluster_mm"]==1.0 and c ["dedupe_mm"]==2.0 and c ["vote_pool_mm"]==2.0 
    assert "_superseded_2026_08_06_values"in c ,"geri alma degerleri kayitli not"


def test_birlesme_yaricaplari_KILIT ():
    """P3-b: 1/2/2 KILIT. Eski 3/10/5 high-CP agizlarini BIRLESTIRIYOR.

    Olculdu (P1 taramasi, D6, candidate kahini bire-a Macar):
      3/10/5 -> kahin 0.8808 | very-CP 0.4740
      1/2/2  -> kahin 0.8966 | very-CP 0.5071 (+0.0331)
    Bu degerler DEGISTIRILMEZ; degistirilecekse ONCE very-CP kahini olculur.
    """
    import json ,io as _io 
    c =json .load (_io .open ("cp_config.json",encoding ="utf-8"))
    pp =c .get ("prediction_postproc",{})
    assert abs (float (pp .get ("cluster_mm",5.0 ))-1.0 )<1e-9 ,pp .get ("cluster_mm")
    assert abs (float (pp .get ("dedupe_mm",10.0 ))-2.0 )<1e-9 ,pp .get ("dedupe_mm")
    assert abs (float (pp .get ("vote_pool_mm",5.0 ))-2.0 )<1e-9 ,pp .get ("vote_pool_mm")
