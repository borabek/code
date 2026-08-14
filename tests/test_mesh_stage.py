# -*- coding: utf-8 -*-
"""UCUNCU MESH KADEMESI + YARIM-MESH KORUMASI (2026-08-04).

DataSet5 turetmesinde two part two kademeyi de gecemedi. Ucuncu kademe (gevsek OCC
tolerans) 7 ayarlik olcumun kazanani; half-mesh korumasi whereas "kismi mesh'i kullanalim"
fikrinin OLCUMLE reddedilmesinin kalicilastirilmasidir (see. infer_step_cp docstring'leri).
"""
import glob 
import os 
import sys 

import numpy as np 
import pytest 

sys .path .insert (0 ,os .path .dirname (os .path .dirname (os .path .abspath (__file__ ))))


def test_tamlik_esigi_calisan_parcalari_kirmiyor ():
    """Olculdu: running 25 parcanin mesh'lenen surface orani TAM 1.0. Esik 0.95 < 1.0."""
    import infer_step_cp as I 
    assert 0.0 <I .MESH_TAMLIK <1.0 ,"threshold oransal olmali"
    assert I .MESH_TAMLIK <=0.95 ,"threshold 0.95'in ustune cikarsa running parts riske girer"


def test_yarim_mesh_reddedilir (monkeypatch ):
    """202 yuzeyin 92'si mesh'lenmis DST2.5_GY vakasi: RuntimeError atmali, sessizce gecmemeli."""
    import infer_step_cp as I 

    class SahteMesh :
        @staticmethod 
        def getElements (dim ,tag ):
            return ([2 ],[[1 ]],[[1 ,2 ,3 ]])if tag <=92 else ([],[],[])

    class SahteModel :
        mesh =SahteMesh 

        @staticmethod 
        def getEntities (dim ):
            return [(2 ,t )for t in range (1 ,203 )]

    monkeypatch .setattr (I .gmsh ,"model",SahteModel )
    with pytest .raises (RuntimeError ,match ="YARIM MESH"):
        I ._tamlik_kontrol (202 )


def test_tam_mesh_gecer (monkeypatch ):
    import infer_step_cp as I 

    class SahteMesh :
        @staticmethod 
        def getElements (dim ,tag ):
            return ([2 ],[[1 ]],[[1 ,2 ,3 ]])

    class SahteModel :
        mesh =SahteMesh 

        @staticmethod 
        def getEntities (dim ):
            return [(2 ,t )for t in range (1 ,203 )]

    monkeypatch .setattr (I .gmsh ,"model",SahteModel )
    I ._tamlik_kontrol (202 )# istisna ATMAMALI


@pytest .mark .slow 
def test_ucuncu_kademe_gercek_parcayi_kurtariyor ():
    """5D.202.0055.6: kademe 1 and 2 duser, kademe 3 (tol 1e-2) full mesh produces."""
    f =glob .glob ("all_wscad_stp/*5D.202.0055.6*")
    if not f :
        pytest .skip ("part none")
    from infer_step_cp import step_to_mesh 
    V ,F =step_to_mesh (f [0 ])
    assert len (V )>3000 and len (F )>6000 
    assert np .all (V .max (0 )-V .min (0 )>1.0 )
