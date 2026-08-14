# -*- coding: utf-8 -*-
"""M10: FIZIKSEL DEFECT KANALI -- ayri dugum, M6'yi BOZMAZ (2026-08-05).

Sozlesmenin M6'si "RENK = ESLESME DURUMU" diyor and denetci renk sayimlarini TP/FP/FN with
karsilastiriyor. Kusurlari new RENKLERLE gostermek M6'yi bozardi. Bu testler ayrimin
korundugunu and bayraksiz ciktinin BIT-OZDES kaldigini garanti eder.
"""
import os 
import sys 

import numpy as np 

sys .path .insert (0 ,os .path .dirname (os .path .dirname (os .path .abspath (__file__ ))))


def _kup ():
    import trimesh 
    m =trimesh .creation .box (extents =(20.0 ,20.0 ,20.0 ))
    return np .asarray (m .vertices ,float ),np .asarray (m .faces ,np .int64 )


def _cp (x ,bayrak =None ):
    d ={"entry_point":[float (x ),0.0 ,12.0 ],"approach_vector":[0.0 ,0.0 ,1.0 ],
    "confidence_score":0.9 }
    if bayrak :
        d ["fiz_bayrak"]=bayrak 
    return d 


def _yaz (tmp ,ml ,ad ="t.glb"):
    import export_glb as E 
    V ,F =_kup ()
    g =E ._GlbBuilder ()
    E .build_scene (g ,V ,F ,ml ,None )
    p =os .path .join (tmp ,ad )
    g .write (p )# _GlbBuilder.write(path) -- `build()` YOK
    return p 


def test_bayraksiz_cikti_bit_ozdes (tmp_path ):
    """Bayrak YOKSA no sey cizilmez -- old GLB'ler aynen kalmali."""
    a =open (_yaz (str (tmp_path ),[_cp (0 ),_cp (5 )],"a.glb"),"rb").read ()
    b =open (_yaz (str (tmp_path ),[_cp (0 ),_cp (5 )],"b.glb"),"rb").read ()
    assert a ==b ,"same input same output vermeli (M9)"


def test_bayrak_ayri_dugum_uretir (tmp_path ):
    """Bayrakli CP, `fiz_<name>` adli AYRI dugum adds."""
    import json 
    p =_yaz (str (tmp_path ),[_cp (0 ,["govde_ici"]),_cp (5 )],"c.glb")
    ham =open (p ,"rb").read ()
    assert b"fiz_govde_ici"in ham ,"fiz_ dugumu yazilmadi"
    assert b"ml_spheres"in ham ,"ana kure kanali kaybolmus -- M6 bozulur"


def test_bilinmeyen_bayrak_yok_sayilir (tmp_path ):
    """Tanimsiz bayrak sessizce ATLANIR, output bayraksizla AYNI becomes."""
    a =open (_yaz (str (tmp_path ),[_cp (0 ,["olmayan_bayrak"])],"d.glb"),"rb").read ()
    b =open (_yaz (str (tmp_path ),[_cp (0 )],"e.glb"),"rb").read ()
    assert a ==b 


def test_dort_bayrak_dort_ayri_dugum (tmp_path ):
    p =_yaz (str (tmp_path ),[_cp (0 ,["govde_ici"]),_cp (3 ,["onu_kapali"]),
    _cp (6 ,["dar_agiz"]),_cp (9 ,["ters_yon"])],"f.glb")
    ham =open (p ,"rb").read ()
    for ad in ("govde_ici","onu_kapali","dar_agiz","ters_yon"):
        assert ("fiz_"+ad ).encode ()in ham ,f"{ad } dugumu yok"


def test_ayni_cp_birden_cok_bayrak (tmp_path ):
    """Bir CP hem body ici hem onu closed may be -- ikisi de cizilir."""
    p =_yaz (str (tmp_path ),[_cp (0 ,["govde_ici","onu_kapali"])],"g.glb")
    ham =open (p ,"rb").read ()
    assert b"fiz_govde_ici"in ham and b"fiz_onu_kapali"in ham 
