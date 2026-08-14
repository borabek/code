# -*- coding: utf-8 -*-
"""TEL-B: acikligin ICINE bak -- channel profili ozellikleri (tel girisi vs alet agzi).

WHY YENI: bugune up to acikligi hep DISARIDAN tarif ettik (cap, depth, egrilik, komsuluk).
Tel/alet farki fiziksel as ICERIDE: tel girisi a kontakta biter, alet agzi a arm/yay
yuvasinda. Bunu olcmek isin-mesh kesisimi gerektiriyordu and that bugune up to CALISMIYORDU
(rtree yoklugunda trimesh.ray each cagrida patliyor, except yutuyordu -- see trimesh-rtree-silent-failure).
cp_geometry.ray_hits with residual mumkun.

TEL-A'NIN GEREKCESI: hayatta kalan 263 FP'de mevcut 41 ozelligin EN IYISI AUC 0.36/0.61 -- set kor.

URETILEN OZELLIKLER (each candidate for, axis along ICERI correct):
  1  serbest_derinlik_mm     agizdan first yuzeye up to empty distance (channel ne up to aciliyor)
  2  kanal_kesisim_sayisi    axis uzerindeki total surface gecisi (kor channel vs tunel)
  3  dip_var_mi              menzil inside channel kapaniyor mu
  4-6 kesit_r2/r4/r6         3 derinlikte channel genisligi (dik yonde first yuzeye distance)
  7  daralma_orani           kesit derinlikle daraliyor mu genisliyor mu (konik/silindirik)
  8  eksen_disi_sapma        channel ekseni gercekten duz mu (arm yuvasi egik becomes)
  9-10 dip_CE / dip_CT       channel dibindeki vertexlerin sinif olasiliklari (KONTAKTA MI BITIYOR)
"""
import numpy as np 

from cp_geometry import ray_hits ,outward_along_axis 


def channel_feats (V ,F ,probs ,point ,direction ,CE ,CT ,max_mm =25.0 ):
    """Tek candidate for channel profili (10 feature). Mesh frame, mm."""
    mesh =(V ,F )
    p =np .asarray (point ,float )
    d =np .asarray (direction ,float )
    n =float (np .linalg .norm (d ))
    if n <1e-9 :
        return [0.0 ]*10 
    d =d /n 

    # channel ICERI correct = DISARI yonun tersi. cp_openings'in verdigi yonun isareti guvenilmez.
    # NOTE (sentetik testte yakalandi): "more very bosluk which is direction iceridir" YANLIS a criterion --
    # parcadan DISARI cikarken bosluk sonsuzdur, that yuzden that criterion hep disariyi secip kanali never
    # gormuyordu. Govde-temelli outward_along_axis is used.
    din =-outward_along_axis (mesh ,p ,d ,max_mm =max_mm *3.0 )

    # ESITLIK DURUMU: bastan sona hole (tunel) whereas axis HIC malzemeye carpmaz and two direction de
    # ayirt edilemez -- outward_along_axis rastgele birini selects. Fiziksel threshold kurali: kanalin
    # ICI, DIK yonde malzemenin DAHA YAKIN oldugu taraftir (hole dar, disarisi genis).
    if not len (ray_hits (mesh ,p ,din ,max_mm ))and not len (ray_hits (mesh ,p ,-din ,max_mm )):
        def _perp_free (dirv ):
            a =np .array ([1.0 ,0.0 ,0.0 ])
            if abs (float (dirv @a ))>0.9 :a =np .array ([0.0 ,1.0 ,0.0 ])
            uu =np .cross (dirv ,a );uu /=(np .linalg .norm (uu )+1e-9 )
            ww =np .cross (dirv ,uu )
            q =p +dirv *2.0 
            r =[ray_hits (mesh ,q ,v ,max_mm )for v in (uu ,-uu ,ww ,-ww )]
            return float (np .median ([float (h [0 ])if len (h )else max_mm for h in r ]))
        if _perp_free (-din )<_perp_free (din ):
            din =-din 

    hits_in =ray_hits (mesh ,p ,din ,max_mm )
    free_in =float (hits_in [0 ])if len (hits_in )else max_mm 

    n_cross =int (len (hits_in ))
    has_bottom =1.0 if n_cross >0 else 0.0 

    # channel genisligi: axis ten belli derinliklerde, DIK yonlerde first yuzeye distance
    a =np .array ([1.0 ,0.0 ,0.0 ])
    if abs (float (din @a ))>0.9 :
        a =np .array ([0.0 ,1.0 ,0.0 ])
    u =np .cross (din ,a );u /=(np .linalg .norm (u )+1e-9 )
    w =np .cross (din ,u )

    widths =[]
    for dep in (2.0 ,4.0 ,6.0 ):
        q =p +din *dep 
        rs =[]
        for vdir in (u ,-u ,w ,-w ):
            h =ray_hits (mesh ,q ,vdir ,max_mm )
            rs .append (float (h [0 ])if len (h )else max_mm )
        widths .append (float (np .median (rs )))
    w2 ,w4 ,w6 =widths 
    taper =float ((w6 -w2 )/max (w2 ,1e-6 ))# <0 daraliyor, >0 genisliyor

    # axis duzlugu: 2mm and 6mm derinlikte kesit merkezleri ne up to kayiyor
    off =0.0 
    for dep in (2.0 ,6.0 ):
        q =p +din *dep 
        cs =[]
        for vdir in (u ,-u ,w ,-w ):
            h =ray_hits (mesh ,q ,vdir ,max_mm )
            cs .append (float (h [0 ])if len (h )else max_mm )
        off +=abs ((cs [0 ]-cs [1 ])+(cs [2 ]-cs [3 ]))/4.0 
    axis_off =float (off /2.0 )

    # KANAL DIBI hangi sinif: dibe yakin vertexlerin CE/CT olasiligi
    ce_b =ct_b =0.0 
    if n_cross >0 :
        bottom =p +din *float (hits_in [0 ])
        dist =np .linalg .norm (np .asarray (V ,float )-bottom ,axis =1 )
        near =dist <=3.0 
        if near .any ():
            ce_b =float (np .asarray (probs )[near ,CE ].mean ())
            ct_b =float (np .asarray (probs )[near ,CT ].mean ())

    return [float (free_in ),float (n_cross ),has_bottom ,
    w2 ,w4 ,w6 ,taper ,axis_off ,ce_b ,ct_b ]


FEAT_NAMES =["serbest_derinlik","kesisim_sayisi","dip_var","kesit_r2","kesit_r4",
"kesit_r6","daralma","eksen_sapma","dip_CE","dip_CT"]


def feats_for (V ,F ,probs ,cps ,CE ,CT ):
    """cp_openings ciktisi -> (n_aday, 10) channel ozellikleri."""
    if not cps :
        return np .zeros ((0 ,len (FEAT_NAMES )))
    return np .array ([channel_feats (V ,F ,probs ,c ["point"],c ["direction"],CE ,CT )
    for c in cps ],float )
