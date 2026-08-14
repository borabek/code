# -*- coding: utf-8 -*-
"""A0: FIZIKSEL BAYRAKLARIN ALETLERI DOGRU MU? (B1-B4'ten ONCE)

WHY: otopsi (2026-08-04) "uretilen CP'lerin %40/%50/%96'si fiziksel as kusurlu"
dedi and this B1-B4 red kriterlerinin dayanagi. Ama that hukmu veren IKI FONKSIYON never
dogrulanmadi:
  * `cp_geometry.is_inside` : 5 rastgele isinla PARITE oyu. Parite testi WATERTIGHT
    mesh varsayar; urun ~6000 tepeye REMESH edilmis, this gecirmez OLMAYAN orgu kullaniyor.
  * `cp_geometry.mouth_width`: 24 yonde isin atip MINIMUMU aliyor. Otopside gorulen
    0.19mm "ic cap" real a wall yakinligi da may be, single a isinin facet kenarina
    denk gelmesi de.

[[trimesh-rtree-silent-failure]]: this proje sessizce wrong cevap veren a geometri
fonksiyonuyla aylarca kostu. Dogrulanmamis a aleti RED KRITERINE terfi ettirmek,
iyi CP'leri silmek demektir.

SINAV: known geometriler (full kutu / gecisli hole / kor cep / ince yarik). Her biri
IKI halde olculur -- TAM (CSG, watertight) and REMESH (urunun gercekten gordugu hal).

KILL (onceden yazildi):
    is_inside  yanilma <= %5
    mouth_width sapmasi <= %10 (known capa according to) VE isin sayisina duyarliligi <= %10
GECMEZSE: fiziksel bayraklar only RAPOR kalir; B1-B4'te RED KRITERI OLAMAZ.
"""
import io 
import json 
import os 
import sys 

import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))

KUTU =(24.0 ,24.0 ,12.0 )# x,y,z
R_DELIK =3.0 # gecisli hole yaricapi
R_CEP =2.5 # kor cep yaricapi
CEP_DERINLIK =6.0 
YARIK =(1.6 ,9.0 )# ince yarik: width x uzunluk


def geometriler ():
    """(name, mesh, sinav_noktalari) -- exam noktasi: (point, iceride_mi_DOGRUSU)."""
    import trimesh 
    g ={}
    z =KUTU [2 ]/2.0 

    kutu =trimesh .creation .box (KUTU )
    g ["dolu_kutu"]=(kutu ,[
    (np .array ([0. ,0. ,0. ]),True ),# full ortada -> ICERIDE
    (np .array ([8. ,8. ,0. ]),True ),
    (np .array ([0. ,0. ,z +5 ]),False ),# disarida
    (np .array ([30. ,0. ,0. ]),False ),
    ])

    sil =trimesh .creation .cylinder (radius =R_DELIK ,height =KUTU [2 ]*3 )
    delik =kutu .difference (sil )
    g ["gecisli_delik"]=(delik ,[
    (np .array ([0. ,0. ,0. ]),False ),# DELIGIN ICI -> malzeme DEGIL
    (np .array ([0. ,0. ,z -0.5 ]),False ),# agza yakin, yine bosluk
    (np .array ([9. ,9. ,0. ]),True ),# kose -> malzeme
    (np .array ([0. ,0. ,z +6 ]),False ),# disarida
    ])

    # KOR CEP -- NOTE (first surumde KENDI TESTIM hataliydi): silindir IKI KEZ otelenmisti
    # (z, after +CEP_DERINLIK), total 12. Kutu z<=6'da bittigi for silindir [6,18] araligina
    # gidiyor and HIC KESMIYORDU: "kor cep" aslinda DOLU KUTUYDU. is_inside'in "wrong" dedigi
    # point full that full bolgedeydi -- ALET DOGRUYDU, GEOMETRI YANLISTI. mouth_width'in 24.00
    # (= kutu genisligi) vermesi de same sebeptendi.
    # DOGRUSU: height 2*CEP_DERINLIK which is silindiri z=+z'ye otele -> [0, 12] araligini kaplar,
    # kutuyla kesisimi z in [0, 6] = upper yuzeyden CEP_DERINLIK derinliginde KOR cep.
    cep_sil =trimesh .creation .cylinder (radius =R_CEP ,height =CEP_DERINLIK *2 )
    cep_sil .apply_translation ([0 ,0 ,z ])
    cep =kutu .difference (cep_sil )
    g ["kor_cep"]=(cep ,[
    (np .array ([0. ,0. ,z -1.0 ]),False ),# cebin ici -> bosluk
    (np .array ([0. ,0. ,-z +2.0 ]),True ),# cebin ALTINDA malzeme present
    (np .array ([9. ,9. ,0. ]),True ),
    ])

    y =trimesh .creation .box ((YARIK [0 ],YARIK [1 ],KUTU [2 ]*3 ))
    yarik =kutu .difference (y )
    g ["ince_yarik"]=(yarik ,[
    (np .array ([0. ,0. ,0. ]),False ),# yarigin ici
    (np .array ([9. ,0. ,0. ]),True ),
    ])
    return g 


def olc (mesh ,noktalar ,is_inside ):
    correct =sum (1 for p ,t in noktalar if bool (is_inside (mesh ,p ))==t )
    return correct ,len (noktalar ),[(p .tolist (),t ,bool (is_inside (mesh ,p )))
    for p ,t in noktalar if bool (is_inside (mesh ,p ))!=t ]


def main ():
    import trimesh 
    import thesis_remesh 
    from cp_geometry import is_inside ,mouth_width 
    import protocol 
    protocol .tez_dogrula ()

    G =geometriler ()
    rapor ={"is_inside":{},"mouth_width":{},"kill":{}}
    print (f"{'geometri':<16}{'hal':<10}{'vertex':>7}{'watertight':>12}{'is_inside':>12}")
    top_d =top_n =0 
    for ad ,(m ,nk )in G .items ():
        for hal in ("TAM","REMESH"):
            if hal =="TAM":
                mm =m 
            else :
                V ,F =thesis_remesh .remesh_uniform (np .asarray (m .vertices ,float ),
                np .asarray (m .faces ,np .int64 ),target =6000 )
                mm =trimesh .Trimesh (vertices =np .ascontiguousarray (V ,float ),
                faces =np .ascontiguousarray (F ,np .int64 ),process =False )
            d ,n ,wrong =olc (mm ,nk ,is_inside )
            top_d +=d ;top_n +=n 
            rapor ["is_inside"][f"{ad }_{hal }"]={"correct":d ,"total":n ,"wrong":wrong ,
            "vertex":len (mm .vertices ),
            "watertight":bool (mm .is_watertight )}
            print (f"{ad :<16}{hal :<10}{len (mm .vertices ):>7}{str (mm .is_watertight ):>12}"
            f"{d }/{n }".rjust (12 ))
    yanilma =1.0 -top_d /max (top_n ,1 )
    print (f"\nis_inside TOPLAM: {top_d }/{top_n } correct -> YANILMA %{100 *yanilma :.1f}")

    # --- mouth_width: BILINEN cap
    print (f"\n{'geometri':<16}{'hal':<10}{'beklenen':>10}{'12 isin':>9}{'24 isin':>9}"
    f"{'48 isin':>9}{'deviation':>9}")
    mw_sapma =[]
    for ad ,r_bek in (("gecisli_delik",R_DELIK ),("kor_cep",R_CEP )):
        m =G [ad ][0 ]
        z =KUTU [2 ]/2.0 
        p =np .array ([0. ,0. ,z -0.3 ]);d =np .array ([0. ,0. ,1. ])
        for hal in ("TAM","REMESH"):
            if hal =="TAM":
                mm =m 
            else :
                V ,F =thesis_remesh .remesh_uniform (np .asarray (m .vertices ,float ),
                np .asarray (m .faces ,np .int64 ),target =6000 )
                mm =trimesh .Trimesh (vertices =np .ascontiguousarray (V ,float ),
                faces =np .ascontiguousarray (F ,np .int64 ),process =False )
            ol ={}
            for nd in (12 ,24 ,48 ):
                ic ,ort =mouth_width (mm ,p ,d ,n_dirs =nd )
                ol [nd ]=float (ic )
            bek =2 *r_bek 
            sap =max (abs (v -bek )/bek for v in ol .values ()if v >0 )if any (
            v >0 for v in ol .values ())else 1.0 
            tut =(max (ol .values ())-min (ol .values ()))/max (bek ,1e-9 )
            mw_sapma .append (sap )
            rapor ["mouth_width"][f"{ad }_{hal }"]={"beklenen":bek ,"measurement":ol ,
            "deviation":sap ,"isin_duyarliligi":tut }
            print (f"{ad :<16}{hal :<10}{bek :>10.2f}{ol [12 ]:>9.2f}{ol [24 ]:>9.2f}{ol [48 ]:>9.2f}"
            f"{100 *sap :>8.0f}%")

    mw_max =max (mw_sapma )if mw_sapma else 1.0 
    g1 =yanilma <=0.05 
    g2 =mw_max <=0.10 
    rapor ["kill"]={"is_inside_yanilma":yanilma ,"is_inside_gecti":bool (g1 ),
    "mouth_width_max_sapma":mw_max ,"mouth_width_gecti":bool (g2 ),
    "sonuc":"RED KRITERI OLABILIR"if (g1 and g2 )else "YALNIZ RAPOR"}
    print (f"\nKILL: is_inside yanilma %{100 *yanilma :.1f} (<=%5 -> {'GECTI'if g1 else 'KALDI'})")
    print (f"      mouth_width deviation %{100 *mw_max :.0f} (<=%10 -> {'GECTI'if g2 else 'KALDI'})")
    print (f"\nHUKUM: fiziksel bayraklar -> {rapor ['kill']['sonuc']}")
    if not (g1 and g2 ):
        print ("  B1-B4'te this bayraklar RED KRITERI OLAMAZ; only raporlanir.")
    with io .open ("results/a0_alet_calibration.json","w",encoding ="utf-8")as f :
        json .dump (rapor ,f ,indent =1 ,ensure_ascii =False )
    print ("receipt -> results/a0_alet_calibration.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
