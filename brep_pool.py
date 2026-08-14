# -*- coding: utf-8 -*-
"""B-rep GENISLETILMIS ADAY HAVUZU -- single source.

WHY: measured (`results/havuz_recall_d7.json`) ki D7 brand-disi pool recall'u
segmentasyon single basina **0.6654**. Donusum ~%48 oldugundan robot tavani ~0.32;
i.e. MEVCUT HAVUZLA 0.50 IMKANSIZ. Gate/selector/count kollarinin all of them this tavanin
altindaydi and all of them closed. Baglayici kisit HAVUZ.

B-rep silindir agizlari + duzlemsel opening merkezleri EK candidate kaynagi as
eklenince (D7, `results/brep_filtre_taramasi.json`):
    only seg        recall 0.6654   13.5 candidate/part
    + B-rep (ham)     recall 0.8465  379.9
    + B-rep (dedupe3) recall 0.7852   98.2   <-- DIZ
CWT 0.3595 -> 0.6579, KLM 0.5663 -> 0.8313 (tabani ceken markalar).

DURUSTLUK: this TEZ TURETMESI DEGIL. Tezin `v_o` mouth-ortasi turetmesi, 5 sinif
segmentasyon and ~6000 remesh AYNEN durur; B-rep onerileri ONLARIN YANINA eklenen
ikinci a candidate kaynagidir and `source` alaniyla isaretlenir. Sonuclar "tez
sonucu" as DEGIL, "tez-omurgali geometrik genisletme" as raporlanir.
"""
import os 

import numpy as np 

DEDUPE_MM =3.0 


def _dedupe (P ,mm ):
    tut =np .ones (len (P ),bool )
    if mm <=0 or len (P )<2 :
        return tut 
    for i in range (len (P )):
        if not tut [i ]:
            continue 
        d =np .linalg .norm (P [i +1 :]-P [i ],axis =1 )
        tut [i +1 :][(d <mm )&tut [i +1 :]]=False 
    return tut 


def brep_adaylari (cyl ,acik ,dedupe_mm =DEDUPE_MM ,meta =False ):
    """Silindir agizlari (two three, direction = +/- axis) + opening merkezleri (direction = normal).

    Doner: (P, D) -- `meta=True` whereas (P, D, kaynak_kaydi) ucluSU. `kaynak_kaydi`
    each candidate for onu ureten silindir/opening sozlugudur; mouth tanimlayicilari
    (radius, hole derinligi, es-eksenli kardesler) ORADAN okunur. Ayni dedupe
    maskesi uygulandigi for siralamalar BIREBIR ortusur.

    Yon TEZIN `v_o`'suyla same sozlesmede: disari bakan axis.
    """
    P ,D ,MET =[],[],[]
    for c in cyl or []:
        ax =np .asarray (c ["axis"],float )
        n =np .linalg .norm (ax )
        if n <1e-9 :
            continue 
        ax =ax /n 
        for u ,s in ((c .get ("mouth_a"),1.0 ),(c .get ("mouth_b"),-1.0 )):
            if u is not None :
                P .append (np .asarray (u ,float ));D .append (s *ax )
                MET .append (c )
    for o in acik or []:
        if isinstance (o ,dict )and o .get ("center")is not None :
            nn =np .asarray (o .get ("normal",[0.0 ,0.0 ,1.0 ]),float )
            m =np .linalg .norm (nn )
            P .append (np .asarray (o ["center"],float ))
            D .append (nn /m if m >1e-9 else np .array ([0.0 ,0.0 ,1.0 ]))
            MET .append (o )
    if not P :
        empty_ =(np .zeros ((0 ,3 )),np .zeros ((0 ,3 )))
        return empty_ +([],)if meta else empty_ 
    P =np .asarray (P ,float );D =np .asarray (D ,float )
    k =_dedupe (P ,dedupe_mm )
    if meta :
        return P [k ],D [k ],[m for m ,t in zip (MET ,k )if t ]
    return P [k ],D [k ]


def merged_pool (P_seg ,D_seg ,cyl ,acik ,dedupe_mm =DEDUPE_MM ):
    """Segmentasyon havuzu + B-rep onerileri. Doner: (P, D, source).

    `source`: 0 = segmentasyon (tezin `v_o`'this), 1 = B-rep onerisi.
    Segmentasyon adaylari HER ZAMAN before gelir and ASLA elenmez -- tezin cevabi
    havuzda bozulmadan durur.
    """
    P_seg =np .asarray (P_seg ,float ).reshape (-1 ,3 )
    D_seg =np .asarray (D_seg ,float ).reshape (-1 ,3 )
    Pb ,Db =brep_adaylari (cyl ,acik ,dedupe_mm )
    if len (Pb )and len (P_seg ):
    # segmentasyon adayina very yakin B-rep onerisi GEREKSIZ: same mouth
        uz =np .linalg .norm (Pb [:,None ]-P_seg [None ],axis =-1 ).min (1 )
        k =uz >=dedupe_mm 
        Pb ,Db =Pb [k ],Db [k ]
    P =np .vstack ([P_seg ,Pb ])if len (Pb )else P_seg 
    D =np .vstack ([D_seg ,Db ])if len (Db )else D_seg 
    src_ =np .concatenate ([np .zeros (len (P_seg ),int ),np .ones (len (Pb ),int )])
    return P ,D ,src_ 


    # --- MESH TEPESI KAYNAGI ---------------------------------------------------
    # Olculdu (`results/tavan_080_eksensiz.json`, D7 brand-disi, signed angle):
    #   only B-rep havuzu          ceiling 0.7472   102 candidate/part
    #   + mesh p>=0.50, 2mm seyrelt  ceiling 0.8347   318 candidate/part
    #   + mesh p>=0.05, 2mm seyrelt  ceiling 0.9235   829 candidate/part
    # Yon kaynagi as tepenin YEREL NORMALI is used; this, "direction no kaynakta
    # absent" kovasini %23.0'ten %3.1'e indirdi.
    #
    # EKSEN BOYU ORNEKLEME KASTEN YOK: part basina ~670 candidate ekleyip tavani very few
    # oynatiyordu (1162 adayda 0.7798 vs 102 adayda 0.7472).
    # `BH_MESH_ESIK` with dusurulebilir. MEASURED (D7 teshis): CWT'de konum recall
    # 0.50 esiginde 0.5532, 0.05'te 0.7914. Segmentasyon zayif markalarda CP
    # bolgelerindeki vertices 0.50'yi GECEMIYOR and pool orayi never gormuyor.
    # Esigi dusurmek candidate EKLER, never CIKARMAZ -> pool tavanini MONOTON yukseltir.
MESH_ESIK =float (os .environ .get ("BH_MESH_ESIK","0.50"))
MESH_DEDUPE_MM =2.0 


def vertex_normals_at (V ,F ):
    """Alan agirlikli vertex normalleri (agizda disari bakar)."""
    V =np .asarray (V ,float )
    F =np .asarray (F ,np .int64 )
    N =np .zeros_like (V )
    tri =V [F ]
    fn =np .cross (tri [:,1 ]-tri [:,0 ],tri [:,2 ]-tri [:,0 ])
    for k in range (3 ):
        np .add .at (N ,F [:,k ],fn )
    n =np .linalg .norm (N ,axis =1 ,keepdims =True )
    return N /np .maximum (n ,1e-12 )


def mesh_adaylari (V ,F ,ppos ,threshold =MESH_ESIK ,dedupe_mm =MESH_DEDUPE_MM ):
    """p_pos esigini gecen mesh tepeleri, uzamsal seyreltmeyle.

    Doner: (P, D) -- D tepenin YEREL NORMALI (disari). Seyreltmede p_pos'u
    high which is tutulur; tolerans YANAL 2mm oldugu for 2mm'de single vertex yeter.
    """
    V =np .asarray (V ,float )
    ppos =np .asarray (ppos ,float )
    k =ppos >=threshold 
    if not k .any ():
        return np .zeros ((0 ,3 )),np .zeros ((0 ,3 ))
    P =V [k ]
    N =vertex_normals_at (V ,F )[k ]
    s =ppos [k ]
    if dedupe_mm >0 and len (P )>1 :
        rank_ =np .argsort (-s )
        tut =np .ones (len (P ),bool )
        for i in rank_ :
            if not tut [i ]:
                continue 
            uz =np .linalg .norm (P -P [i ],axis =1 )
            yakin =(uz <dedupe_mm )
            yakin [i ]=False 
            tut [yakin ]=False 
        P ,N =P [tut ],N [tut ]
    return P ,N 


def tam_havuz (P_seg ,D_seg ,cyl ,acik ,V =None ,F =None ,ppos =None ,
dedupe_mm =DEDUPE_MM ,mesh_esik =MESH_ESIK ,
mesh_dedupe_mm =MESH_DEDUPE_MM ):
    """Segmentasyon + B-rep + (varsa) mesh tepeleri. Doner: (P, D, source).

    `source`: 0 = segmentasyon (tezin `v_o`'this), 1 = B-rep, 2 = mesh tepesi.
    Tezin adaylari HER ZAMAN basta and ASLA elenmez.
    """
    P ,D ,kay =merged_pool (P_seg ,D_seg ,cyl ,acik ,dedupe_mm )
    if V is None or F is None or ppos is None :
        return P ,D ,kay 
    Pm ,Dm =mesh_adaylari (V ,F ,ppos ,mesh_esik ,mesh_dedupe_mm )
    if len (Pm )and len (P ):
        uz =np .linalg .norm (Pm [:,None ]-P [None ],axis =-1 ).min (1 )
        k =uz >=mesh_dedupe_mm 
        Pm ,Dm =Pm [k ],Dm [k ]
    if not len (Pm ):
        return P ,D ,kay 
    return (np .vstack ([P ,Pm ]),np .vstack ([D ,Dm ]),
    np .concatenate ([kay ,np .full (len (Pm ),2 ,int )]))
