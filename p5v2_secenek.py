# -*- coding: utf-8 -*-
"""p5-v2 SECENEK URETICI -- gate'ten ONCE, ORTAK secim for.

MEASURED (`gate-before-poz-after-tavani-kirpiyor`): gate'i poz seciminden ONCE
uygulamak ortak kahin tavanini 0.3781 -> 0.3155 kirpiyor. Bu modul, HAM candidate
havuzu for secenekleri produces; secim and gate SONRA gelir.

SECENEK TURLERI (tez sadakati: MEVCUT always 0 numarali secenek):
  0 MEVCUT   : `v_o` -- tezin own turetmesi, GERCEK fallback
  1 SILINDIR : B-rep silindir agzi (+/- axis)
  2 PLANAR   : duzlemsel face ic halkasi (slot/kare giris) (+/- normal)
  3 NULL     : adayi AT (p5-v2 a adayi elemekte serbest)

Bir FIZIKSEL AGIZ most extra a adaya verilir -- this kisit selector tarafinda
(bipartite) uygulanir; here only secenekler and OZNITELIKLERI uretilir.
"""
import numpy as np 

import os 
MM_MAX =8.0 # mouth-candidate distance siniri
# ABLASYON: planar (duzlemsel ic halka) seceneklerini KAPAT -> kazancin
# silindirlerden mi planarlardan mi geldigini separates.
P5V2_PLANAR =os .environ .get ("P5V2_PLANAR","1")=="1"
TUR_MEVCUT ,TUR_SILINDIR ,TUR_PLANAR ,TUR_NULL =0 ,1 ,2 ,3 
OZ_AD =["tur_silindir","tur_planar","tur_null",
"distance","mesafe_norm","aci_mevcut","radius","uzunluk",
"esd_r","cevre","alan","eksen_hiza","komsu_uyum",
"gate_skoru","votes","n_aday","n_secenek"]
# PARCA-ICI GORELI OZNITELIKLER: DENENDI, REVERTED (2026-08-09).
# Gate tarihindeki most large single kazanc part-ici z-skordu
# ([[part-ici-zskor-deployed]]) and same fikri buraya tasidim:
# candidate-ici ranking (distance/angle/radius) + part genelinde z-skor, 6 column.
# MEASURED (LOMO, same kurulum, single degisken):
#     ham 23 column -> secim +0.0391 | uctan uca 0.1465
#     ham 17 column -> secim +0.0501 | uctan uca 0.1649   <- IYI OLAN
# BOZDU. Muhtemel reason: gate'in oznitelikleri markalar arasi kiyaslanamaz
# MUTLAK buyukluklerdi; p5-v2'ninkiler ZATEN goreli (this adaydan this agza distance).
# Siralama bilgi katmiyor, RF'nin asiri uyacagi 6 column noise katiyor.
# TEKRAR DENEME -- before feature uzayini degistir, after normalizasyonu.
# NOT: `agiz_kimlik` OZ_AD'de YOK -- that a OZNITELIK not, bipartite KISIT
# anahtaridir and tuple'in 4. ogesi as ayri returns. (Ilk surumde listeye
# koymustum: 18 name / 17 value uyusmazligi olurdu.)


def _birim (v ):
    v =np .asarray (v ,float )
    n =np .linalg .norm (v )
    return v /n if n >1e-9 else np .array ([0.0 ,0.0 ,1.0 ])


def secenekler (P ,D ,cyls ,acik ,diag ,gate_s =None ,votes =None ,komsu =None ):
    """Her candidate for secenek listesi. Doner: list[list[(konum, direction, oz, agiz_id)]].

    `agiz_id`: same fiziksel agzi kullanan secenekler AYNI kimligi carries; selector
    a agzi two adaya veremesin diye. MEVCUT and NULL for -1 (kisit disi).
    """
    P =np .asarray (P ,float );D =np .asarray (D ,float )
    n =len (P )
    gate_s =np .zeros (n )if gate_s is None else np .asarray (gate_s ,float )
    votes =np .zeros (n )if votes is None else np .asarray (votes ,float )
    # fiziksel agizlar: (konum, direction, kind, radius, uzunluk, esd_r, cevre, alan)
    agizlar =[]
    for c in (cyls or []):
        a =_birim (c ["axis"])
        uzn =float (np .linalg .norm (np .asarray (c ["mouth_b"],float )-
        np .asarray (c ["mouth_a"],float )))
        for m in (c ["mouth_a"],c ["mouth_b"]):
            agizlar .append ((np .asarray (m ,float ),a ,TUR_SILINDIR ,
            float (c ["radius"]),uzn ,0.0 ,0.0 ,0.0 ))
    for o in ((acik or [])if P5V2_PLANAR else []):
        nrm =_birim (o .get ("normal",[0 ,0 ,1 ]))
        agizlar .append ((np .asarray (o ["center"],float ),nrm ,TUR_PLANAR ,
        0.0 ,0.0 ,float (o .get ("esd_r",0.0 )),
        float (o .get ("cevre",0.0 )),float (o .get ("alan",0.0 ))))
    out =[]
    for i in range (n ):
        o =[]
        u0 =_birim (D [i ])

        def oz (mes ,aci ,yar ,uzn ,esd ,cev ,aln ,tur ,direction ,n_sec ):
            return [float (tur ==TUR_SILINDIR ),float (tur ==TUR_PLANAR ),
            float (tur ==TUR_NULL ),
            mes ,mes /max (diag ,1e-6 ),aci ,yar ,uzn ,esd ,cev ,aln ,
            float (np .max (np .abs (direction ))),
            abs (float (direction @komsu ))if komsu is not None else 0.0 ,
            float (gate_s [i ]),float (votes [i ]),float (n ),float (n_sec )]

            # 0) MEVCUT -- tezin cevabi, HER ZAMAN first
        o .append ((P [i ],u0 ,oz (0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,
        TUR_MEVCUT ,u0 ,0 ),-1 ))
        for k ,(m ,a ,tur ,yar ,uzn ,esd ,cev ,aln )in enumerate (agizlar ):
            mes =float (np .linalg .norm (m -P [i ]))
            if mes >MM_MAX :
                continue 
            for sg in (1.0 ,-1.0 ):
                y =sg *a 
                aci =float (np .degrees (np .arccos (np .clip (abs (float (y @u0 )),-1 ,1 ))))
                o .append ((m ,y ,oz (mes ,aci ,yar ,uzn ,esd ,cev ,aln ,tur ,y ,0 ),k ))
                # 3) NULL -- adayi at
        o .append ((None ,None ,oz (0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,
        TUR_NULL ,u0 ,0 ),-1 ))
        for t in o :# n_secenek'i geriye yaz
            t [2 ][-1 ]=float (len (o ))
        out .append (o )
    return out 
