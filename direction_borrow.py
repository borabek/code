# -*- coding: utf-8 -*-
"""YON ODUNC ALMA: konum `v_o`'da KALIR, only YON secilir.

WHY SEPARATE BIR KOL: bugune up to denenen five mimari ADAY EKLIYORDU and all of them
kesinligi seyreltip tabanin under kaldi
([[brep-pool-ceiling-acildi-selector-acamadi]]). Bu arm **candidate sayisini
DEGISTIRMEZ** -- each adayin only yonunu degistirebilir. Yapisal as
farklidir: FP count artamaz, only present which is a candidate robot-hazir hale gelir.

MEASURED (`results/konum_yon_{val,d7}.json`, mukemmel selector tavani, TEZ-SAF pool):
  VAL  robot F1 tavani 0.5928 -> 0.7008   (only direction odunc alarak)
  D7   robot F1 tavani 0.3412 -> 0.4401
Kaynak kirilimi (recall): komsu candidate yonu VAL +0.027 / D7 +0.030,
B-rep silindir ekseni VAL +0.086 / D7 +0.040, baskin direction +0.005.

TEZE SADIK: `v_o` mouth-ortasi KONUMU, 5 sinif segmentasyon and ~6000 remesh
DEGISMEZ. Tezin own yonu HER ZAMAN 0 numarali secenektir and real fallback'tir:
no secenek more iyi gorunmezse candidate tezin yonuyle cikar.
"""
import numpy as np 

KOMSU_R =10.0 # komsu adayin yonunu odunc alma yaricapi (mm)
EKSEN_R =10.0 # B-rep silindir eksenini odunc alma yaricapi (mm)
AYIRT_ACI =5.0 # this aciya more yakin secenekler AYNI sayilir (tekrar absent)

OZ_AD =["mevcut","aci_mevcuda","kaynak_komsu","kaynak_eksen","kaynak_baskin",
"distance","mesafe_diag","aci_baskina","eksen_hizasi","gate_skoru",
"n_aday","destek"]


def _birim (V ):
    V =np .asarray (V ,float ).reshape (-1 ,3 )
    return V /np .maximum (np .linalg .norm (V ,axis =1 ,keepdims =True ),1e-12 )


def _aci (a ,b ):
    return float (np .degrees (np .arccos (np .clip (abs (float (a @b )),-1.0 ,1.0 ))))


def baskin_yon (D ):
    """Parcanin sign-hizali baskin yonu (tum adaylarin ortalamasi)."""
    D =_birim (D )
    if len (D )==0 :
        return None 
    if len (D )==1 :
        return D [0 ]
    B =D *np .sign (D @D [0 ])[:,None ]
    return _birim ([B .mean (0 )])[0 ]


def secenekler (P ,D ,i ,cyl ,gate_s ,bask ,komsu_r =KOMSU_R ,eksen_r =EKSEN_R ):
    """`i` numarali candidate for (direction, feature) listesi. Ilk oge HER ZAMAN MEVCUT.

    Konum DEGISMEZ -- returns degerde konum absent, only direction.
    """
    P =np .asarray (P ,float ).reshape (-1 ,3 )
    D =_birim (D )
    p ,d =P [i ],D [i ]
    n =len (P )
    diag =float (np .linalg .norm (P .max (0 )-P .min (0 )))if n >1 else 1.0 

    candidates =[(d ,"mevcut",0.0 )]
    if n >1 :
        uz =np .linalg .norm (P -p ,axis =1 )
        for j in np .argsort (uz ):
            if j ==i or uz [j ]>komsu_r :
                continue 
            candidates .append ((D [j ],"komsu",float (uz [j ])))
    for c in cyl or []:
        a =np .asarray (c ["axis"],float )
        na =np .linalg .norm (a )
        if na <1e-9 :
            continue 
        a =a /na 
        m =np .asarray (c .get ("center",p ),float )
        u =float (np .linalg .norm (m -p ))
        if u <=eksen_r :
            candidates .append ((a ,"axis",u ))
            candidates .append ((-a ,"axis",u ))
    if bask is not None :
        candidates .append ((bask ,"baskin",0.0 ))

        # TEKRAR AYIKLAMA: each other `AYIRT_ACI`'dan yakin yonler AYNI secenektir.
        # Mevcut HER ZAMAN korunur (first sirada oldugu for dogal as kazanir).
    secili ,oz =[],[]
    for v ,src_ ,mes in candidates :
        v =_birim ([v ])[0 ]
        if any (_aci (v ,w )<AYIRT_ACI for w ,_ ,_ in secili ):
            continue 
        secili .append ((v ,src_ ,mes ))
    for v ,src_ ,mes in secili :
        destek =int (sum (1 for w in D if _aci (v ,w )<AYIRT_ACI ))
        oz .append ([
        float (src_ =="mevcut"),
        _aci (v ,d ),
        float (src_ =="komsu"),
        float (src_ =="axis"),
        float (src_ =="baskin"),
        mes ,
        mes /max (diag ,1e-6 ),
        _aci (v ,bask )if bask is not None else 0.0 ,
        float (np .max (np .abs (v ))),
        float (gate_s ),
        float (n ),
        float (destek ),
        ])
    return [v for v ,_ ,_ in secili ],np .asarray (oz ,float )


def uygula (P ,D ,cyl ,gate_skorlari ,puanla ):
    """Her candidate for most iyi yonu sec. Doner: new D (konum DOKUNULMAZ).

    `puanla(X)` -> each secenek for skor. MEVCUT (0. secenek) esitlikte KAZANIR:
    new direction however KESIN more iyiyse alinir, so tezin cevabi varsayilan kalir.
    """
    P =np .asarray (P ,float ).reshape (-1 ,3 )
    D =_birim (D )
    if len (P )==0 :
        return D 
    bask =baskin_yon (D )
    new_ =D .copy ()
    for i in range (len (P )):
        V ,X =secenekler (P ,D ,i ,cyl ,float (gate_skorlari [i ]),bask )
        if len (V )<2 :
            continue 
        s =np .asarray (puanla (X ),float )
        j =int (np .argmax (s ))
        if j !=0 and s [j ]>s [0 ]:
            new_ [i ]=V [j ]
    return new_ 
