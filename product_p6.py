# -*- coding: utf-8 -*-
"""URUN P6 YOLU: (konum x direction) ortak siralayici.

`product_genis.output` with AYNI imza -- `canonical_chain.product_output` bunu a config
anahtariyla cagirabilsin diye. Kol calisamazsa (model/STEP/B-rep absent) None returns
and cagiran BIR ONCEKI yola duser; sessizce bozuk output URETILMEZ.

FARKI (single cumleyle): `product_genis` each adayi havuzun verdigi TEK yonle puanlar;
this modul each adaya `direction_bank` seceneklerini takar and (konum, direction) ciftini TEK
skorla siralar. Yon residual SECILIR.

ISARET DUZELTME YOK -- and this kasten. `product_genis.isaret_duzelt` fiziksel a
kuralla yonu ters cevirir (+0.0316 olculmustu); here +u and -u ZATEN ayri two
secenek as bankada and siralayici hangisinin correct oldugunu ogrenir. Kurali
ustune koymak, ogrenilen karari eziyor. `P6_ISARET=1` with acilir (ablasyon).
"""
import os 

import numpy as np 

import p6_decision 
import product_genis 
import direction_bank as YB 

MODEL_YOL =os .environ .get ("P6_MODEL","results/p6_kademe2_model.pkl")
_MODEL =None 
ISARET =os .environ .get ("P6_ISARET","0")=="1"
MESH_HAVUZ =os .environ .get ("P6_MESH_HAVUZ","1")=="1"


def _cfg (ad ,cevre ,vars_ ):
    v =os .environ .get (cevre )
    if v is not None :
        return v not in ("0","","false","False")
    try :
        import json 
        return bool (json .load (open ("cp_config.json",encoding ="utf-8"))
        .get (ad ,vars_ ))
    except Exception :
        return vars_ 


ACIK =_cfg ("robot_p6_ortak","URUN_P6",False )


def model_yukle (yol =MODEL_YOL ):
    """Egitimin yazdigi PAKETI oku. Doner: dictionary ya da None.

    Paket: kademe1 (+ istege bagli kademe2), karar kurali, NMS, seed kurali.
    Tek a yerden okunur ki urun with training AYNI kurali kullansin.
    """
    global _MODEL 
    if _MODEL is None :
        if not os .path .exists (yol ):
            return None 
        import pickle 
        _MODEL =pickle .load (open (yol ,"rb"))
    return _MODEL 


def secenek_tablosu (V ,F ,probs ,cps_seg ,step_path ,CE ,CT ):
    """Havuz + direction bankasi + 92 sutunluk feature. Doner: (P, idx, YD, X) ya da None.

    Egitim betikleri de bunu cagirabilsin diye ayri: so egitimdeki feature
    with urundeki feature AYNI koddan cikar.
    """
    import trimesh 

    import brep_pool 
    import thin_pool 
    import wire_gate 
    cyl ,acik =product_genis .brep_cikar (step_path )
    if cyl is None :
        return None 
    Ps =np .asarray ([c ["point"]for c in cps_seg ],float )
    Ds =np .asarray ([c ["direction"]for c in cps_seg ],float )
    P ,D ,src_ =product_genis .pool (Ps ,Ds ,cyl ,acik )
    if len (P )<2 :
        return None 
    if MESH_HAVUZ :
    # MESH TEPESI HAVUZU. Olculdu (D6): only-konum recall 0.5371 -> 0.9768.
    # Tarihte UC times zarar vermisti because direction bankasi yoktu; single basina
    # only FP uretiyor. Burada yonu ortak siralayici seciyor.
    # Seyreltme kurali `thin_pool` -- EGITIMDEKIYLE AYNI FONKSIYON.
        pp =thin_pool .ppos (probs ,CE ,CT )
        Pm ,Dm =brep_pool .mesh_adaylari (V ,F ,pp ,brep_pool .MESH_ESIK ,
        brep_pool .MESH_DEDUPE_MM )
        if len (Pm ):
            uz =np .linalg .norm (Pm [:,None ]-P [None ],axis =-1 ).min (1 )
            k =uz >=brep_pool .MESH_DEDUPE_MM 
            Pm ,Dm =Pm [k ],Dm [k ]
        if len (Pm ):
            sm =pp [np .argmin (np .linalg .norm (
            Pm [:,None ,:]-np .asarray (V ,float )[None ,:,:],
            axis =-1 ),axis =1 )]if len (Pm )*len (V )<6e7 else np .zeros (len (Pm ))
            s_ =thin_pool .seyrelt (Pm ,sm ,len (P ))
            if len (s_ ):
                P =np .vstack ([P ,Pm [s_ ]])
                D =np .vstack ([D ,Dm [s_ ]])
                src_ =np .concatenate ([src_ ,np .full (len (s_ ),2 ,int )])
    V =np .asarray (V ,float )
    diag =float (np .linalg .norm (V .max (0 )-V .min (0 )))
    mesh =trimesh .Trimesh (V ,np .asarray (F ,np .int64 ),process =False )
    A =np .asarray (wire_gate .feats_for (
    V ,F ,probs ,[{"point":P [i ],"direction":D [i ]}
    for i in range (len (P ))],CE ,CT ,step_path =step_path ),
    float )
    B =product_genis .tanimlayici (P ,D ,cyl ,mesh ,diag )
    # YELPAZE only mesh OLMAYAN adaylara -- egitimdekiyle AYNI rule.
    idx ,YD ,C =YB .secenekler (P ,D ,cyl ,V ,mesh =mesh ,diag =diag ,
    fan_maske =(np .asarray (src_ ,int )!=2 ))
    if not len (idx ):
        return None 
    Dblok =product_genis .tanimlayici (P [idx ],YD ,cyl ,mesh ,diag )
    return P ,idx ,YD ,np .hstack ([A [idx ],B [idx ],C ,Dblok ]),src_ 


    # SESSIZ GERI DUSME SAYACI. Kol calisamazsa `None` returns and cagiran ESKI yola
    # duser -- this correct davranis, but SIK olursa "P6 sonucu" aslinda baseline sonucudur
    # and this SESSIZ becomes. Olcum betikleri this sayaci makbuza writes.
SAYAC ={"cagri":0 ,"p6":0 ,"model_yok":0 ,"aday_yok":0 ,"tablo_yok":0 ,
"rejim_disi":0 }


def _rejim_gecer (pk ,P ,src_ ):
    """REJIM KAPISI: this parcada P6 mi, DAGITILAN TABAN mi?

    MEASURED (real training, brand katlari):
        WEI  TABAN 0.2797 -> P6 0.3795  (+0.0998)
        PXC  TABAN 0.5850 -> P6 0.3972  (-0.1878)
        SIE  TABAN 0.6226 -> P6 0.4681  (-0.1545)
    Mesh havuzu HER PARCADA correct arac not: tabanin already guclu oldugu
    sparse/temiz parcalarda havuzu uce katlamak kesinligi boguyor. Kapi,
    inference aninda gorulebilen a istatistige (candidate sayilari) bakar; threshold
    `run_p6_rejim.py` with MARKA KATLARINDA secilir, sinavda taranmaz.

    Kapiyi gecemeyen parcada `output` None returns and `canonical_chain` ZATEN VAR
    OLAN geri-dusmeyle old yola gecer -- new a kod yolu acilmaz.
    """
    r =pk .get ("regime")
    if not r :
        return True 
    k =np .asarray (src_ ,int )
    n01 =int ((k !=2 ).sum ())
    nm =int ((k ==2 ).sum ())
    val_ ={"n01":float (n01 ),"mesh_oran":nm /max (n01 ,1 ),
    "n_aday":float (len (P )),
    "n_secenek":float (len (P ))}.get (r ["istatistik"])
    if val_ is None :
        return True 
    return val_ >=float (r ["threshold"])


def out_ (V ,F ,probs ,cps_seg ,step_path ,CE ,CT ):
    SAYAC ["cagri"]+=1 
    pk =model_yukle ()
    if pk is None :
        SAYAC ["model_yok"]+=1 
        return None 
    if not cps_seg :
        SAYAC ["aday_yok"]+=1 
        return None 
    tab =secenek_tablosu (V ,F ,probs ,cps_seg ,step_path ,CE ,CT )
    if tab is None :
        SAYAC ["tablo_yok"]+=1 
        return None 
    P ,idx ,YD ,X ,src_ =tab 
    if not _rejim_gecer (pk ,P ,src_ ):
        SAYAC ["rejim_disi"]+=1 
        return None 
    SAYAC ["p6"]+=1 # BU parcada P6 gercekten kullanildi
    zskor =pk .get ("zskor","ab")
    # EGITIMDEKI SUTUN SIRASI: [donusturulmus 92] + [source gostergesi 3]
    Xd =np .hstack ([p6_decision .donustur (X ,zskor ),
    p6_decision .kaynak_blok (src_ [idx ])])
    if pk .get ("arm")=="P6_GEO":
    # SEGMENTASYON BLOGU (first 58 column) ATILIR -- egitimdekiyle AYNI dilim.
        ab =int (pk .get ("AB",67 ))
        Xd =np .hstack ([Xd [:,58 :ab ],Xd [:,ab :]])
    s =np .asarray (pk ["kademe1"].predict_proba (Xd )[:,1 ],float )
    if pk .get ("kademe2")is not None :
    # IKINCI KADEME = KISA LISTE UZERINDE FP REDDEDICI.
    # Birinci gecisin YUKSEK GUVENLI secimleri TOHUM becomes, periyodik yapi
    # olculeri cikar; ikinci model only `kisa_esik`i gecen secenekleri
    # yeniden puanlar and birinci kademe skorunu da OZNITELIK as takes.
    # Kisa list DISI satirlar 0 kalir -- ikinci kademe birinci kademeyi
    # EZEMEZ, only icinden selects. Tohumlar only tahminden gelir; GT this
    # yola HIC girmez.
        import lattice 
        Pt ,Dt =p6_decision .sec_ayrintili (
        P ,idx ,YD ,s ,tuple (pk ["tohum_kural"]),
        nms_mm =float (pk ["tohum_nms"]))[:2 ]
        kb =lattice .oznitelik (P [idx ],YD ,Pt ,Dt )
        k =np .where (s >=float (pk .get ("kisa_esik",0.20 )))[0 ]
        s2 =np .zeros (len (s ))
        if len (k ):
            par =[Xd [k ],kb [k ]]
            if pk .get ("sira"):# EGITIMDEKI column sirasiyla AYNI
                import order_stamp 
                par .append (order_stamp .oznitelik (P [idx ],YD ,Pt ,Dt )[k ])
            par .append (s [k ][:,None ])
            X2 =np .hstack (par )
            s2 [k ]=pk ["kademe2"].predict_proba (X2 .astype (np .float32 ))[:,1 ]
        s =s2 
    P2 ,D2 ,_ai ,S2 =p6_decision .sec_ayrintili (P ,idx ,YD ,s ,tuple (pk ["rule"]),
    nms_mm =float (pk ["nms"]))
    if ISARET and len (P2 ):
        T2 =product_genis .tanimlayici (P2 ,D2 ,*_mesh_arg (V ,F ))
        D2 =product_genis .isaret_duzelt (D2 ,T2 )
        # `wire_score` GERCEK skordur (eskiden sabit 1.0 idi). Guven kapili GLB
        # bunun uzerine kurulur: precision >=0.90 verecek threshold kalibre edilir,
        # ustundekiler ONAYLI, altindakiler ONERI becomes.
    return [{"point":P2 [i ],"direction":D2 [i ],"wire_score":float (S2 [i ])}
    for i in range (len (P2 ))]


def _mesh_arg (V ,F ):
    import trimesh 
    V =np .asarray (V ,float )
    return (None ,trimesh .Trimesh (V ,np .asarray (F ,np .int64 ),process =False ),
    float (np .linalg .norm (V .max (0 )-V .min (0 ))))
