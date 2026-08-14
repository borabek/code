# -*- coding: utf-8 -*-
"""EK OZNITELIK BLOGU: single degiskenli kiyas cercevesi.

`EK_BLOK` cevre degiskeniyle secilen a blok, mevcut 95 sutunun UZERINE
eklenir and `full` MARKA KATLARINDA blogu OLAN / OLMAYAN two arm karsilastirilir.
Tek degisken bloktur: same corpus, same katlar, same kurallar, same seed.

Bloklar:
  simetri   4 column  -- ayna simetri esi present mi (klemensler simetriktir)
  depth 12 column  -- axis boyu radius profili (tel/vida/alet ayrimi)
  kafes_adet 3 column -- lattice adiminden beklenen CP count and secim baskisi
  ozkalib   3 column  -- part-ici oz-kalibrasyon (transduktif yeniden ranking)

KAPI: +0.01 under kalan blok ATILIR. Oznitelik sisirmek modeli bozuyor --
P6_GEO deneyi (-0.0154) and p5-v2'nin goreli sutunlari bunu showed.

D7'ye BAKILMAZ.
"""
import collections 
import json 
import os 
import sys 
import time 

import numpy as np 
from sklearn .ensemble import HistGradientBoostingClassifier 

import makbuz_hash 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1"
os .environ ["WG_TOPO"]="1"
os .environ ["WG_ZENGIN"]="1"
os .environ .setdefault ("P6_DIZIN","results/_p6_oz_u25")
sys .path .insert (0 ,".")
import lattice # noqa: E402
import canonical_d7 as K # noqa: E402
import p6_decision # noqa: E402
from run_p6_ortak import yukle # noqa: E402
from sina_cluster import match_hungarian # noqa: E402

BLOK =os .environ .get ("EK_BLOK","simetri")
KURALLAR =([("mutlak",e )for e in (0.20 ,0.40 ,0.60 ,0.80 ,0.90 ,0.95 ,0.97 )]+
[("goreli",o ,t )for o in (0.50 ,0.70 ,0.85 ,0.95 )
for t in (0.05 ,0.20 ,0.40 )])
NMS =5.0 
ARAMA_N =int (os .environ .get ("P6_ARAMA_N","250"))
ITER =int (os .environ .get ("P6_ITER","200"))
NEG_KAT =int (os .environ .get ("P6_NEG_KAT","6"))
KAT_MIN =int (os .environ .get ("P6_KAT_MIN","200"))
MESH_DIZ ={"tam":"results/_p1_olasilik_brepegit","d6":"results/_p1_olasilik"}


def temel (d ):
    return np .hstack ([p6_decision .donustur (d ["X"]),
    p6_decision .kaynak_blok (d ["kaynak"][d ["idx"]])])


def yigin_f32 (ogeler ,uret ,satir ):
    """float64 ARA YIGIN OLMADAN float32 matris kur.

    `np.vstack([...]).astype(np.float32)` before HEPSINI float64 merges:
    full-open korpusta 8.414.677 row x 162 column = **10.2 GiB** and
    MemoryError -- gece 04:15'te `kanonik` blogu full boyle dustu. Satir count
    onceden bilindigi for array DOGRUDAN float32 ayrilir and part part
    doldurulur: vertex bellek yariya iner, ara kopya kalmaz.

    `uret(oge)` matris, `row(oge)` that ogenin row sayisini gives.
    """
    n_satir =sum (satir (o )for o in ogeler )
    ilk =np .asarray (uret (ogeler [0 ]),np .float32 )
    M =np .empty ((n_satir ,ilk .shape [1 ]),np .float32 )
    M [:len (ilk )]=ilk 
    y =len (ilk )
    for o in ogeler [1 :]:
        b =uret (o )
        M [y :y +len (b )]=b 
        y +=len (b )
    if y !=n_satir :# SESSIZ UYUMSUZLUK OLMASIN
        raise ValueError (f"satir sayisi tutmadi: {y } != {n_satir }")
    return M 


def ek_blok (d ,s1 ):
    """Secilen blogu SECENEK BASINA uret. Tohumlar HER ZAMAN tahminden."""
    P =d ["P"][d ["idx"]]
    YD =d ["YD"]
    if BLOK =="simetri":
        import simetri 
        mf =f"{MESH_DIZ [d ['_kume']]}/{d ['pid']}.npz"
        V =(np .asarray (np .load (mf )["V"],float )
        if os .path .exists (mf )else np .zeros ((0 ,3 )))
        return simetri .oznitelik (P ,V ,skor =s1 )
    if BLOK =="depth":
        import trimesh 

        import derinlik_profili as DP 
        mf =f"{MESH_DIZ [d ['_kume']]}/{d ['pid']}.npz"
        if not os .path .exists (mf ):
            return np .zeros ((len (P ),len (DP .OZ_AD )))
        z =np .load (mf )
        V =np .ascontiguousarray (z ["V"],np .float64 )
        F =np .ascontiguousarray (z ["F"],np .int64 )
        mesh =trimesh .Trimesh (V ,F ,process =False )
        diag =float (np .linalg .norm (V .max (0 )-V .min (0 )))
        return DP .profil (P ,YD ,mesh ,diag )
    if BLOK =="kafes_adet":
    # lattice adimindan BEKLENEN CP count -> secim baskisi
        Pt ,Dt =p6_decision .sec_ayrintili (
        d ["P"],d ["idx"],d ["YD"],s1 ,("mutlak",0.6 ),nms_mm =5.0 )[:2 ]
        t =lattice .otelemeler (Pt )if len (Pt )>=2 else []
        if not t :
            return np .zeros ((len (P ),3 ))
        adim =float (np .linalg .norm (t [0 ][0 ]))
        u =t [0 ][0 ]/max (adim ,1e-9 )
        pr =Pt @u 
        uzanim =float (pr .max ()-pr .min ())if len (pr )>1 else 0.0 
        bek =uzanim /adim +1.0 
        n_tohum =float (len (Pt ))
        return np .tile ([bek ,n_tohum ,bek -n_tohum ],(len (P ),1 ))
    if BLOK =="cluster":
    # ADAYLAR ARASI BAGLAM: noktasal selector "same delige bakan digerleri"
    # and "5mm otedeki more guclu rakip" bilgisini HIC gormuyor. D2
    # (candidate-kumesi transformer) kolunun ucuz yaklasimidir.
        import kume_baglami as KM 
        return KM .oznitelik (P ,YD ,d ["idx"],s1 ,d ["diag"])
    if BLOK =="kanonik":
    # PARCANIN KENDI EKSEN SISTEMI: gorulmemis markada modelleme ekseni
    # bizimkiyle same olmak zorunda not; dunya koordinati ogrenilen each
    # konumsal kalibi bozuyor.
        import kanonik_hizalama as KH 
        mf =f"{MESH_DIZ [d ['_kume']]}/{d ['pid']}.npz"
        V =(np .asarray (np .load (mf )["V"],float )
        if os .path .exists (mf )else d ["P"])
        return KH .oznitelik (P ,YD ,V )
    if BLOK =="topoloji":
    # ES-EKSENLI AILE: candidate a dizinin uyesi mi, only mi. Mesh/isin
    # GEREKMEZ -- only candidate konumlari and secenek yonu.
        import topoloji_ailesi as TA 
        return TA .oznitelik (P ,YD ,d ["P"],d ["diag"])
    if BLOK =="ozkalib":
    # PARCA-ICI OZ-KALIBRASYON: skorun part icindeki yuzdeligi, most
    # yuksekten farki and yerel komsulukta kacinci oldugu
        s =np .asarray (s1 ,float )
        if not len (s ):
            return np .zeros ((len (P ),3 ))
        sira =np .argsort (np .argsort (-s ))/max (len (s )-1 ,1 )
        return np .stack ([sira ,s /max (s .max (),1e-9 ),
        s -float (np .median (s ))],axis =1 )
    raise ValueError (BLOK )


    # ---------------------------------------------------------------- PARALEL
    # `depth` blogu part basina ~10 s (candidate x 6 depth x 8 isin). 3051
    # parcada 8.5 SAAT -> single basina butun geceyi yer. Blok part basina bagimsiz
    # oldugu for havuza dagitilir.
    #
    # ISCILERE TAM PARCA SOZLUGU GONDERILMEZ: `X` single basina part basina
    # yuzbinlerce float and 3051 parcayi pickle'lamak gigabaytlar demek. `ek_blok`
    # only this five alani okuyor; slim yuk onlari carries.
    # DOGRULANDI (2026-08-12): paralel path seri yolla BIT-AYNI sonuc veriyor
    # (kanonik and topoloji bloklarinda maks difference 0). Once "paralel path oluyor"
    # sanilmisti; that olumlerin sebebi pool DEGIL, arka plan gorevi kapaninca
    # cocuk surecin de kapanmasiydi -- test ON PLAN cagrisindan kosunca gecti.
    # YALNIZ `depth` for acin: ucuz bloklarda process acma maliyeti kazanci
    # yiyor (kanonik 0.0s -> 1.9s). depth ~10 s/part, orada kazanc real.
ISCI =int (os .environ .get ("EK_ISCI","1"))


def _slim (d ):
    return {"pid":d ["pid"],"_kume":d ["_kume"],"P":d ["P"],
    "idx":d ["idx"],"YD":d ["YD"],"diag":d ["diag"]}


def _ek_bir (a ):
    return ek_blok (a [0 ],a [1 ])


def ek_hepsi (veri ,oof ):
    isler =[(_slim (d ),np .asarray (s ,float ))for d ,s in zip (veri ,oof )]
    if ISCI <=1 :
        return [_ek_bir (a )for a in isler ]
    import multiprocessing as mp 
    with mp .Pool (ISCI )as p :
        return p .map (_ek_bir ,isler ,chunksize =2 )


def puanla (veri ,skor ,kural ):
    per =collections .defaultdict (lambda :[0 ,0 ,0 ])
    for d ,s in zip (veri ,skor ):
        P ,D =p6_decision .sec (d ["P"],d ["idx"],d ["YD"],s ,kural ,nms_mm =NMS )
        a ,b ,c =match_hungarian (P ,D ,d ["G"],d ["Gd"],d ["diag"],K .YANAL ,K .ACI ,
        False ,signed =True )[:3 ]
        q =per [d ["mfg"]]
        q [0 ]+=a ;q [1 ]+=b ;q [2 ]+=c 
    pm ={m :2 *q [0 ]/max (2 *q [0 ]+q [1 ]+q [2 ],1 )for m ,q in per .items ()}
    T =[sum (q [i ]for q in per .values ())for i in range (3 )]
    return {"robot":2 *T [0 ]/max (2 *T [0 ]+T [1 ]+T [2 ],1 ),
    "makro":float (np .mean (list (pm .values ())))if pm else 0.0 ,
    "TP":T [0 ],"FP":T [1 ],"FN":T [2 ]}


def main ():
    t0 =time .time ()
    veri =[]
    for cluster in os .environ .get ("P6_KUME","tam,d6").split (","):
        cluster =cluster .strip ()
        for d in yukle (cluster ,int (os .environ .get ("P6_TR","0"))):
            d ["_kume"]=cluster 
            veri .append (d )
    for d in veri :
        d ["y"]=np .asarray (d ["y"],int )
    print (f"BLOK={BLOK } | {len (veri )} part ({time .time ()-t0 :.0f} s)",
    flush =True )

    brand =collections .Counter (d ["mfg"]for d in veri )
    katlar =[m for m ,n in brand .items ()if n >=KAT_MIN ]
    print (f"katlar: {katlar }",flush =True )

    # OOF birinci kademe skorlari (blok tohumu for) -- brand katli
    oof =[None ]*len (veri )
    for b in katlar :
        ic =[i for i ,d in enumerate (veri )if d ["mfg"]!=b ]
        dis =[i for i ,d in enumerate (veri )if d ["mfg"]==b ]
        M =yigin_f32 (ic ,lambda i :temel (veri [i ]),
        lambda i :len (veri [i ]["y"]))
        Y =np .concatenate ([veri [i ]["y"]for i in ic ])
        rng =np .random .default_rng (0 )
        poz =np .where (Y ==1 )[0 ]
        neg =np .where (Y ==0 )[0 ]
        sec =np .concatenate ([poz ,rng .choice (
        neg ,min (len (neg ),NEG_KAT *max (len (poz ),1 )),replace =False )])
        m =HistGradientBoostingClassifier (
        max_iter =ITER ,learning_rate =0.06 ,max_leaf_nodes =63 ,
        l2_regularization =1.0 ,random_state =0 ).fit (M [sec ],Y [sec ])
        for i in dis :
            oof [i ]=m .predict_proba (
            temel (veri [i ]).astype (np .float32 ))[:,1 ]
        print (f"  OOF {b } ({time .time ()-t0 :.0f} s)",flush =True )
    for i ,s in enumerate (oof ):
        if s is None :
            oof [i ]=np .full (len (veri [i ]["X"]),0.5 )

    print (f"ek blok hesaplaniyor ({ISCI } isci)...",flush =True )
    EK =ek_hepsi (veri ,oof )
    print (f"blok {np .vstack (EK ).shape } ({time .time ()-t0 :.0f} s)",flush =True )

    top ={"YOK":collections .Counter (),"VAR":collections .Counter ()}
    for b in katlar :
        ic =[i for i ,d in enumerate (veri )if d ["mfg"]!=b ]
        dis =[i for i ,d in enumerate (veri )if d ["mfg"]==b ]
        for ad in ("YOK","VAR"):
            def mat (i ):
                return (np .hstack ([temel (veri [i ]),EK [i ]])if ad =="VAR"
                else temel (veri [i ])).astype (np .float32 )
            M =yigin_f32 (ic ,mat ,lambda i :len (veri [i ]["y"]))
            Y =np .concatenate ([veri [i ]["y"]for i in ic ])
            rng =np .random .default_rng (0 )
            poz =np .where (Y ==1 )[0 ]
            neg =np .where (Y ==0 )[0 ]
            sec =np .concatenate ([poz ,rng .choice (
            neg ,min (len (neg ),NEG_KAT *max (len (poz ),1 )),replace =False )])
            m =HistGradientBoostingClassifier (
            max_iter =ITER ,learning_rate =0.06 ,max_leaf_nodes =63 ,
            l2_regularization =1.0 ,random_state =0 ).fit (M [sec ],Y [sec ])
            s_ic =[m .predict_proba (mat (i ))[:,1 ]for i in ic ]
            s_dis =[m .predict_proba (mat (i ))[:,1 ]for i in dis ]
            ar =(np .random .default_rng (0 ).choice (len (ic ),ARAMA_N ,False )
            if len (ic )>ARAMA_N else np .arange (len (ic )))
            AR =[veri [ic [i ]]for i in ar ]
            AS =[s_ic [i ]for i in ar ]
            en =max (KURALLAR ,key =lambda k :puanla (AR ,AS ,k )["makro"])
            r =puanla ([veri [i ]for i in dis ],s_dis ,en )
            for k_ in ("TP","FP","FN"):
                top [ad ][k_ ]+=r [k_ ]
        print (f"  {b :<6} YOK {2 *top ['YOK']['TP']:.0f}TP | "
        f"VAR {2 *top ['VAR']['TP']:.0f}TP ({time .time ()-t0 :.0f} s)",
        flush =True )

    son ={}
    for ad in ("YOK","VAR"):
        c =top [ad ]
        son [ad ]=2 *c ["TP"]/max (2 *c ["TP"]+c ["FP"]+c ["FN"],1 )
    fark =son ["VAR"]-son ["YOK"]
    print (f"\nBLOK {BLOK }: YOK {son ['YOK']:.4f} -> VAR {son ['VAR']:.4f} "
    f"({fark :+.4f})")
    print (f"KAPI: +0.01 -> {'GECTI'if fark >=0.01 else 'GECMEDI'}")
    json .dump ({"damga":makbuz_hash .damga (),"blok":BLOK ,
    "yok":son ["YOK"],"var":son ["VAR"],"fark":fark ,
    "gecti":bool (fark >=0.01 ),"katlar":katlar ,
    "n_parca":len (veri ),
    "not":"Tek degiskenli ek-oznitelik kiyasi. tam brand katlari, "
    "MAKRO kural secimi. D7'ye BAKILMADI."},
    open (f"results/ek_blok_{BLOK }.json","w"),indent =1 )
    print (f"receipt -> results/ek_blok_{BLOK }.json")


if __name__ =="__main__":
    main ()
