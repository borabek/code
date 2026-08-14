# -*- coding: utf-8 -*-
"""YENI-8 -- KONUM TOPLAMA: 24 direction secenegi = 24 PIYANGO BILETI

DIAGNOSIS (results/aday_auc_d6.json). Aday duzeyinde secicinin AUC'si iyi
(NIT 0.8854, UPUN 0.9877). Bagliyan sey AUC not, ADAY/GT orani:
  NIT  6322 secenek / 24 GT   -> gereken AUC 0.9968
  MOR 11555 secenek /  9 GT   -> gereken AUC 0.9997
Ama this 6322 count KONUM count not: each konum MAX_SEC=24 direction secenegi
uretiyor, i.e. ~260 konum x 24 direction.

MEKANIZMA. Bir konumu "most high skorlu secenegiyle" siralamak, each YANLIS
konuma 24 piyango bileti vermektir. Yanlis a konumun 24 denemeden birinde
high skor kapma olasiligi, correct konumun single real sinyalini bastirir.
Bu klasik coklu-karsilastirma sismesidir and MAX_SEC 12->24'un tavani acip
gerceklesen F1'i acmamasini da aciklar.

COZUM. Konum skorunu MAX with not, sisme yapmayan a toplamayla kur.
Yanlis konumda skorlar rastgele dagilir (high max, low median);
correct konumda BIRCOK secenek makul skor takes (high median).

KOLLAR (konum siralamasi; direction always that konumun most iyi secenegi):
  max        : bugunku
  median    : medyan -- piyangoyu tamamen sondurur
  ort        : mean
  q75        : 75. yuzdelik -- max with median arasi
  say        : skoru 0.5 ustunde which is secenek SAYISI
  maxxort    : max * mean (gucbirligi)
  ust2       : most high IKI secenegin ortalamasi

OLCULEN: konum duzeyinde first-k orani and UCTAN UCA robot F1 (threshold kurali).
KAPI: mikro robot F1'de +0.01.
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
os .environ .setdefault ("P6_DIZIN","results/_p6_oz_tam4")
sys .path .insert (0 ,".")
import canonical_d7 as K # noqa: E402
import p6_decision # noqa: E402
from run_p6_ortak import yukle # noqa: E402
from sina_cluster import match_hungarian # noqa: E402

KUME =os .environ .get ("KT_KUME","d6")
KAT_MIN =int (os .environ .get ("KT_KAT_MIN","40"))
ITER =int (os .environ .get ("P6_ITER","200"))
NEG_KAT =int (os .environ .get ("P6_NEG_KAT","6"))
NMS =5.0 
KOLLAR =("max","ortanca","ort","q75","say","maxxort","ust2")
ESIKLER =(0.20 ,0.40 ,0.60 ,0.80 ,0.90 ,0.95 )


def temel (d ):
    return np .hstack ([p6_decision .donustur (d ["X"]),
    p6_decision .kaynak_blok (d ["kaynak"][d ["idx"]])]).astype (
    np .float32 )


def konum_skoru (s ,idx ,arm ):
    """each BENZERSIZ konum for (skor, that konumun most iyi secenek indisi)."""
    rank_ =np .argsort (idx ,kind ="stable")
    idx_s ,s_s =idx [rank_ ],s [rank_ ]
    bound_ =np .flatnonzero (np .diff (idx_s ))+1 
    parts =np .split (np .arange (len (idx_s )),bound_ )
    konum ,skor ,en_iyi =[],[],[]
    for p in parts :
        q =s_s [p ]
        if arm =="max":
            v =q .max ()
        elif arm =="ortanca":
            v =np .median (q )
        elif arm =="ort":
            v =q .mean ()
        elif arm =="q75":
            v =np .percentile (q ,75 )
        elif arm =="say":
            v =float ((q >=0.5 ).sum ())/len (q )
        elif arm =="maxxort":
            v =q .max ()*q .mean ()
        elif arm =="ust2":
            v =np .sort (q )[-2 :].mean ()
        else :
            raise ValueError (arm )
        konum .append (idx_s [p [0 ]])
        skor .append (float (v ))
        en_iyi .append (int (rank_ [p [int (np .argmax (q ))]]))
    return (np .asarray (konum ,int ),np .asarray (skor ,float ),
    np .asarray (en_iyi ,int ))


def konum_dogru (y ,idx ):
    """each BENZERSIZ konum (konum_skoru with AYNI sirada) correct mu.

    Dongusuz: konum_skoru like idx'e according to siralayip parcalara boler.
    Naif hali (each konum for `y[idx == ki]`) 6322 secenek x 260 konum x
    7 arm x 468 part = milyarlarca islem ederdi; also kola bagli
    olmadigi for part basina BIR times is computed.
    """
    rank_ =np .argsort (idx ,kind ="stable")
    idx_s ,y_s =idx [rank_ ],y [rank_ ]
    bound_ =np .flatnonzero (np .diff (idx_s ))+1 
    return np .asarray ([bool (q .max ())if len (q )else False 
    for q in np .split (y_s ,bound_ )],bool )


def sec_esik (P ,YD ,idx ,ks ,en_iyi ,threshold ):
    """konum skoru esigi + NMS; direction = that konumun most iyi secenegi."""
    tut =np .where (ks >=threshold )[0 ]
    if not len (tut ):
        tut =np .array ([int (np .argmax (ks ))])
    tut =tut [np .argsort (-ks [tut ])]
    sp ,sd =[],[]
    for j in tut :
        p =P [j ]
        if sp and min (np .linalg .norm (np .asarray (sp )-p ,axis =1 ))<NMS :
            continue 
        sp .append (p )
        sd .append (YD [en_iyi [j ]])
    return (np .asarray (sp ).reshape (-1 ,3 ),np .asarray (sd ).reshape (-1 ,3 ))


def main ():
    t0 =time .time ()
    data_ =yukle (KUME ,int (os .environ .get ("P6_TR","0")))
    for d in data_ :
        d ["y"]=np .asarray (d ["y"],int )
        d ["_M"]=temel (d )
    brand =collections .Counter (d ["mfg"]for d in data_ )
    katlar =[m for m ,n in brand .items ()if n >=KAT_MIN ]
    print (f"{len (data_ )} part | katlar {katlar }",flush =True )

    oof =[None ]*len (data_ )
    for b in katlar :
        ic =[i for i ,d in enumerate (data_ )if d ["mfg"]!=b ]
        dis =[i for i ,d in enumerate (data_ )if d ["mfg"]==b ]
        n_s =sum (len (data_ [i ]["y"])for i in ic )
        M =np .empty ((n_s ,data_ [0 ]["_M"].shape [1 ]),np .float32 )
        o =0 
        for i in ic :
            m_ =data_ [i ]["_M"]
            M [o :o +len (m_ )]=m_ 
            o +=len (m_ )
        Y =np .concatenate ([data_ [i ]["y"]for i in ic ])
        rng =np .random .default_rng (0 )
        poz ,neg =np .where (Y ==1 )[0 ],np .where (Y ==0 )[0 ]
        sec =np .concatenate ([poz ,rng .choice (
        neg ,min (len (neg ),NEG_KAT *max (len (poz ),1 )),replace =False )])
        m =HistGradientBoostingClassifier (
        max_iter =ITER ,learning_rate =0.06 ,max_leaf_nodes =63 ,
        l2_regularization =1.0 ,random_state =0 ).fit (M [sec ],Y [sec ])
        del M 
        for i in dis :
            oof [i ]=m .predict_proba (data_ [i ]["_M"])[:,1 ]
        print (f"  OOF {b } ({time .time ()-t0 :.0f} s)",flush =True )

        # threshold, KAT-DISI secilir: each arm for training markalarindan not,
        # here single cluster oldugu for TUM kollara AYNI threshold listesi uygulanip
        # most iyisi AYRI raporlanir (arm karsilastirmasi esikten bagimsiz olsun).
    agg ={k :{e :collections .defaultdict (collections .Counter )
    for e in ESIKLER }for k in KOLLAR }
    ustk =collections .defaultdict (lambda :collections .defaultdict (list ))
    n =0 
    for d ,s in zip (data_ ,oof ):
        if s is None :
            continue 
        G =np .asarray (d ["G"],float )
        Gd =np .asarray (d ["Gd"],float )
        P =np .asarray (d ["P"],float )
        idx =np .asarray (d ["idx"],int )
        YD =np .asarray (d ["YD"],float )
        s =np .asarray (s ,float )
        y =d ["y"]
        dogru =konum_dogru (y ,idx )
        for arm in KOLLAR :
            kon ,ks ,en_iyi =konum_skoru (s ,idx ,arm )
            # OLCEK DUZELTMESI. Ilk kosuda threshold izgarasi (0.20-0.95) TUM
            # kollara ham olcekte uygulandi. Ortanca/mean skorlarin
            # olcegi very more sikisik oldugu for same threshold bambaska a
            # yerden kesiyordu -- kollar kiyaslanamazdi (max 0.3012 vs
            # ort 0.0790). Parca ici YUZDELIK siralamaya cevirince butun
            # kollar same olcege gelir and threshold anlamli becomes.
            if os .environ .get ("KT_NORM","1")=="1"and len (ks )>1 :
                ks =np .argsort (np .argsort (ks ))/(len (ks )-1.0 )
            assert len (dogru )==len (kon ),"konum sirasi tutmuyor"
            # konum duzeyinde first-k dogruluk orani
            rank_ =np .argsort (-ks )
            k =max (len (G ),1 )
            ustk [d ["mfg"]][arm ].append (float (dogru [rank_ [:k ]].sum ())/k )
            for e in ESIKLER :
                Ps ,Ds =sec_esik (P [kon ],YD ,idx ,ks ,en_iyi ,e )
                tp ,fp ,fn =match_hungarian (Ps ,Ds ,G ,Gd ,d ["diag"],K .YANAL ,
                K .ACI ,False ,signed =True )[:3 ]
                q =agg [arm ][e ][d ["mfg"]]
                q ["tp"]+=tp ;q ["fp"]+=fp ;q ["fn"]+=fn 
        n +=1 
        if n %80 ==0 :
            print (f"  {n } part ({time .time ()-t0 :.0f} s)",flush =True )

    def f1 (c ):
        return 2 *c ["tp"]/max (2 *c ["tp"]+c ["fp"]+c ["fn"],1 )

    print (f"\n{n } part | KONUM duzeyi ilk-k dogruluk orani")
    print (f"{'brand':<7}"+"".join (f"{k :>10}"for k in KOLLAR ))
    for m_ in sorted (ustk ):
        print (f"{m_ :<7}"+"".join (
        f"{np .mean (ustk [m_ ][k ]):>10.3f}"for k in KOLLAR ))

    print (f"\nUCTAN UCA mikro robot F1 (threshold taramasi)")
    print (f"{'threshold':<7}"+"".join (f"{k :>10}"for k in KOLLAR ))
    en ={}
    for e in ESIKLER :
        line_ ={}
        for arm in KOLLAR :
            T =collections .Counter ()
            for q in agg [arm ][e ].values ():
                T +=q 
            line_ [arm ]=f1 (T )
            en [arm ]=max (en .get (arm ,0.0 ),line_ [arm ])
        print (f"{e :<7.2f}"+"".join (f"{line_ [k ]:>10.4f}"for k in KOLLAR ))
    print (f"\n{'EN IYI':<7}"+"".join (f"{en [k ]:>10.4f}"for k in KOLLAR ))
    print ("\n=== max'A GORE ===")
    for arm in KOLLAR [1 :]:
        f =en [arm ]-en ["max"]
        print (f"  {arm :<10}{en [arm ]:.4f}   {f :+.4f}"
        +("  <- KAPI GECTI"if f >=0.01 else ""))
    json .dump ({"damga":makbuz_hash .damga (),"cluster":KUME ,"en_iyi":en ,
    "ustk":{m_ :{k :float (np .mean (ustk [m_ ][k ]))for k in KOLLAR }
    for m_ in ustk },
    "not":"KONUM skoru toplama kurali. max = bugunku (her yanlis "
    "konuma 24 piyango bileti). D7'ye BAKILMADI."},
    open (f"results/konum_toplama_{KUME }.json","w"),indent =1 )
    print (f"receipt -> results/konum_toplama_{KUME }.json")


if __name__ =="__main__":
    main ()
