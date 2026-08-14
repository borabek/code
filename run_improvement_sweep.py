# -*- coding: utf-8 -*-
"""GELISTIRME TARAMASI — four kaldirac TEK kosuda, UCTAN UCA

Olcum fasli closed. Bu betik only F1'i YUKSELTMEYE works.
Taranan four kaldirac (none of them sistematik denenmedi):

  1. HPO          : ogrenme hizi / yaprak / iterasyon / L2
  2. ZOR NEGATIF  : negatif orani and SECIMI (rastgele vs skor-yakin)
  3. AGIRLIK      : part-esitleyici weight -- NIT GT'nin %51'i, egitimi
                    eziyor; each part equal weight alirsa sparse markalar
                    bogulmaz
  4. KANONIK      : gunun single uctan uca kazanci (+0.0093) baseline kabul edilir

Her yapilandirma AYNI brand-disi katlarda, AYNI karar kuraliyla, UCTAN UCA
robot F1 with olculur. Kazanan, `full` katlarinda AYRICA dogrulanmadan urune
girmez (d6'ya ayar yapmak d6'ya ezberlemektir).

D7'ye BAKILMAZ.
"""
import collections 
import itertools 
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
import kanonik_hizalama as KH # noqa: E402
import p6_decision # noqa: E402
from run_p6_ortak import yukle # noqa: E402
from sina_cluster import match_hungarian # noqa: E402

KUME =os .environ .get ("GT_KUME","d6")
KAT_MIN =int (os .environ .get ("GT_KAT_MIN","40"))
MESH ={"d6":"results/_p1_olasilik",
"tam":"results/_p1_olasilik_brepegit"}[KUME ]
NMS =5.0 
KURAL =("goreli",0.85 ,0.20 )


def temel (d ):
    return np .hstack ([p6_decision .donustur (d ["X"]),
    p6_decision .kaynak_blok (d ["source"][d ["idx"]])]).astype (
    np .float32 )


def kanonik_blok (d ):
    mf =f"{MESH }/{d ['pid']}.npz"
    V =(np .asarray (np .load (mf )["V"],float )
    if os .path .exists (mf )else np .asarray (d ["P"],float ))
    return KH .oznitelik (d ["P"][d ["idx"]],d ["YD"],V ).astype (np .float32 )


def negatif_sec (M ,Y ,agirlik_p ,neg_kat ,zor ,rng ,on_skor =None ):
    """negatif ornekleme. hard=True whereas SKOR-YAKIN negatifler tercih edilir."""
    poz =np .where (Y ==1 )[0 ]
    neg =np .where (Y ==0 )[0 ]
    n_al =min (len (neg ),int (neg_kat *max (len (poz ),1 )))
    if zor and on_skor is not None :
    # ZOR NEGATIF: ten gecisin most high skorlu negatifleri. Yarisini
    # hard, yarisini rastgele al -- salt hard secim dagilimi breaks.
        rank_ =neg [np .argsort (-on_skor [neg ])]
        z =rank_ [:n_al //2 ]
        kalan =np .setdiff1d (neg ,z ,assume_unique =False )
        r =rng .choice (kalan ,min (len (kalan ),n_al -len (z )),replace =False )
        sec =np .concatenate ([poz ,z ,r ])
    else :
        sec =np .concatenate ([poz ,rng .choice (neg ,n_al ,replace =False )])
    return sec ,agirlik_p [sec ]


def f1 (c ):
    return 2 *c ["tp"]/max (2 *c ["tp"]+c ["fp"]+c ["fn"],1 )


def main ():
    t0 =time .time ()
    data_ =yukle (KUME ,int (os .environ .get ("P6_TR","0")))
    for d in data_ :
        d ["y"]=np .asarray (d ["y"],int )
        d ["_M"]=np .hstack ([temel (d ),kanonik_blok (d )])
    brand =collections .Counter (d ["mfg"]for d in data_ )
    katlar =[m for m ,n in brand .items ()if n >=KAT_MIN ]
    print (f"{len (data_ )} part | katlar {katlar } | baseline = temel + kanonik",
    flush =True )

    # ---- taranan yapilandirmalar (TABAN first sirada)
    TABAN =dict (lr =0.06 ,yaprak =63 ,iter =200 ,l2 =1.0 ,neg =6 ,zor =False ,
    esit =False )
    if os .environ .get ("GT_MOD")=="4":
    # URETIM TABANI DUZELTMESI. Taramanin tabani neg=6 idi, but URETIM
    # egiticisi (`run_p6_kademe2.py`) `P6_NEG_KAT` varsayilani **8**
    # kullaniyor. Yani olculen "+0.0088" 6->12 farkidir; uretimin real
    # kazanci 8->12 and DAHA KUCUK must be. Dagitilan number this must be.
        ADAYLAR =[{**TABAN ,"neg":8 },{**TABAN ,"neg":12 }]
    elif os .environ .get ("GT_MOD")=="3":
    # VERIFICATION TURU. d6'da secilen ayar (neg=12, +0.0088) `full`
    # katlarinda tekrar edilir. d6'da ayar yapip d6'da ilan etmek
    # d6'ya ezberlemektir; gate ONCE ilan edilmisti.
        ADAYLAR =[TABAN ,{**TABAN ,"neg":12 }]
    elif os .environ .get ("GT_MOD")=="2":
    # IKINCI TUR. Birinci kind single a ekseni acti: NEGATIF ORANI
    # (neg=12 +0.0088, neg=3 -0.0498). Ogrenme hizi/yaprak/L2 notr ya da
    # zararli; hard negatif -0.0195; part-esitleyici weight -0.1120
    # (CLOSED -- NIT baskinligi egitime zarar not FAYDA veriyor).
    # Bu kind only negatif oranini sonuna up to surer and single yardimci
    # eksenle (lr=0.03/iter=400, +0.0045) merges.
        ADAYLAR =[TABAN ]
        for ng in (12 ,18 ,24 ,36 ):
            ADAYLAR .append ({**TABAN ,"neg":ng })
        for ng in (12 ,24 ):
            ADAYLAR .append ({**TABAN ,"neg":ng ,"lr":0.03 ,"iter":400 })
    else :
        ADAYLAR =[TABAN ]
        for lr ,yap in itertools .product ((0.03 ,0.10 ),(31 ,127 )):
            ADAYLAR .append ({**TABAN ,"lr":lr ,"yaprak":yap })
        ADAYLAR +=[
        {**TABAN ,"iter":400 ,"lr":0.03 },
        {**TABAN ,"l2":10.0 },
        {**TABAN ,"neg":12 },
        {**TABAN ,"neg":3 },
        {**TABAN ,"zor":True },
        {**TABAN ,"esit":True },
        {**TABAN ,"zor":True ,"esit":True },
        {**TABAN ,"lr":0.03 ,"iter":400 ,"esit":True },
        ]

    res_ =[]
    for ci ,c in enumerate (ADAYLAR ):
        agg =collections .Counter ()
        for b in katlar :
            ic =[i for i ,d in enumerate (data_ )if d ["mfg"]!=b ]
            dis =[i for i ,d in enumerate (data_ )if d ["mfg"]==b ]
            n_s =sum (len (data_ [i ]["y"])for i in ic )
            M =np .empty ((n_s ,data_ [0 ]["_M"].shape [1 ]),np .float32 )
            W =np .empty (n_s ,np .float32 )
            o =0 
            for i in ic :
                m_ =data_ [i ]["_M"]
                M [o :o +len (m_ )]=m_ 
                # PARCA-ESITLEYICI AGIRLIK: each parcanin total agirligi 1.
                # Boylece 24 CP'li NIT parcasi 3 CP'li parcadan 8 fold extra
                # soz sahibi olmaz.
                W [o :o +len (m_ )]=(1.0 /len (m_ ))if c ["esit"]else 1.0 
                o +=len (m_ )
            Y =np .concatenate ([data_ [i ]["y"]for i in ic ])
            rng =np .random .default_rng (0 )
            on =None 
            if c ["zor"]:
            # ten gecis: small and fast a model, KAT-ICI capraz not --
            # amaci only "hangi negatifler kafa karistirici" demek
                alt =np .concatenate ([
                np .where (Y ==1 )[0 ],
                rng .choice (np .where (Y ==0 )[0 ],
                min ((Y ==0 ).sum (),3 *max ((Y ==1 ).sum (),1 )),
                replace =False )])
                m0 =HistGradientBoostingClassifier (
                max_iter =60 ,learning_rate =0.1 ,max_leaf_nodes =31 ,
                random_state =0 ).fit (M [alt ],Y [alt ])
                on =m0 .predict_proba (M )[:,1 ]
            sec ,w =negatif_sec (M ,Y ,W ,c ["neg"],c ["zor"],rng ,on )
            m =HistGradientBoostingClassifier (
            max_iter =c ["iter"],learning_rate =c ["lr"],
            max_leaf_nodes =c ["yaprak"],l2_regularization =c ["l2"],
            random_state =0 ).fit (M [sec ],Y [sec ],sample_weight =w )
            del M ,W 
            for i in dis :
                s =m .predict_proba (data_ [i ]["_M"])[:,1 ]
                d =data_ [i ]
                P ,D =p6_decision .sec (d ["P"],d ["idx"],d ["YD"],s ,KURAL ,
                nms_mm =NMS )
                tp ,fp ,fn =match_hungarian (P ,D ,d ["G"],d ["Gd"],d ["diag"],
                K .YANAL ,K .ACI ,False ,
                signed =True )[:3 ]
                agg ["tp"]+=tp ;agg ["fp"]+=fp ;agg ["fn"]+=fn 
        r =f1 (agg )
        res_ .append ((r ,c ))
        label_ =("TABAN"if ci ==0 else 
        " ".join (f"{k }={v }"for k ,v in c .items ()
        if v !=TABAN [k ]))
        print (f"  {r :.4f}  {label_ }   ({time .time ()-t0 :.0f} s)",flush =True )

    baseline =res_ [0 ][0 ]
    res_ .sort (key =lambda x :-x [0 ])
    print (f"\n=== TABAN {baseline :.4f} ===")
    for r ,c in res_ [:5 ]:
        fark =r -baseline 
        et =" ".join (f"{k }={v }"for k ,v in c .items ()if v !=TABAN [k ])
        print (f"  {r :.4f}  {fark :+.4f}  {et or 'TABAN'}"
        +("  <- KAZANC"if fark >=0.01 else ""))
    json .dump ({"damga":makbuz_hash .damga (),"cluster":KUME ,"baseline":baseline ,
    "en_iyi":{"f1":res_ [0 ][0 ],"yapilandirma":res_ [0 ][1 ]},
    "hepsi":[{"f1":r ,"c":c }for r ,c in res_ ],
    "not":"HPO + zor negatif + part-esitleyici agirlik, baseline "
    "temel+kanonik. UCTAN UCA robot F1. Kazanan `tam` "
    "katlarinda AYRICA dogrulanmadan urune girmez. "
    "D7'ye BAKILMADI."},
    open (f"results/gelistirme_taramasi_{KUME }.json","w"),indent =1 )
    print (f"receipt -> results/gelistirme_taramasi_{KUME }.json")


if __name__ =="__main__":
    main ()
