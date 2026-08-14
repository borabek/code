# -*- coding: utf-8 -*-
"""S4 OLCUMU: candidate-kumesi modeli (DeepSets) vs noktasal HGB -- AYNI protokolde.

TEK DEGISKEN MODEL SINIFI. Ayni corpus, same brand katlari, same features,
same rule aramasi, same kabul kutusu. Fark only skorlayici:
  HGB      : each secenegi TEK BASINA puanlar (bugunku urun)
  DeepSets : secenegi parcanin ORTALAMA/MAKSIMUM ozetiyle birlikte puanlar

RATIONALE (S7). Secici verimliligi two kutuplu (UPUN %65.5 / NIT %0.5) and sebep
POZ-NEG SKOR AYRIMI'nin cokmesi (0.847 -> 0.050). Noktasal model "this secenek
this parcadaki digerlerine according to iyi mi" sorusunu goremiyor.

YETENEK DOGRULANDI (tests/test_kume_modeli.py): part baglami gerektiren
sentetik gorevde DeepSets 0.483, noktasal 0.025 (sans 0.04).

KAPI: LOMO MIKRO robot F1'de HGB'ye **+0.05**. Alti kalirsa arm KAPANIR.

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
os .environ .setdefault ("P6_DIZIN","results/_p6_oz_tam3")
sys .path .insert (0 ,".")
import canonical_d7 as K # noqa: E402
import kume_modeli as KM # noqa: E402
import p6_decision # noqa: E402
from run_p6_ortak import yukle # noqa: E402
from sina_cluster import match_hungarian # noqa: E402

KURALLAR =([("mutlak",e )for e in (0.20 ,0.40 ,0.60 ,0.80 ,0.90 ,0.95 ,0.97 )]+
[("goreli",o ,t )for o in (0.50 ,0.70 ,0.85 ,0.95 )
for t in (0.05 ,0.20 ,0.40 )])
NMS =5.0 
ARAMA_N =int (os .environ .get ("P6_ARAMA_N","200"))
ITER =int (os .environ .get ("P6_ITER","200"))
NEG_KAT =int (os .environ .get ("P6_NEG_KAT","6"))
KAT_MIN =int (os .environ .get ("P6_KAT_MIN","200"))
DEVIR =int (os .environ .get ("S4_DEVIR","25"))
LR =float (os .environ .get ("S4_LR","3e-3"))
LAM =float (os .environ .get ("S4_LAM","1.0"))
BOYUT =int (os .environ .get ("S4_D","128"))
# Cok large parts GPU belleğini zorlar; egitimde secenek ORNEKLENIR
# (pozitifler HER ZAMAN tutulur, negatifler seyreltilir). Tahminde TAM cluster.
EGIT_MAKS =int (os .environ .get ("S4_EGIT_MAKS","3000"))
# BCE'yi HGB koluyla AYNI sinif dengesinde hesapla (pozitif basina N negatif).
# 0 = closed (v1 davranisi: tum secenekler, ~300:1 dengesizlik).
S4_NEG_KAT =int (os .environ .get ("S4_NEG_KAT","0"))


def temel (d ):
    return np .hstack ([p6_decision .donustur (d ["X"]),
    p6_decision .kaynak_blok (d ["kaynak"][d ["idx"]])]).astype (
    np .float32 )


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


def en_iyi_kural (veri ,skor ,rng ):
    ar =(rng .choice (len (veri ),ARAMA_N ,replace =False )
    if len (veri )>ARAMA_N else np .arange (len (veri )))
    AR =[veri [i ]for i in ar ]
    AS =[skor [i ]for i in ar ]
    return max (KURALLAR ,key =lambda k :puanla (AR ,AS ,k )["makro"])


def main ():
    t0 =time .time ()
    veri =[]
    for cluster in os .environ .get ("P6_KUME","tam,d6").split (","):
        for d in yukle (cluster .strip (),int (os .environ .get ("P6_TR","0"))):
            veri .append (d )
    for d in veri :
        d ["y"]=np .asarray (d ["y"],int )
        d ["_M"]=temel (d )
        # HAM `X` ARTIK GEREKMIYOR: puanlama P/idx/YD/G/Gd with calisiyor.
        # Ikisini birden tutmak vertex bellegi ~10 GB'a cikariyordu (corpus
        # 3051 part x ~2600 secenek x 162 column), this da this gece three sureci
        # olduren bellek darligini tekrar dogururdu.
        d ["X"]=None 
    brand =collections .Counter (d ["mfg"]for d in veri )
    katlar =[m for m ,n in brand .items ()if n >=KAT_MIN ]
    n_giris =veri [0 ]["_M"].shape [1 ]
    print (f"{len (veri )} part | {n_giris } sutun | katlar {katlar } "
    f"({time .time ()-t0 :.0f} s)",flush =True )

    top ={"HGB":collections .Counter (),"KUME":collections .Counter ()}
    kat_sonuc ={}
    rng0 =np .random .default_rng (0 )
    for b in katlar :
        ic =[i for i ,d in enumerate (veri )if d ["mfg"]!=b ]
        dis =[i for i ,d in enumerate (veri )if d ["mfg"]==b ]

        # ---------- NOKTASAL HGB (bugunku urun)
        n_satir =sum (len (veri [i ]["y"])for i in ic )
        M =np .empty ((n_satir ,n_giris ),np .float32 )
        o =0 
        for i in ic :
            m_ =veri [i ]["_M"]
            M [o :o +len (m_ )]=m_ 
            o +=len (m_ )
        Y =np .concatenate ([veri [i ]["y"]for i in ic ])
        rng =np .random .default_rng (0 )
        poz =np .where (Y ==1 )[0 ]
        neg =np .where (Y ==0 )[0 ]
        sec =np .concatenate ([poz ,rng .choice (
        neg ,min (len (neg ),NEG_KAT *max (len (poz ),1 )),replace =False )])
        hgb =HistGradientBoostingClassifier (
        max_iter =ITER ,learning_rate =0.06 ,max_leaf_nodes =63 ,
        l2_regularization =1.0 ,random_state =0 ).fit (M [sec ],Y [sec ])
        del M 
        s_ic_h =[hgb .predict_proba (veri [i ]["_M"])[:,1 ]for i in ic ]
        s_dis_h =[hgb .predict_proba (veri [i ]["_M"])[:,1 ]for i in dis ]
        print (f"  {b } HGB egitildi ({time .time ()-t0 :.0f} s)",flush =True )

        # ---------- ADAY-KUMESI (DeepSets)
        training =[]
        for i in ic :
            X ,y =veri [i ]["_M"],veri [i ]["y"].astype (np .float32 )
            if len (X )>EGIT_MAKS :
                p_ =np .where (y >0 )[0 ]
                n_ =np .where (y ==0 )[0 ]
                k =max (EGIT_MAKS -len (p_ ),10 )
                n_ =rng0 .choice (n_ ,min (len (n_ ),k ),replace =False )
                se =np .sort (np .concatenate ([p_ ,n_ ]))
                X ,y =X [se ],y [se ]
            training .append ((X ,y ))
        m =KM .egit (training ,n_giris ,d =BOYUT ,devir =DEVIR ,lr =LR ,lam =LAM ,
        seed =0 ,neg_kat =S4_NEG_KAT )
        s_ic_k =[KM .tahmin (m ,veri [i ]["_M"])for i in ic ]
        s_dis_k =[KM .tahmin (m ,veri [i ]["_M"])for i in dis ]
        print (f"  {b } KUME egitildi ({time .time ()-t0 :.0f} s)",flush =True )

        # ---------- AYNI rule aramasi, AYNI kabul kutusu
        kat_sonuc [b ]={}
        for ad ,s_ic ,s_dis in (("HGB",s_ic_h ,s_dis_h ),
        ("KUME",s_ic_k ,s_dis_k )):
            rng =np .random .default_rng (0 )
            kural =en_iyi_kural ([veri [i ]for i in ic ],s_ic ,rng )
            r =puanla ([veri [i ]for i in dis ],s_dis ,kural )
            for k_ in ("TP","FP","FN"):
                top [ad ][k_ ]+=r [k_ ]
            kat_sonuc [b ][ad ]={"robot":r ["robot"],"kural":list (kural )}
        print (f"  {b :<6} HGB {kat_sonuc [b ]['HGB']['robot']:.4f} | "
        f"KUME {kat_sonuc [b ]['KUME']['robot']:.4f} "
        f"({time .time ()-t0 :.0f} s)",flush =True )

    son ={}
    for ad in ("HGB","KUME"):
        c =top [ad ]
        son [ad ]=2 *c ["TP"]/max (2 *c ["TP"]+c ["FP"]+c ["FN"],1 )
    fark =son ["KUME"]-son ["HGB"]
    print (f"\n=== S4 SONUC (MIKRO, {len (katlar )} brand kati) ===")
    print (f"  HGB (noktasal)   {son ['HGB']:.4f}")
    print (f"  KUME (DeepSets)  {son ['KUME']:.4f}")
    print (f"  FARK             {fark :+.4f}   KAPI +0.05 -> "
    f"{'GECTI'if fark >=0.05 else 'GECMEDI'}")
    json .dump ({"damga":makbuz_hash .damga (),"dizin":os .environ ["P6_DIZIN"],
    "katlar":katlar ,"n_parca":len (veri ),
    "hgb":son ["HGB"],"cluster":son ["KUME"],"fark":fark ,
    "gecti":bool (fark >=0.05 ),"fold":kat_sonuc ,
    "ayar":{"devir":DEVIR ,"lr":LR ,"lam":LAM ,"d":BOYUT ,
    "egit_maks":EGIT_MAKS ,"neg_kat":S4_NEG_KAT },
    "not":"S4: candidate-kumesi modeli vs noktasal HGB, AYNI protocol "
    "(ayni katlar, oznitelikler, kural aramasi, kabul "
    "kutusu). Tek degisken MODEL SINIFI. D7'ye BAKILMADI."},
    open ("results/s4_kume_modeli.json","w"),indent =1 )
    print ("receipt -> results/s4_kume_modeli.json")


if __name__ =="__main__":
    main ()
