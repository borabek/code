# -*- coding: utf-8 -*-
"""SAHA GUVEN KAPISI: "robot hangi isaretlere KENDI BASINA guvenebilir?"

SORU. Urun each parcada a array CP isaretliyor but all of them same kalitede not.
Robotun otonom davranabilmesi for "this sign %X dogrudur" diyebilmek is required.
Bugun GLB'deki kirmizi/turuncu ayrimi `robot_conf_auto=0.5` + 3 oy like KEYFI
a esikle yapiliyor -- olculmus a kesinlige BAGLI DEGIL.

BU BETIK. `full` MARKA KATLARINDA (each fold: a brand disarida, gorulmemis
brand kosulu) each SECILEN prediction for (skor, correct mu) kaydeder and
precision-kapsama egrisini removes. Boylece "ONAYLI" katmani for threshold, olculmus
kesinlige according to secilir.

WHY D7 DEGIL. D7 SINAV kumesidir and butcesi 2 okumadir. Isletme esigi ayarlamak
a okumayi HARCAR and dahasi esigi sinava UYDURMAK becomes. Kat olcumu same soruyu
(gorulmemis brand) sinavi harcamadan yanitlar.

DURUSTLUK NOTU. Buradaki precision, SECILEN tahminler icindir. Kapsama = ONAYLI
isaretlerin GT'ye orani (i.e. isin ne kadari otonom yapilabilir). Kesinligi
yukseltmek kapsamayi DUSURUR; ikisi same anda buyumez.

Cikti: results/saha_kapisi_tam.json + ekrana tablo.
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
import p6_decision # noqa: E402
from run_p6_ortak import yukle # noqa: E402
from sina_cluster import match_hungarian # noqa: E402

NMS =5.0 
ITER =int (os .environ .get ("P6_ITER","200"))
NEG_KAT =int (os .environ .get ("P6_NEG_KAT","6"))
KAT_MIN =int (os .environ .get ("P6_KAT_MIN","200"))
# Genis a rule: ONAYLI katmani already skor esigiyle daraltilacak, that is why
# secim kurali GENIS tutulur (high recall) and daraltmayi threshold yapar.
KURAL =("goreli",0.50 ,0.05 )
HEDEFLER =(0.70 ,0.80 ,0.90 ,0.95 )


def temel (d ):
    return np .hstack ([p6_decision .donustur (d ["X"]),
    p6_decision .kaynak_blok (d ["kaynak"][d ["idx"]])])


def yigin_f32 (ogeler ,uret ,line_ ):
    """float64 ara yigin OLMADAN float32 matris. `run_extra_feature` with same
    rationale: full-open korpusta `vstack(...).astype(float32)` before float64
    birlestirip 10.2 GiB istiyor and MemoryError veriyor."""
    n_satir =sum (line_ (o )for o in ogeler )
    first_ =np .asarray (uret (ogeler [0 ]),np .float32 )
    M =np .empty ((n_satir ,first_ .shape [1 ]),np .float32 )
    M [:len (first_ )]=first_ 
    y =len (first_ )
    for o in ogeler [1 :]:
        b =uret (o )
        M [y :y +len (b )]=b 
        y +=len (b )
    if y !=n_satir :
        raise ValueError (f"satir sayisi tutmadi: {y } != {n_satir }")
    return M 


def main ():
    t0 =time .time ()
    data_ =[]
    for cluster in os .environ .get ("P6_KUME","tam,d6").split (","):
        for d in yukle (cluster .strip (),int (os .environ .get ("P6_TR","0"))):
            d ["_kume"]=cluster .strip ()
            data_ .append (d )
    for d in data_ :
        d ["y"]=np .asarray (d ["y"],int )
    brand =collections .Counter (d ["mfg"]for d in data_ )
    katlar =[m for m ,n in brand .items ()if n >=KAT_MIN ]
    print (f"{len (data_ )} part | katlar {katlar } ({time .time ()-t0 :.0f} s)",
    flush =True )

    rec_ =[]# (skor, dogru_mu)
    gt_top =0 
    for b in katlar :
        ic =[i for i ,d in enumerate (data_ )if d ["mfg"]!=b ]
        dis =[i for i ,d in enumerate (data_ )if d ["mfg"]==b ]
        M =yigin_f32 (ic ,lambda i :temel (data_ [i ]),
        lambda i :len (data_ [i ]["y"]))
        Y =np .concatenate ([data_ [i ]["y"]for i in ic ])
        rng =np .random .default_rng (0 )
        poz =np .where (Y ==1 )[0 ]
        neg =np .where (Y ==0 )[0 ]
        sec =np .concatenate ([poz ,rng .choice (
        neg ,min (len (neg ),NEG_KAT *max (len (poz ),1 )),replace =False )])
        m =HistGradientBoostingClassifier (
        max_iter =ITER ,learning_rate =0.06 ,max_leaf_nodes =63 ,
        l2_regularization =1.0 ,random_state =0 ).fit (M [sec ],Y [sec ])
        for i in dis :
            d =data_ [i ]
            s =m .predict_proba (temel (d ).astype (np .float32 ))[:,1 ]
            P ,D ,_ ,sk =p6_decision .sec_ayrintili (
            d ["P"],d ["idx"],d ["YD"],s ,KURAL ,nms_mm =NMS )
            gt_top +=len (d ["G"])
            if not len (P ):
                continue 
            tp ,fp ,fn ,bilgi =match_hungarian (P ,D ,d ["G"],d ["Gd"],d ["diag"],
            K .YANAL ,K .ACI ,False ,signed =True )
            dogru =np .zeros (len (P ),bool )
            for e in bilgi ["eslesme"]:
                dogru [e [0 ]]=True 
            rec_ .extend (zip (np .asarray (sk ,float ).tolist (),dogru .tolist ()))
        print (f"  {b :<6} biriken tahmin {len (rec_ )} ({time .time ()-t0 :.0f} s)",
        flush =True )

    if not rec_ :
        sys .exit ("hic tahmin yok")
    sk =np .array ([a for a ,_ in rec_ ])
    dg =np .array ([b for _ ,b in rec_ ],bool )
    rank_ =np .argsort (-sk )
    dg_s =dg [rank_ ]
    sk_s =sk [rank_ ]
    kum_tp =np .cumsum (dg_s )
    n =np .arange (1 ,len (dg_s )+1 )
    precision =kum_tp /n 
    kapsama =kum_tp /max (gt_top ,1 )

    print (f"\ntoplam tahmin {len (sk )} | toplam GT {gt_top } | "
    f"ham precision {dg .mean ():.4f}")
    print (f"\n{'hedef':<8}{'threshold':>8}{'ONAYLI':>9}{'precision':>10}"
    f"{'kapsama':>9}")
    oneri ={}
    for h in HEDEFLER :
        uy =np .where (precision >=h )[0 ]
        # most BUYUK k: threshold dustukce precision dusuyor; hedefi saglayan most genis
        # onek aranir (at least 20 prediction olsun ki noise olmasin)
        uy =uy [uy >=19 ]
        if not len (uy ):
            print (f"{h :<8.2f}{'-':>8}{'-':>9}{'ULASILMIYOR':>10}{'-':>9}")
            oneri [str (h )]=None 
            continue 
        k =int (uy .max ())
        oneri [str (h )]={"threshold":float (sk_s [k ]),"n":int (k +1 ),
        "precision":float (precision [k ]),
        "kapsama":float (kapsama [k ])}
        print (f"{h :<8.2f}{sk_s [k ]:>8.3f}{k +1 :>9d}{precision [k ]:>10.4f}"
        f"{kapsama [k ]:>9.4f}")

    json .dump ({"damga":makbuz_hash .damga (),"dizin":os .environ ["P6_DIZIN"],
    "katlar":katlar ,"n_parca":len (data_ ),"n_tahmin":len (sk ),
    "gt_toplam":int (gt_top ),"ham_kesinlik":float (dg .mean ()),
    "kural":list (KURAL ),"nms":NMS ,"oneri":oneri ,
    "not":"GORULMEMIS MARKA katlarinda (LOMO) precision-kapsama. "
    "ONAYLI katmani esigi buradan secilir. D7'ye BAKILMADI. "
    "Kapsama = ONAYLI sign / toplam GT."},
    open ("results/saha_kapisi_tam.json","w"),indent =1 )
    print ("\nmakbuz -> results/saha_kapisi_tam.json")


if __name__ =="__main__":
    main ()
