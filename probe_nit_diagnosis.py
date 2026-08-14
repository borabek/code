# -*- coding: utf-8 -*-
"""NIT COKUSU: ranking mi, kalibrasyon mi, temsil mi?

DURUM. NIT 50 part / 1222 GT (24.4 CP/part -- D6'nin most dense markasi).
Dagitilan urun 1222 GT'den **2** tanesini buluyor (F1 0.0032). Yeni pool that
markada yonlu recall **0.5254** veriyor, i.e. cevabin yarisi HAVUZDA. Kural
kahini (that markadaki EN IYI threshold/NMS) only 0.0349 -- demek ki loss ESIKTE
not, modelin SIRALAMASINDA.

Bu betik three soruyu separates:

  1. SIRALAMA NE KADAR KOTU?  Esikten bagimsiz criterion: mean precision (AP) and
     `recall@k` (part basina GT count up to secenek al). Rastgele siralamanin
     beklenen degeriyle kiyaslanir.
  2. PARCA-ICI Z-SKOR MU BOZUYOR?  Egitim markalarinda part basina ~150 candidate
     present, NIT'te 407. Z-skor each parcayi own ortalamasina according to kaydiriyor;
     candidate count and dagilimi very different olunca NIT egitimin HIC GORMEDIGI a
     bolgeye dusebilir. `zskor=absent` (ham feature) kolu bunu sinar.
  3. HANGI OZNITELIK BLOGU?  A (segmentasyon, 58) / B+D (mouth olculeri, 18) /
     C (direction bankasi, 16) bloklari single single verilerek hangisinin NIT'te bilgi
     tasidigi olculur.

Egitim: NIT DISI D6 markalari. Sinav: NIT. (Bu a DIAGNOSIS betigidir; buradan
dagitim karari CIKMAZ, arm secimi `full` korpusunun katlarinda is done.)
"""
import collections 
import json 
import os 
import sys 

import numpy as np 
from sklearn .ensemble import HistGradientBoostingClassifier 

import makbuz_hash 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1"
os .environ ["WG_TOPO"]="1"
os .environ ["WG_ZENGIN"]="1"
os .environ .setdefault ("P6_DIZIN","results/_p6_oz_u25")
sys .path .insert (0 ,".")
import p6_decision # noqa: E402
from run_p6_ortak import yukle # noqa: E402

AB =p6_decision .AB 
A_SUT =58 
HEDEF =os .environ .get ("NIT_MARKA","NIT")


def yap ():
    return HistGradientBoostingClassifier (
    max_iter =250 ,learning_rate =0.06 ,max_leaf_nodes =63 ,
    l2_regularization =1.0 ,random_state =0 )


def alt (M ,Y ,fold =6 ):
    rng =np .random .default_rng (0 )
    p =np .where (Y ==1 )[0 ]
    n =np .where (Y ==0 )[0 ]
    k =min (len (n ),fold *max (len (p ),1 ))
    s =np .concatenate ([p ,rng .choice (n ,k ,replace =False )])
    rng .shuffle (s )
    return M [s ],Y [s ]


def blok (d ,ad ,zskor ):
    X =np .hstack ([p6_decision .donustur (d ["X"],zskor ),
    p6_decision .kaynak_blok (d ["kaynak"][d ["idx"]])])
    if ad =="hepsi":
        return X 
    if ad =="A (segmentasyon)":
        return np .hstack ([X [:,:A_SUT ],X [:,AB +16 :]])# A + source
    if ad =="B+D (mouth olculeri)":
        return np .hstack ([X [:,A_SUT :AB ],X [:,AB +16 :]])
    if ad =="C (direction bankasi)":
        return X [:,AB :]
    raise ValueError (ad )


def ap_ve_recall_k (veri ,skor ):
    """Esikten BAGIMSIZ ranking olcutleri. Doner: (AP, recall@k, rastgele)."""
    aps ,rk ,rnd =[],[],[]
    for d ,s in zip (veri ,skor ):
        y =np .asarray (d ["y"],int )
        if not y .any ():
            continue 
        sira =np .argsort (-np .asarray (s ,float ))
        ys =y [sira ]
        kum =np .cumsum (ys )
        kes =kum /np .arange (1 ,len (ys )+1 )
        aps .append (float ((kes *ys ).sum ()/max (ys .sum (),1 )))
        k =int (len (d ["G"]))
        rk .append (float (ys [:k ].sum ())/max (k ,1 ))
        rnd .append (float (y .mean ()))# rastgele siralamanin beklentisi
    return (float (np .mean (aps )),float (np .mean (rk )),float (np .mean (rnd )))


def main ():
    dev =yukle ("d6")
    for d in dev :
        d ["y"]=np .asarray (d ["y"],int )
    tr =[d for d in dev if d ["mfg"]!=HEDEF ]
    te =[d for d in dev if d ["mfg"]==HEDEF ]
    if not te :
        sys .exit (f"{HEDEF } yok")
    print (f"training {len (tr )} part ({len (set (d ['mfg']for d in tr ))} brand) | "
    f"exam {HEDEF } {len (te )} part / "
    f"{sum (len (d ['G'])for d in te )} GT",flush =True )
    print (f"candidate/part: training {np .mean ([len (d ['P'])for d in tr ]):.0f} | "
    f"{HEDEF } {np .mean ([len (d ['P'])for d in te ]):.0f}")
    print (f"secenek/part: training {np .mean ([len (d ['idx'])for d in tr ]):.0f} | "
    f"{HEDEF } {np .mean ([len (d ['idx'])for d in te ]):.0f}\n")

    out ={}
    print (f"{'kurulum':<34}{'AP':>8}{'recall@k':>10}{'rastgele':>10}{'fold':>7}")
    for zskor in ("ab","sira","ikisi","yok"):
        for ad in ("hepsi","C (direction bankasi)"):
            M =np .vstack ([blok (d ,ad ,zskor )for d in tr ]).astype (np .float32 )
            Y =np .concatenate ([d ["y"]for d in tr ])
            M ,Y =alt (M ,Y )
            m =yap ().fit (M ,Y )
            sk =[m .predict_proba (blok (d ,ad ,zskor ).astype (np .float32 ))[:,1 ]
            for d in te ]
            ap ,rk ,rnd =ap_ve_recall_k (te ,sk )
            etiket =f"{ad } / zskor={zskor }"
            out [etiket ]={"AP":ap ,"recall@k":rk ,"rastgele":rnd ,
            "fold":rk /max (rnd ,1e-9 )}
            print (f"{etiket :<34}{ap :>8.4f}{rk :>10.4f}{rnd :>10.4f}"
            f"{rk /max (rnd ,1e-9 ):>7.1f}x",flush =True )

    json .dump ({"damga":makbuz_hash .damga (),"brand":HEDEF ,"sonuc":out ,
    "not":"TESHIS. Esikten bagimsiz siralama olcutleri. `fold` = "
    "recall@k'nin rastgele siralamaya orani; 1.0x rastgele "
    "demektir."},
    open (f"results/nit_teshis_{HEDEF }.json","w"),indent =1 )
    print (f"\nmakbuz -> results/nit_teshis_{HEDEF }.json")
    print ("YORUM: `fold` 1.0x civarindaysa model o markada RASTGELE siralama "
    "yapiyor demektir (temsil sorunu). Yuksekse siralama iyi, loss "
    "threshold/kalibrasyonda.")


if __name__ =="__main__":
    main ()
