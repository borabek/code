# -*- coding: utf-8 -*-
"""S7 COKUS TESHISI: coken brand with running brand arasindaki difference NE?

FINDING (2026-08-12). Secici verimliligi (gerceklesen F1 / pool tavani) two
kutuplu: UPUN %65.5, SUPU %47.5 -- but MOR %5.7, NIT %0.5. Arada a sey absent.
NIT'te pool cevabi TASIYOR (ceiling 0.9147) but selector bulamiyor.

ILK HIPOTEZ (YOGUNLUK) HEMEN CURUYOR: NIT dense (24.4 CP/part) but MOR
3.4 CP/part and UPUN 3.2 -- ikisi neredeyse same, biri cokuyor obru calisiyor.

BU SONDA this eksenlerde brand basina measurement yapar:
  * CP yogunlugu, candidate/part, option/part
  * DOGRU secenegin part ICINDEKI SKOR SIRASI (yuzdelik) -- asil soru this:
    model correct secenegi YUKARI koyuyor mu, otherwise gomuyor mu?
  * first-10 / first-50 inside found GT orani
  * pozitif and negatif skorlarin AYRILIGI (medyan difference)
  * skorun part ici DAGILIM GENISLIGI (goreli rule buna duyarli)

Skorlar OOF: each brand disarida birakilarak egitilir (unseen brand kosulu).

Kullanim:  P6_DIZIN=results/_p6_oz_tam4 python probe_cokus.py
"""
import collections 
import json 
import os 
import sys 
import time 

import numpy as np 
from sklearn .ensemble import HistGradientBoostingClassifier 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1"
os .environ ["WG_TOPO"]="1"
os .environ ["WG_ZENGIN"]="1"
os .environ .setdefault ("P6_DIZIN","results/_p6_oz_tam4")
sys .path .insert (0 ,".")
import canonical_d7 as K # noqa: E402
import p6_decision # noqa: E402
from run_p6_ortak import yukle # noqa: E402

KUME =os .environ .get ("CK_KUME","d6")
KAT_MIN =int (os .environ .get ("CK_KAT_MIN","40"))
ITER =int (os .environ .get ("P6_ITER","200"))
NEG_KAT =int (os .environ .get ("P6_NEG_KAT","6"))
EKSEN_TOL =40.0 


def temel (d ):
    return np .hstack ([p6_decision .donustur (d ["X"]),
    p6_decision .kaynak_blok (d ["source"][d ["idx"]])])


def dogru_maske (d ):
    """Her option for: a GT'yi kabul kutusunda tutuyor mu?"""
    P =d ["P"][d ["idx"]]
    YD =d ["YD"]
    G ,Gd =np .asarray (d ["G"],float ),np .asarray (d ["Gd"],float )
    # GT YONU BIRIM OLMAYABILIR: 6074 kayittan 27'sinde not and EN KUCUGU
    # TAM SIFIR. Birimlestirmeden axial/lateral ayristirma bozulur.
    # (`probe_pool_ceiling` bunu `YB.unit` with already yapiyordu.)
    if len (Gd ):
        Gd =Gd /np .maximum (np .linalg .norm (Gd ,axis =1 ,keepdims =True ),1e-12 )
    if not len (P )or not len (G ):
        return np .zeros (len (P ),bool ),np .zeros ((0 ,),bool )
    v =P [:,None ,:]-G [None ,:,:]
    al =(v *Gd [None ,:,:]).sum (-1 )
    pe =np .linalg .norm (v -al [...,None ]*Gd [None ,:,:],axis =-1 )
    an =np .degrees (np .arccos (np .clip (YD @Gd .T ,-1 ,1 )))
    kabul =(pe <=K .YANAL )&(np .abs (al )<=EKSEN_TOL )&(an <=K .ACI )
    return kabul .any (1 ),kabul .any (0 )# (option correct mu, GT ulasildi mi)


def main ():
    t0 =time .time ()
    data_ =[d for d in yukle (KUME ,int (os .environ .get ("P6_TR","0")))]
    for d in data_ :
        d ["y"]=np .asarray (d ["y"],int )
    brand =collections .Counter (d ["mfg"]for d in data_ )
    katlar =[m for m ,n in brand .items ()if n >=KAT_MIN ]
    print (f"{len (data_ )} part | katlar {katlar } ({time .time ()-t0 :.0f} s)",
    flush =True )

    oof =[None ]*len (data_ )
    for b in katlar :
        ic =[i for i ,d in enumerate (data_ )if d ["mfg"]!=b ]
        dis =[i for i ,d in enumerate (data_ )if d ["mfg"]==b ]
        n_satir =sum (len (data_ [i ]["y"])for i in ic )
        first_ =temel (data_ [ic [0 ]])
        M =np .empty ((n_satir ,first_ .shape [1 ]),np .float32 )
        M [:len (first_ )]=first_ 
        o =len (first_ )
        for i in ic [1 :]:
            b_ =temel (data_ [i ])
            M [o :o +len (b_ )]=b_ 
            o +=len (b_ )
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
            oof [i ]=m .predict_proba (temel (data_ [i ]).astype (np .float32 ))[:,1 ]
        print (f"  OOF {b } ({time .time ()-t0 :.0f} s)",flush =True )

    ist =collections .defaultdict (lambda :collections .defaultdict (list ))
    for d ,s in zip (data_ ,oof ):
        if s is None :
            continue 
        mfg =d ["mfg"]
        dg ,ulasilan =dogru_maske (d )
        n =len (s )
        a =ist [mfg ]
        a ["part"].append (1 )
        a ["cp"].append (len (d ["G"]))
        a ["candidate"].append (len (np .unique (d ["idx"])))
        a ["option"].append (n )
        # MIKRO topla (part oranlarinin ORTALAMASI DEGIL). Ilk surumde
        # `ulasilan.mean()` part basina ratio biriktiriyordu; this MAKRO'dur and
        # KAPI A'nin MIKRO sayisiyla kiyaslanamaz. MOR'da 0.653 vs 0.8686
        # farki full buydu -- celiski not, BIRIM UYUSMAZLIGI. Bu projede
        # headline always MIKRO.
        a ["gt_ulasilan"].append (int (ulasilan .sum ()))
        a ["gt_toplam"].append (int (len (ulasilan )))
        if not n or not dg .any ():
            continue 
            # DOGRU secenegin part ICINDEKI order yuzdeligi (0 = most vertex)
        rank_ =np .argsort (np .argsort (-s ))
        en_iyi_dogru =int (rank_ [dg ].min ())
        a ["en_iyi_sira"].append (en_iyi_dogru )
        a ["en_iyi_yuzdelik"].append (en_iyi_dogru /max (n -1 ,1 ))
        a ["ilk10"].append (1.0 if en_iyi_dogru <10 else 0.0 )
        a ["ilk50"].append (1.0 if en_iyi_dogru <50 else 0.0 )
        a ["poz_med"].append (float (np .median (s [dg ])))
        a ["neg_med"].append (float (np .median (s [~dg ])))
        a ["skor_maks"].append (float (s .max ()))
        a ["skor_araligi"].append (float (s .max ()-np .median (s )))

    print (f"\n{'brand':<7}{'part':>6}{'CP/p':>7}{'candidate/p':>8}{'sec/p':>8}"
    f"{'havuzda':>9}{'ilk10':>7}{'ilk50':>7}{'sira%':>8}"
    f"{'poz-neg':>9}{'skor_ar':>8}")
    out ={}
    for mfg in sorted (ist ,key =lambda m :-np .mean (ist [m ]["ilk10"]or [0 ])):
        a =ist [mfg ]
        if not a ["en_iyi_sira"]:
            continue 
        r ={"part":len (a ["part"]),"cp_parca":float (np .mean (a ["cp"])),
        "aday_parca":float (np .mean (a ["candidate"])),
        "secenek_parca":float (np .mean (a ["option"])),
        "havuzda":float (sum (a ["gt_ulasilan"])/
        max (sum (a ["gt_toplam"]),1 )),# MIKRO
        "ilk10":float (np .mean (a ["ilk10"])),
        "ilk50":float (np .mean (a ["ilk50"])),
        "sira_yuzdelik":float (np .mean (a ["en_iyi_yuzdelik"])),
        "poz_neg_fark":float (np .mean (a ["poz_med"])-
        np .mean (a ["neg_med"])),
        "skor_araligi":float (np .mean (a ["skor_araligi"]))}
        out [mfg ]=r 
        print (f"{mfg :<7}{r ['part']:>6}{r ['cp_parca']:>7.1f}"
        f"{r ['aday_parca']:>8.0f}{r ['secenek_parca']:>8.0f}"
        f"{r ['havuzda']:>9.3f}{r ['ilk10']:>7.3f}{r ['ilk50']:>7.3f}"
        f"{r ['sira_yuzdelik']:>8.3f}{r ['poz_neg_fark']:>9.3f}"
        f"{r ['skor_araligi']:>8.3f}")

    json .dump ({"dizin":os .environ ["P6_DIZIN"],"cluster":KUME ,"brand":out ,
    "not":"S7 cokus teshisi. `havuzda` = GT'nin kabul kutusunda "
    "en az a secenegi olma orani (pool tavani). "
    "`ilk10/ilk50` = DOGRU secenegin part icinde ilk 10/50'ye "
    "girme orani. `sira%` = correct secenegin average sira "
    "yuzdeligi (0 = tepe). D7'ye BAKILMADI."},
    open (f"results/cokus_teshisi_{KUME }.json","w"),indent =1 )
    print (f"\nmakbuz -> results/cokus_teshisi_{KUME }.json")


if __name__ =="__main__":
    main ()
