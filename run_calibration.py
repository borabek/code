# -*- coding: utf-8 -*-
"""GUVEN KALIBRASYONU: "this sign correct mu?" sorusuna AYRI a model.

MEASURED (D7, 2512 prediction): ham skorla precision HICBIR esikte >=0.90 olmuyor,
most high 0.6429 and egri vertex sonrasi GERI DONUYOR. Sebep, karar kuralinin
GORELI olmasi (`0.85 x part-maks`): secilen tahminlerin all of them already part
maksimumuna yakin, therefore MUTLAK skor parts arasi ayirt edici not.

Bu modul SECILMIS tahminler on ikinci a soru sorar: "this prediction DOGRU
mu?" Girdi, secim aninda already hesaplanmis which is sinyallerdir:

  * ham skor and part-ici GORELI konumu (skor / part-maks, order yuzdeligi)
  * secenek bankasinin that konumda ne up to HEMFIKIR oldugu
  * lattice / order tutarliligi
  * mouth olculeri (girme, erisim, narinlik)
  * part baglami (candidate count, secilen prediction count)

Cikti kalibre a olasiliktir; confidence kapisi ONUN uzerine kurulur. Boylece
"robotun kullandigi isaretler >=0.90 kesinliktedir" sozu, KAPSAMA bedeliyle
birlikte verilebilir hale gelir.

Egitim `full`+`d6`, dogrulama MARKA KATLARINDA. D7'ye BAKILMAZ.
"""
import collections 
import json 
import os 
import pickle 
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
import direction_bank as YB # noqa: E402
from run_p6_ortak import yukle # noqa: E402
from sina_cluster import match_hungarian # noqa: E402

PAKET =os .environ .get ("P6_MODEL","results/p6_kademe2_model.pkl")
HEDEF =float (os .environ .get ("KAL_HEDEF","0.90"))
OZ_AD =["skor","skor_orani","skor_sira","skor_marj",
"banka_hemfikir","banka_n","kafes_mesafe","kafes_yon",
"girme","erisim","narinlik","yaricap",
"n_aday","n_secilen","secim_sirasi"]


def secim_ve_oznitelik (d ,pk ):
    """P6 secimini yap and HER SECILEN TAHMIN for kalibrasyon ozniteligi uret."""
    Xd =np .hstack ([p6_decision .donustur (d ["X"],pk .get ("zskor","ab")),
    p6_decision .kaynak_blok (d ["kaynak"][d ["idx"]])])
    if pk .get ("arm")=="P6_GEO":
        Xd =np .hstack ([Xd [:,58 :p6_decision .AB ],Xd [:,p6_decision .AB :]])
    s =np .asarray (pk ["kademe1"].predict_proba (
    Xd .astype (np .float32 ))[:,1 ],float )
    P ,D ,ai ,sc =p6_decision .sec_ayrintili (
    d ["P"],d ["idx"],d ["YD"],s ,tuple (pk ["kural"]),
    nms_mm =float (pk ["nms"]))
    if not len (P ):
        return P ,D ,np .zeros ((0 ,len (OZ_AD )))
    smax =float (s .max ())if len (s )else 1.0 
    sira =np .argsort (np .argsort (-s ))/max (len (s )-1 ,1 )
    Pt ,Dt =p6_decision .sec_ayrintili (
    d ["P"],d ["idx"],d ["YD"],s ,tuple (pk ["tohum_kural"]),
    nms_mm =float (pk ["tohum_nms"]))[:2 ]
    kb =lattice .oznitelik (d ["P"][d ["idx"]],d ["YD"],Pt ,Dt )
    D_blok =d ["X"][:,p6_decision .AB +len (YB .OZ_AD ):]# mouth olculeri (9)
    X =[]
    for r ,i in enumerate (ai ):
        j =np .where (d ["idx"]==i )[0 ]# this adayin secenegi
        sj =s [j ]
        # bankanin hemfikirligi: this adayin secenekleri ne up to single noktada
        hem =float (sj .max ()-np .median (sj ))if len (sj )>1 else 0.0 
        # secilen secenegin satiri: skoru sc[r] which is
        t =j [int (np .argmin (np .abs (sj -sc [r ])))]
        X .append ([sc [r ],sc [r ]/max (smax ,1e-9 ),float (sira [t ]),hem ,
        float (sj .max ()-sj .min ())if len (sj )>1 else 0.0 ,
        float (len (sj )),float (kb [t ,1 ]),float (kb [t ,6 ]),
        float (D_blok [t ,3 ]),float (D_blok [t ,5 ]),
        float (D_blok [t ,2 ]),float (D_blok [t ,0 ]),
        float (len (d ["P"])),float (len (P )),float (r )])
    return P ,D ,np .asarray (X ,float )


def dogru_maskesi (P ,D ,d ):
    tp ,fp ,fn ,bilgi =match_hungarian (P ,D ,d ["G"],d ["Gd"],d ["diag"],
    K .YANAL ,K .ACI ,False ,signed =True )
    e ={int (x [0 ])for x in bilgi .get ("eslesme",[])}
    return np .array ([1 if i in e else 0 for i in range (len (P ))],int )


def egri (S ,Y ):
    i =np .argsort (-S )
    y =Y [i ]
    d =np .cumsum (y )
    n =np .arange (1 ,len (y )+1 )
    return S [i ],d /n ,n /len (y )


def main ():
    t0 =time .time ()
    pk =pickle .load (open (PAKET ,"rb"))
    veri =[]
    for cluster in os .environ .get ("P6_KUME","tam,d6").split (","):
        veri +=yukle (cluster .strip (),int (os .environ .get ("P6_TR","0")))
    print (f"{len (veri )} part ({time .time ()-t0 :.0f} s)",flush =True )

    X ,Y ,M =[],[],[]
    for i ,d in enumerate (veri ,1 ):
        P ,D ,x =secim_ve_oznitelik (d ,pk )
        if not len (P ):
            continue 
        y =dogru_maskesi (P ,D ,d )
        X .append (x )
        Y .append (y )
        M +=[d ["mfg"]]*len (y )
        if i %600 ==0 :
            print (f"  {i }/{len (veri )} ({time .time ()-t0 :.0f} s)",flush =True )
    X =np .vstack (X )
    Y =np .concatenate (Y )
    M =np .asarray (M )
    print (f"tahmin {len (Y )} | ham precision {Y .mean ():.4f}",flush =True )

    brand =collections .Counter (M )
    KAT_MIN =int (os .environ .get ("KAL_KAT_MIN","400"))
    katlar =[m for m ,n in brand .items ()if n >=KAT_MIN ]
    if not katlar :# small kosularda becomes
        katlar =[m for m ,_ in brand .most_common (3 )]
        print (f"  (fold esigi {KAT_MIN } kimseyi gecmedi -> en buyuk 3 brand)")
    print (f"katlar: {katlar }",flush =True )

    # HAM SKOR with kiyas (first column)
    hs ,hk ,hc =egri (X [:,0 ],Y )
    print (f"\nHAM SKOR: en yuksek precision {hk .max ():.4f}")

    Sk =np .zeros (len (Y ))
    for b in katlar :
        ic =M !=b 
        dis =M ==b 
        m =HistGradientBoostingClassifier (
        max_iter =300 ,learning_rate =0.06 ,max_leaf_nodes =31 ,
        l2_regularization =1.0 ,random_state =0 ).fit (X [ic ],Y [ic ])
        Sk [dis ]=m .predict_proba (X [dis ])[:,1 ]
    kd =np .isin (M ,katlar )
    if not kd .any ():
        sys .exit ("fold-disi tahmin YOK -- daha buyuk cluster ile kos")
    ks ,kk ,kc =egri (Sk [kd ],Y [kd ])
    # HAM SKOR kiyasi AYNI fold-disi altkumede yapilmali; tum veride yapmak
    # kalibre modeli haksiz avantajli/dezavantajli gosterirdi.
    hs ,hk ,hc =egri (X [kd ,0 ],Y [kd ])
    print (f"HAM SKOR (ayni altkume): en yuksek precision {hk .max ():.4f}")
    print (f"KALIBRE  : en yuksek precision {kk .max ():.4f}")

    print (f"\n{'hedef':>7}{'ham kapsama':>14}{'kalibre kapsama':>18}")
    out ={}
    for h in (0.60 ,0.70 ,0.80 ,0.90 ):
        a =hc [hk >=h ]
        b =kc [kk >=h ]
        ha =float (a .max ())if len (a )else 0.0 
        kb_ =float (b .max ())if len (b )else 0.0 
        out [str (h )]={"ham_kapsama":ha ,"kalibre_kapsama":kb_ }
        print (f"{h :>7.2f}{ha :>14.4f}{kb_ :>18.4f}")

    son =HistGradientBoostingClassifier (
    max_iter =300 ,learning_rate =0.06 ,max_leaf_nodes =31 ,
    l2_regularization =1.0 ,random_state =0 ).fit (X ,Y )
    pk ["kalibrasyon"]={"model":son ,"oz_ad":OZ_AD }
    with open (PAKET ,"wb")as f :
        pickle .dump (pk ,f )
    json .dump ({"damga":makbuz_hash .damga (),"n_tahmin":int (len (Y )),
    "ham_kesinlik":float (Y .mean ()),
    "ham_maks_kesinlik":float (hk .max ()),
    "kalibre_maks_kesinlik":float (kk .max ()),
    "kapsama":out ,"katlar":katlar ,
    "not":"Guven kalibrasyonu: secilmis tahminler uzerinde ikinci "
    "model. Kat-disi measurement; D7'ye BAKILMADI."},
    open ("results/kalibrasyon.json","w"),indent =1 )
    print (f"\nmakbuz -> results/kalibrasyon.json ({time .time ()-t0 :.0f} s)")


if __name__ =="__main__":
    main ()
