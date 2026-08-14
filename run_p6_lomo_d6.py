# -*- coding: utf-8 -*-
"""P6 ERKEN OKUMA: D6-ICI LOMO (brand-birak-disarida) + ESIK x NMS taramasi.

WHY: `full` korpusunun secenek oznitelikleri still cikiyor. Ama D6'nin 8 markasi
elde and LOMO with ADIL a kiyas kurulabilir -- each kivrimda same parts, same
training buyuklugu, TEK DEGISKEN feature+secim kurali.

KOLLAR
  TABAN     candidate basina TEK row (own yonu), A+B (67 column), threshold + NMS +
            sign correction                      -- dagitilan kuralin ta kendisi
  P6        candidate basina TUM direction secenekleri, A+B+C+D (92 column), ortak skor +
            acgozlu secim + konum NMS
  P6_KAHIN  P6 secimi, but SECILEN adaylarin yonu KAHIN'den. Tavan not DIAGNOSIS:
            "yonu mu kaciriyoruz, konumu mu?"

ILK KOSU BULGUSU (2026-08-11):
  TABAN 0.2093 -> P6 0.2649 (+0.0556), direction kahini only +0.0064 EKLIYOR.
  Yani secilen adaylarda direction secimi ZATEN neredeyse mukemmel; bankanin actigi
  +0.2392'lik ceiling HIC SECILMEYEN adaylarda duruyor. Recall %17 / precision %58
  -- i.e. very AZ prediction uretiyoruz.
  Olculdu: D6 GT'lerinin %16.1'inin most yakin komsusu 5mm'den yakin. NMS yaricapi
  5mm whereas KOMSU KONTAKLARI birbirini bastiriyor and recall'a ceiling koyuyor.
  Bu yuzden threshold and NMS BIRLIKTE taranir; ikisi de EGITIM markalarinda secilir.

Esik/NMS HER KOL and HER KIVRIM for EGITIM markalarinda secilir; disarida
birakilan markada TARANMAZ. Aksi halde number sisik becomes.

Bu a ON OKUMADIR: 468 part / 8 brand small a baseline and `full` korpusunun
buyuklugunu temsil etmez. Karar `full` with egitilmis modelde verilir.
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
sys .path .insert (0 ,".")
import canonical_d7 as K # noqa: E402
import p6_decision # noqa: E402
import product_genis # noqa: E402
import wire_gate # noqa: E402
import direction_bank as YB # noqa: E402
from run_p6_ortak import yukle # noqa: E402
from sina_cluster import match_hungarian # noqa: E402

AB =p6_decision .AB 
C0 =AB # C blogunun first sutunu = k_kendi
# DECISION KURALLARI. Ilk taramada MUTLAK threshold neredeyse each kivrimda IZGARANIN EN
# UST degerini (0.50) sectti -- i.e. optimum SINIRDAYDI and real optimumu
# bulamiyordum. Izgara yukari acildi and urunun own GORELI kurali
# (`p1c_threshold.maske`: part-ici most high skorun `ratio` kati VE `baseline` ustu) de
# candidate kurallar arasina alindi.
KURALLAR =([("mutlak",e )for e in 
(0.05 ,0.10 ,0.20 ,0.30 ,0.40 ,0.50 ,0.60 ,0.70 ,0.80 )]+
[("goreli",o ,t )for o in (0.30 ,0.50 ,0.70 )
for t in (0.05 ,0.15 ,0.30 )])
NMSLER =(0.0 ,1.5 ,2.5 ,3.5 ,5.0 )
KOLLAR =("TABAN","P6","P6_KAHIN")


def yap ():
    return HistGradientBoostingClassifier (
    max_iter =400 ,learning_rate =0.06 ,max_leaf_nodes =63 ,
    l2_regularization =1.0 ,random_state =0 )


def kendi (d ):
    """Aday basina TEK row: own direction secenegi (C blogunun k_kendi=1 satiri)."""
    return np .where (d ["X"][:,C0 ]==1.0 )[0 ]


def _nms (P ,s ,mm ):
    """Skor sirali konum bastirma. mm<=0 whereas bastirma YOK."""
    if mm <=0 or len (P )<2 :
        return np .ones (len (P ),bool )
    tut =np .zeros (len (P ),bool )
    alinan =[]
    for i in np .argsort (-s ):
        if alinan and float (np .min (np .linalg .norm (
        np .asarray (alinan )-P [i ],axis =1 )))<mm :
            continue 
        alinan .append (P [i ])
        tut [i ]=True 
    return tut 


def yon_kahini (d ,ai ):
    """Secilen each candidate for KENDI bankasindan most iyi direction (KAHIN, teshis for)."""
    if not len (ai ):
        return np .zeros ((0 ,3 ))
    G =np .asarray (d ["G"],float )
    Gn =YB .birim (d ["Gd"])
    out =[]
    for i in ai :
        Y =d ["YD"][np .where (d ["idx"]==i )[0 ]]
        w =d ["P"][i ][None ]-G 
        al =(w *Gn ).sum (-1 )
        yan =np .linalg .norm (w -al [:,None ]*Gn ,axis =-1 )
        uy =(yan <=K .YANAL )&(np .abs (al )<=40.0 )
        if not uy .any ():
            out .append (Y [0 ])
            continue 
        an =np .degrees (np .arccos (np .clip (YB .birim (Y )@Gn [uy ].T ,-1.0 ,1.0 )))
        out .append (Y [int (np .unravel_index (np .argmin (an ),an .shape )[0 ])])
    return np .asarray (out ,float ).reshape (-1 ,3 )


def puanla (d ,s ,threshold ,nms ,arm ):
    if arm =="TABAN":
        k =kendi (d )
        sk =s [k ]
        m =p6_decision .kabul_maskesi (sk ,threshold )
        if not m .any ():
            return np .zeros ((0 ,3 )),np .zeros ((0 ,3 ))
        P ,D =d ["P"][m ],d ["D"][m ]
        T =d ["X"][k ][m ][:,AB +len (YB .OZ_AD ):]
        n =_nms (P ,sk [m ],nms )
        return P [n ],product_genis .isaret_duzelt (D [n ],T [n ])
    P ,D ,ai ,_sc =p6_decision .sec_ayrintili (d ["P"],d ["idx"],d ["YD"],s ,threshold ,
    nms_mm =nms )
    if arm =="P6_KAHIN":
        return P ,yon_kahini (d ,ai )
    return P ,D 


def olc (veri ,skor ,threshold ,nms ,arm ):
    tp =fp =fn =0 
    tes =[]
    for d ,s in zip (veri ,skor ):
        P ,D =puanla (d ,s ,threshold ,nms ,arm )
        a ,b ,c =match_hungarian (P ,D ,d ["G"],d ["Gd"],d ["diag"],K .YANAL ,K .ACI ,
        False ,signed =True )[:3 ]
        tp +=a ;fp +=b ;fn +=c 
        tes .append ((len (d ["G"]),)+match_hungarian (
        P ,D ,d ["G"],d ["Gd"],d ["diag"],max (3.0 ,0.06 *d ["diag"]),
        180.0 ,True )[:3 ])
    return {"robot":2 *tp /max (2 *tp +fp +fn ,1 ),"tespit":K .mikro (tes ),
    "TP":tp ,"FP":fp ,"FN":fn }


def egit (tr ,baseline ):
    if baseline :
        M =np .vstack ([p6_decision .donustur (d ["X"][kendi (d )][:,:AB ],"hepsi")
        for d in tr ])
        Y =np .concatenate ([d ["y"][kendi (d )]for d in tr ])
    else :
        M =np .vstack ([p6_decision .donustur (d ["X"])for d in tr ])
        Y =np .concatenate ([d ["y"]for d in tr ])
    return yap ().fit (M ,Y )


def skorla (m ,veri ,baseline ):
    out =[]
    for d in veri :
        if baseline :
            k =kendi (d )
            s =np .zeros (len (d ["X"]))
            s [k ]=m .predict_proba (
            p6_decision .donustur (d ["X"][k ][:,:AB ],"hepsi"))[:,1 ]
        else :
            s =m .predict_proba (p6_decision .donustur (d ["X"]))[:,1 ]
        out .append (np .asarray (s ,float ))
    return out 


def main ():
    dev =yukle ("d6",int (os .environ .get ("P6_DEV","0")))
    for d in dev :
        d ["y"]=np .asarray (d ["y"],int )
    brand =collections .Counter (d ["mfg"]for d in dev )
    kivrimlar =[m for m ,n in brand .items ()if n >=8 ]
    print (f"D6 {len (dev )} part | LOMO kivrimi {kivrimlar }",flush =True )

    top ={k :collections .Counter ()for k in KOLLAR }
    ayrinti ={}
    for b in kivrimlar :
        tr =[d for d in dev if d ["mfg"]!=b ]
        te =[d for d in dev if d ["mfg"]==b ]
        modeller ={False :egit (tr ,False ),True :egit (tr ,True )}
        ayrinti [b ]={}
        for arm in KOLLAR :
            tb =(arm =="TABAN")
            m =modeller [tb ]
            s_tr ,s_te =skorla (m ,tr ,tb ),skorla (m ,te ,tb )
            en =max (((e ,n )for e in KURALLAR for n in NMSLER ),
            key =lambda en :olc (tr ,s_tr ,en [0 ],en [1 ],arm )["robot"])
            r =olc (te ,s_te ,en [0 ],en [1 ],arm )
            for k in ("TP","FP","FN"):
                top [arm ][k ]+=r [k ]
            ayrinti [b ][arm ]=dict (r ,kural =list (en [0 ]),nms =en [1 ])
        t ,p ,o =(ayrinti [b ]["TABAN"],ayrinti [b ]["P6"],ayrinti [b ]["P6_KAHIN"])
        print (f"  {b :<6} n={len (te ):<4} TABAN {t ['robot']:.4f} -> "
        f"P6 {p ['robot']:.4f} ({p ['kural']}, nms {p ['nms']})  "
        f"({p ['robot']-t ['robot']:+.4f})  [direction kahini {o ['robot']:.4f}]",
        flush =True )

    print (f"\n{'arm':<10} {'robot':>8} {'TP':>6} {'FP':>6} {'FN':>6} {'recall':>8} {'precision':>9}")
    son ={}
    for arm in KOLLAR :
        c =top [arm ]
        f1 =2 *c ["TP"]/max (2 *c ["TP"]+c ["FP"]+c ["FN"],1 )
        rc =c ["TP"]/max (c ["TP"]+c ["FN"],1 )
        pr =c ["TP"]/max (c ["TP"]+c ["FP"],1 )
        son [arm ]={"robot":f1 ,"recall":rc ,"precision":pr ,**dict (c )}
        print (f"{arm :<10} {f1 :>8.4f} {c ['TP']:>6} {c ['FP']:>6} {c ['FN']:>6} "
        f"{rc :>8.4f} {pr :>9.4f}")
    print (f"\nYON SECIM KAYBI (P6_KAHIN - P6): "
    f"{son ['P6_KAHIN']['robot']-son ['P6']['robot']:+.4f}")
    d =son ["P6"]["robot"]-son ["TABAN"]["robot"]
    art =sum (1 for b in kivrimlar 
    if ayrinti [b ]["P6"]["robot"]>ayrinti [b ]["TABAN"]["robot"])
    print (f"P6 - TABAN = {d :+.4f} | {art }/{len (kivrimlar )} markada ARTI")
    json .dump ({"damga":makbuz_hash .damga (),"toplam":son ,"brand":ayrinti ,
    "n_parca":len (dev ),"kivrim":kivrimlar ,
    "kurallar":[list (k )for k in KURALLAR ],"nmsler":list (NMSLER ),
    "not":"D6-ICI LOMO on okumasi. Esik VE NMS her arm/kivrim icin "
    "EGITIM markalarinda secildi. Onbellek havuzu -- urunun "
    "canli havuzu DEGIL."},
    open ("results/p6_lomo_d6.json","w"),indent =1 )
    print ("receipt -> results/p6_lomo_d6.json")


if __name__ =="__main__":
    main ()
