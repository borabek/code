# -*- coding: utf-8 -*-
"""C2: YAPISAL FP BASTIRMA -- real CP'ler SIRA olusturur.

HATA BANKASI v2 (sign duzeltmeli yigin, `results/hata_bankasi_v2.json`):
TP 786 / FN 2301 / **FP 1247** (B-rep 639, seg 382) -> F1 0.3070.
Kova aritmetigi: **FP yariya inerse F1 0.350** (+0.043).

FIZIK: klemensler tekrarli yapilardir; real tel girisleri part on
DUZENLI ARALIKLI SIRALAR olusturur. Yalitik duran a candidate (komsusu absent, order
uyumu absent) large olasilikla vida deligi / alet yuvasi.

KOL: gate+NMS sonrasi each candidate for YAPI SKORU is computed --
  komsu_sayisi   parcanin baskin ekseni along es-aralikli komsu adedi
  aralik_uyumu   most yakin komsu araliginin part MEDYAN araligina uyumu
  dizilim        adayin, komsularinin olusturdugu dogruya diklik sapmasi
Skoru low olanlar ATILIR. Esik D6'da secilir, D7'de YENIDEN TARANMAZ.

KIYAS: K2.1 periyodik spread more before robot -0.0100 vermisti, but BOZUK
etiketli gate and ESKI havuzla; also that arm candidate EKLIYORDU, this arm candidate SILIYOR.
Dayanagi cokmus, yeniden olculur.

TEZE SADIK: konum/direction uretimi does not change, only last kabul.
"""
import collections 
import json 
import os 
import pickle 
import sys 

import numpy as np 

import makbuz_hash 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1"
os .environ ["WG_TOPO"]="1"
os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,".")
import d6_record # noqa: E402
import canonical_d7 as K # noqa: E402
import product_zinciri # noqa: E402
import wire_gate # noqa: E402
from sina_cluster import match_hungarian # noqa: E402

OZ ="results/_tam_oz"
TAN ="results/_tan_hizali"
OB ={"d7":"results/_p1_olasilik_d7","d6":"results/_p1_olasilik_g7"}
KAYNAKLAR =(0 ,1 )
ESIK =0.05 
YANAL ,ACI ,EKSENEL =2.0 ,10.0 ,40.0 
GIRME ,ERISIM =3 ,5 
ARALIK_TOL =0.25 # medyan araliktan this goreli sapmaya up to "uyumlu"
YAPI_ESIKLERI =(None ,0 ,1 ,2 )# None = arm KAPALI


def yapi_skoru (P ):
    """Her candidate for: es-aralikli komsu count (0..n).

    Parcanin baskin ekseni bulunur, candidates that eksene yansitilir, ardisik
    araliklarin MEDYANI referans alinir. Bir adayin skoru, kendisine medyan
    aralik +/- tolerans mesafede duran komsu sayisidir. Yalitik candidate -> 0.
    """
    n =len (P )
    if n <3 :
        return np .full (n ,99 )# few adayli parcada arm devre disi
    Q =P -P .mean (0 )
    u =np .linalg .svd (Q ,full_matrices =False )[2 ][0 ]
    t =Q @u 
    s =np .sort (t )
    fark =np .diff (s )
    fark =fark [fark >1e-6 ]
    if not len (fark ):
        return np .full (n ,99 )
    med =float (np .median (fark ))
    if med <=1e-6 :
        return np .full (n ,99 )
    d =np .abs (t [:,None ]-t [None ,:])
    uyum =np .abs (d -med )<=ARALIK_TOL *med 
    np .fill_diagonal (uyum ,False )
    return uyum .sum (1 )


def yukle (on ,pid ):
    z =np .load (f"{OZ }/{on }_{pid }.npz")
    m =np .isin (np .asarray (z ["kaynak"],int ),KAYNAKLAR )
    T =np .asarray (np .load (f"{TAN }/{on }_{pid }.npz")["T"],float )
    X =np .hstack ([np .asarray (z ["X"],float ),T ])[m ]
    if len (X )<2 :
        return None 
    f =f"{OB [on ]}/{pid }.npz"
    if not os .path .exists (f ):
        return None 
    zz =np .load (f )
    return {"X":X ,"T":T [m ],"P":np .asarray (z ["P"],float )[m ],
    "D":np .asarray (z ["D"],float )[m ],
    "V":np .ascontiguousarray (zz ["V"],np .float64 ),
    "F":np .ascontiguousarray (zz ["F"],np .int64 ),
    "pb":np .asarray (zz ["pbs"],float ).mean (0 )}


def cluster (on ):
    d6 ={str (p ):r for p ,r in 
    d6_record .yukle (set (d6_record .exam ()["pidler"])).items ()}
    Rk =K .yukle (None )
    out =[]
    for f in sorted (os .listdir (OZ )):
        if not (f .startswith (on +"_")and f .endswith (".npz")):
            continue 
        pid =f [len (on )+1 :-4 ]
        r =Rk .get (pid )or d6 .get (pid )
        if r is None or not len (r .get ("G",[])):
            continue 
        d =yukle (on ,pid )
        if d is None :
            continue 
        d .update ({"pid":pid ,"mfg":r ["mfg"],"G":np .asarray (r ["G"],float ),
        "Gd":np .asarray (r ["Gd"],float ),"diag":float (r ["diag"])})
        out .append (d )
    return out 


def kos (gate ,data_ ,yapi_esik ,S ,tam =False ):
    rob =collections .defaultdict (lambda :[0 ,0 ,0 ])
    tes =[]
    silinen =0 
    for d in data_ :
        s =np .asarray (gate .predict_proba (
        wire_gate .within_part (d ["X"],"zskor"))[:,1 ],float )
        k =s >=ESIK 
        if not k .any ():
            P ,D =d ["P"][:0 ],d ["D"][:0 ]
        else :
            P ,D ,T ,sk =d ["P"][k ],d ["D"][k ],d ["T"][k ],s [k ]
            if len (P )>1 :
                nm =wire_gate .crowd_mask (P ,sk )
                P ,D ,T =P [nm ],D [nm ],T [nm ]
            D =np .where ((T [:,ERISIM ]<T [:,GIRME ])[:,None ],-D ,D )
            if yapi_esik is not None and len (P )>=3 :
                ys =yapi_skoru (P )
                tut =ys >=yapi_esik 
                silinen +=int ((~tut ).sum ())
                P ,D =P [tut ],D [tut ]
        if tam and len (P ):
            P ,D =product_zinciri .tam_poz (d ["V"],d ["F"],d ["pb"],P ,D ,
            step_path =S .get (d ["pid"]))
        tp ,fp ,fn =match_hungarian (P ,D ,d ["G"],d ["Gd"],d ["diag"],YANAL ,ACI ,
        False ,signed =True )[:3 ]
        a =rob [d ["mfg"]]
        a [0 ]+=tp ;a [1 ]+=fp ;a [2 ]+=fn 
        tes .append ((len (d ["G"]),)+match_hungarian (
        P ,D ,d ["G"],d ["Gd"],d ["diag"],max (3.0 ,0.06 *d ["diag"]),
        180.0 ,True )[:3 ])
    pm ={m :2 *v [0 ]/max (2 *v [0 ]+v [1 ]+v [2 ],1 )for m ,v in rob .items ()}
    mi =float (2 *sum (v [0 ]for v in rob .values ())/
    max (sum (2 *v [0 ]+v [1 ]+v [2 ]for v in rob .values ()),1 ))
    return {"robot":mi ,"tespit":K .mikro (tes ),
    "makro":float (np .mean (list (pm .values ()))),
    "en_kotu":float (min (pm .values ())),
    "TP":int (sum (v [0 ]for v in rob .values ())),
    "FP":int (sum (v [1 ]for v in rob .values ())),
    "silinen":silinen ,"brand":pm }


def main ():
    gate =pickle .load (open ("results/kazanan_hgb_derin.pkl","rb"))["HGB-derin"]
    S =K .step_map ()
    dev ,te =cluster ("d6"),cluster ("d7")
    print (f"D6 {len (dev )} | D7 {len (te )}\n",flush =True )
    en =None 
    for ye in YAPI_ESIKLERI :
        r =kos (gate ,dev ,ye ,S )
        print (f"  D6 yapi_esik={ye }  robot {r ['robot']:.4f} | TP {r ['TP']} "
        f"FP {r ['FP']} | silinen {r ['silinen']}",flush =True )
        if ye is not None and (en is None or r ["robot"]>en [1 ]["robot"]):
            en =(ye ,r )
    ye =en [0 ]
    baseline =kos (gate ,te ,None ,S ,tam =True )
    new_ =kos (gate ,te ,ye ,S ,tam =True )
    print (f"\nTABAN (arm kapali)   robot {baseline ['robot']:.4f} | tespit "
    f"{baseline ['tespit']:.4f} | TP {baseline ['TP']} FP {baseline ['FP']}")
    print (f"YAPI  (threshold={ye })       robot {new_ ['robot']:.4f} | tespit "
    f"{new_ ['tespit']:.4f} | TP {new_ ['TP']} FP {new_ ['FP']} | "
    f"silinen {new_ ['silinen']}")
    print (f"\nFARK {new_ ['robot']-baseline ['robot']:+.4f} | KAPI >= +0.02 "
    f"(threshold gurultusu ~0.015)")
    json .dump ({"damga":makbuz_hash .damga (),"baseline":baseline ,"yapi":new_ ,
    "secilen_esik":ye ,
    "not":"Yapisal FP bastirma: parcanin baskin ekseninde es-aralikli "
    "komsu sayisi. Esik D6'da secildi. Isaret duzeltmesi IKI "
    "kolda acik. D7 brand-disi, TAM ZINCIR, MIKRO."},
    open ("results/c2_yapisal_fp.json","w"),indent =1 )
    print ("receipt -> results/c2_yapisal_fp.json")


if __name__ =="__main__":
    main ()
