# -*- coding: utf-8 -*-
"""P6 ORTAK SIRALAYICI: (konum x direction) seceneklerini TEK skorla puanla.

BUGUNKU URUN two ayri karar veriyor:
  1. gate  -> this candidate CP mi?           (`product_wide.sec`, HGB-derin, threshold 0.05)
  2. direction   -> havuzun verdigi direction + sign duzeltmesi + poz kafasi
Yon no zaman SECILMIYOR; havuzdan ne geldiyse that.

BU BETIK ikisini merges: each (konum, direction) secenegi single modelle puanlanir,
secim skor sirasina according to acgozlu + NMS with is done. Boylece
  * YON_YOK kovasi (D7'de 539 GT) dogrudan hedeflenir,
  * GATE_REDDI kovasi (580) da faydalanir: correct yonu bulunan a candidate more
    high skor takes, esigi gecer.

OLCUM: D6. D7'ye BAKILMAZ (butce 3 okuma, all of them kapida).

BLOK AYRIMI (`within_part` z-skoru):
  A+B (58+9) pool/mouth olculeri MUTLAK buyukluklerdir -> part-ici z-skor
             dagitilan gate'te most large single kazancti, KORUNUR.
  C+D (16+9) direction olculeri ZATEN goreli (aciya, destege, orana dayali) ->
             HAM birakilir. p5-v2'de goreli olculeri a more normalize etmek
             uctan uca 0.1649 -> 0.1465'e DUSURMUSTU.
  `P6_ZSKOR=all of them` with ikisi de z-skorlanir (ablasyon for).
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
sys .path .insert (0 ,".")
import d6_record # noqa: E402
import canonical_d7 as K # noqa: E402
import p6_decision # noqa: E402
import product_wide # noqa: E402
import wire_gate # noqa: E402
import direction_bank as YB # noqa: E402
from sina_cluster import match_hungarian # noqa: E402

P6 =os .environ .get ("P6_DIZIN","results/_p6_oz")
# HAVUZ KAYNAGI SUZGECI: cache `source` alanini carries (0 seg / 1 B-rep /
# 2 mesh tepesi), so TEK cikarimdan different pool kollari egitilebilir.
_KS =os .environ .get ("P6_KAYNAK_EGIT","")
KAYNAK_SUZ =tuple (int (c )for c in _KS )if _KS else None 
AB =p6_decision .AB # A(58) + B(9) -- z-skorlanan blok
ZSKOR =os .environ .get ("P6_ZSKOR","ab")
ESIKLER =(0.02 ,0.05 ,0.10 ,0.15 ,0.20 ,0.30 ,0.40 ,0.50 )
_D6 =None 


def kayitlar (pidler ):
    """pid -> kayit (G, Gd, diag, mfg). Once kanonik, after D6 yedegi."""
    global _D6 
    kay =K .yukle (pidler )
    eksik =[p for p in pidler if p not in kay ]
    if eksik :
        if _D6 is None :
            _D6 ={str (p ):r for p ,r in d6_record .yukle ().items ()}
        kay .update ({p :_D6 [p ]for p in eksik if p in _D6 })
    return kay 


def yukle (on ,bound_ =0 ):
    """Secenek tablolarini oku and etiketle. Doner: list[part sozlugu]."""
    fs =sorted (f for f in os .listdir (P6 )
    if f .startswith (on +"_")and f .endswith (".npz"))
    if bound_ :
        fs =fs [:bound_ ]
    pidler =[f [len (on )+1 :-4 ]for f in fs ]
    kay =kayitlar (pidler )
    out =[]
    # SESSIZ KORPUS DARALMASI this projede two times became (X genisligi 58-vs-22,
    # birlestir globu). Atlanan part count HER ZAMAN basilir.
    yok_kayit =yok_gt =0 
    for f ,pid in zip (fs ,pidler ):
        r =kay .get (pid )
        if r is None :
            yok_kayit +=1 
            continue 
        if not len (r .get ("G",[])):
            yok_gt +=1 
            continue 
        z =np .load (f"{P6 }/{f }")
        X =np .asarray (z ["X"],float )
        idx =np .asarray (z ["idx"],int )
        YD =np .asarray (z ["YD"],float )
        P =np .asarray (z ["P"],float )
        Dham =np .asarray (z ["D"],float )
        kayn =(np .asarray (z ["source"],int )if "source"in z 
        else np .zeros (len (P ),int ))
        if KAYNAK_SUZ is not None and "source"in z :
            kay =kayn 
            tut =np .isin (kay ,KAYNAK_SUZ )
            # secenekler ADAY indeksine bagli; before secenekleri suz, after
            # candidate indekslerini YENIDEN NUMARALA (aksi halde `idx` empty adaylara
            # sign eder and secim sessizce wrong konumu returns).
            ysec =tut [idx ]
            new_ =-np .ones (len (P ),int )
            new_ [np .where (tut )[0 ]]=np .arange (int (tut .sum ()))
            X ,YD =X [ysec ],YD [ysec ]
            idx =new_ [idx [ysec ]]
            P =P [tut ]
            Dham =Dham [tut ]
            kayn =kayn [tut ]
        G =np .asarray (r ["G"],float )
        Gd =np .asarray (r ["Gd"],float )
        y ,_ =YB .etiketle (P [idx ],YD ,G ,Gd )
        out .append ({"pid":pid ,"mfg":r ["mfg"],"X":X ,"idx":idx ,"YD":YD ,
        "P":P ,"D":Dham ,"y":y ,"source":kayn ,
        "G":G ,"Gd":Gd ,"diag":float (r ["diag"])})
    print (f"  {on }: {len (fs )} dosya -> {len (out )} part "
    f"(kayit yok {yok_kayit }, GT yok {yok_gt })",flush =True )
    return out 


def donustur (X ):
    """URUNLE AYNI donusum -- `p6_decision` single kaynaktir, here kopyalanmaz."""
    return p6_decision .donustur (X ,ZSKOR )


def olc (data_ ,skorlar ,threshold ):
    rob =collections .defaultdict (lambda :[0 ,0 ,0 ])
    tes =[]
    for d ,s in zip (data_ ,skorlar ):
        P ,D =p6_decision .sec (d ["P"],d ["idx"],d ["YD"],s ,threshold )
        tp ,fp ,fn =match_hungarian (P ,D ,d ["G"],d ["Gd"],d ["diag"],K .YANAL ,
        K .ACI ,False ,signed =True )[:3 ]
        a =rob [d ["mfg"]]
        a [0 ]+=tp ;a [1 ]+=fp ;a [2 ]+=fn 
        tes .append ((len (d ["G"]),)+match_hungarian (
        P ,D ,d ["G"],d ["Gd"],d ["diag"],max (3.0 ,0.06 *d ["diag"]),
        180.0 ,True )[:3 ])
    pm ={m :2 *a [0 ]/max (2 *a [0 ]+a [1 ]+a [2 ],1 )for m ,a in rob .items ()}
    return {"robot":float (2 *sum (a [0 ]for a in rob .values ())/
    max (sum (2 *a [0 ]+a [1 ]+a [2 ]
    for a in rob .values ()),1 )),
    "tespit":K .mikro (tes ),
    "makro":float (np .mean (list (pm .values ()))),
    "en_kotu":float (min (pm .values ())),"brand":pm ,
    "TP":sum (a [0 ]for a in rob .values ()),
    "FP":sum (a [1 ]for a in rob .values ())}


def baseline (data_ ):
    """DAGITILAN path, same onbellekten yeniden kurulmus.

    `product_wide.sec` with same: X = A+B, part-ici z-skor, HGB-derin, threshold 0.05,
    kalabalik NMS, sign correction. Tek difference: here onbellekten okunuyor.
    """
    model =product_wide .model_yukle ()
    if model is None :
        return None 
    rob =collections .defaultdict (lambda :[0 ,0 ,0 ])
    tes =[]
    for d in data_ :
        kendi =np .where (d ["X"][:,AB ]==1.0 )[0 ]# C blogunun first sutunu
        AB_ =d ["X"][kendi ,:AB ]
        s =np .asarray (model .predict_proba (
        wire_gate .within_part (AB_ ,"zskor"))[:,1 ],float )
        k =s >=product_wide .ESIK 
        P ,D =(d ["P"][k ],d ["D"][k ])if k .any ()else (d ["P"][:0 ],d ["D"][:0 ])
        T =d ["X"][kendi ][k ][:,AB +len (YB .OZ_AD ):]
        if len (P )>1 :
            nm =wire_gate .crowd_mask (P ,s [k ])
            P ,D ,T =P [nm ],D [nm ],T [nm ]
        D =product_wide .isaret_duzelt (D ,T )if len (D )else D 
        tp ,fp ,fn =match_hungarian (P ,D ,d ["G"],d ["Gd"],d ["diag"],K .YANAL ,
        K .ACI ,False ,signed =True )[:3 ]
        a =rob [d ["mfg"]]
        a [0 ]+=tp ;a [1 ]+=fp ;a [2 ]+=fn 
        tes .append ((len (d ["G"]),)+match_hungarian (
        P ,D ,d ["G"],d ["Gd"],d ["diag"],max (3.0 ,0.06 *d ["diag"]),
        180.0 ,True )[:3 ])
    pm ={m :2 *a [0 ]/max (2 *a [0 ]+a [1 ]+a [2 ],1 )for m ,a in rob .items ()}
    return {"robot":float (2 *sum (a [0 ]for a in rob .values ())/
    max (sum (2 *a [0 ]+a [1 ]+a [2 ]
    for a in rob .values ()),1 )),
    "tespit":K .mikro (tes ),
    "makro":float (np .mean (list (pm .values ()))),
    "en_kotu":float (min (pm .values ())),"brand":pm ,
    "TP":sum (a [0 ]for a in rob .values ()),
    "FP":sum (a [1 ]for a in rob .values ())}


def main ():
    t0 =time .time ()
    tr =yukle ("tam",int (os .environ .get ("P6_TR","0")))
    dev =yukle ("d6",int (os .environ .get ("P6_DEV","0")))
    print (f"training {len (tr )} part | dev {len (dev )} part "
    f"({time .time ()-t0 :.0f} s)",flush =True )
    if not tr or not dev :
        sys .exit ("VERI YOK -- first `python run_p6_feature.py tam` kos.")

    M =np .vstack ([donustur (d ["X"])for d in tr ])
    Y =np .concatenate ([d ["y"]for d in tr ])
    print (f"secenek {M .shape } | pozitif {Y .mean ():.4f}",flush =True )

    m =HistGradientBoostingClassifier (max_iter =600 ,learning_rate =0.06 ,
    max_leaf_nodes =63 ,l2_regularization =1.0 ,
    random_state =0 ).fit (M ,Y )
    sk =[np .asarray (m .predict_proba (donustur (d ["X"]))[:,1 ],float )
    for d in dev ]

    tb =baseline (dev )
    print (f"\nTABAN (dagitilan yol, ayni parts): robot {tb ['robot']:.4f} | "
    f"tespit {tb ['tespit']:.4f} | makro {tb ['makro']:.4f} | "
    f"TP {tb ['TP']} FP {tb ['FP']}",flush =True )
    print (f"\n{'threshold':>6} {'robot':>8} {'tespit':>8} {'makro':>8} {'TP':>6} {'FP':>6}")
    res_ ={}
    en =None 
    for e in ESIKLER :
        r =olc (dev ,sk ,e )
        res_ [f"{e :.2f}"]=r 
        print (f"{e :>6.2f} {r ['robot']:>8.4f} {r ['tespit']:>8.4f} "
        f"{r ['makro']:>8.4f} {r ['TP']:>6} {r ['FP']:>6}",flush =True )
        if en is None or r ["robot"]>en [1 ]["robot"]:
            en =(e ,r )
    e ,r =en 
    print (f"\nEN IYI threshold {e :.2f} -> robot {r ['robot']:.4f} "
    f"(baseline {tb ['robot']:.4f}, fark {r ['robot']-tb ['robot']:+.4f})")
    with open ("results/p6_ortak_model.pkl","wb")as f :
        pickle .dump ({"model":m ,"threshold":e ,"zskor":ZSKOR ,"AB":AB },f )
    json .dump ({"damga":makbuz_hash .damga (),"baseline":tb ,"esik_taramasi":res_ ,
    "en_iyi_esik":e ,"zskor":ZSKOR ,"n_dev":len (dev ),
    "n_egitim":len (tr ),
    "not":"P6 ortak (konum x direction) siralayici. D6 DEV -- D7'ye "
    "BAKILMADI. Poz kafasi UYGULANMADI (ayri olculur)."},
    open ("results/p6_ortak_d6.json","w"),indent =1 )
    print ("receipt -> results/p6_ortak_d6.json")


if __name__ =="__main__":
    main ()
