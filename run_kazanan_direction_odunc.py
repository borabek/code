# -*- coding: utf-8 -*-
"""KAZANAN KOL + YON ODUNC ALMA (duzeltilmis etiketle).

Kazanan: B-rep havuzu + mouth tanimlayicilari, D7 TAM ZINCIR robot **0.2649**
(urun 0.2029). Donusum %60 (0.2649/0.4409) -- urunun %45'inin very ustunde,
i.e. pool adaylari more iyi KONUMLANIYOR. Kalan loss TESPITTE.

Yon odunc alma this gece -0.0044 olculmustu but that measurement BOZUK ETIKETLI gate with
yapilmisti ([[gate-label-tanimi-hatasi-and-v6-kunyesi]]); dayanagi cokmus
durumda. Kol candidate EKLEMEZ, only secilen adayin yonunu changes -> tespit
YAPISAL OLARAK bozulamaz.

Yon selector, gate'in KENDI skorunu and `yon_odunc.OZ_AD` ozniteliklerini kullanir;
training etiketi "this direction adayi ROBOT-hazir yapar mi" (signed angle <=10).
"""
import collections 
import json 
import os 
import pickle 
import sys 

import numpy as np 
from sklearn .ensemble import RandomForestClassifier 

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
import yon_odunc as YO # noqa: E402
from p1c_threshold import maske # noqa: E402
from sina_cluster import match_hungarian # noqa: E402

OZ ="results/_tam_oz"
TAN ="results/_tan_hizali"
OB ="results/_p1_olasilik_d7"
KAYNAKLAR =(0 ,1 )
KURAL =("mutlak",0.15 )# kazanan kolda D6'da secilmisti
YANAL ,ACI ,EKSENEL =2.0 ,10.0 ,40.0 
_D6 =None 


def oku (on ):
    v =[]
    for f in sorted (os .listdir (OZ )):
        if not (f .startswith (on +"_")and f .endswith (".npz")):
            continue 
        z =np .load (f"{OZ }/{f }")
        kay =np .asarray (z ["source"],int )
        m =np .isin (kay ,KAYNAKLAR )
        if int (m .sum ())<2 :
            continue 
        T =np .asarray (np .load (f"{TAN }/{f }")["T"],float )
        X =np .hstack ([np .asarray (z ["X"],float ),T ])[m ]
        v .append ({"pid":f [len (on )+1 :-4 ],"X":X ,
        "P":np .asarray (z ["P"],float )[m ],
        "D":np .asarray (z ["D"],float )[m ]})
    return v 


def kimlikle (v ):
    global _D6 
    kay =K .yukle ([d ["pid"]for d in v ])
    eksik =[d ["pid"]for d in v if d ["pid"]not in kay ]
    if eksik :
        if _D6 is None :
            _D6 ={str (p ):r for p ,r in 
            d6_record .yukle (set (d6_record .exam ()["pidler"])).items ()}
        kay .update ({p :_D6 [p ]for p in eksik if p in _D6 })
    out =[]
    for d in v :
        r =kay .get (d ["pid"])
        if r is None :
            continue 
        d .update ({"mfg":r ["mfg"],"G":np .asarray (r ["G"],float ),
        "Gd":np .asarray (r ["Gd"],float ),"diag":float (r ["diag"])})
        out .append (d )
    return out 


def warn_position (p ,g ,gd ):
    n =np .linalg .norm (gd )
    if n <1e-9 :
        return False 
    u =gd /n 
    v =p -g 
    e =float (v @u )
    return (np .linalg .norm (v -e *u )<=YANAL )and (abs (e )<=EKSENEL )


def yon_egitim (data_ ,gate ,cyl ):
    X ,Y =[],[]
    for d in data_ :
        s =np .asarray (wire_gate .decision_score (gate ,d ["X"]),float )
        k =s >=KURAL [1 ]
        if not k .any ():
            continue 
        P ,D ,sk =d ["P"][k ],d ["D"][k ],s [k ]
        if len (P )>1 :
            nm =wire_gate .crowd_mask (P ,sk )
            P ,D ,sk =P [nm ],D [nm ],sk [nm ]
        if not len (P ):
            continue 
        bask =YO .baskin_yon (D )
        cy =cyl .get (d ["pid"])
        for i in range (len (P )):
            j =-1 
            for t in range (len (d ["G"])):
                if warn_position (P [i ],d ["G"][t ],d ["Gd"][t ]):
                    j =t 
                    break 
            if j <0 :
                continue 
            V ,F =YO .secenekler (P ,D ,i ,cy ,float (sk [i ]),bask )
            if len (V )<2 :
                continue 
            u =d ["Gd"][j ]/max (np .linalg .norm (d ["Gd"][j ]),1e-12 )
            for q ,v in enumerate (V ):
                a =np .degrees (np .arccos (np .clip (float (v @u ),-1.0 ,1.0 )))
                X .append (F [q ]);Y .append (int (a <=ACI ))
    return np .asarray (X ,float ),np .asarray (Y ,int )


def olc (data_ ,gate ,cyl ,S ,yon_clf =None ):
    rob =collections .defaultdict (lambda :[0 ,0 ,0 ])
    tes =[]
    for d in data_ :
        s =np .asarray (wire_gate .decision_score (gate ,d ["X"]),float )
        k =s >=KURAL [1 ]
        P ,D ,sk =((d ["P"][k ],d ["D"][k ],s [k ])if k .any ()
        else (d ["P"][:0 ],d ["D"][:0 ],s [:0 ]))
        if len (P )>1 :
            nm =wire_gate .crowd_mask (P ,sk )
            P ,D ,sk =P [nm ],D [nm ],sk [nm ]
        if yon_clf is not None and len (P ):
            D =YO .uygula (P ,D ,cyl .get (d ["pid"]),sk ,
            lambda Z :yon_clf .predict_proba (Z )[:,1 ])
        if len (P ):
            f =f"{OB }/{d ['pid']}.npz"
            if os .path .exists (f ):
                z =np .load (f )
                P ,D =product_zinciri .tam_poz (
                np .ascontiguousarray (z ["V"],np .float64 ),
                np .ascontiguousarray (z ["F"],np .int64 ),
                np .asarray (z ["pbs"],float ).mean (0 ),P ,D ,
                step_path =S .get (d ["pid"]))
        tp ,fp ,fn =match_hungarian (P ,D ,d ["G"],d ["Gd"],d ["diag"],K .YANAL ,
        K .ACI ,False ,signed =True )[:3 ]
        a =rob [d ["mfg"]]
        a [0 ]+=tp ;a [1 ]+=fp ;a [2 ]+=fn 
        tes .append ((len (d ["G"]),)+match_hungarian (
        P ,D ,d ["G"],d ["Gd"],d ["diag"],max (3.0 ,0.06 *d ["diag"]),
        180.0 ,True )[:3 ])
    pm ={m :2 *a [0 ]/max (2 *a [0 ]+a [1 ]+a [2 ],1 )for m ,a in rob .items ()}
    mi =float (2 *sum (a [0 ]for a in rob .values ())/
    max (sum (2 *a [0 ]+a [1 ]+a [2 ]for a in rob .values ()),1 ))
    return {"robot":mi ,"tespit":K .mikro (tes ),
    "makro":float (np .mean (list (pm .values ()))),
    "en_kotu":float (min (pm .values ())),"brand":pm }


def main ():
    gate =pickle .load (open ("results/tg_tanimli.pkl","rb"))["B-REP +TANIM"]
    S =K .step_map ()
    cyl ={}
    for f in ("results/_brepegit_silindirler.pkl","results/_d6_silindirler.pkl",
    "results/_d7_silindirler.pkl"):
        cyl .update (pickle .load (open (f ,"rb")))
    tr =kimlikle (oku ("tam"))
    te =kimlikle (oku ("d7"))
    print (f"training {len (tr )} part | exam {len (te )} part",flush =True )
    X ,Y =yon_egitim (tr ,gate ,cyl )
    print (f"direction secenegi {X .shape } | pozitif {Y .mean ():.4f}",flush =True )
    yc =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
    class_weight ="balanced_subsample",
    random_state =0 ).fit (X ,Y )
    out ={}
    out ["YON SABIT"]=olc (te ,gate ,cyl ,S ,None )
    print (f"YON SABIT   robot {out ['YON SABIT']['robot']:.4f} | tespit "
    f"{out ['YON SABIT']['tespit']:.4f} | makro "
    f"{out ['YON SABIT']['makro']:.4f}",flush =True )
    out ["YON ODUNC"]=olc (te ,gate ,cyl ,S ,yc )
    print (f"YON ODUNC   robot {out ['YON ODUNC']['robot']:.4f} | tespit "
    f"{out ['YON ODUNC']['tespit']:.4f} | makro "
    f"{out ['YON ODUNC']['makro']:.4f}",flush =True )
    d =out ["YON ODUNC"]["robot"]-out ["YON SABIT"]["robot"]
    print (f"\nFARK {d :+.4f} | urun 0.2029")
    json .dump ({"damga":makbuz_hash .damga (),"sonuc":out ,"fark":d ,
    "not":"Kazanan arm (B-rep + tanimlayici) uzerinde direction odunc alma, "
    "DUZELTILMIS etiketli gate with. D7 brand-disi, TAM ZINCIR."},
    open ("results/kazanan_yon_odunc.json","w"),indent =1 )
    print ("receipt -> results/kazanan_yon_odunc.json")


if __name__ =="__main__":
    main ()
