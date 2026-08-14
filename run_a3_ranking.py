# -*- coding: utf-8 -*-
"""A3: SIRALAMA hedefi -- GATE_REDDI a threshold not SIRALAMA hatasi.

Olculdu: karar kurali ailesinin HEPSI tabanin under kaldi
(`results/a1_esik_kalibrasyon.json`) -> esigi nasil kurarsan kur GATE_REDDI
kovasi (531 GT, %17.2) bosalmiyor. Demek ki uygun candidates yanlislarin ALTINDA
siralaniyor; sorun modelin OGRENME HEDEFINDE.

Pointwise BCE each adayi BAGIMSIZ ogreniyor and very adayli parts kaybi domine
ediyor. Denenen agirliklandirmalar (single degisken: `sample_weight`):
  duz          weight absent (kazanan baseline, 0.3070/0.3090)
  parca_esit   1/n_parca -- each part kayba ESIT katkida bulunur
  poz_dengeli  part ICINDE pozitif/negatif dengelenir
  karma        parca_esit x poz_dengeli
Ayrica PAIRWISE arm: part inside (pozitif, negatif) ciftlerinin FARK vektoru
on ikili siniflandirici -- dogrudan ranking ogrenir.

Esik each model for D6'da AYRI secilir. Isaret duzeltmesi HER kolda open.
"""
import collections 
import json 
import os 
import pickle 
import sys 

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
import product_zinciri # noqa: E402
import wire_gate # noqa: E402
from sina_cluster import match_hungarian # noqa: E402

OZ ="results/_tam_oz"
TAN ="results/_tan_hizali"
OB ={"d7":"results/_p1_olasilik_d7","d6":"results/_p1_olasilik_g7",
"tam":"results/_p1_olasilik_brepegit"}
KAYNAKLAR =(0 ,1 )
YANAL ,ACI ,EKSENEL =2.0 ,10.0 ,40.0 
GIRME ,ERISIM =3 ,5 
ESIKLER =(0.03 ,0.05 ,0.08 ,0.12 ,0.20 ,0.30 ,0.45 )
RNG =np .random .default_rng (0 )


def yukle (on ,pid ,mesh =False ):
    z =np .load (f"{OZ }/{on }_{pid }.npz")
    m =np .isin (np .asarray (z ["kaynak"],int ),KAYNAKLAR )
    T =np .asarray (np .load (f"{TAN }/{on }_{pid }.npz")["T"],float )
    X =np .hstack ([np .asarray (z ["X"],float ),T ])[m ]
    if len (X )<2 :
        return None 
    d ={"X":X ,"T":T [m ],"y":np .asarray (z ["y"],int )[m ],
    "P":np .asarray (z ["P"],float )[m ],"D":np .asarray (z ["D"],float )[m ]}
    if mesh :
        f =f"{OB [on ]}/{pid }.npz"
        if not os .path .exists (f ):
            return None 
        zz =np .load (f )
        d .update ({"V":np .ascontiguousarray (zz ["V"],np .float64 ),
        "F":np .ascontiguousarray (zz ["F"],np .int64 ),
        "pb":np .asarray (zz ["pbs"],float ).mean (0 )})
    return d 


def cluster (on ,mesh =False ):
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
        d =yukle (on ,pid ,mesh )
        if d is None :
            continue 
        d .update ({"pid":pid ,"mfg":r ["mfg"],"G":np .asarray (r ["G"],float ),
        "Gd":np .asarray (r ["Gd"],float ),"diag":float (r ["diag"])})
        out .append (d )
    return out 


def wgt_ (tr ,kip ):
    w =[]
    for d in tr :
        n =len (d ["y"])
        p =max (int (d ["y"].sum ()),1 )
        a =np .ones (n )
        if kip in ("parca_esit","karma"):
            a =a /n 
        if kip in ("poz_dengeli","karma"):
            a =a *np .where (d ["y"]==1 ,(n -p )/p ,1.0 )
        w .append (a )
    return np .concatenate (w )


def olc (skorla ,data_ ,e ,S ,tam =False ):
    rob =collections .defaultdict (lambda :[0 ,0 ,0 ])
    tes =[]
    for d in data_ :
        s =skorla (d ["X"])
        k =s >=e 
        if not k .any ():
            P ,D =d ["P"][:0 ],d ["D"][:0 ]
        else :
            P ,D ,T ,sk =d ["P"][k ],d ["D"][k ],d ["T"][k ],s [k ]
            if len (P )>1 :
                nm =wire_gate .crowd_mask (P ,sk )
                P ,D ,T =P [nm ],D [nm ],T [nm ]
            D =np .where ((T [:,ERISIM ]<T [:,GIRME ])[:,None ],-D ,D )
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
    "en_kotu":float (min (pm .values ())),"brand":pm }


def main ():
    S =K .step_map ()
    tr =cluster ("tam")
    dev =cluster ("d6",mesh =True )
    te =cluster ("d7",mesh =True )
    M =np .vstack ([wire_gate .within_part (d ["X"],"zskor")for d in tr ])
    Y =np .concatenate ([d ["y"]for d in tr ])
    print (f"training {M .shape } pozitif {Y .mean ():.4f} | D6 {len (dev )} | "
    f"D7 {len (te )}\n",flush =True )
    res_ ={}
    for kip in ("duz","parca_esit","poz_dengeli","karma"):
        w =None if kip =="duz"else wgt_ (tr ,kip )
        m =HistGradientBoostingClassifier (
        max_iter =600 ,learning_rate =0.06 ,max_leaf_nodes =63 ,
        l2_regularization =1.0 ,random_state =0 ).fit (M ,Y ,sample_weight =w )

        def sk (X ,_m =m ):
            return np .asarray (_m .predict_proba (
            wire_gate .within_part (X ,"zskor"))[:,1 ],float )

        en =None 
        for e in ESIKLER :
            r =olc (sk ,dev ,e ,S )
            if en is None or r ["robot"]>en [1 ]["robot"]:
                en =(e ,r )
        e ,_ =en 
        r7 =olc (sk ,te ,e ,S ,tam =True )
        res_ [kip ]=dict (r7 ,threshold =e )
        print (f"{kip :<12} threshold {e :.2f} -> D7 robot **{r7 ['robot']:.4f}** | "
        f"tespit {r7 ['tespit']:.4f} | makro {r7 ['makro']:.4f}",flush =True )

        # --- PAIRWISE: part inside (poz - neg) difference vektorleri
    A ,B =[],[]
    for d in tr :
        Md =wire_gate .within_part (d ["X"],"zskor")
        pi =np .where (d ["y"]==1 )[0 ]
        ni =np .where (d ["y"]==0 )[0 ]
        if not len (pi )or not len (ni ):
            continue 
        ns =ni if len (ni )<=12 else RNG .choice (ni ,12 ,replace =False )
        for i in pi :
            for j in ns :
                A .append (Md [i ]-Md [j ]);B .append (1 )
                A .append (Md [j ]-Md [i ]);B .append (0 )
    A =np .asarray (A ,float );B =np .asarray (B ,int )
    print (f"\npairwise {A .shape }",flush =True )
    pm_ =HistGradientBoostingClassifier (
    max_iter =400 ,learning_rate =0.08 ,max_leaf_nodes =63 ,
    random_state =0 ).fit (A ,B )

    def sk_pair (X ):
        """Siralama skoru: adayin PARCA ORTALAMASINA according to ustunlugu."""
        Md =wire_gate .within_part (X ,"zskor")
        ort =Md .mean (0 ,keepdims =True )
        return np .asarray (pm_ .predict_proba (Md -ort )[:,1 ],float )

    en =None 
    for e in (0.3 ,0.4 ,0.5 ,0.6 ,0.7 ,0.8 ):
        r =olc (sk_pair ,dev ,e ,S )
        if en is None or r ["robot"]>en [1 ]["robot"]:
            en =(e ,r )
    e ,_ =en 
    r7 =olc (sk_pair ,te ,e ,S ,tam =True )
    res_ ["pairwise"]=dict (r7 ,threshold =e )
    print (f"{'pairwise':<12} threshold {e :.2f} -> D7 robot **{r7 ['robot']:.4f}** | "
    f"tespit {r7 ['tespit']:.4f} | makro {r7 ['makro']:.4f}",flush =True )

    iyi =max (res_ ,key =lambda k :res_ [k ]["robot"])
    print (f"\nEN IYI: {iyi } {res_ [iyi ]['robot']:.4f} | baseline (duz) "
    f"{res_ ['duz']['robot']:.4f} | fark "
    f"{res_ [iyi ]['robot']-res_ ['duz']['robot']:+.4f}")
    print ("KAPI: >= +0.02")
    json .dump ({"damga":makbuz_hash .damga (),"sonuc":res_ ,"en_iyi":iyi ,
    "not":"Tek degisken: OGRENME HEDEFI (agirlik / pairwise). Esik "
    "each model for D6'da secildi. Isaret duzeltmesi HER kolda "
    "acik. D7 brand-disi, TAM ZINCIR, MIKRO."},
    open ("results/a3_siralama.json","w"),indent =1 )
    print ("receipt -> results/a3_siralama.json")


if __name__ =="__main__":
    main ()
