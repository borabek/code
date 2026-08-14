# -*- coding: utf-8 -*-
"""SECICI AILESI: same pool/feature/label, FARKLI siniflandirici.

Kazanan arm: B-rep havuzu + mouth tanimlayicilari, RF 400/leaf3 -> D7 TAM ZINCIR
robot **0.2649** (urun 0.2029). Havuz and feature tarafi tuketildi; this betik
SECICININ KENDISINI changes. Tek degisken: model.

Kollar:
  RF-400/3   (kazanan baseline)
  RF-800/1   more high kapasite
  HGB        HistGradientBoosting (tabular veride RF'yi sik geciyor)
  HGB derin  more very yaprak / iterasyon
  ENSEMBLE   RF + HGB skor ortalamasi

Esik HER MODEL ICIN D6'da AYRI secilir (model kalibrasyonu different becomes; sabit
threshold kullanmak zayif modeli haksiz cezalandirirdi). D7'de yeniden TARANMAZ.
"""
import collections 
import json 
import os 
import pickle 
import sys 

import numpy as np 
from sklearn .ensemble import HistGradientBoostingClassifier ,RandomForestClassifier 

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
from p1c_threshold import maske # noqa: E402
from sina_cluster import match_hungarian # noqa: E402

OZ ="results/_tam_oz"
TAN ="results/_tan_hizali"
OB ="results/_p1_olasilik_d7"
KAYNAKLAR =(0 ,1 )
DONUSUM ="zskor"
KURALLAR =[("mutlak",x )for x in (0.05 ,0.10 ,0.15 ,0.20 ,0.25 ,0.30 )]
_D6 =None 


def oku (on ):
    v =[]
    for f in sorted (os .listdir (OZ )):
        if not (f .startswith (on +"_")and f .endswith (".npz")):
            continue 
        z =np .load (f"{OZ }/{f }")
        kay =np .asarray (z ["kaynak"],int )
        m =np .isin (kay ,KAYNAKLAR )
        if int (m .sum ())<2 :
            continue 
        T =np .asarray (np .load (f"{TAN }/{f }")["T"],float )
        X =np .hstack ([np .asarray (z ["X"],float ),T ])
        if len (T )!=len (z ["X"]):
            raise SystemExit (f"{f }: HIZALAMA BOZUK")
        v .append ({"pid":f [len (on )+1 :-4 ],"X":X [m ],
        "y":np .asarray (z ["y"],int )[m ],
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


def skorla (modeller ,X ):
    Xd =wire_gate .within_part (X ,DONUSUM )
    s =np .mean ([m .predict_proba (Xd )[:,1 ]for m in modeller ],axis =0 )
    return np .asarray (s ,float )


def olc (modeller ,veri ,e ,tam_zincir =False ,S =None ):
    rob =collections .defaultdict (lambda :[0 ,0 ,0 ])
    tes =[]
    for d in veri :
        s =skorla (modeller ,d ["X"])
        k =s >=e 
        P ,D =(d ["P"][k ],d ["D"][k ])if k .any ()else (d ["P"][:0 ],d ["D"][:0 ])
        if len (P )>1 :
            nm =wire_gate .crowd_mask (P ,s [k ])
            P ,D =P [nm ],D [nm ]
        if tam_zincir and len (P ):
            f =f"{OB }/{d ['pid']}.npz"
            if os .path .exists (f ):
                z =np .load (f )
                P ,D =product_zinciri .tam_poz (
                np .ascontiguousarray (z ["V"],np .float64 ),
                np .ascontiguousarray (z ["F"],np .int64 ),
                np .asarray (z ["pbs"],float ).mean (0 ),P ,D ,
                step_path =(S or {}).get (d ["pid"]))
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
    S =K .step_map ()
    tr =kimlikle (oku ("tam"))
    dev =kimlikle (oku ("d6"))
    te =kimlikle (oku ("d7"))
    M =np .vstack ([wire_gate .within_part (d ["X"],DONUSUM )for d in tr ])
    Y =np .concatenate ([d ["y"]for d in tr ])
    print (f"training {M .shape } pozitif {Y .mean ():.4f} | dev {len (dev )} | "
    f"exam {len (te )}",flush =True )

    kurulum ={
    "RF-400/3":lambda :RandomForestClassifier (
    n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,random_state =0 ),
    "RF-800/1":lambda :RandomForestClassifier (
    n_estimators =800 ,min_samples_leaf =1 ,n_jobs =-1 ,random_state =0 ),
    "HGB":lambda :HistGradientBoostingClassifier (
    max_iter =300 ,learning_rate =0.1 ,random_state =0 ),
    "HGB-derin":lambda :HistGradientBoostingClassifier (
    max_iter =600 ,learning_rate =0.06 ,max_leaf_nodes =63 ,
    l2_regularization =1.0 ,random_state =0 ),
    }
    egitilmis ={}
    sonuc ={}
    for ad ,yap in kurulum .items ():
        m =yap ().fit (M ,Y )
        egitilmis [ad ]=m 
        en =None 
        for _t ,e in KURALLAR :
            r =olc ([m ],dev ,e )
            if en is None or r ["robot"]>en [1 ]["robot"]:
                en =(e ,r )
        e ,_ =en 
        r7 =olc ([m ],te ,e ,tam_zincir =True ,S =S )
        sonuc [ad ]=dict (r7 ,threshold =e )
        print (f"{ad :<12} threshold {e :.2f} -> D7 robot **{r7 ['robot']:.4f}** | tespit "
        f"{r7 ['tespit']:.4f} | makro {r7 ['makro']:.4f} | en kotu "
        f"{r7 ['en_kotu']:.4f}",flush =True )
    ens =[egitilmis ["RF-400/3"],egitilmis ["HGB-derin"]]
    en =None 
    for _t ,e in KURALLAR :
        r =olc (ens ,dev ,e )
        if en is None or r ["robot"]>en [1 ]["robot"]:
            en =(e ,r )
    e ,_ =en 
    r7 =olc (ens ,te ,e ,tam_zincir =True ,S =S )
    sonuc ["ENSEMBLE RF+HGB"]=dict (r7 ,threshold =e )
    print (f"{'ENSEMBLE':<12} threshold {e :.2f} -> D7 robot **{r7 ['robot']:.4f}** | "
    f"tespit {r7 ['tespit']:.4f} | makro {r7 ['makro']:.4f}",flush =True )
    iyi =max (sonuc ,key =lambda k :sonuc [k ]["robot"])
    print (f"\nEN IYI: {iyi } robot {sonuc [iyi ]['robot']:.4f} | urun 0.2029 "
    f"({sonuc [iyi ]['robot']-0.2029 :+.4f})")
    with open ("results/secici_ailesi_modeller.pkl","wb")as f :
        pickle .dump (egitilmis ,f )
    json .dump ({"damga":makbuz_hash .damga (),"sonuc":sonuc ,"en_iyi":iyi ,
    "urun":0.2029 ,
    "not":"Ayni pool (B-rep + tanimlayici), ayni etiket; TEK DEGISKEN "
    "siniflandirici. Esik her model icin D6'da ayri secildi. "
    "D7 brand-disi, TAM ZINCIR, MIKRO."},
    open ("results/secici_ailesi.json","w"),indent =1 )
    print ("receipt -> results/secici_ailesi.json")


if __name__ =="__main__":
    main ()
