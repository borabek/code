# -*- coding: utf-8 -*-
"""A1+A2: GATE_REDDI kovasini bosalt -- goreli threshold and SKOR kalibrasyonu.

HATA BANKASI (`results/kazanan_hata_bankasi.json`): kazanan yiginda
GATE_REDDI = 531 GT (%17.2). Bunlar for havuzda ROBOT-UYGUN candidate VARDI but
skoru esigin under kaldi. Kova aritmetigi: only this kova kurtarilirsa
F1 0.2773 -> **0.439**.

GATE_REDDI a SIRALAMA/KALIBRASYON hatasi. Uc karar kurali kiyaslanir:
  MUTLAK    s >= e                       (kazananin kullandigi, e=0.05)
  GORELI    urunun `p1c_threshold.maske`      (part-ici ratio + mutlak baseline);
            measured ki distribution kaymasini emiyor (gate-manufacturer-disi-cokusu:
            0.2799 -> 0.4402)
  Z-SKOR    skoru PARCA ICINDE z-skorla, after esikle. Oznitelikte `within_part`
            present but SKORDA absent; mutlak skor markadan markaya kayiyorsa this
            dogrudan fixes.
  YUZDELIK  part inside skor yuzdeligi >= q (count-serbest ranking karari)

Esik/parametre D6'da secilir, D7'de YENIDEN TARANMAZ. TAM ZINCIR, MIKRO.
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
import product_chain # noqa: E402
import wire_gate # noqa: E402
from p1c_threshold import maske # noqa: E402
from sina_cluster import match_hungarian # noqa: E402

OZ ="results/_tam_oz"
TAN ="results/_tan_hizali"
OB ="results/_p1_olasilik_d7"
KAYNAKLAR =(0 ,1 )
_D6 =None 

KURALLAR =(
[("mutlak",e )for e in (0.03 ,0.05 ,0.08 ,0.12 ,0.20 )]+
[("goreli",(o ,t ))for o in (0.3 ,0.5 ,0.7 )for t in (0.02 ,0.05 ,0.10 )]+
[("zskor",z )for z in (0.0 ,0.5 ,1.0 ,1.5 ,2.0 )]+
[("yuzdelik",q )for q in (0.80 ,0.90 ,0.94 ,0.97 )]
)


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


def sec (s ,tip ,p ):
    """Karar kurali -> bool maske."""
    if tip =="mutlak":
        return s >=p 
    if tip =="goreli":
        return maske (s ,p [0 ],p [1 ])
    if tip =="zskor":
        sd =s .std ()
        if sd <1e-9 :
            return s >=s .max ()
        return (s -s .mean ())/sd >=p 
    if tip =="yuzdelik":
        return s >=np .quantile (s ,p )
    raise ValueError (tip )


def olc (model ,data_ ,tip ,p ,tam_zincir =False ,S =None ):
    rob =collections .defaultdict (lambda :[0 ,0 ,0 ])
    tes =[]
    for d in data_ :
        s =np .asarray (model .predict_proba (
        wire_gate .within_part (d ["X"],"zskor"))[:,1 ],float )
        k =sec (s ,tip ,p )
        P ,D =(d ["P"][k ],d ["D"][k ])if k .any ()else (d ["P"][:0 ],d ["D"][:0 ])
        if len (P )>1 :
            nm =wire_gate .crowd_mask (P ,s [k ])
            P ,D =P [nm ],D [nm ]
        if tam_zincir and len (P ):
            f =f"{OB }/{d ['pid']}.npz"
            if os .path .exists (f ):
                z =np .load (f )
                P ,D =product_chain .tam_poz (
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
    model =pickle .load (open ("results/kazanan_hgb_derin.pkl","rb"))["HGB-derin"]
    S =K .step_map ()
    dev =kimlikle (oku ("d6"))
    te =kimlikle (oku ("d7"))
    print (f"D6 {len (dev )} | D7 {len (te )}\n",flush =True )
    d6skor ={}
    for tip ,p in KURALLAR :
        r =olc (model ,dev ,tip ,p )
        d6skor [f"{tip } {p }"]=r ["robot"]
        print (f"  D6 {tip :<9}{str (p ):<14} robot {r ['robot']:.4f}",flush =True )
        # each AILE for D6'nin most iyisi -> D7'de olc (aile kiyasi for)
    aileler ={}
    for tip ,p in KURALLAR :
        aileler .setdefault (tip ,[]).append ((d6skor [f"{tip } {p }"],p ))
    out ={}
    for tip ,v in aileler .items ():
        p =max (v )[1 ]
        r7 =olc (model ,te ,tip ,p ,tam_zincir =True ,S =S )
        out [tip ]=dict (r7 ,secilen =str (p ),d6 =max (v )[0 ])
        print (f"\n{tip :<9} secilen {str (p ):<14} -> D7 robot **{r7 ['robot']:.4f}** "
        f"| tespit {r7 ['tespit']:.4f} | makro {r7 ['makro']:.4f} | "
        f"en kotu {r7 ['en_kotu']:.4f}",flush =True )
    iyi =max (out ,key =lambda k :out [k ]["robot"])
    print (f"\nEN IYI KURAL: {iyi } -> {out [iyi ]['robot']:.4f} "
    f"(baseline mutlak 0.05 = 0.2773, fark {out [iyi ]['robot']-0.2773 :+.4f})")
    print ("KAPI: >= +0.02 whereas A1/A2 KABUL")
    json .dump ({"damga":makbuz_hash .damga (),"D6":d6skor ,"D7":out ,
    "baseline":0.2773 ,"en_iyi":iyi ,
    "not":"Karar kurali ailesi kiyasi. Parametre D6'da secildi, D7'de "
    "yeniden taranmadi. TAM ZINCIR, MIKRO, D7 brand-disi."},
    open ("results/a1_esik_kalibrasyon.json","w"),indent =1 )
    print ("receipt -> results/a1_esik_kalibrasyon.json")


if __name__ =="__main__":
    main ()
