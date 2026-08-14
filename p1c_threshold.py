# -*- coding: utf-8 -*-
"""P1-GATE-C: goreli threshold (ratio/baseline) new candidate yogunluguna according to yeniden secilir.

Dagitilan rule: candidate, own PARCASINDAKI most high skorun **0.5 katini** gecmeli VE
**0.25 tabanini** asmali. Bu two number ESKI candidate havuzuyla secilmisti; that zamandan beri
seg agi degisti (candidate recall %31.7 -> %92.1) and gate v5 yeniden egitildi. Otopside
GATE_REDDI still GT'nin %10.5'i.

SECIM YANLILIGI ENGELI: 8 exam ureticisi IKIYE bolunur.
  DEV  : SUPU, NIT, S+S, SE      (threshold BURADA secilir)
  SINAV: UPUN, MOR, UTL, ONV     (selected threshold BURADA TEK ATIS olculur)
Ikisi de "unseen manufacturer" ozelligini korur; selected number with raporlanan number
AYNI parcalardan gelmez. ([[uclu-split-and-fake-kazanclar]])
"""
import argparse 
import collections 
import io 
import json 
import os 
import pickle 
import sys 

import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1";os .environ ["WG_TOPO"]="1";os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))

import d6_record 

MAKBUZ ="results/p1c_threshold.json"
DEV_MFG ={"SUPU","NIT","S+S","SE"}
ROBOT_YANAL ,ROBOT_ACI =2.0 ,10.0 


def maske (score ,ratio ,baseline ):
    """Urunun goreli karar kurali -- single source here TEKRAR EDILMEZ, same formul."""
    if not len (score ):
        return np .zeros (0 ,bool )
    return (score >=ratio *float (np .max (score )))&(score >=baseline )


def puanla (rec_ ,model ,ratio ,baseline ,match_greedy ,f1w ,mfgler =None ):
    import wire_gate 
    T ,R =[],[]
    for pid ,r in rec_ .items ():
        if mfgler is not None and r ["mfg"]not in mfgler :
            continue 
        G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
        rj ="very"if r ["n"]>=8 else "dusuk"
        P =np .zeros ((0 ,3 ));D =np .zeros ((0 ,3 ))
        M =d6_record .x58 (r )
        if M is not None and r .get ("P")is not None and len (r ["P"])and M .shape [1 ]*2 ==model ["n_feat"]:
            s =wire_gate .decision_score (model ,M )
            k =maske (np .asarray (s ,float ),ratio ,baseline )
            if k .any ():
                P =np .asarray (r ["P"],float )[k ];D =np .asarray (r ["Pd"],float )[k ]
        T .append ((rj ,)+match_greedy (P ,D ,G ,Gd ,r ["diag"],0.0 ,180.0 ,True )[:3 ])
        R .append ((rj ,)+match_greedy (P ,D ,G ,Gd ,r ["diag"],ROBOT_YANAL ,ROBOT_ACI ,
        False ,signed =True )[:3 ])
    return f1w (T ),f1w (R )


def main ():
    ap =argparse .ArgumentParser ()
    ap .add_argument ("--model",default ="results/wire_gate_v5.pkl")
    a =ap .parse_args ()
    import protocol 
    protocol .tez_dogrula ()
    from sina_cluster import match_greedy ,f1w 

    sv =d6_record .exam ()
    rec_ =d6_record .yukle (set (sv ["pidler"]))
    with open (a .model ,"rb")as f :
        model =pickle .load (f )
    dev ={p :r for p ,r in rec_ .items ()if r ["mfg"]in DEV_MFG }
    sin ={p :r for p ,r in rec_ .items ()if r ["mfg"]not in DEV_MFG }
    print (f"model: {a .model }")
    print (f"DEV   {len (dev )} part {sorted ({r ['mfg']for r in dev .values ()})}")
    print (f"SINAV {len (sin )} part {sorted ({r ['mfg']for r in sin .values ()})}\n")

    oranlar =[0.30 ,0.40 ,0.50 ,0.60 ,0.70 ]
    tabanlar =[0.15 ,0.20 ,0.25 ,0.30 ,0.35 ]
    print (f"{'ratio/baseline':<12}"+"".join (f"{t :>9.2f}"for t in tabanlar ))
    en_iyi ,en_iyi_skor =None ,-1.0 
    izgara ={}
    for o in oranlar :
        line_ =[]
        for t in tabanlar :
            tf ,rf =puanla (dev ,model ,o ,t ,match_greedy ,f1w )
            izgara [f"{o }/{t }"]={"detection":tf ,"robot":rf }
            line_ .append (tf )
            if tf >en_iyi_skor :
                en_iyi_skor ,en_iyi =tf ,(o ,t )
        print (f"{o :<12.2f}"+"".join (f"{v :>9.4f}"for v in line_ ))

    o0 ,t0 =0.50 ,0.25 
    dt0 ,dr0 =puanla (dev ,model ,o0 ,t0 ,match_greedy ,f1w )
    print (f"\nDEV'de selected: ratio {en_iyi [0 ]:.2f} / baseline {en_iyi [1 ]:.2f} "
    f"-> detection {en_iyi_skor :.4f}  (mevcut 0.50/0.25: {dt0 :.4f})")

    st0 ,sr0 =puanla (sin ,model ,o0 ,t0 ,match_greedy ,f1w )
    st1 ,sr1 =puanla (sin ,model ,en_iyi [0 ],en_iyi [1 ],match_greedy ,f1w )
    print (f"\n--- SINAV YARISI (TEK ATIS, secimde KULLANILMADI) ---")
    print (f"{'setting':<18}{'TESPIT':>9}{'ROBOT':>9}")
    print (f"{'mevcut 0.50/0.25':<18}{st0 :>9.4f}{sr0 :>9.4f}")
    print (f"{f'yeni {en_iyi [0 ]:.2f}/{en_iyi [1 ]:.2f}':<18}{st1 :>9.4f}{sr1 :>9.4f}")
    print (f"{'FARK':<18}{st1 -st0 :>+9.4f}{sr1 -sr0 :>+9.4f}")
    karar ="DAGIT"if st1 >st0 else "GERI AL (secim DEV'e ozgu cikti)"
    print (f"\nKARAR: {karar }")
    with io .open (MAKBUZ ,"w",encoding ="utf-8")as f :
        json .dump ({"model":a .model ,"dev_mfg":sorted (DEV_MFG ),"izgara":izgara ,
        "dev_secim":{"ratio":en_iyi [0 ],"baseline":en_iyi [1 ],
        "dev_tespit":en_iyi_skor },
        "sinav_mevcut":{"detection":st0 ,"robot":sr0 },
        "sinav_yeni":{"detection":st1 ,"robot":sr1 },
        "karar":karar },f ,indent =1 ,ensure_ascii =False )
    print (f"receipt -> {MAKBUZ }")


if __name__ =="__main__":
    main ()
