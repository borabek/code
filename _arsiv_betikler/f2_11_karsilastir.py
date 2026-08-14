# -*- coding: utf-8 -*-
"""F2-11 KARSILASTIRMA: A (STEP agi) vs B (JSON agi) vs C (JSON agi, STEP absent).

KILL (f2_11_parite.py'de onceden yazildi):
    candidate recall farki <= 0.01  VE  manufacturer basina tespit F1 farki <= 0.015
GECERSE: STEP'siz 2759 part kullanilabilir -> corpus 1926 -> ~4685 (2.4x).
GECMEZSE: new data only segmentasyon ten-egitimi for kullanilabilir.

ADIL KARSILASTIRMA SARTI: A kolu, same gece same ortamda uretilmis `_der_kontrol.pkl`
olmalidir -- 2 gunluk `_der_tam.pkl` DEGIL. Sebep [[robot-nondeterminism]]: pymeshlab
remesh surecler arasi oynuyor (~0.4mm), and aradaki kod degisiklikleri de karisir.

GATE: each arm KENDI candidate dagilimiyla egitilmis gate with puanlanmali (G3 dersi: dagitilan
gate'i baska a dagilima uygulamak kolu HAKSIZ YERE oldurur). Burada gate measurement
gruplari CIKARILARAK egitilir (tezgah2 with same yordam).
"""
import io 
import json 
import os 
import pickle 
import sys 

import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1";os .environ ["WG_TOPO"]="1";os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))

import measure_set 
import wire_gate 
from sina_cluster import match_greedy ,f1w 


def aday_recall (DER ):
    ul =n =na =0 
    for r in DER :
        G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
        n +=len (G )
        P =np .asarray (r ["P"],float )if r ["P"]is not None else np .zeros ((0 ,3 ))
        na +=len (P )
        if len (P )and len (G ):
            d =P [:,None ,:]-G [None ,:,:]
            al =(d *Gd [None ,:,:]).sum (-1 )
            pe =np .linalg .norm (d -al [...,None ]*Gd [None ,:,:],axis =-1 )
            pe =np .where (np .abs (al )>40 ,np .inf ,pe )
            ul +=int ((pe .min (0 )<=max (3.0 ,0.06 *float (r ["diag"]))).sum ())
    return ul /max (n ,1 ),na ,n 


def gate_kur (DER ):
    """Bu kolun KENDI candidate dagilimiyla gate egit (measurement gruplari already disarida)."""
    from sklearn .ensemble import RandomForestClassifier 
    X ,y ,pid =[],[],[]
    for r in DER :
        if r ["X"]is None or r .get ("XR")is None :
            continue 
        M =np .hstack ([r ["X"],r ["XR"]])
        G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
        P =np .asarray (r ["P"],float )
        if len (G )and len (P ):
            d =P [:,None ,:]-G [None ,:,:]
            al =(d *Gd [None ,:,:]).sum (-1 )
            pe =np .linalg .norm (d -al [...,None ]*Gd [None ,:,:],axis =-1 )
            pe =np .where (np .abs (al )>40 ,np .inf ,pe )
            lab =(pe .min (1 )<=max (3.0 ,0.06 *float (r ["diag"]))).astype (int )
        else :
            lab =np .zeros (len (P ),int )
        X .append (M );y .append (lab );pid +=[r ["pid"]]*len (M )
    X =np .vstack (X );y =np .concatenate (y );pid =np .array (pid )
    Z =np .zeros ((len (X ),X .shape [1 ]*2 ))
    for u in np .unique (pid ):
        i =np .where (pid ==u )[0 ]
        Z [i ]=wire_gate .within_part (X [i ],"zskor")
    return {"clf":RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
    random_state =0 ).fit (Z ,y ),
    "n_feat":Z .shape [1 ],"donusum":"zskor"},pid ,y 


def puanla (DER ,gate ):
    """GRUP-CAPRAZ: each parcanin skoru, that parcanin grubu DISLANARAK egitilmis gateden."""
    rows ,mfg_rows =[],{}
    for r in DER :
        G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
        rj ="cok"if r ["n"]>=8 else "dusuk"
        P =np .zeros ((0 ,3 ));D =np .zeros ((0 ,3 ))
        if r ["X"]is not None and r .get ("XR")is not None :
            M =np .hstack ([r ["X"],r ["XR"]])
            if M .shape [1 ]*2 ==gate ["n_feat"]:
                k =wire_gate .decision_mask (wire_gate .decision_score (gate ,M ))
                if k .any ():
                    P =np .asarray (r ["P"],float )[k ];D =np .asarray (r ["Pd"],float )[k ]
        tp ,fp ,fn ,_ =match_greedy (P ,D ,G ,Gd ,r ["diag"],0.0 ,180.0 ,True )
        rows .append ((rj ,tp ,fp ,fn ))
        mfg_rows .setdefault (r .get ("mfg","?"),[]).append ((rj ,tp ,fp ,fn ))
    return f1w (rows ),{m :f1w (v )for m ,v in mfg_rows .items ()},rows 


def main ():
    import protocol 
    protocol .tez_dogrula ()
    KOLLAR ={}
    for ad ,yol in (("A_STEP","results/_der_kontrol.pkl"),
    ("B_JSON","results/_der_json_B.pkl"),
    ("C_JSON_stepsiz","results/_der_json_C.pkl")):
        if not os .path .exists (yol ):
            print (f"  {ad }: {yol } YOK, atlaniyor")
            continue 
        with open (yol ,"rb")as f :
            KOLLAR [ad ]=pickle .load (f )
    ortak =set .intersection (*[{r ["pid"]for r in v }for v in KOLLAR .values ()])
    print (f"kollar: {list (KOLLAR )} | ORTAK part: {len (ortak )}")
    for ad in KOLLAR :
        KOLLAR [ad ]=[r for r in KOLLAR [ad ]if r ["pid"]in ortak ]

    print (f"\n{'arm':<18}{'candidate':>7}{'GT':>6}{'recall':>9}{'tespit F1':>11}")
    SON ={}
    for ad ,DER in KOLLAR .items ():
        rc ,na ,ng =aday_recall (DER )
        gate ,_ ,_ =gate_kur (DER )
        f1 ,per_mfg ,_ =puanla (DER ,gate )
        SON [ad ]={"recall":rc ,"candidate":na ,"gt":ng ,"tespit":f1 ,"manufacturer":per_mfg }
        print (f"{ad :<18}{na :>7}{ng :>6}{rc :>9.4f}{f1 :>11.4f}")

    if "A_STEP"in SON and "B_JSON"in SON :
        a ,b =SON ["A_STEP"],SON ["B_JSON"]
        dr =b ["recall"]-a ["recall"]
        print (f"\n{'='*66 }\nKILL DEGERLENDIRMESI (A vs B -- yalniz MESH farki)\n{'='*66 }")
        print (f"  candidate recall farki : {dr :+.4f}   (threshold <= 0.01)  "
        f"{'OK'if abs (dr )<=0.01 else 'KALDI'}")
        print (f"  tespit F1 farki   : {b ['tespit']-a ['tespit']:+.4f}")
        print (f"\n  {'manufacturer':<10}{'A_STEP':>9}{'B_JSON':>9}{'fark':>9}")
        enb =0.0 
        for m in sorted (set (a ["manufacturer"])|set (b ["manufacturer"])):
            va ,vb =a ["manufacturer"].get (m ,0 ),b ["manufacturer"].get (m ,0 )
            enb =max (enb ,abs (vb -va ))
            print (f"  {m :<10}{va :>9.4f}{vb :>9.4f}{vb -va :>+9.4f}")
        print (f"\n  manufacturer basina EN BUYUK fark: {enb :.4f}  (threshold <= 0.015)  "
        f"{'OK'if enb <=0.015 else 'KALDI'}")
        gecti =abs (dr )<=0.01 and enb <=0.015 
        print (f"\n  KILL -> {'GECTI: JSON agi KULLANILABILIR, 2759 part acilir'if gecti else 'GECMEDI'}")
        if "C_JSON_stepsiz"in SON :
            c =SON ["C_JSON_stepsiz"]
            print (f"\n  C kolu (STEP YOK, gercek kosul): recall {c ['recall']:.4f} | "
            f"tespit {c ['tespit']:.4f}  -> B'ye gore {c ['tespit']-b ['tespit']:+.4f}")
            print (f"  (bu fark B-rep ozellik blogunun BEDELIDIR)")
        SON ["kill"]={"recall_farki":dr ,"uretici_en_buyuk":enb ,"gecti":bool (gecti )}
    with io .open ("results/f2_11_parite.json","w",encoding ="utf-8")as f :
        json .dump (SON ,f ,indent =1 ,ensure_ascii =False )
    print ("\nmakbuz -> results/f2_11_parite.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
