# -*- coding: utf-8 -*-
"""D6-6b: TEMIZ gorulmemis-manufacturer sinavinda DURUST TABAN.

WHY: old exam kumesi (`d5_4_sinav_kumesi.json`) URETICI duzeyinde kirlenmisti --
seg oto-label korpusu that kumenin 250 parcasindan 169'unu and exam ureticilerinden 958
parcayi iceriyordu. Orada olculen 0.6050 SISIK. Yeni cluster (`d6_sinav_kumesi.json`,
468 part / 8 manufacturer) no training yapitinda GECMEYEN ureticilerden kuruldu.

YENIDEN TURETME YOK: turetme kayitlari candidate noktalarini (P), yonlerini (Pd) and gate
feature matrislerini (X, XR) already tasiyor. Gate karari and two metrik BUNLARDAN
is computed -- urunun KENDI karar kodu (`wire_gate.decision_score` + goreli threshold) cagrilir,
elde yeniden kurulmaz ([[measurement-zaafiyetleri-kapatildi]]).

IKI METRIK AYRI ([[brep-axis-and-two-metrics]]):
  TESPIT    : lateral <= max(3mm, %6*kosegen), angle SERBEST, axial <= 40mm
  ROBOT     : lateral <= 2mm, angle <= 10 derece, ISARETLI direction
"""
import collections 
import glob 
import io 
import json 
import os 
import pickle 
import sys 

import numpy as np 

import d6_record 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1";os .environ ["WG_TOPO"]="1";os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))

KUME ="results/d6_sinav_kumesi.json"
MAKBUZ ="results/d6_temiz_olc.json"
ROBOT_YANAL =2.0 
ROBOT_ACI =10.0 


def main ():
    import protocol 
    protocol .tez_dogrula ()
    import wire_gate 
    from sina_cluster import match_greedy ,f1w 

    sv =json .load (io .open (KUME ,encoding ="utf-8"))
    PID =set (sv ["pidler"])
    print (f"TEMIZ SINAV: {sv ['n_parca']} part | {len (sv ['manufacturer'])} manufacturer | "
    f"GT {sv ['gt_toplam']} CP | muhur {sv ['sha16']}")
    print (f"  manufacturer: {sv ['manufacturer']}\n")

    import d6_record 
    kayit =d6_record .yukle (PID )
    print (f"turetme kaydi bulunan: {len (kayit )}/{len (PID )}")

    with open ("results/wire_gate.pkl","rb")as f :
        gate =pickle .load (f )
    print (f"gate: n_feat={gate ['n_feat']} donusum={gate .get ('donusum')}")

    T ,R =[],[]# (regime, tp, fp, fn)
    Tm =collections .defaultdict (list );Rm =collections .defaultdict (list )
    atlanan =collections .Counter ()
    for pid ,r in kayit .items ():
        G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
        rj ="cok"if r ["n"]>=8 else "dusuk"
        P =np .zeros ((0 ,3 ));D =np .zeros ((0 ,3 ))
        if r .get ("X")is not None :
        # DAGITILAN gate 58 sutunla egitildi and part-ici z-skor onu 116'ya removes.
        # (X + XR birlestirmek 94 column gives -> 188 != 116 and HER part sessizce
        #  elenir; first kosuda full da this became, F1 0.0018 output.)
            M =d6_record .x58 (r )
            if M is not None and M .shape [1 ]*2 ==gate ["n_feat"]:
                k =wire_gate .decision_mask (wire_gate .decision_score (gate ,M ))
                if k .any ():
                    P =np .asarray (r ["P"],float )[k ];D =np .asarray (r ["Pd"],float )[k ]
            else :
                atlanan ["genislik"]+=1 
        else :
            atlanan ["aday_yok"]+=1 
            # TESPIT: angle serbest, unsigned
        tp ,fp ,fn ,_ =match_greedy (P ,D ,G ,Gd ,r ["diag"],0.0 ,180.0 ,True )
        T .append ((rj ,tp ,fp ,fn ));Tm [r ["mfg"]].append ((rj ,tp ,fp ,fn ))
        # ROBOT: lateral 2mm SABIT (pct=False -- pct=True 'tol'u YOK SAYAR and first kosuda
        # full this hatayi yaptim: 0.2353 aslinda "tespit toleransi + angle<=10" idi),
        # angle 10 derece, ISARETLI. Resmi cagri s0s4_triyaj.py:103 with BIREBIR.
        tp2 ,fp2 ,fn2 ,_ =match_greedy (P ,D ,G ,Gd ,r ["diag"],ROBOT_YANAL ,ROBOT_ACI ,
        False ,signed =True )
        R .append ((rj ,tp2 ,fp2 ,fn2 ));Rm [r ["mfg"]].append ((rj ,tp2 ,fp2 ,fn2 ))

    if atlanan :
        print (f"ATLANAN: {dict (atlanan )}")
    tf =f1w (T );rf =f1w (R )
    print (f"\n{'':<8}{'TESPIT':>10}{'ROBOT':>10}")
    print (f"{'TOPLAM':<8}{tf :>10.4f}{rf :>10.4f}\n")
    print (f"{'manufacturer':<8}{'n':>5}{'TESPIT':>10}{'ROBOT':>10}")
    for m in sorted (Tm ,key =lambda x :-len (Tm [x ])):
        print (f"{m :<8}{len (Tm [m ]):>5}{f1w (Tm [m ]):>10.4f}{f1w (Rm [m ]):>10.4f}")
    yay =[f1w (Tm [m ])for m in Tm if len (Tm [m ])>=5 ]
    print (f"\nURETICI YAYILIMI (n>=5): en kotu {min (yay ):.4f} | en iyi {max (yay ):.4f}")

    with io .open (MAKBUZ ,"w",encoding ="utf-8")as f :
        json .dump ({"cluster":KUME ,"muhur":sv ["sha16"],"n":len (kayit ),
        "tespit":tf ,"robot":rf ,
        "uretici_tespit":{m :f1w (v )for m ,v in Tm .items ()},
        "uretici_robot":{m :f1w (v )for m ,v in Rm .items ()},
        "not":("TEMIZ cluster. Eski d5_4 kumesindeki sayilarla KIYASLANAMAZ -- "
        "o cluster manufacturer duzeyinde kirliydi.")},f ,indent =1 ,
        ensure_ascii =False )
    print (f"receipt -> {MAKBUZ }")


if __name__ =="__main__":
    main ()
