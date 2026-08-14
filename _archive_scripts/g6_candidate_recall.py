# -*- coding: utf-8 -*-
"""G6-a: ADAY_YOK KURTARILDI MI? -- gate'e dokunmadan, DOGRUDAN measurement.

SORU: G1 unseen-manufacturer sinavinda FN'lerin **%68.1'inin ADAY_YOK** oldugunu showed
(563/827) -- i.e. network that bolgede HIC candidate uretmemis. G5 (oto-label + seg yeniden training)
full this kovayi hedefliyordu. Duzeldi mi?

OLCU: **ADAY RECALL** = toleransta at least a candidate found GT orani. Bu, `1 - ADAY_YOK payi`
demektir and gate'ten BAGIMSIZDIR. Gate'i isin icine katmak two degisikligi karistirirdi.

ADIL KARSILASTIRMA: urun 4 checkpoint'lik TOPLULUK kullaniyor; new modelde single seed present.
Toplulukla single modeli kiyaslamak new modeli haksiz yere kotu gosterirdi. Bu yuzden
ESKI TEK MODEL (same k_eig 96) with YENI TEK MODEL karsilastirilir. Topluluk etkisi
also olculur (three seed tamamlaninca).

TEZ DEGISMEZ: same ~6000 uniform remesh, same `v_o` candidate ureticisi, same 5 sinif.
Degisen TEK sey agin egitildigi ETIKET SAYISI (203 -> 1826).
"""
import argparse 
import collections 
import io 
import json 
import os 
import pickle 
import sys 
import time 

import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1";os .environ ["WG_TOPO"]="1";os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))

SINAV ="results/_der_sinav_yeni.pkl"
MAKBUZ ="results/g6_candidate_recall.json"
YENI ="results/seg_g5/g5_s0.pt"


def main ():
    ap =argparse .ArgumentParser ()
    ap .add_argument ("--boundary",type =int ,default =120 )
    a =ap .parse_args ()
    import protocol 
    protocol .tez_dogrula ()
    import torch 
    import cad_eval 
    import diffusionnet as D_ 
    import robot_cp as RC 
    import thesis_remesh 
    from big_arbiter import eligible 
    from infer_step_cp import load_any ,step_to_mesh 

    with open (SINAV ,"rb")as f :
        S =pickle .load (f )
    hedef ={r ["pid"]:r for r in S }
    path ={p :s for m ,p ,jf ,s in eligible ()if p in hedef }
    jf_ ={p :jf for m ,p ,jf ,s in eligible ()if p in hedef }
    with io .open ("cp_config.json",encoding ="utf-8")as f :
        cfg =json .load (f )
    ESKI =[c for c in (cfg ["current_product"].get ("checkpoints")or [])
    if "keig96"in c ][:1 ]# ESKI TEK MODEL (same k_eig)
    print (f"ESKI tek model: {ESKI }\nYENI tek model: [{YENI }]\n")

    dev ="cuda"if torch .cuda .is_available ()else "cpu"
    MOD ={"ESKI (203 part training)":[load_any (c ,dev =dev )[:2 ]for c in ESKI ],
    "YENI (1826 part training)":[load_any (YENI ,dev =dev )[:2 ]]}

    pid_l =sorted (hedef )[:a .bound_ ]
    R ={k :collections .defaultdict (lambda :[0 ,0 ])for k in MOD }# mfg -> [kapsanan, GT]
    t0 =time .time ();error =0 
    for i ,p in enumerate (pid_l ,1 ):
        if i %20 ==0 :
            print (f"  {i }/{len (pid_l )}  {(time .time ()-t0 )/i :.1f}s/part",flush =True )
        try :
            r =hedef [p ]
            Vr ,Fr =step_to_mesh (path [p ])
            V ,F =thesis_remesh .remesh_uniform (Vr ,Fr ,target =6000 )
            V =np .ascontiguousarray (V ,np .float64 );F =np .ascontiguousarray (F ,np .int64 )
            G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
            tol =max (3.0 ,0.06 *float (np .linalg .norm (V .max (0 )-V .min (0 ))))
            for ad ,models in MOD .items ():
                pbs =[]
                for model ,meta in models :
                    _ ,pb =D_ .predict (model ,meta ,V ,F ,device =dev ,
                    op_cache_dir =None ,return_probs =True )
                    pbs .append (np .asarray (pb ,float ))
                cps ,probs ,_ ,_u =RC .derive_candidates (V ,F ,pbs ,path [p ],cfg =cfg )
                P =np .array ([c ["point"]for c in cps ],float )if cps else np .zeros ((0 ,3 ))
                if len (P )and len (G ):
                    d =P [:,None ,:]-G [None ,:,:]
                    al =(d *Gd [None ,:,:]).sum (-1 )
                    pe =np .linalg .norm (d -al [...,None ]*Gd [None ,:,:],axis =-1 )
                    pe =np .where (np .abs (al )<=40.0 ,pe ,np .inf )
                    kap =int ((pe .min (0 )<=tol ).sum ())
                else :
                    kap =0 
                R [ad ][r ["mfg"]][0 ]+=kap ;R [ad ][r ["mfg"]][1 ]+=len (G )
        except Exception as e :
            error +=1 
            if error <=3 :
                print (f"    {p }: {type (e ).__name__ }: {str (e )[:60 ]}",flush =True )

    print (f"\n{'manufacturer':<8}"+"".join (f"{k [:20 ]:>24}"for k in MOD ))
    urs =sorted ({m for k in R for m in R [k ]})
    for u in urs :
        sat ="".join (f"{100 *R [k ][u ][0 ]/max (R [k ][u ][1 ],1 ):>22.1f}%"for k in MOD )
        print (f"{u :<8}{sat }")
    print (f"\n{'TOPLAM':<8}",end ="")
    top ={}
    for k in MOD :
        kap =sum (v [0 ]for v in R [k ].values ());gt =sum (v [1 ]for v in R [k ].values ())
        top [k ]=kap /max (gt ,1 )
        print (f"{100 *top [k ]:>22.1f}%",end ="")
    ks =list (MOD )
    d =top [ks [1 ]]-top [ks [0 ]]
    print (f"\n\nADAY RECALL FARKI: {d :+.4f}  ({100 *d :+.1f} puan)")
    print (f"  ADAY_YOK payi: %{100 *(1 -top [ks [0 ]]):.1f} -> %{100 *(1 -top [ks [1 ]]):.1f}")
    print (f"  {'KURTARILDI'if d >0.02 else ('DEGISMEDI'if abs (d )<=0.02 else 'KOTULESTI')}")
    with io .open (MAKBUZ ,"w",encoding ="utf-8")as f :
        json .dump ({"n_parca":len (pid_l ),"total":top ,
        "manufacturer":{k :{m :v for m ,v in R [k ].items ()}for k in R },
        "difference":d ,"error":error },f ,indent =1 ,ensure_ascii =False )
    print (f"receipt -> {MAKBUZ }")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
