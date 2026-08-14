# -*- coding: utf-8 -*-
"""T2: manufacturer-disi cokus SIRALAMA hatasi mi, ESIK (kalibrasyon) hatasi mi?

T1 OLCTU: 18 sutunlu gate manufacturer-disi bolmede F1 0.7422 -> 0.6399 / 0.2799 (mean -0.282).
Iki very different hastalik may be and tedavileri zit:
  SIRALAMA bozuksa  -> model that ureticide correct/yanlisi ayirt EDEMIYOR; feature/data isi.
  ESIK bozuksa      -> ranking iyi but 0.40 wrong places; KALIBRASYON isi (ucuz cozum).

Ayrim: AUC siralamayi olcer (esikten bagimsiz). Sabit esikteki F1 with EN IYI esikteki F1
arasindaki difference, only kalibrasyonun bedelidir.
"""
import json 
import numpy as np 
from sklearn .ensemble import RandomForestClassifier 
from sklearn .model_selection import GroupKFold 
from t1_uretici_disi import auc_mw ,f1_at 


def main ():
    d =np .load ("results/gate_regrow_data_fiz.npz",allow_pickle =True )
    X =d ["X"][:,:18 ];y =d ["y"].astype (bool )
    mfg =np .array ([str (x )for x in d ["mfg"]])
    pids =np .array ([str (x )for x in d ["pids"]])
    gk =json .load (open ("results/_strict_geometry_keys.json"))
    grp =np .array ([gk .get (p ,"yok:"+p )for p in pids ])
    THR =float (json .load (open ("cp_config.json",encoding ="utf-8"))["robot_wire_gate_threshold"])
    grid =np .arange (0.05 ,0.96 ,0.05 )

    def rapor (ad ,s ,yy ):
        f_sabit ,p_ ,r_ =f1_at (s ,yy ,THR )
        best =max (((f1_at (s ,yy ,t )[0 ],t )for t in grid ))
        a =auc_mw (s ,yy )
        print (f"{ad :<26}{a :>8.4f}{f_sabit :>9.4f}{best [0 ]:>10.4f}{best [1 ]:>8.2f}"
        f"{best [0 ]-f_sabit :>10.4f}{float ((s >=THR ).mean ()):>10.3f}{float (yy .mean ()):>9.3f}")
        return {"auc":a ,"f1_sabit":f_sabit ,"f1_enIyi":best [0 ],"enIyi_esik":best [1 ],
        "kalibrasyon_bedeli":best [0 ]-f_sabit }

    print (f"{'split':<26}{'AUC':>8}{'F1@0.40':>9}{'F1@enIyi':>10}{'threshold':>8}"
    f"{'kalib.bedeli':>10}{'pozitif%':>10}{'gercek%':>9}")
    out ={}
    o =np .zeros (len (y ))
    for tr ,te in GroupKFold (n_splits =5 ).split (X ,y ,grp ):
        o [te ]=RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
        random_state =0 ).fit (X [tr ],y [tr ]).predict_proba (X [te ])[:,1 ]
    out ["geometri-disi"]=rapor ("geometri-disi (referans)",o ,y )
    for u in sorted (set (mfg )):
        te =mfg ==u 
        s =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
        random_state =0 ).fit (X [~te ],y [~te ]).predict_proba (X [te ])[:,1 ]
        out [f"manufacturer{u }"]=rapor (f"  -> manufacturer {u } disarida",s ,y [te ])

    print ("\nAYRISTIRMA:")
    ref =out ["geometri-disi"]
    for u in sorted (set (mfg )):
        k =out [f"manufacturer{u }"]
        toplam =k ["f1_sabit"]-ref ["f1_sabit"]
        kalib =k ["kalibrasyon_bedeli"]
        siralama =toplam +kalib # threshold duzeltilse bile kalan loss
        print (f"  manufacturer {u }: toplam {toplam :+.4f} = KALIBRASYON {-kalib :+.4f} "
        f"+ SIRALAMA {siralama :+.4f}   (AUC {ref ['auc']:.3f} -> {k ['auc']:.3f})")
    json .dump (out ,open ("results/t2_esik_siralama.json","w"),indent =1 )
    print ("\nmakbuz -> results/t2_esik_siralama.json")


if __name__ =="__main__":
    main ()
