# -*- coding: utf-8 -*-
"""T4: kalibrasyon kurallari TANIDIK veride ne KAYBETTIRIYOR?

T3 OLCTU: part-ici kurallar manufacturer-disi most kotu durumu 0.2799 -> 0.4553 does (+0.175).
AMA that kurallar very more GEVSEK (adaylarin %38-53'une pozitif; sabit threshold %10-36).
Gevseklik tanidik dagilimda KESINLIGI dusurur. Bir rule however
   (a) manufacturer-disi belirgin kazandiriyorsa VE
   (b) tanidik veride loss kucukse
alinabilir. Bu betik (b)'yi olcer -- same geometri-disi protocol, same referans.

KILL: tanidik veride loss 0.02'den buyukse rule ALINMAZ (manseti bozmaya degmez).
"""
import json 
import numpy as np 
from sklearn .ensemble import RandomForestClassifier 
from sklearn .model_selection import GroupKFold 


def main ():
    d =np .load ("results/gate_regrow_data_fiz.npz",allow_pickle =True )
    X =d ["X"][:,:18 ];y =d ["y"].astype (bool )
    pids =np .array ([str (x )for x in d ["pids"]])
    gk =json .load (open ("results/_strict_geometry_keys.json"))
    grp =np .array ([gk .get (p ,"yok:"+p )for p in pids ])
    THR =float (json .load (open ("cp_config.json",encoding ="utf-8"))["robot_wire_gate_threshold"])

    o =np .zeros (len (y ))
    for tr ,te in GroupKFold (n_splits =5 ).split (X ,y ,grp ):
        o [te ]=RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
        random_state =0 ).fit (X [tr ],y [tr ]).predict_proba (X [te ])[:,1 ]

    def part (fn ):
        m =np .zeros (len (o ),bool )
        for u in np .unique (pids ):
            i =pids ==u 
            m [i ]=fn (o [i ])
        return m 

    kur ={
    "sabit (mevcut)":o >=THR ,
    "parca_orani 0.5":part (lambda v :v >=0.5 *max (v .max (),1e-9 )),
    "parca_orani 0.6":part (lambda v :v >=0.6 *max (v .max (),1e-9 )),
    "parca_z >= 0.0":part (lambda v :(v -v .mean ())/(v .std ()+1e-9 )>=0.0 ),
    "VE(sabit, parca_orani 0.5)":(o >=THR )&part (lambda v :v >=0.5 *max (v .max (),1e-9 )),
    "VEYA(sabit, parca_orani 0.5)":(o >=THR )|part (lambda v :v >=0.5 *max (v .max (),1e-9 )),
    }
    print (f"{'kural':<30}{'F1':>8}{'kesin':>8}{'recall':>8}{'pozitif%':>10}{'fark':>9}")
    out ={}
    baseline =None 
    for ad ,m in kur .items ():
        tp =int ((y &m ).sum ());fp =int ((~y &m ).sum ());fn_ =int ((y &~m ).sum ())
        pr =tp /max (tp +fp ,1 );rc =tp /max (tp +fn_ ,1 )
        f =2 *pr *rc /max (pr +rc ,1e-9 )
        if baseline is None :
            baseline =f 
        print (f"{ad :<30}{f :>8.4f}{pr :>8.3f}{rc :>8.3f}{float (m .mean ()):>10.3f}{f -baseline :>+9.4f}")
        out [ad ]={"f1":f ,"precision":pr ,"recall":rc ,"fark":f -baseline }
    print ("\nKILL: tanidik veride loss > 0.02 ise kural ALINMAZ.")
    json .dump (out ,open ("results/t4_calibration_bedeli.json","w"),indent =1 )
    print ("receipt -> results/t4_calibration_bedeli.json")


if __name__ =="__main__":
    main ()
