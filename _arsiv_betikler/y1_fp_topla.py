# -*- coding: utf-8 -*-
"""Y1: YANLIS POZITIFLERI TOPLA -- GT TAMLIK DENETIMI'nin birinci adimi.

PLANDAN GELIYOR (ROAD_TO_085.md, "W1. GT-TAMLIK AUDIT"): "WEI FP'lerinin ne kadari GERCEK
listelenmemis opening? (PXC'de %96-98'di)... NOTE: sisme tuzagi -> SADECE dogrulanirsa
(insan/gorsel denetim). OLCUM duzeltmesi, model not."

MODELE HIC DOKUNULMAZ. Ag, remesh, candidate uretimi, gate, threshold, correction zinciri AYNEN kalir.
Yapilan single sey: urunun "wrong" sayilan 313 tahminini toplayip INSANA sormak.

CIKTI: results/fp_denetim.json
    part basina each FP for: point, direction, confidence, most yakin GT'ye uzaklik, regime, manufacturer
    + TABAKALI RASTGELE EXAMPLE (manufacturer x regime), so prediction YANSIZ and GA'li becomes.

WHY TABAKALI RASTGELE: "most emin FP'leri sec" demek sismedir. Rastgele ornek + confidence araligi
single durust path. Ornek buyuklugu 80 -> %50 civari a ratio for GA yarı genisligi ~%11.
"""
import io 
import json 
import os 
import sys 

import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1"
os .environ ["WG_TOPO"]="1"
os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))

ORNEK =80 
OUT ="results/fp_denetim.json"


def main ():
    import gate_bench as T 
    import measure_set 
    import wire_gate 
    from sklearn .ensemble import RandomForestClassifier 

    D =T .yukle ()
    measure_set .rapor_bas (D ["rap"])
    cfg =D ["cfg"]
    X ,y ,pid ,keep =D ["X"],D ["y"],D ["pid"],D ["keep"]
    Z =np .zeros ((len (X ),X .shape [1 ]*2 ))
    for u in np .unique (pid ):
        i =np .where (pid ==u )[0 ]
        Z [i ]=wire_gate .within_part (X [i ],D ["donusum"])
    gate ={"clf":RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
    random_state =0 ).fit (Z [keep ],y [keep ]),
    "n_feat":Z .shape [1 ],"donusum":D ["donusum"]}

    FP ,TP_N ,GT_N =[],0 ,0 
    for r in D ["DER"]:
        if r ["X"]is None or r .get ("XR")is None :
            continue 
        Xr =np .hstack ([r ["X"],r ["XR"]])
        k =wire_gate .decision_mask (wire_gate .decision_score (gate ,Xr ))
        if not k .any ():
            continue 
        idx =np .where (k )[0 ]
        P =r ["P"][k ].copy ();Pd =r ["Pd"][k ].copy ()
        if cfg .get ("robot_pose_head"):
            c =[{"point":P [i ],"direction":Pd [i ]}for i in range (len (P ))]
            c =wire_gate .pose_correct (Xr [k ],c )
            if cfg .get ("robot_aci_secici"):
                c =wire_gate .angle_correct (Xr [k ],c )
            if cfg .get ("robot_uye_secici")and r .get ("UYE"):
                c =wire_gate .pick_member_direction (Xr [k ],c ,r ["UYE"])
            P =np .array ([x ["point"]for x in c ],float )
            Pd =np .array ([x ["direction"]for x in c ],float )
        G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
        GT_N +=len (G )
        if not len (G ):
            for j in range (len (P )):
                FP .append ({"pid":r ["pid"],"mfg":r ["mfg"],"geo":r ["geo"],
                "point":P [j ].tolist (),"direction":Pd [j ].tolist (),
                "aday_i":int (idx [j ]),"gt_uzaklik":None ,
                "regime":"very"if r ["n"]>=8 else "low",
                "diag":float (r ["diag"]),"n_gt":0 })
            continue 
            # URUNLE AYNI eslestirme (lateral distance, +-40mm axial pencere, greedy)
        diff =P [:,None ,:]-G [None ,:,:]
        al =(diff *Gd [None ,:,:]).sum (-1 )
        pe =np .linalg .norm (diff -al [...,None ]*Gd [None ,:,:],axis =-1 )
        pe =np .where (np .abs (al )<=40.0 ,pe ,np .inf )
        tol =max (3.0 ,0.06 *float (r ["diag"]))
        up ,ug =set (),set ()
        for d_ ,a_ ,b_ in sorted ((pe [a ,b ],a ,b )for a in range (len (P ))
        for b in range (len (G ))):
            if not np .isfinite (d_ )or d_ >tol or a_ in up or b_ in ug :
                continue 
            up .add (a_ );ug .add (b_ )
        TP_N +=len (up )
        for j in range (len (P )):
            if j in up :
                continue 
            dmin =float (np .min (np .linalg .norm (G -P [j ],axis =1 )))
            FP .append ({"pid":r ["pid"],"mfg":r ["mfg"],"geo":r ["geo"],
            "point":P [j ].tolist (),"direction":Pd [j ].tolist (),
            "aday_i":int (idx [j ]),"gt_uzaklik":dmin ,
            "regime":"very"if r ["n"]>=8 else "low",
            "diag":float (r ["diag"]),"n_gt":int (len (G ))})
    print (f"\nTP {TP_N } | FP {len (FP )} | GT {GT_N }")
    print (f"  precision {TP_N /max (TP_N +len (FP ),1 ):.4f}  recall {TP_N /max (GT_N ,1 ):.4f}")

    # --- TABAKALI RASTGELE EXAMPLE (manufacturer x regime)
    rng =np .random .default_rng (0 )
    tab ={}
    for i ,f in enumerate (FP ):
        tab .setdefault ((f ["mfg"],f ["regime"]),[]).append (i )
    sec =[]
    total_ =len (FP )
    for k_ ,v in sorted (tab .items ()):
        n =max (1 ,int (round (ORNEK *len (v )/total_ )))
        n =min (n ,len (v ))
        sec +=list (rng .choice (v ,size =n ,replace =False ))
        print (f"  tabaka {k_ }: {len (v )} FP -> {n } ornek")
    sec =sorted (int (x )for x in sec )
    print (f"ORNEK: {len (sec )} FP / {len (FP )} ({len (sec )/len (FP ):.1%})")

    with io .open (OUT ,"w",encoding ="utf-8")as f :
        json .dump ({"tp":TP_N ,"fp":len (FP ),"gt":GT_N ,
        "precision":TP_N /max (TP_N +len (FP ),1 ),
        "recall":TP_N /max (GT_N ,1 ),
        "all of them":FP ,"ornek_idx":sec ,
        "not":("Ornek TABAKALI RASTGELE (manufacturer x regime). 'En emin FP'leri secmek "
        "sismedir; yansiz prediction + confidence araligi single durust path.")},f ,indent =1 )
    print (f"receipt -> {OUT }")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
