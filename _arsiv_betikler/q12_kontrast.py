# -*- coding: utf-8 -*-
"""Q12: renkten BAGIMSIZ kontrast olcutu, metal-tonu olcutunden iyi mi?

DUZELTILEN: `METAL_RGB` single a gumus tonuna sabitti; measured (40 part) ki parcalarin
%22'sinde renk VAR but that tona uyan face YOK. "Gri = metal" demek de wrong olurdu (gri PLASTIK
body klemenslerde very yaygin). Renk-bagimsiz criterion: parcanin BASKIN renginden different yuzler.
Ayrica kati-seviyesi renk kurtarma eklendi (dogrulanan part 23->34/40).

OLCULEN: kapsama 14 -> 19 part (+%36). AMA KAPSAMA AYIRT EDICILIK DEGILDIR -- this betik
ayirt ediciligi olcer. 100 dakikalik yeniden uretime girmeden ONCE.

KILL: kontrast olcutu metal-tonunu |AUC-0.5|'te gecmezse yeniden uretim ACILMAZ.
"""
import os ,sys ,json ,pickle 
import numpy as np 
os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
from q7_renk_degeri import auc_mw 


def main ():
    import cp_openings ,robot_cp 
    import step_face_colors as SC 
    from connector_constants import CABLE_ENTRY as CE ,CONTACT as CT 
    cfg =json .load (open ("cp_config.json",encoding ="utf-8"));pp =cfg ["prediction_postproc"]
    MINV =int (pp ["min_vertices"]);VC =float (pp ["vertex_confidence_mask"]);CL =float (pp ["cluster_mm"])
    n =int (sys .argv [1 ])if len (sys .argv )>1 else 70 
    parts =[]
    for k in ("dev","val"):
        cf =f"results/_probs_{k }.pkl"
        if not os .path .exists (cf )and k =="dev":
            cf ="results/_h_probs.pkl"
        for r in pickle .load (open (cf ,"rb")):
            r .setdefault ("pid",os .path .basename (r ["stp"]).split ("_")[1 ]);parts .append (r )
    rng =np .random .RandomState (0 )
    parts =[parts [i ]for i in rng .choice (len (parts ),min (n ,len (parts )),replace =False )]
    X ,Y =[],[]
    kap_m =kap_k =0 
    for r in parts :
        try :
            faces ,ok =SC .read_all_by_order (r ["stp"])
        except Exception :
            continue 
        if not ok :
            continue 
        mt =np .array ([f ["centroid"]for f in faces if f ["is_metal"]],float )
        K ,baskin =SC .kontrast_yuzler (faces )
        kap_m +=len (mt )>0 ;kap_k +=len (K )>0 
        V =np .ascontiguousarray (r ["V"],np .float64 );F =np .ascontiguousarray (r ["F"],np .int64 )
        plist =[np .asarray (pb ,np .float64 )for pb in r ["pbs"]]
        der =[cp_openings .connection_points (V ,F ,pb .argmax (-1 ),min_v =MINV ,classes =(CE ,CT ),
        dedupe_mm =10.0 ,probs =pb ,vertex_conf =VC ,ct_depth_min_mm =1.0 ,cluster_mm =CL ,
        step_path =r ["stp"])for pb in plist ]
        cps =robot_cp ._vote2 (der ,min_votes =1 )
        if not cps :
            continue 
        P =np .array ([c ["point"]for c in cps ],float )
        G ,Gd =r ["G"],r ["Gd"]
        lab =np .zeros (len (P ),bool )
        if len (G ):
            diff =P [:,None ,:]-G [None ,:,:]
            a_ =(diff *Gd [None ,:,:]).sum (-1 )
            pe =np .linalg .norm (diff -a_ [...,None ]*Gd [None ,:,:],axis =-1 )
            pe =np .where (np .abs (a_ )<=40.0 ,pe ,np .inf )
            t0 =max (3.0 ,0.06 *r ["diag"]);us ,ug =set (),set ()
            for dv ,x ,y in sorted ((pe [i ,j ],i ,j )for i in range (len (P ))for j in range (len (G ))):
                if dv >t0 or x in us or y in ug :
                    continue 
                us .add (x );ug .add (y );lab [x ]=True 
        for k2 in range (len (P )):
            dm =float (np .min (np .linalg .norm (mt -P [k2 ],axis =1 )))if len (mt )else 99.0 
            dk =float (np .min (np .linalg .norm (K -P [k2 ],axis =1 )))if len (K )else 99.0 
            X .append ([dm ,dk ]);Y .append (bool (lab [k2 ]))
    X =np .array (X ,float );Y =np .array (Y ,bool )
    print (f"{len (Y )} candidate | TP {int (Y .sum ())} | kapsama: metal-tonu {kap_m }, kontrast {kap_k }")
    rng2 =np .random .RandomState (0 )
    print (f"\n{'criterion':<28}{'AUC':>8}{'null p95':>10}{'TP ort':>10}{'FP ort':>10}{'|AUC-.5|':>10}")
    res ={}
    for i ,nm in enumerate (["metal-tonu mesafesi (eski)","KONTRAST mesafesi (yeni)"]):
        a =auc_mw (X [:,i ],Y )
        nl =np .array ([auc_mw (X [:,i ],rng2 .permutation (Y ))for _ in range (300 )])
        p95 =float (np .percentile (np .abs (nl -0.5 ),95 )+0.5 )
        print (f"{nm :<28}{a :>8.3f}{p95 :>10.3f}{X [Y ,i ].mean ():>10.1f}{X [~Y ,i ].mean ():>10.1f}"
        f"{abs (a -0.5 ):>10.4f}")
        res [nm ]={"auc":float (a ),"null_p95":p95 ,"ayirt":float (abs (a -0.5 ))}
    d =res ["KONTRAST mesafesi (yeni)"]["ayirt"]-res ["metal-tonu mesafesi (eski)"]["ayirt"]
    print (f"\nayirt edicilik farki: {d :+.4f}")
    print (f"KILL: kontrast eskiyi gecmezse yeniden uretim ACILMAZ -> {'AC'if d >0 else 'ACMA'}")
    json .dump (res |{"fark":float (d ),"kapsama_metal":kap_m ,"kapsama_kontrast":kap_k },
    open ("results/q12_kontrast.json","w"),indent =1 )
    print ("receipt -> results/q12_kontrast.json")


if __name__ =="__main__":
    main ()
