# -*- coding: utf-8 -*-
"""Q11: `metal_mesafe` TUM metal yuzlerle mi, otherwise only metal SILINDIRLERLE mi olculmeli?

RATIONALE: tel kelepcesi silindirik a surface DEGIL, duz metal a yuzeydir. Mevcut EK blogu
metal mesafesini only metal SILINDIRLERDEN hesapliyor (q8'de olculen AUC 0.639 da oyleydi).
Tum yuzlere gecince metal face count orneklemde 8 -> 38 cikiyor (most duzlem).

KILL: tum-face surumu silindir-surumunu AUC'de gecmezse degisiklik YAPILMAZ (uretim yeniden
baslatilmaz). Gecerse uretim yeniden baslatilir -- ~20dk hesap COPE gider, that yuzden before this.
"""
import os ,sys ,json ,pickle 
import numpy as np 
os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
from q7_colour_degeri import auc_mw 


def main ():
    import cp_openings ,robot_cp 
    import step_face_colors as SC 
    from connector_constants import CABLE_ENTRY as CE ,CONTACT as CT 
    cfg =json .load (open ("cp_config.json",encoding ="utf-8"));pp =cfg ["prediction_postproc"]
    MINV =int (pp ["min_vertices"]);VC =float (pp ["vertex_confidence_mask"]);CL =float (pp ["cluster_mm"])
    n =int (sys .argv [1 ])if len (sys .argv )>1 else 60 
    parts =[]
    for k in ("dev","val"):
        cf =f"results/_probs_{k }.pkl"
        if not os .path .exists (cf )and k =="dev":
            cf ="results/_h_probs.pkl"
        for r in pickle .load (open (cf ,"rb")):
            r .setdefault ("pid",os .path .basename (r ["stp"]).split ("_")[1 ]);parts .append (r )
    rng =np .random .RandomState (0 )
    parts =[parts [i ]for i in rng .choice (len (parts ),min (n ,len (parts )),replace =False )]
    Xs ,Y =[],[]
    kab_c =kab_t =0 
    for r in parts :
        try :
            cy ,okc =SC .read_by_order (r ["stp"])
            al ,okt =SC .read_all_by_order (r ["stp"])
        except Exception :
            continue 
        mc =np .array ([f ["com"]for f in cy if okc and f ["is_metal"]],float )
        mt =np .array ([f ["centroid"]for f in al if okt and f ["is_metal"]],float )
        kab_c +=int (okc and len (mc )>0 );kab_t +=int (okt and len (mt )>0 )
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
        for k in range (len (P )):
            dc =float (np .min (np .linalg .norm (mc -P [k ],axis =1 )))if len (mc )else 99.0 
            dt =float (np .min (np .linalg .norm (mt -P [k ],axis =1 )))if len (mt )else 99.0 
            Xs .append ([dc ,dt ]);Y .append (bool (lab [k ]))
    X =np .array (Xs ,float );Y =np .array (Y ,bool )
    print (f"{len (Y )} candidate | TP {int (Y .sum ())} | renk cozulen part: silindir-yol {kab_c }, "
    f"tum-yuz yol {kab_t } / {len (parts )}")
    rng2 =np .random .RandomState (0 )
    print (f"\n{'version':<26}{'AUC':>8}{'null p95':>10}{'TP ort':>10}{'FP ort':>10}")
    res ={}
    for i ,nm in enumerate (["metal SILINDIR (mevcut)","TUM metal yuzler (new)"]):
        a =auc_mw (X [:,i ],Y )
        nl =np .array ([auc_mw (X [:,i ],rng2 .permutation (Y ))for _ in range (300 )])
        p95 =float (np .percentile (np .abs (nl -0.5 ),95 )+0.5 )
        print (f"{nm :<26}{a :>8.3f}{p95 :>10.3f}{X [Y ,i ].mean ():>10.1f}{X [~Y ,i ].mean ():>10.1f}")
        res [nm ]={"auc":float (a ),"null_p95":p95 }
    d =abs (res ["TUM metal yuzler (new)"]["auc"]-0.5 )-abs (res ["metal SILINDIR (mevcut)"]["auc"]-0.5 )
    print (f"\nayirt edicilik farki (|AUC-0.5|): {d :+.4f}")
    print (f"KILL: yeni version eskiyi gecmezse DEGISIKLIK YOK -> {'DEGISTIR'if d >0 else 'BIRAK'}")
    json .dump (res |{"difference":float (d ),"kabul_silindir":kab_c ,"kabul_tum":kab_t ,
    "n_parca":len (parts )},
    open ("results/q11_metal_duzlem.json","w"),indent =1 )
    print ("receipt -> results/q11_metal_duzlem.json")


if __name__ =="__main__":
    main ()
