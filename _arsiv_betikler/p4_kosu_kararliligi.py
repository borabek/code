# -*- coding: utf-8 -*-
"""P4: KOSUDAN KOSUYA KARARLILIK -- "0.7584" ne up to saglam a number?

WHY: [[robot-nondeterminism]] kaydi remesh'in SURECLER ARASI degistigini says
(pymeshlab). Yani same part, same model, same kod -- different a Python surecinde different
a mesh, therefore different candidates. Bu bugune up to never NOT MEASURED: mansetin own
kosu-varyansini bilmeden "+0.008 kazandi" demek anlamsizdir.

YONTEM: N parcayi AYNI surecte two times turet (process-ici kararlilik) and also this betigi
IKI KEZ calistirip makbuzlari karsilastir (process-arasi). Burada process-ICI olculur;
`--kind 2` with ikinci kosu ayri a surecte is done and difference raporlanir.

Olculen: candidate count farki, point eslesme orani, and part basina tespit F1 farki.
"""
import io 
import json 
import os 
import sys 

import numpy as np 
import torch 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
N_PARCA =30 
OUT ="results/p4_kararlilik_tur{}.json"


def main ():
    import diffusionnet as D_ 
    import measure_set 
    import robot_cp as RC 
    import thesis_remesh 
    from big_arbiter import eligible 
    from infer_step_cp import load_any ,step_to_mesh 
    from gece_kilit import guard 

    tur =int (sys .argv [sys .argv .index ("--tur")+1 ])if "--tur"in sys .argv else 1 
    guard ("p4")
    with io .open ("cp_config.json",encoding ="utf-8")as f :
        cfg =json .load (f )
    stp ={p :s for m ,p ,jf ,s in eligible ()}
    DER ,_ =measure_set .cluster ("results/_der_tam.pkl")
    rng =np .random .default_rng (0 )
    sec =list (rng .permutation (len (DER ))[:N_PARCA ])
    cks =cfg ["current_product"].get ("checkpoints")or cfg ["robot_vote2_checkpoints"]
    dev ="cuda"if torch .cuda .is_available ()else "cpu"
    models =[load_any (c ,dev =dev )[:2 ]for c in cks ]

    KAYIT ={}
    for k ,i in enumerate (sec ,1 ):
        r =DER [i ]
        try :
            Vr ,Fr =step_to_mesh (stp [r ["pid"]])
            V ,F =thesis_remesh .remesh_uniform (Vr ,Fr ,target =6000 )
            V =np .ascontiguousarray (V ,np .float64 );F =np .ascontiguousarray (F ,np .int64 )
            pbs =[]
            for model ,meta in models :
                _ ,pb =D_ .predict (model ,meta ,V ,F ,device =dev ,
                op_cache_dir =f"results/step_infer/ops_k{int (meta .get ('k_eig',64 ))}",
                return_probs =True )
                pbs .append (np .asarray (pb ,float ))
            cps ,_ ,_ ,_ =RC .derive_candidates (V ,F ,pbs ,stp [r ["pid"]],cfg =cfg )
            KAYIT [r ["pid"]]={"n_vert":int (len (V )),"n_aday":len (cps ),
            "noktalar":[list (map (float ,c ["point"]))for c in cps ]}
        except Exception as e :
            KAYIT [r ["pid"]]={"error":f"{type (e ).__name__ }"}
        if k %10 ==0 :
            print (f"  {k }/{len (sec )}",flush =True )
    with io .open (OUT .format (tur ),"w",encoding ="utf-8")as f :
        json .dump (KAYIT ,f )
    print (f"tur {tur } -> {OUT .format (tur )}")

    if tur ==2 and os .path .exists (OUT .format (1 )):
        with io .open (OUT .format (1 ),encoding ="utf-8")as f :
            A =json .load (f )
        B =KAYIT 
        dv ,da ,esl =[],[],[]
        for pid in A :
            a ,b =A [pid ],B .get (pid ,{})
            if "error"in a or "error"in b :
                continue 
            dv .append (abs (a ["n_vert"]-b ["n_vert"]))
            da .append (abs (a ["n_aday"]-b ["n_aday"]))
            PA =np .array (a ["noktalar"],float );PB =np .array (b ["noktalar"],float )
            if len (PA )and len (PB ):
                d =np .linalg .norm (PA [:,None ,:]-PB [None ,:,:],axis =-1 )
                esl .append (float ((d .min (1 )<=1.0 ).mean ()))
        print (f"\nSUREC-ARASI KARARLILIK ({len (dv )} part)")
        print (f"  tepe sayisi farki : medyan {np .median (dv ):.0f}  max {max (dv )if dv else 0 }")
        print (f"  candidate sayisi farki : medyan {np .median (da ):.0f}  max {max (da )if da else 0 }"
        f"  ({float (np .mean (np .array (da )>0 )):.0%} parcada DEGISTI)")
        print (f"  nokta eslesmesi   : medyan {np .median (esl ):.1%}  "
        f"en kotu {min (esl )if esl else 0 :.1%}")
        with io .open ("results/p4_kararlilik.json","w",encoding ="utf-8")as f :
            json .dump ({"n":len (dv ),"vert_fark_medyan":float (np .median (dv )),
            "aday_fark_medyan":float (np .median (da )),
            "aday_degisen_pay":float (np .mean (np .array (da )>0 )),
            "eslesme_medyan":float (np .median (esl )),
            "eslesme_en_kotu":float (min (esl ))if esl else None },f ,indent =1 )
        print ("receipt -> results/p4_kararlilik.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
