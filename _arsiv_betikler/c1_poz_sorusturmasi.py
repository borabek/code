# -*- coding: utf-8 -*-
"""C1: FIZIKSEL KUSURLARIN SUCLUSU KIM? -- zincirin asamalarina according to bayrak sayimi.

A4 olctu: uretilen 1186 CP'nin %18.7'sinde fiziksel kusur present (body ici / onu closed /
duvara yapisik) and bunlar FP'de 2-3 fold more sik. AMA kusurun NEREDE olustugu bilinmiyor.

URUN ZINCIRI:
    gate kabulu -> pose_correct(NOKTAYI oynatir) -> angle_correct -> pick_member_direction
                -> pick_direction_from_dictionary (three direction kolu YALNIZ YONU oynatir)

Dolayisiyla:
  * govde_ici and duvara_yapisik   -> only `pose_correct` sorumlu may be
  * onu_kapali                    -> hem point hem direction etkiler

`pose_correct` robot-hazira +0.0428 kazandirmisti (2026-08-02) but FIZIKSEL yan etkisi
HIC NOT MEASURED. Bu betik three asamada bayrak sayar:
    HAM   : gate kabul etti, no correction absent
    POZ   : + pose_correct
    TAM   : + angle/uye/direction kollari (= A4'un olctugu hal)

VERDICT: HAM -> POZ between govde_ici/duvara_yapisik ARTIYORSA suclu poz kafasidir and
S1 onarimindan ONCE poz kafasi yeniden ayarlanmalidir (kaldirilmaz -- before measurement).
"""
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

import tezgah2 as T2 
import wire_gate 

ONB ="results/_c1_asamalar.pkl"
ILERI_MIN =5.0 
IC_CAP_MIN =0.8 


def asamalar (r ,gate ,cfg ):
    """(HAM, POZ, TAM) -> each biri (P, Pd)."""
    if r ["X"]is None or r .get ("XR")is None :
        return None 
    X =np .hstack ([r ["X"],r ["XR"]])
    if X .shape [1 ]*2 !=gate ["n_feat"]:
        return None 
    k =wire_gate .decision_mask (wire_gate .decision_score (gate ,X ))
    if not k .any ():
        return None 
    P0 =np .asarray (r ["P"],float )[k ].copy ()
    D0 =np .asarray (r ["Pd"],float )[k ].copy ()
    Xk =X [k ]
    ham =(P0 .copy (),D0 .copy ())

    c =[{"point":P0 [i ].copy (),"direction":D0 [i ].copy ()}for i in range (len (P0 ))]
    c =wire_gate .pose_correct (Xk ,c )
    poz =(np .array ([x ["point"]for x in c ],float ),
    np .array ([x ["direction"]for x in c ],float ))

    c =wire_gate .angle_correct (Xk ,c )
    if r .get ("UYE"):
        c =wire_gate .pick_member_direction (Xk ,c ,r ["UYE"])
    tam =(np .array ([x ["point"]for x in c ],float ),
    np .array ([x ["direction"]for x in c ],float ))
    return ham ,poz ,tam 


def main ():
    import protocol 
    protocol .tez_dogrula ()
    import trimesh 
    import thesis_remesh 
    from big_arbiter import eligible 
    from cp_geometry import is_inside ,mouth_width ,ray_hits 
    from infer_step_cp import step_to_mesh 

    with io .open ("cp_config.json",encoding ="utf-8")as f :
        cfg =json .load (f )
    DER ,gate ,ek =T2 .yukle ()
    stp ={p :s for m ,p ,jf ,s in eligible ()}

    if os .path .exists (ONB ):
        with open (ONB ,"rb")as f :
            SAY =pickle .load (f )
        print (f"onbellekten: {len (SAY )} part",flush =True )
    else :
        SAY ={}
        t0 =time .time ()
        for kk ,r in enumerate (DER ,1 ):
            if kk %20 ==0 :
                print (f"  {kk }/{len (DER )}  {time .time ()-t0 :.0f}s",flush =True )
                with open (ONB ,"wb")as f :
                    pickle .dump (SAY ,f )
            a =asamalar (r ,gate ,cfg )
            if a is None or r ["pid"]not in stp :
                continue 
            try :
                Vr ,Fr =step_to_mesh (stp [r ["pid"]])
                V ,F =thesis_remesh .remesh_uniform (Vr ,Fr ,target =6000 )
                mesh =trimesh .Trimesh (vertices =np .ascontiguousarray (V ,float ),
                faces =np .ascontiguousarray (F ,np .int64 ),process =False )
            except Exception :
                continue 
            rec_ =[]
            for ad ,(P ,D )in zip (("HAM","POZ","TAM"),a ):
                sat =[]
                for i in range (len (P )):
                    p ,d =P [i ],D [i ]
                    try :
                        ic =bool (is_inside (mesh ,p ))
                    except Exception :
                        ic =False 
                    try :
                        h =ray_hits (mesh ,p +1e-3 *d ,d ,60.0 )
                        il =float (min (h ))if len (h )else float ("inf")
                    except Exception :
                        il =float ("inf")
                    try :
                        cp =float (mouth_width (mesh ,p ,d )[0 ])
                    except Exception :
                        cp =float ("nan")
                    sat .append ((ic ,il ,cp ))
                rec_ .append (sat )
                # poz kafasinin NOKTAYI ne up to oynattigi
            kay =float (np .linalg .norm (a [1 ][0 ]-a [0 ][0 ],axis =1 ).mean ())if len (a [0 ][0 ])else 0.0 
            SAY [r ["pid"]]={"asama":rec_ ,"poz_kayma":kay ,
            "regime":"cok"if r ["n"]>=8 else "dusuk"}
        with open (ONB ,"wb")as f :
            pickle .dump (SAY ,f )
        print (f"-> {ONB }",flush =True )

    def bayrak (s ):
        ic ,il ,cp =s 
        return (bool (ic ),
        bool (np .isfinite (il )and il <ILERI_MIN ),
        bool (np .isfinite (cp )and 0 <cp <IC_CAP_MIN ))

    ADLAR =("govde_ici","onu_kapali","duvara_yapisik")
    print (f"\n{'='*74 }\nC1 -- ZINCIR ASAMALARINA GORE FIZIKSEL KUSUR\n{'='*74 }")
    print (f"{'asama':<8}{'CP':>7}"+"".join (f"{a :>16}"for a in ADLAR )+f"{'HERHANGI':>11}")
    tab ={}
    for j ,ad in enumerate (("HAM","POZ","TAM")):
        n =c =0 ;k =[0 ,0 ,0 ]
        for pid ,d in SAY .items ():
            for s in d ["asama"][j ]:
                n +=1 
                b =bayrak (s )
                for q in range (3 ):
                    k [q ]+=b [q ]
                c +=any (b )
        tab [ad ]={"n":n ,"bayrak":k ,"herhangi":c }
        print (f"{ad :<8}{n :>7}"+"".join (f"{k [q ]:>9} %{100 *k [q ]/max (n ,1 ):>4.1f}"for q in range (3 ))
        +f"{c :>7} %{100 *c /max (n ,1 ):>3.1f}")

    kay =np .array ([d ["poz_kayma"]for d in SAY .values ()])
    print (f"\npoz kafasinin NOKTAYI oynatma miktari: medyan {np .median (kay ):.2f}mm | "
    f"%90 {np .percentile (kay ,90 ):.2f}mm | max {kay .max ():.2f}mm")

    print (f"\n{'='*74 }\nHUKUM\n{'='*74 }")
    for q ,ad in enumerate (ADLAR ):
        h ,p ,t =(tab [a ]["bayrak"][q ]for a in ("HAM","POZ","TAM"))
        d1 =p -h ;d2 =t -p 
        suc =("POZ KAFASI"if d1 >0 and d1 >=abs (d2 )else 
        "YON KOLLARI"if d2 >0 and d2 >d1 else "ZINCIR DEGIL (gate/candidate)")
        print (f"  {ad :<16} HAM {h :>4} -> POZ {p :>4} ({d1 :+d}) -> TAM {t :>4} ({d2 :+d})   suclu: {suc }")
    hh ,hp ,ht =(tab [a ]["herhangi"]for a in ("HAM","POZ","TAM"))
    print (f"  {'HERHANGI':<16} HAM {hh :>4} -> POZ {hp :>4} ({hp -hh :+d}) -> TAM {ht :>4} ({ht -hp :+d})")
    if hp >hh :
        print (f"\n  -> poz kafasi kusuru {hp -hh } ARTIRIYOR: S1 onariminden ONCE yeniden ayarlanmali")
    else :
        print (f"\n  -> poz kafasi kusuru ARTIRMIYOR ({hp -hh :+d}); kusur GATE/ADAY duzeyinden geliyor")

    with io .open ("results/c1_poz_sorusturmasi.json","w",encoding ="utf-8")as f :
        json .dump ({"tablo":tab ,"adlar":list (ADLAR ),
        "poz_kayma_medyan":float (np .median (kay )),
        "esikler":{"ileri_min":ILERI_MIN ,"ic_cap_min":IC_CAP_MIN }},
        f ,indent =1 )
    print ("\nmakbuz -> results/c1_poz_sorusturmasi.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
