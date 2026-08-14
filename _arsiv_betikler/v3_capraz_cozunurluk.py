# -*- coding: utf-8 -*-
"""V3: CAPRAZ-COZUNURLUK UYUMU -- niteliksel as YENI bilgi denemesi.

TESPIT 0.85 OPERASYONU, ikinci arm.

BUGUN OLEN UC BLOGUN ORTAK YANI: all of them AYNI 6k mesh uzerindeki geometriden turedi
(B-rep graf, uzamsal duzen, uye zenginlestirme). Ayni kuyudan a kova more.

BU KOL FARKLI: four model de AYNI mesh'te calisiyor, i.e. hatalari ILISKILI. Farkli a
COZUNURLUKTE yeniden orneklemek BAGIMSIZ evidence produces:
    real a opening yeniden ornekleme SONRASI da bulunur;
    segmentasyon artefakti bulunmayabilir.
Bu, tezin uniform-yogunluk kuralini bozmaz -- same network, same sinif, same remesh yordami,
only ikinci a uniform hedef (9000). "Multires" more before ADAY BIRLESIMI as
denenmisti (very-CP'de +0.0283); OZELLIK as never denenmedi.

OZELLIKLER (6):
    mr_var      9k'da 5mm inside candidate VAR mi (0/1)
    mr_mes      most yakin 9k adayina distance (otherwise 99)
    mr_aci      yonler arasi angle (otherwise 90)
    mr_conf     that adayin guveni
    mr_n        5mm icindeki 9k candidate count
    mr_oy       that adayin oy count (_votes)

ONCE UCUZ SINAMA: only measurement kumesinde (194 part) grup-capraz OOF with sinyal present mi?
Varsa full korpusa yatirim is done.
"""
import io 
import json 
import os 
import pickle 
import sys 
import time 

import numpy as np 
import torch 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1"
os .environ ["WG_TOPO"]="1"
os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
OUT ="results/capraz_coz.pkl"
YAKIN =5.0 


def main ():
    import measure_set 
    import robot_cp as RC 
    import thesis_remesh 
    import wire_gate 
    import diffusionnet as D 
    from big_arbiter import eligible 
    from infer_step_cp import load_any ,step_to_mesh 

    with io .open ("cp_config.json",encoding ="utf-8")as f :
        cfg =json .load (f )
    stp_of ={p :s for m ,p ,jf ,s in eligible ()}
    DER ,rap =measure_set .cluster ("results/_der_tam.pkl")
    measure_set .rapor_bas (rap )
    cks =cfg ["current_product"].get ("checkpoints")or cfg ["robot_vote2_checkpoints"]
    dev ="cuda"if torch .cuda .is_available ()else "cpu"
    models =[load_any (c ,dev =dev )[:2 ]for c in cks ]

    if os .path .exists (OUT ):
        with open (OUT ,"rb")as f :
            MR =pickle .load (f )
        print (f"onbellekten: {len (MR )} part",flush =True )
    else :
        MR ={}
        t0 =time .time ()
        for i_ ,r in enumerate (DER ,1 ):
            if i_ %20 ==0 :
                print (f"  9k {i_ }/{len (DER )}  {time .time ()-t0 :.0f}s",flush =True )
            try :
                stp =stp_of .get (r ["pid"])
                Vr ,Fr =step_to_mesh (stp )
                V9 ,F9 =thesis_remesh .remesh_uniform (Vr ,Fr ,target =9000 )
                V9 =np .ascontiguousarray (V9 ,np .float64 )
                F9 =np .ascontiguousarray (F9 ,np .int64 )
                pbs =[]
                for model ,meta in models :
                    _ ,pb =D .predict (model ,meta ,V9 ,F9 ,device =dev ,
                    op_cache_dir =f"results/step_infer/ops9_k{int (meta .get ('k_eig',64 ))}",
                    return_probs =True )
                    pbs .append (np .asarray (pb ,float ))
                cps9 ,_ ,_ ,_ =RC .derive_candidates (V9 ,F9 ,pbs ,stp ,cfg =cfg )
                MR [r ["pid"]]=[(np .asarray (c ["point"],float ),
                np .asarray (c ["direction"],float ),
                float (c .get ("confidence",1.0 )),
                float (c .get ("_votes",1 )))for c in cps9 ]
            except Exception as e :
                MR [r ["pid"]]=None 
                print (f"    {r ['pid']}: {type (e ).__name__ }")
        with open (OUT ,"wb")as f :
            pickle .dump (MR ,f )
        print (f"-> {OUT }",flush =True )

        # --- CAPRAZ-COZUNURLUK sutunlari
    def mr_ozellik (P ,Pd ,pid ):
        n =len (P )
        F =np .zeros ((n ,6 ));F [:,1 ]=99.0 ;F [:,2 ]=90.0 
        lst =MR .get (pid )
        if not lst :
            return F 
        Q =np .array ([x [0 ]for x in lst ]);QD =np .array ([x [1 ]for x in lst ])
        QC =np .array ([x [2 ]for x in lst ]);QV =np .array ([x [3 ]for x in lst ])
        for i in range (n ):
            d =np .linalg .norm (Q -P [i ],axis =1 )
            m =d <=YAKIN 
            F [i ,4 ]=float (m .sum ())
            if m .any ():
                j =int (np .argmin (d ))
                F [i ,0 ]=1.0 ;F [i ,1 ]=float (d [j ])
                F [i ,2 ]=float (np .degrees (np .arccos (np .clip (
                abs (float (QD [j ]@Pd [i ])),0 ,1 ))))
                F [i ,3 ]=float (QC [j ]);F [i ,5 ]=float (QV [j ])
        return F 

    kap =sum (1 for r in DER if MR .get (r ["pid"]))
    print (f"9k adayi olan part: {kap }/{len (DER )}",flush =True )

    # --- SINYAL VAR MI: eslesen (TP) vs eslesmeyen (FP) adaylarda distribution
    from sina_cluster import esle ,f1w 
    from sklearn .ensemble import RandomForestClassifier 
    from sklearn .model_selection import GroupKFold 
    RX ,RY ,RG ,RP =[],[],[],[]
    for r in DER :
        if r ["X"]is None or r .get ("XR")is None or not len (r ["G"]):
            continue 
        P =r ["P"];Pd =r ["Pd"]
        G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
        diff =P [:,None ,:]-G [None ,:,:]
        al =(diff *Gd [None ,:,:]).sum (-1 )
        pe =np .linalg .norm (diff -al [...,None ]*Gd [None ,:,:],axis =-1 )
        pe =np .where (np .abs (al )>40 ,np .inf ,pe )
        tt =max (3.0 ,0.06 *float (r ["diag"]))
        yy =np .zeros (len (P ),int );us ,hi =set (),set ()
        for dd ,a_ ,b_ in sorted ((pe [a ,b ],a ,b )
        for a in range (len (P ))for b in range (len (G ))):
            if dd >tt or a_ in us or b_ in hi :
                continue 
            us .add (a_ );hi .add (b_ );yy [a_ ]=1 
        MRX =mr_ozellik (P ,Pd ,r ["pid"])
        X58 =np .hstack ([r ["X"],r ["XR"]])
        for i in range (len (P )):
            RX .append (np .concatenate ([X58 [i ],MRX [i ]]))
            RY .append (yy [i ]);RG .append (r ["geo"]);RP .append (r ["pid"])
    RX =np .array (RX ,float );RY =np .array (RY );RG =np .array (RG );RP =np .array (RP )
    print (f"\n{len (RY )} candidate | pozitif {RY .mean ():.1%}",flush =True )

    from t1_uretici_disi import auc_mw 
    AD =["mr_var","mr_mes","mr_aci","mr_conf","mr_n","mr_oy"]
    print (f"\n{'sutun':<10}{'AUC':>8}{'TP ort':>10}{'FP ort':>10}")
    for j ,a in enumerate (AD ):
        v =RX [:,58 +j ]
        print (f"{a :<10}{auc_mw (v ,RY .astype (bool )):>8.3f}"
        f"{v [RY ==1 ].mean ():>10.3f}{v [RY ==0 ].mean ():>10.3f}")

    def olc (M ):
        o =np .zeros (len (RY ))
        for tr ,te in GroupKFold (n_splits =5 ).split (M ,RY ,RG ):
            o [te ]=RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
            random_state =0 ).fit (M [tr ],RY [tr ]).predict_proba (M [te ])[:,1 ]
        m =np .zeros (len (o ),bool )
        for u in np .unique (RP ):
            i =RP ==u ;v =o [i ]
            m [i ]=(v >=0.5 *max (v .max (),1e-9 ))&(v >=0.25 )
        tp =int ((RY .astype (bool )&m ).sum ());fp =int ((~RY .astype (bool )&m ).sum ())
        fn =int ((RY .astype (bool )&~m ).sum ())
        p_ =tp /max (tp +fp ,1 );r_ =tp /max (tp +fn ,1 )
        return 2 *p_ *r_ /max (p_ +r_ ,1e-9 )
    a58 =olc (RX [:,:58 ]);a64 =olc (RX )
    print (f"\nADAY DUZEYI: 58 sutun {a58 :.4f} | 58+capraz {a64 :.4f} | fark {a64 -a58 :+.4f}")
    print (f"KARAR: {'SINYAL VAR -> tam korpusa yatirim'if a64 -a58 >=0.01 else 'SINYAL YOK -> arm OLU'}")
    with io .open ("results/v3_capraz_coz.json","w",encoding ="utf-8")as f :
        json .dump ({"aday_58":float (a58 ),"aday_64":float (a64 ),"fark":float (a64 -a58 ),
        "kapsama":kap /len (DER )},f ,indent =1 )
    print ("receipt -> results/v3_capraz_coz.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
