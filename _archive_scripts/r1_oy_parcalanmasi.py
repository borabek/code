# -*- coding: utf-8 -*-
"""R1: OY PARCALANMASI -- genellesen TEK sinyali bozuyor muyuz?

RATIONALE: gate'in reddettigi 248 DOGRU adayin ayirt edici ozelligi DUSUK OY (mean 1.33
vs kabul edilenlerde 1.98). Ve `_vote2`'nin own yorumunda yazili: "dürüst (geometri)
bolmede gate'in GENELLESEN TEK ozelligi votes (AUC dususu 0.018)".

HIPOTEZ: `_vote2` uyeleri 5mm'lik acgozlu kumelemeyle birlestiriyor. Iki uye AYNI acikligi
bulup noktalari 5mm'den ayri dusuyorsa, sonuc IKI ayri candidate x 1 oy becomes -- oysa real
BIR candidate x 2 oy. Yani "low oy" a zayiflik isareti not, OLCUM PARCALANMASI may be.

BU BETIK: each manufacturer CP'si for, that CP'ye yakin duşen UYE noktalarinin YAYILIMINI olcer.
Yayilim 5mm'yi asiyorsa oy parcalaniyor demektir.

Ayrica different cluster_mm degerlerinde oy dagilimi nasil degisiyor, onu da gives.
Bu a OLCUM; no sey dagitilmaz.
"""
import io 
import json 
import os 
import sys 
import time 

import numpy as np 
import torch 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
N_PARCA =45 


def main ():
    import cp_openings 
    import diffusionnet as D_ 
    import measure_set 
    import thesis_remesh 
    from big_arbiter import eligible 
    from connector_constants import CABLE_ENTRY as CE ,CONTACT as CT 
    from infer_step_cp import load_any ,step_to_mesh 

    with io .open ("cp_config.json",encoding ="utf-8")as f :
        cfg =json .load (f )
    pp =cfg .get ("prediction_postproc",{})
    mv =int (pp .get ("min_vertices",4 ));vc =float (pp .get ("vertex_confidence_mask",0.3 ))
    cl =float (pp .get ("cluster_mm",3.0 ))
    stp ={p :s for m ,p ,jf ,s in eligible ()}
    DER ,rap =measure_set .cluster ("results/_der_tam.pkl")
    measure_set .rapor_bas (rap )
    rng =np .random .default_rng (0 )
    sec =[DER [i ]for i in rng .permutation (len (DER ))[:N_PARCA ]if len (DER [i ]["G"])]
    cks =cfg ["current_product"].get ("checkpoints")or cfg ["robot_vote2_checkpoints"]
    dev ="cuda"if torch .cuda .is_available ()else "cpu"
    models =[load_any (c ,dev =dev )[:2 ]for c in cks ]
    print (f"\n{len (sec )} part | {len (models )} uye | turetme: min_v {mv } vc {vc } cluster {cl }",
    flush =True )

    YAY ,N_UYE =[],[]
    OY ={c :[]for c in (3.0 ,5.0 ,8.0 ,12.0 )}
    t0 =time .time ()
    for k ,r in enumerate (sec ,1 ):
        if k %10 ==0 :
            print (f"  {k }/{len (sec )}  {time .time ()-t0 :.0f}s",flush =True )
        try :
            Vr ,Fr =step_to_mesh (stp [r ["pid"]])
            V ,F =thesis_remesh .remesh_uniform (Vr ,Fr ,target =6000 )
            V =np .ascontiguousarray (V ,np .float64 );F =np .ascontiguousarray (F ,np .int64 )
            UYE =[]
            for model ,meta in models :
                _ ,pb =D_ .predict (model ,meta ,V ,F ,device =dev ,
                op_cache_dir =f"results/step_infer/ops_k{int (meta .get ('k_eig',64 ))}",
                return_probs =True )
                q =np .asarray (pb ,float )
                lst =cp_openings .connection_points (
                V ,F ,q .argmax (-1 ),min_v =mv ,classes =(CE ,CT ),dedupe_mm =10.0 ,
                probs =q ,vertex_conf =vc ,ct_depth_min_mm =1.0 ,cluster_mm =cl ,
                step_path =stp [r ["pid"]])
                UYE .append (np .array ([c ["point"]for c in lst ],float )if lst 
                else np .zeros ((0 ,3 )))
        except Exception as e :
            print (f"    {r ['pid']}: {type (e ).__name__ }");continue 
        G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
        for b in range (len (G )):
        # each uyenin this GT'ye EN YAKIN adayi (dik distance, detection toleransi inside)
            yakin =[]
            for P in UYE :
                if not len (P ):
                    continue 
                v =P -G [b ]
                al =v @Gd [b ]
                pe =np .linalg .norm (v -al [:,None ]*Gd [b ],axis =1 )
                pe =np .where (np .abs (al )>40 ,np .inf ,pe )
                i =int (np .argmin (pe ))
                if np .isfinite (pe [i ])and pe [i ]<=max (3.0 ,0.06 *float (r ["diag"])):
                    yakin .append (P [i ])
            N_UYE .append (len (yakin ))
            if len (yakin )>=2 :
                Y =np .array (yakin )
                d =[np .linalg .norm (Y [a ]-Y [c ])for a in range (len (Y ))
                for c in range (a +1 ,len (Y ))]
                YAY .append (float (max (d )))
                for cm in OY :
                # this cluster_mm with kac KUME olusurdu (acgozlu, single gecis)
                    cluster =[]
                    for p in Y :
                        h =next ((q for q in cluster if np .linalg .norm (p -q )<=cm ),None )
                        if h is None :
                            cluster .append (p )
                    OY [cm ].append (len (yakin )/max (len (cluster ),1 ))# cluster basina ort. oy

    N_UYE =np .array (N_UYE );YAY =np .array (YAY )
    print (f"\n{len (N_UYE )} manufacturer CP incelendi")
    print (f"  hicbir uye bulamadi   : {int ((N_UYE ==0 ).sum ()):>4}  ({float ((N_UYE ==0 ).mean ()):.1%})")
    for n in (1 ,2 ,3 ,4 ):
        print (f"  {n } uye buldu           : {int ((N_UYE ==n ).sum ()):>4}  ({float ((N_UYE ==n ).mean ()):.1%})")
    print (f"\n>=2 uyenin bulduğu {len (YAY )} CP'de UYE NOKTALARININ YAYILIMI:")
    for q in (50 ,75 ,90 ,95 ):
        print (f"   %{q :<3} {np .percentile (YAY ,q ):>6.2f} mm")
    print (f"   5mm'yi ASAN  : {float ((YAY >5.0 ).mean ()):.1%}   <- bunlar PARCALANIYOR")
    print (f"   8mm'yi asan  : {float ((YAY >8.0 ).mean ()):.1%}")
    print ()
    print (f"{'cluster_mm':<12}{'cluster basina ort. oy':>22}")
    for cm in sorted (OY ):
        print (f"{cm :<12.1f}{np .mean (OY [cm ]):>22.3f}")
    par =float ((YAY >5.0 ).mean ())
    print ()
    if par >=0.15 :
        print (f"HUKUM: >=2 uyenin buldugu CP'lerin %{100 *par :.0f}'inde yayilim 5mm'yi asiyor")
        print ("       -> OY PARCALANIYOR. Genellesen single sinyal bozuluyor.")
        print ("       Sonraki step: cluster_mm'yi buyutup UCTAN UCA olc.")
    else :
        print (f"HUKUM: only %{100 *par :.0f} parcalaniyor -> cluster_mm 5.0 makul, hipotez ZAYIF.")
    with io .open ("results/r1_oy_parcalanmasi.json","w",encoding ="utf-8")as f :
        json .dump ({"n_cp":int (len (N_UYE )),"uye_dagilimi":
        {str (n ):int ((N_UYE ==n ).sum ())for n in range (5 )},
        "yayilim_p50":float (np .percentile (YAY ,50 ))if len (YAY )else None ,
        "yayilim_p90":float (np .percentile (YAY ,90 ))if len (YAY )else None ,
        "parcalanan_pay":par ,
        "oy_per_kume":{str (c ):float (np .mean (OY [c ]))for c in OY }},f ,indent =1 )
    print ("receipt -> results/r1_oy_parcalanmasi.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
