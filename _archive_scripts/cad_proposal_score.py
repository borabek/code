# -*- coding: utf-8 -*-
"""ONER-SONRA-PUANLA kaskadi (fbi_recall bulgusundan dogdu; tez-sadik: YOLOv6 kolu da region onerip
after segmentasyonla dogruluyor -- Abbildung 45/46).

FINDING: CAD-geometri model-FN'lerinin %60'ini kapsiyor and model bunlarin %63'unde ZATEN atesliyor,
but ham CAD-union precision'i cokertiyor (P 0.590->0.210) because step_openings HER deligi buluyor
(montaj/vida dahil) -- hangisi TEL GIRISI ayirt edemiyor.
COZUM: CAD acikliklarini ADAY as al, each adayi MODELIN baglanti-olasiligiyla PUANLA
(opening ekseni along silindirde), esigin ustundekileri tut. Modelin own CP'leriyle union.
Esigi sup: precision/recall sinirini cikar -> single-model urunu (F1 0.563) geciyor mu?
"""
import os ,sys ,json ,time 
import numpy as np ,torch 
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
import diffusionnet as D ,cp_openings ,thesis_remesh ,step_openings 
from cad_eval import align_frames 
from infer_step_cp import step_to_mesh ,load_any 
from big_arbiter import greedy ,eligible ,CE ,CT ,OP 

CKPT ="results/seg_extra/recall_hard_s2.pt"
HELD =set (open ("_hw_r3.txt").read ().split ())
THRS =[0.0 ,0.20 ,0.30 ,0.40 ,0.50 ,0.60 ,0.70 ,0.80 ]


def score_candidate (V ,conn_p ,entry ,axis ,radius_mm ):
    """Modelin this CAD acikligina verdigi destek: opening ekseni along silindirde baglanti
    olasiliginin upper yuzdeligi. entry/axis STEP(=mesh) frame'inde."""
    a =np .asarray (axis ,float );n =np .linalg .norm (a )
    if n <1e-9 :return 0.0 
    a =a /n 
    rel =V -np .asarray (entry ,float )[None ,:]
    al =rel @a 
    perp =np .linalg .norm (rel -al [:,None ]*a [None ,:],axis =1 )
    rad =max (3.0 ,2.0 *float (radius_mm ))
    near =(al >=-6.0 )&(al <=20.0 )&(perp <=rad )# agizdan ice correct
    if near .sum ()<3 :return 0.0 
    return float (np .percentile (conn_p [near ],90 ))# upper yuzdelik = "model burayi baglanti goruyor mu"


def main ():
    dev ="cuda"if torch .cuda .is_available ()else "cpu"
    model ,meta ,_ =load_any (CKPT ,dev =dev )
    os .environ ["BA_ALLOW_SEEN"]="1"
    raw =[p for p in eligible ()if p [0 ]=="WEI"and p [1 ]in HELD ]
    print (f"{len (raw )} WEI held-out | ONER-SONRA-PUANLA kaskadi",flush =True )

    parts =[];t0 =time .time ()
    for k ,(mfg ,pid ,jf ,stp )in enumerate (raw ,1 ):
        try :
            j =json .load (open (jf ,encoding ="utf-8-sig"))
            Vj =np .array ([[p ["X"],p ["Y"],p ["Z"]]for p in j ["Graphic3d"]["Points"]],float )
            G =np .array ([[c ["Point"]["X"],c ["Point"]["Y"],c ["Point"]["Z"]]for c in j .get ("ConnectionPoints",[])],float )
            Gd =np .array ([[c ["InsertDirection"]["X"],c ["InsertDirection"]["Y"],c ["InsertDirection"]["Z"]]for c in j .get ("ConnectionPoints",[])],float )
            if not len (G ):continue 
            Gd =Gd /(np .linalg .norm (Gd ,axis =1 ,keepdims =True )+1e-9 )
            Vr ,Fr =step_to_mesh (stp )
            V ,F =thesis_remesh .remesh_uniform (Vr ,Fr ,target =6000 )
            V =np .ascontiguousarray (V ,np .float64 );F =np .ascontiguousarray (F ,np .int64 )
            _ ,pb =D .predict (model ,meta ,V ,F ,device =dev ,op_cache_dir =OP ,return_probs =True )
            probs =np .asarray (pb ,float );lab =probs .argmax (-1 );conn_p =probs [:,CE ]+probs [:,CT ]
            cps =cp_openings .connection_points (V ,F ,lab ,min_v =30 ,classes =(CE ,CT ),dedupe_mm =10.0 ,
            probs =probs ,vertex_conf =0.5 ,ct_depth_min_mm =1.0 ,cluster_mm =5.0 )
            Pm =np .array ([np .asarray (c ["point"])for c in cps ],float )if cps else np .zeros ((0 ,3 ))
            try :
                cad =step_openings .openings_from_step (stp ,auto =True ,slot_pairs =True ,rect_clusters =True )
            except Exception :
                cad =[]
            cand =[]# (point_step_frame, score)
            for o in cad :
                s =score_candidate (V ,conn_p ,o ["entry_point"],o ["approach_vector"],o .get ("radius_mm",1.5 ))
                cand .append ((np .asarray (o ["entry_point"],float ),s ))
            R ,t ,_ =align_frames (Vr ,Vj )
            tol =max (3.0 ,0.06 *float (np .linalg .norm (Vj .max (0 )-Vj .min (0 ))))
            parts .append ({"Pm":Pm ,"cand":cand ,"R":R ,"t":t ,"G":G ,"Gd":Gd ,"tol":tol })
        except Exception :
            continue 
        if k %25 ==0 :print (f"  {k }/{len (raw )}  {time .time ()-t0 :.0f}s",flush =True )

    print (f"\n=== ONER-SONRA-PUANLA ({len (parts )} part) | urun tek-model F1 0.563 ===")
    print (f"{'CAD threshold':>9s} | {'P':>6s} {'R':>6s} {'F1':>6s}")
    best =None 
    for thr in THRS :
        T =Fp =Fn =0 
        for c in parts :
            keep =[p for p ,s in c ["cand"]if s >=thr ]if thr >0 else [p for p ,s in c ["cand"]]
            allp =list (c ["Pm"])+keep 
            U =[]
            for pnt in allp :
                if all (np .linalg .norm (pnt -u )>5.0 for u in U ):U .append (pnt )
            U =np .array (U ,float )if U else np .zeros ((0 ,3 ))
            P =(U @c ["R"].T +c ["t"])if len (U )else np .zeros ((0 ,3 ))
            tp ,fp ,fn =greedy (P ,c ["G"],c ["tol"],c ["Gd"],40.0 )
            T +=tp ;Fp +=fp ;Fn +=fn 
        p =T /max (T +Fp ,1 );r =T /max (T +Fn ,1 );f =2 *p *r /max (p +r ,1e-9 )
        print (f"{thr :9.2f} | {p :6.3f} {r :6.3f} {f :6.3f}",flush =True )
        if best is None or f >best [0 ]:best =(f ,p ,r ,thr )
    print (f"\n  EN IYI kaskad F1={best [0 ]:.3f} (P{best [1 ]:.3f} R{best [2 ]:.3f}) @ CAD threshold {best [3 ]:.2f}")
    print (f"  (tek-model urun: P0.590 R0.538 F1 0.563 -- kaskad bunu geciyor mu?)")
    json .dump ({"best_f1":best [0 ],"best_thr":best [3 ]},open ("results/cad_cascade.json","w"),indent =1 )


if __name__ =="__main__":
    main ()
