# -*- coding: utf-8 -*-
"""WEI 0.80: agresif WEI-tipik candidate havuzu (oracle 0.972) feature+label+pozisyon+vote with CACHE'le.
Sonra wei_selector_eval.py offline gate+spatial+topology with WEI F1'i olcer. results/wei_aggr_pool.json."""
import os ,sys ,json ,time 
import numpy as np ,torch 
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
import diffusionnet as D ,cp_openings ,thesis_remesh ,connector3d ,wire_gate 
from cad_eval import align_frames 
from infer_step_cp import step_to_mesh ,load_any 
from big_arbiter import eligible ,CE ,CT ,OP 
from robot_cp import _vote2 
from f1_sweep import HELD 

dev ="cuda"if torch .cuda .is_available ()else "cpu";HICP =11 


def derive (V ,F ,pb ,mv ,vc ,cl ):
    return cp_openings .connection_points (V ,F ,pb .argmax (-1 ),min_v =mv ,classes =(CE ,CT ),dedupe_mm =10.0 ,
    probs =pb ,vertex_conf =vc ,ct_depth_min_mm =1.0 ,cluster_mm =cl )


def main ():
    prod =json .load (open ("cp_config.json"))["robot_vote2_checkpoints"]
    extra =["results/seg_extra/recall_hard_keig128_s0.pt","results/seg_extra/recall_hard_keig128_s1.pt","results/seg_extra/recall_hard_keig128_s2.pt"]
    m7 =[load_any (c ,dev =dev )[:2 ]for c in prod +extra ]
    os .environ ["BA_ALLOW_SEEN"]="1"
    wei =[]
    for mfg ,pid ,jf ,stp in eligible ():
        if mfg !="WEI"or pid not in HELD :continue 
        try :
            n =len (json .load (open (jf ,encoding ="utf-8-sig")).get ("ConnectionPoints",[]))
            if 1 <=n <HICP :wei .append ((pid ,jf ,stp ))
        except Exception :pass 
    print (f"{len (wei )} WEI tipik part | agresif pool + ozellik",flush =True )

    out ={};t0 =time .time ()
    for k ,(pid ,jf ,stp )in enumerate (wei ,1 ):
        try :
            j =json .load (open (jf ,encoding ="utf-8-sig"))
            Vj =np .array ([[p ["X"],p ["Y"],p ["Z"]]for p in j ["Graphic3d"]["Points"]],float )
            G =np .array ([[c ["Point"]["X"],c ["Point"]["Y"],c ["Point"]["Z"]]for c in j ["ConnectionPoints"]],float )
            Gd =np .array ([[c ["InsertDirection"]["X"],c ["InsertDirection"]["Y"],c ["InsertDirection"]["Z"]]for c in j ["ConnectionPoints"]],float )
            if not len (G ):continue 
            Gd =Gd /(np .linalg .norm (Gd ,axis =1 ,keepdims =True )+1e-9 )
            Vr ,Fr =step_to_mesh (stp )
            V6 ,F6 =thesis_remesh .remesh_uniform (Vr ,Fr ,target =6000 );V6 =np .ascontiguousarray (V6 ,np .float64 );F6 =np .ascontiguousarray (F6 ,np .int64 )
            per =[];acc =None 
            for model ,meta in m7 :
                _ ,pb =D .predict (model ,meta ,V6 ,F6 ,device =dev ,op_cache_dir =f"{OP }_k{int (meta .get ('k_eig',64 ))}",return_probs =True )
                pb =np .asarray (pb ,float );acc =pb if acc is None else acc +pb 
                per .append (derive (V6 ,F6 ,pb ,30 ,0.5 ,5.0 ));per .append (derive (V6 ,F6 ,pb ,10 ,0.20 ,3.0 ))
            avg =acc /len (m7 )
            for target ,params ,opd in [(9000 ,[(8 ,0.10 ,0.0 ),(4 ,0.05 ,0.0 )],"_9k"),(12000 ,[(6 ,0.10 ,0.0 )],"_12k")]:
                try :
                    Vt ,Ft =thesis_remesh .remesh_uniform (Vr ,Fr ,target =target );Vt =np .ascontiguousarray (Vt ,np .float64 );Ft =np .ascontiguousarray (Ft ,np .int64 )
                    for model ,meta in m7 :
                        _ ,pb =D .predict (model ,meta ,Vt ,Ft ,device =dev ,op_cache_dir =f"{OP }_k{int (meta .get ('k_eig',64 ))}{opd }",return_probs =True )
                        pb =np .asarray (pb ,float )
                        for mv ,vc ,cl in params :per .append (derive (Vt ,Ft ,pb ,mv ,vc ,cl ))
                except Exception :pass 
            cps =_vote2 ([c for c in per if c ])
            if not cps :continue 
            Xf =wire_gate .feats_for (V6 ,None ,avg ,cps ,CE ,CT )
            R ,t ,_ =align_frames (Vr ,Vj )
            P =np .array ([np .asarray (c ["point"])for c in cps ],float )@R .T +t 
            Dv =np .array ([np .asarray (c ["direction"],float )for c in cps ],float )
            vts =np .array ([int (c .get ("_votes",1 ))for c in cps ],float )
            tol =max (3.0 ,0.06 *float (np .linalg .norm (Vj .max (0 )-Vj .min (0 ))))
            diff =P [:,None ,:]-G [None ,:,:];al =(diff *Gd [None ,:,:]).sum (-1 )
            pe =np .where (np .abs (al )<=40.0 ,np .linalg .norm (diff -al [...,None ]*Gd [None ,:,:],axis =-1 ),np .inf )
            yy =np .zeros (len (P ),int );order =sorted ((pe [a ,b ],a ,b )for a in range (len (P ))for b in range (len (G ))if pe [a ,b ]<=tol )
            up ,ug =set (),set ()
            for dd ,a ,b in order :
                if a in up or b in ug :continue 
                up .add (a );ug .add (b );yy [a ]=1 
            out [pid ]={"P":P .tolist (),"dir":Dv .tolist (),"X":Xf .tolist (),"votes":vts .tolist (),
            "y":yy .tolist (),"N":int (len (G ))}
        except Exception :
            continue 
        if k %20 ==0 :print (f"  {k }/{len (wei )}  {len (out )} part  {time .time ()-t0 :.0f}s",flush =True )
    json .dump (out ,open ("results/wei_aggr_pool.json","w"))
    tot =sum (d ["N"]for d in out .values ());cov =sum (int (np .array (d ["y"]).sum ())for d in out .values ())
    print (f"-> results/wei_aggr_pool.json ({len (out )} part, {tot } GT, matched {cov } = {cov /max (tot ,1 ):.3f})  {time .time ()-t0 :.0f}s")


if __name__ =="__main__":
    main ()
