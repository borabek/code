# -*- coding: utf-8 -*-
"""LEVER B (0.80 hedefi): per-model post-proc (min_v, vertex_conf) wire-gate ILE yeniden tara.

min_v 30 / vc 0.5 gate YOKKEN secilmisti (small/low-confidence fragmentler noise diye atiliyordu). Gate
residual precision'i temizledigi for, union'a DAHA COK recall sokup gate'e filtreletmek F1'i artirabilir.
Her part: 4 model probs BIR times -> each (min_v, vc) kombosunda per-model CP turet -> union(vote>=1) ->
gate feature -> OOF gate skoru -> F1. Sizinti-siz (GroupKFold). 823 part.
"""
import os ,sys ,json ,time ,itertools 
import numpy as np ,torch 
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
import diffusionnet as D ,cp_openings ,thesis_remesh ,wire_gate 
from cad_eval import align_frames 
from infer_step_cp import step_to_mesh ,load_any 
from big_arbiter import eligible ,CE ,CT ,OP 
from robot_cp import _vote2 

HELD =set (open ("_hw_r3.txt").read ().split ())
GRID =list (itertools .product ([30 ,20 ,10 ],[0.50 ,0.40 ,0.35 ]))# (min_v, vc)


def main ():
    dev ="cuda"if torch .cuda .is_available ()else "cpu"
    cks =json .load (open ("cp_config.json"))["robot_vote2_checkpoints"]
    models =[load_any (c ,dev =dev )[:2 ]for c in cks ]
    os .environ ["BA_ALLOW_SEEN"]="1"
    parts =[p for p in eligible ()if (p [0 ]=="WEI"and p [1 ]in HELD )]+[p for p in eligible ()if p [0 ]=="PXC"]
    print (f"{len (parts )} part | {len (GRID )} (min_v,vc) kombosu x union+gate",flush =True )

    # each kombo for: X listesi, y, votes, grp, mfg + ngt
    data ={g :{"X":[],"y":[],"v":[],"grp":[],"mfg":[]}for g in GRID }
    ngt ={};t0 =time .time ()
    for k ,(mfg ,pid ,jf ,stp )in enumerate (parts ,1 ):
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
            probs_list =[];acc =None 
            for model ,meta in models :
                _opd =f"{OP }_k{int (meta .get ('k_eig',64 ))}"
                _ ,pb =D .predict (model ,meta ,V ,F ,device =dev ,op_cache_dir =_opd ,return_probs =True )
                pb =np .asarray (pb ,float );probs_list .append (pb );acc =pb if acc is None else acc +pb 
            avg =acc /len (probs_list )
            R ,t ,_ =align_frames (Vr ,Vj );ngt [k ]=len (G )
            tol =max (3.0 ,0.06 *float (np .linalg .norm (Vj .max (0 )-Vj .min (0 ))))
            for (mv ,vc )in GRID :
                per =[cp_openings .connection_points (V ,F ,pb .argmax (-1 ),min_v =mv ,classes =(CE ,CT ),
                dedupe_mm =10.0 ,probs =pb ,vertex_conf =vc ,ct_depth_min_mm =1.0 ,cluster_mm =5.0 )
                for pb in probs_list ]
                cps =_vote2 (per )
                if not cps :continue 
                X =wire_gate .feats_for (V ,F ,avg ,cps ,CE ,CT )
                P =np .array ([np .asarray (c ["point"])for c in cps ],float )@R .T +t 
                diff =P [:,None ,:]-G [None ,:,:];al =(diff *Gd [None ,:,:]).sum (-1 )
                pe =np .where (np .abs (al )<=40.0 ,np .linalg .norm (diff -al [...,None ]*Gd [None ,:,:],axis =-1 ),np .inf )
                yy =np .zeros (len (P ),int );order =sorted ((pe [a ,b ],a ,b )for a in range (len (P ))for b in range (len (G ))if pe [a ,b ]<=tol )
                up ,ug =set (),set ()
                for dd ,a ,b in order :
                    if a in up or b in ug :continue 
                    up .add (a );ug .add (b );yy [a ]=1 
                D_ =data [(mv ,vc )]
                D_ ["X"].append (X );D_ ["y"].append (yy );D_ ["v"].append (np .array ([c ["_votes"]for c in cps ]))
                D_ ["grp"]+=[k ]*len (cps );D_ ["mfg"]+=[1 if mfg =="WEI"else 0 ]*len (cps )
        except Exception :
            continue 
        if k %50 ==0 :print (f"  {k }/{len (parts )}  {time .time ()-t0 :.0f}s",flush =True )

    from sklearn .ensemble import GradientBoostingClassifier 
    from sklearn .model_selection import GroupKFold 
    GT =sum (ngt .values ())
    print (f"\n=== POST-PROC + GATE SWEEP (manufacturer {GT }) | mevcut urun min_v30/vc0.5+gate0.35 = 0.693 ===")
    print (f"{'min_v':>5s} {'vc':>4s} | {'ALL_F1':>6s} {'WEI':>5s} {'PXC':>5s} (en iyi gate-esikte)")
    best =None 
    for (mv ,vc )in GRID :
        D_ =data [(mv ,vc )]
        if not D_ ["X"]:continue 
        X =np .vstack (D_ ["X"]);y =np .concatenate (D_ ["y"]);v =np .concatenate (D_ ["v"])
        grp =np .array (D_ ["grp"]);mfg =np .array (D_ ["mfg"])
        oof =np .zeros (len (y ))
        for tr ,te in GroupKFold (5 ).split (X ,y ,grp ):
            c =GradientBoostingClassifier (n_estimators =150 ,max_depth =3 ,learning_rate =0.05 ).fit (X [tr ],y [tr ]);oof [te ]=c .predict_proba (X [te ])[:,1 ]
        def f1 (m ,th ):
            k =(m if m is not None else np .ones (len (y ),bool ))&(v >=1 )&(oof >=th )
            tp =int ((y [k ]==1 ).sum ());gt =sum (ngt [g ]for g in np .unique (grp [(m if m is not None else np .ones (len (y ),bool ))]))
            p =tp /max (k .sum (),1 );r =tp /max (gt ,1 );return 2 *p *r /max (p +r ,1e-9 )
            # each min_v/vc for most iyi gate-threshold
        bth =max ([0.25 ,0.30 ,0.35 ,0.40 ],key =lambda th :f1 (None ,th ))
        fa ,fw ,fp =f1 (None ,bth ),f1 (mfg ==1 ,bth ),f1 (mfg ==0 ,bth )
        mark =""
        if best is None or fa >best [0 ]:best =(fa ,mv ,vc ,bth );mark =" <--"
        print (f"{mv :5d} {vc :4.2f} | {fa :6.3f} {fw :5.3f} {fp :5.3f}  gate{bth }{mark }")
    print (f"\n  EN IYI: min_v{best [1 ]} vc{best [2 ]} + gate{best [3 ]} -> ALL F1 {best [0 ]:.3f} (mevcut 0.693)")


if __name__ =="__main__":
    main ()
