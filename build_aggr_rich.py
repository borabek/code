# -*- coding: utf-8 -*-
"""GECE 3. HAMLE: AGRESIF ADAY (very-cozunurluk, recall tavani 0.969) + ZENGIN FEATURE.
WHY SIMDI: P2 (agresif candidate) more before ASKIYA alinmisti because zayif gate 6.5x adayi kaldiramiyordu
(WEI top-N 0.662 < baz 0.723). Bu night gate MATERYAL guclendi (AUC 0.88 -> 0.955, zengin temsil).
P2'nin ten-kosulu ("precision yukunu more iyi gate carries") ARTIK KARSILANDI -> yeniden test.
Cikti: results/aggr_rich_<mfg>.npz (X13, XR, y, groups, mfg, pos, ngt) -- analiz eval_rich_deploy with same.
Kullanim: python build_aggr_rich.py WEI [limit]"""
import os ,sys ,json ,time 
import numpy as np ,torch ,trimesh 
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
import diffusionnet as D ,cp_openings ,thesis_remesh ,wire_gate 
from cad_eval import align_frames 
from infer_step_cp import step_to_mesh ,load_any 
from big_arbiter import eligible ,CE ,CT ,OP 
from robot_cp import _vote2 
from build_rich_feats import rich_feats 

dev ="cuda"if torch .cuda .is_available ()else "cpu"
MFG =sys .argv [1 ]if len (sys .argv )>1 else "WEI"
LIM =int (sys .argv [2 ])if len (sys .argv )>2 and sys .argv [2 ].isdigit ()else 0 
# --list file.txt: MFG instead of open part listesi (high-CP hibrit for)
PLIST =None 
if "--list"in sys .argv :
    PLIST =set (open (sys .argv [sys .argv .index ("--list")+1 ]).read ().split ())
    MFG ="LIST"
cfg =json .load (open ("cp_config.json"))
PROD =cfg ["robot_vote2_checkpoints"]
EXTRA =[f"results/seg_extra/recall_hard_keig128_s{i }.pt"for i in range (3 )]


def derive (V ,F ,pb ,mv ,vc ,cl ):
    return cp_openings .connection_points (V ,F ,pb .argmax (-1 ),min_v =mv ,classes =(CE ,CT ),dedupe_mm =10.0 ,
    probs =pb ,vertex_conf =vc ,ct_depth_min_mm =1.0 ,cluster_mm =cl )


def main ():
    m7 =[load_any (c ,dev =dev )[:2 ]for c in PROD +[e for e in EXTRA if os .path .exists (e )]]
    os .environ ["BA_ALLOW_SEEN"]="1"
    oos =set (open ("pxc_out_of_scope.txt").read ().split ())if os .path .exists ("pxc_out_of_scope.txt")else set ()
    held =set (open ("_hw_r3.txt").read ().split ())
    if PLIST is not None :
        parts =[(m ,p ,jf ,s )for m ,p ,jf ,s in eligible ()if p in PLIST ]
    else :
        parts =[(m ,p ,jf ,s )for m ,p ,jf ,s in eligible ()
        if m ==MFG and ((m =="WEI"and p in held )or (m =="PXC"and p not in oos ))]
    if LIM :parts =parts [:LIM ]
    print (f"{len (parts )} {MFG } part | {len (m7 )} model | agresif very-cozunurluk + zengin feature",flush =True )
    X13 ,XR ,YY ,GG ,POS ,NGT =[],[],[],[],[],{}
    PIDS =[]
    t0 =time .time ()
    for k ,(mfg ,pid ,jf ,stp )in enumerate (parts ,1 ):
        try :
            j =json .load (open (jf ,encoding ="utf-8-sig"))
            Vj =np .array ([[q ["X"],q ["Y"],q ["Z"]]for q in j ["Graphic3d"]["Points"]],float )
            G =np .array ([[c ["Point"]["X"],c ["Point"]["Y"],c ["Point"]["Z"]]for c in j ["ConnectionPoints"]],float )
            Gd =np .array ([[c ["InsertDirection"]["X"],c ["InsertDirection"]["Y"],c ["InsertDirection"]["Z"]]for c in j ["ConnectionPoints"]],float )
            if not len (G ):continue 
            Gd =Gd /(np .linalg .norm (Gd ,axis =1 ,keepdims =True )+1e-9 )
            Vr ,Fr =step_to_mesh (stp )
            V6 ,F6 =thesis_remesh .remesh_uniform (Vr ,Fr ,target =6000 )
            V6 =np .ascontiguousarray (V6 ,np .float64 );F6 =np .ascontiguousarray (F6 ,np .int64 )
            per =[];acc =None 
            for model ,meta in m7 :
                _ ,pb =D .predict (model ,meta ,V6 ,F6 ,device =dev ,op_cache_dir =f"{OP }_k{int (meta .get ('k_eig',64 ))}",return_probs =True )
                pb =np .asarray (pb ,float );acc =pb if acc is None else acc +pb 
                per .append (derive (V6 ,F6 ,pb ,30 ,0.5 ,5.0 ));per .append (derive (V6 ,F6 ,pb ,10 ,0.20 ,3.0 ))
            avg =acc /len (m7 )
            for target ,params ,opd in [(9000 ,[(8 ,0.10 ,0.0 ),(4 ,0.05 ,0.0 )],"_9k"),(12000 ,[(6 ,0.10 ,0.0 )],"_12k")]:
                try :
                    Vt ,Ft =thesis_remesh .remesh_uniform (Vr ,Fr ,target =target )
                    Vt =np .ascontiguousarray (Vt ,np .float64 );Ft =np .ascontiguousarray (Ft ,np .int64 )
                    for model ,meta in m7 :
                        _ ,pb =D .predict (model ,meta ,Vt ,Ft ,device =dev ,op_cache_dir =f"{OP }_k{int (meta .get ('k_eig',64 ))}{opd }",return_probs =True )
                        pb =np .asarray (pb ,float )
                        for mv ,vc ,cl in params :per .append (derive (Vt ,Ft ,pb ,mv ,vc ,cl ))
                except Exception :pass 
            cps =_vote2 ([c for c in per if c ],min_votes =1 )
            if not cps :continue 
            Nrm =trimesh .Trimesh (V6 ,F6 ,process =False ).vertex_normals .view (np .ndarray )
            x13 =wire_gate .feats_for (V6 ,None ,avg ,cps ,CE ,CT )
            xr =rich_feats (V6 ,F6 ,avg ,cps ,Nrm )
            R ,t ,_ =align_frames (Vr ,Vj )
            P =np .array ([np .asarray (c ["point"])for c in cps ],float )@R .T +t 
            tol =max (3.0 ,0.06 *float (np .linalg .norm (Vj .max (0 )-Vj .min (0 ))))
            diff =P [:,None ,:]-G [None ,:,:];al =(diff *Gd [None ,:,:]).sum (-1 )
            pe =np .where (np .abs (al )<=40.0 ,np .linalg .norm (diff -al [...,None ]*Gd [None ,:,:],axis =-1 ),np .inf )
            yy =np .zeros (len (P ),int )
            up ,ug =set (),set ()
            for dd ,a ,b in sorted ((pe [a ,b ],a ,b )for a in range (len (P ))for b in range (len (G ))if pe [a ,b ]<=tol ):
                if a in up or b in ug :continue 
                up .add (a );ug .add (b );yy [a ]=1 
            gi =len (NGT );NGT [gi ]=len (G )
            X13 .append (x13 );XR .append (xr );YY .append (yy );POS .append (P );GG +=[gi ]*len (cps )
            PIDS .append (pid )
        except Exception :
            continue 
        if k %20 ==0 :print (f"  {k }/{len (parts )}  {len (NGT )} ok  {time .time ()-t0 :.0f}s",flush =True )
    gid =np .array (sorted (NGT ));ng =np .array ([NGT [g ]for g in gid ])
    out =f"results/aggr_rich_{MFG }.npz"
    y =np .concatenate (YY )
    np .savez (out ,X13 =np .vstack (X13 ),XR =np .vstack (XR ),y =y ,groups =np .array (GG ),
    mfg =np .full (len (y ),1 if MFG =="WEI"else 0 ),pos =np .vstack (POS ),grp_ids =gid ,ngt =ng ,
    part_ids =np .array (PIDS ))
    print (f"-> {out }  {len (y )} candidate ({len (y )/max (ng .sum (),1 ):.2f}x GT), {len (NGT )} part, "
    f"candidate-ceiling recall {y .sum ()}/{ng .sum ()} = {y .sum ()/max (ng .sum (),1 ):.3f}  {time .time ()-t0 :.0f}s")


if __name__ =="__main__":
    main ()
