# -*- coding: utf-8 -*-
"""Raise cp-v3 F1 by improving PRECISION ten the arbiter set (recall is capped by segmentation ->
needs labels; precision is a post-processing lever). Current cp-v3: P0.265 R0.722 F1 0.388 --
13 TP but 36 FP, because a single manufacturer terminal has several connection features (clamp,
screw, wire slot) each labelled Contact/CableEntry -> several CPs per terminal. The manufacturer
counts TERMINALS. So the physically-correct fix is per-terminal CLUSTERING: single-linkage-merge
all connection CPs within a radius into ONE CP (the robot inserts before per terminal).

Predict each of the 9 in-scope PXC terminals ONCE (cache plab+probs), then sweep the precision
knobs in pure numpy (no re-inference): min_v, ct_depth_min_mm, per-vertex confidence mask, and the
terminal cluster radius. Report the F1 surface, best combo, and the HONEST 9-part overfitting caveat.

Usage: PYTHONPATH=_diffusion_net_repo/src .venv/Scripts/python.exe improve_cp_precision.py
"""
import os ,glob ,json ,itertools 
import numpy as np 
import torch 
import diffusionnet as D 
import connector3d ,cp_openings ,thesis_remesh 
from cad_eval import align_frames 
from infer_step_cp import step_to_mesh ,load_any 

CE =int (connector3d .CABLE_ENTRY );CT =int (connector3d .CONTACT )
OP ="results/step_infer/ops";JDIR ="C:/Users/DE00024082/Desktop/JSON"
CKPT ="results/scheffler_semantic/refit91.pt"


def predict_probs (model ,meta ,V ,F ,dev ):
    """Use D.predict(return_probs=True) -- do NOT hand-roll the forward: a previous version used
    meta.get('n_eig', 48) while refit91 was trained with n_eig=64, which silently built the WRONG
    eigenbasis and produced invalid (flatteringly high) sweep numbers. D.predict resolves n_eig
    from meta itself."""
    V =np .ascontiguousarray (V ,np .float64 );F =np .ascontiguousarray (F ,np .int64 )
    plab ,probs =D .predict (model ,meta ,V ,F ,device =dev ,op_cache_dir =OP ,return_probs =True )
    return np .asarray (plab ),np .asarray (probs )


def cluster (points ,radius ):
    """Single-linkage merge of points within `radius` -> cluster centroids (one CP per terminal)."""
    if radius <=0 or len (points )<=1 :
        return points 
    n =len (points );parent =list (range (n ))
    def find (x ):
        while parent [x ]!=x :parent [x ]=parent [parent [x ]];x =parent [x ]
        return x 
    d =np .linalg .norm (points [:,None ,:]-points [None ,:,:],axis =2 )
    for i in range (n ):
        for j in range (i +1 ,n ):
            if d [i ,j ]<radius :
                parent [find (i )]=find (j )
    out ={}
    for i in range (n ):
        out .setdefault (find (i ),[]).append (points [i ])
    return np .array ([np .mean (v ,0 )for v in out .values ()])


def greedy_match (P ,G ,tol ):
    if not len (P )or not len (G ):
        return 0 ,len (P ),len (G )
    dm =np .linalg .norm (P [:,None ,:]-G [None ,:,:],axis =2 )
    pairs =sorted ((dm [i ,j ],i ,j )for i in range (len (P ))for j in range (len (G )))
    up ,ug =set (),set ();tp =0 
    for dd ,i ,j in pairs :
        if dd >tol :break 
        if i in up or j in ug :continue 
        up .add (i );ug .add (j );tp +=1 
    return tp ,len (P )-tp ,len (G )-tp 


def main ():
    dev ="cuda"if torch .cuda .is_available ()else "cpu"
    model ,meta ,_ =load_any (CKPT ,dev =dev )
    cache =[]
    for stp in sorted (glob .glob ("_cad_eval_pxc/*.stp")):
        pid =os .path .basename (stp ).split ("_")[1 ]
        jf =os .path .join (JDIR ,f"PXC.{pid }.json")
        if not os .path .exists (jf ):continue 
        j =json .load (open (jf ,encoding ="utf-8-sig"))
        Vj =np .array ([[p ["X"],p ["Y"],p ["Z"]]for p in j ["Graphic3d"]["Points"]],float )
        G =np .array ([[c ["Point"]["X"],c ["Point"]["Y"],c ["Point"]["Z"]]for c in j .get ("ConnectionPoints",[])],float )
        Vr ,Fr =step_to_mesh (stp )
        V ,F =thesis_remesh .remesh_uniform (Vr ,Fr ,target =6000 )
        V =np .ascontiguousarray (V ,np .float64 );F =np .ascontiguousarray (F ,np .int64 )
        plab ,probs =predict_probs (model ,meta ,V ,F ,dev )
        R ,t ,ares =align_frames (Vr ,Vj )
        diag =float (np .linalg .norm (Vj .max (0 )-Vj .min (0 )))
        cache .append (dict (pid =pid ,V =V ,F =F ,plab =plab ,probs =probs ,R =R ,t =t ,G =G ,tol =max (3.0 ,0.06 *diag )))
        print (f"  cached {pid }: mfg={len (G )} align={ares :.2f}mm",flush =True )

    def evaluate (min_v ,depth ,vconf ,radius ):
        TP =FP =FN =0 
        for c in cache :
            cps =cp_openings .connection_points (c ["V"],c ["F"],c ["plab"],min_v =min_v ,
            probs =c ["probs"],vertex_conf =vconf ,
            ct_depth_min_mm =depth ,dedupe_mm =10.0 )
            P =np .array ([np .asarray (x ["point"])for x in cps ],float )if cps else np .zeros ((0 ,3 ))
            P =cluster (P ,radius )
            if len (P ):P =P @c ["R"].T +c ["t"]
            tp ,fp ,fn =greedy_match (P ,c ["G"],c ["tol"])
            TP +=tp ;FP +=fp ;FN +=fn 
        pr =TP /max (TP +FP ,1 );rc =TP /max (TP +FN ,1 )
        return pr ,rc ,(2 *pr *rc /max (pr +rc ,1e-9 )),TP ,FP ,FN 

    print ("\nsweeping precision knobs (predictions cached; pure numpy) ...",flush =True )
    rows =[]
    for min_v ,depth ,vconf ,radius in itertools .product ([20 ,40 ,60 ],[1.0 ,2.0 ,3.0 ],
    [0.0 ,0.7 ,0.9 ],[0.0 ,6.0 ,9.0 ,10.0 ,12.0 ,15.0 ]):
        pr ,rc ,f1 ,tp ,fp ,fn =evaluate (min_v ,depth ,vconf ,radius )
        rows .append ((f1 ,pr ,rc ,tp ,fp ,fn ,min_v ,depth ,vconf ,radius ))
    rows .sort (reverse =True )
    print ("\n=== TOP 12 by F1 (min_v / depth / vconf / cluster_radius) ===")
    for f1 ,pr ,rc ,tp ,fp ,fn ,mv ,dp ,vc ,rr in rows [:12 ]:
        print (f"  F1={f1 :.3f} P={pr :.3f} R={rc :.3f} (tp{tp } fp{fp } fn{fn }) | min_v{mv } depth{dp } vconf{vc } cluster{rr }mm")
    base =[r for r in rows if r [6 ]==20 and r [7 ]==1.0 and r [8 ]==0.0 and r [9 ]==0.0 ]
    if base :
        f1 ,pr ,rc ,tp ,fp ,fn =base [0 ][:6 ]
        print (f"\n  baseline (current cp-v3, no cluster/conf): F1={f1 :.3f} P={pr :.3f} R={rc :.3f} (tp{tp } fp{fp } fn{fn })")
        # the PRINCIPLED combo we actually adopt (no per-part fine-tuning): project-default conf mask
        # 0.9 + one-CP-per-terminal clustering 10mm; min_v/depth left at the cp-v3 defaults.
    pr ,rc ,f1 ,tp ,fp ,fn =evaluate (20 ,1.0 ,0.9 ,10.0 )
    print (f"\n  ADOPTED (principled, not fine-tuned): min_v20 depth1 vconf0.9 cluster10mm -> "
    f"F1={f1 :.3f} P={pr :.3f} R={rc :.3f} (tp{tp } fp{fp } fn{fn })")
    best =rows [0 ]
    print (f"\n  BEST (fine-tuned, OVERFIT to 9 parts): F1={best [0 ]:.3f} @ min_v{best [6 ]} depth{best [7 ]} vconf{best [8 ]} cluster{best [9 ]}mm")
    print ("  HONEST: tuned ten 9 PXC clamp-terminals only (all mfg=2). Overfitting risk -- treat as a")
    print ("  CANDIDATE recipe; validate ten more manufacturer-matched parts (both terminal types) before freezing.")
    json .dump ({"rows":[dict (f1 =r [0 ],p =r [1 ],r =r [2 ],tp =r [3 ],fp =r [4 ],fn =r [5 ],
    min_v =r [6 ],depth =r [7 ],vconf =r [8 ],cluster_mm =r [9 ])for r in rows [:20 ]],
    "caveat":"tuned ten 9 PXC clamp terminals (mfg=2 each); candidate not frozen"},
    open ("results/cp_precision_sweep.json","w"),indent =1 )


if __name__ =="__main__":
    main ()
