# -*- coding: utf-8 -*-
"""The UNMEASURED HALF of the product: CP DIRECTION (the robot's approach vector).

Everything so far scored the CP POSITION (arbiter F1 0.520). But `connection_points` also returns a
direction (axis-snapped outward normal), and the robot needs it to insert the wire. The manufacturer
JSON carries an `InsertDirection` for every ConnectionPoint -- so this is directly checkable and has
never been checked.

For each manufacturer CP that our pipeline MATCHES ten position, compare our direction to theirs
(angle in degrees, after rotating ours into the JSON frame). Reports the angle distribution and the
fraction within 15 deg (the project's direction gate) and 30 deg.

Usage: PYTHONPATH=_diffusion_net_repo/src .venv/Scripts/python.exe measure_cp_direction.py
"""
import os ,glob ,json ,hashlib 
import numpy as np 
import torch 
import diffusionnet as D 
import connector3d ,cp_openings ,thesis_remesh 
from cad_eval import align_frames 
from infer_step_cp import step_to_mesh ,load_any ,CKPT 

CE =int (connector3d .CABLE_ENTRY );CT =int (connector3d .CONTACT )
OP ="results/step_infer/ops";JDIR ="C:/Users/DE00024082/Desktop/JSON"
MIN_V ,VCONF ,CLUSTER ,DEPTH =60 ,0.7 ,10.0 ,1.0 # cp-v3.1 product postproc


def main ():
    dev ="cuda"if torch .cuda .is_available ()else "cpu"
    model ,meta ,_ =load_any (CKPT ,dev =dev )
    rows ,angles =[],[]
    for stp in sorted (glob .glob ("_cad_eval_pxc/*.stp")):
        pid =os .path .basename (stp ).split ("_")[1 ]
        jf =os .path .join (JDIR ,f"PXC.{pid }.json")
        if not os .path .exists (jf ):continue 
        j =json .load (open (jf ,encoding ="utf-8-sig"))
        Vj =np .array ([[p ["X"],p ["Y"],p ["Z"]]for p in j ["Graphic3d"]["Points"]],float )
        cps_j =j .get ("ConnectionPoints",[])
        G =np .array ([[c ["Point"]["X"],c ["Point"]["Y"],c ["Point"]["Z"]]for c in cps_j ],float )
        Gd =np .array ([[c ["InsertDirection"]["X"],c ["InsertDirection"]["Y"],c ["InsertDirection"]["Z"]]
        for c in cps_j ],float )
        Vr ,Fr =step_to_mesh (stp )
        V ,F =thesis_remesh .remesh_uniform (Vr ,Fr ,target =6000 )
        V =np .ascontiguousarray (V ,np .float64 );F =np .ascontiguousarray (F ,np .int64 )
        lab ,probs =D .predict (model ,meta ,V ,F ,device =dev ,op_cache_dir =OP ,return_probs =True )
        cps =cp_openings .connection_points (V ,F ,np .asarray (lab ),min_v =MIN_V ,probs =np .asarray (probs ),
        vertex_conf =VCONF ,ct_depth_min_mm =DEPTH ,cluster_mm =CLUSTER )
        if not cps :continue 
        R ,t ,_ =align_frames (Vr ,Vj )
        P =np .array ([c ["point"]for c in cps ])@R .T +t 
        Dp =np .array ([c ["direction"]for c in cps ])@R .T # directions rotate only
        diag =float (np .linalg .norm (Vj .max (0 )-Vj .min (0 )));tol =max (3.0 ,0.06 *diag )
        for gi ,(g ,gd )in enumerate (zip (G ,Gd )):
            d =np .linalg .norm (P -g ,axis =1 );k =int (np .argmin (d ))
            if d [k ]>tol :continue # unmatched position -> skip
            a =Dp [k ]/(np .linalg .norm (Dp [k ])+1e-9 )
            b =gd /(np .linalg .norm (gd )+1e-9 )
            ang =float (np .degrees (np .arccos (np .clip (a @b ,-1 ,1 ))))
            angles .append (ang )
            rows .append ({"part_id":pid ,"mfg_cp":gi ,"pos_err_mm":round (float (d [k ]),2 ),
            "angle_deg":round (ang ,1 ),
            "ours":[round (float (x ),2 )for x in a ],"mfg":[round (float (x ),2 )for x in b ]})
        print (f"  {pid }: matched {sum (1 for r in rows if r ['part_id']==pid )}/{len (G )}",flush =True )
    if not angles :
        print ("no matched CPs");return 
    A =np .array (angles )
    print (f"\n=== CP DIRECTION vs manufacturer InsertDirection (n={len (A )} matched CPs) ===")
    print (f"  median {np .median (A ):.1f} deg | mean {A .mean ():.1f} | min {A .min ():.1f} | max {A .max ():.1f}")
    for th in (15 ,30 ,45 ,90 ):
        print (f"  within {th :3d} deg: {(A <=th ).sum ():2d}/{len (A )} ({100 *(A <=th ).mean ():.0f}%)")
    print (f"  ~opposite (>150 deg): {(A >150 ).sum ()}/{len (A )}  <- sign/outward-orientation errors")
    json .dump ({"metric":"CP approach-direction vs manufacturer InsertDirection",
    "ckpt":CKPT ,"n_matched":len (A ),
    "median_deg":round (float (np .median (A )),1 ),
    "within_15_deg":int ((A <=15 ).sum ()),"within_30_deg":int ((A <=30 ).sum ()),
    "opposite_gt150":int ((A >150 ).sum ()),"per_cp":rows },
    open ("results/cp_direction_eval.json","w"),indent =1 )
    print ("  -> results/cp_direction_eval.json")


if __name__ =="__main__":
    main ()
