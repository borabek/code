# -*- coding: utf-8 -*-
"""IN-DOMAIN manufacturer-CP detector ten the JSON corpus (audit item 3 = the scaling path).
Reuses the proven DiffusionNet + cluster recipe: per-vertex BINARY label (1 if a vertex is within
`--radius` mm of a manufacturer CP, else 0) -> DiffusionNet segmentation -> connected CP components
-> centroids = predicted CP points -> match to manufacturer CPs. This turns the 11,927 real
manufacturer CPs into a trainable, IN-DOMAIN CP benchmark (metric #3, in-domain).

Split from json_cp_split.json (SHA-deduped 341/74/64). Operators cached in results/json_cp/ops.
NOTE: manufacturer CP points sit ~4.7mm off-surface (standoff); the predicted centroid is ten the
surface, so `--match` is a bit looser (7mm) to absorb the standoff.

Usage: PYTHONPATH=_diffusion_net_repo/src .venv/Scripts/python.exe train_json_cp.py [--limit N --epochs E]
"""
import argparse ,os ,json 
import numpy as np 
from scipy .spatial import cKDTree 
import torch 
import diffusionnet as D 
import connector3d ,metrics 
import json_cp_benchmark as B 

OPS ="results/json_cp/ops";CKPT ="results/json_cp/best.pt"


def labels_for (V ,cp_pts ,radius ):
    if len (cp_pts )==0 :
        return np .zeros (len (V ),int )
    d =cKDTree (cp_pts ).query (V )[0 ]
    return (d <=radius ).astype (int )


def cluster (V ,F ,pred ,min_v ):
    frags =connector3d .build_fragments (V ,F ,pred ,min_vertices =min_v ,skip_label =0 )
    return np .array ([f .center for f in frags if int (f .label )==1 ]or []).reshape (-1 ,3 )


def _to_dev (ops ,dev ):
    return {k :(v .to (dev )if hasattr (v ,"to")else v )for k ,v in ops .items ()}


def prep (parts ,radius ,k_eig ,dev ):
# keep operators ten CPU (a 4GB GPU cannot hold all 415 parts' operators at before -> WDDM
# sysmem-spill = 100% util at ~29W, pathologically slow). Move per-part in the loop instead.
    out =[]
    for p in parts :
        V ,F =p ["V"],p ["F"]
        try :
            ops =D .precompute_operators (V ,F ,k_eig ,op_cache_dir =OPS )
        except Exception :
            continue 
        lab =labels_for (V ,p ["cp_points"],radius )
        out .append ({"pid":p ["part_id"],"V":V ,"F":F ,"cp":p ["cp_points"],
        "ops":ops ,"lab":torch .tensor (lab ,dtype =torch .long )})
    return out 


def evaluate (model ,meta ,data ,match ,min_v ):
    model .eval ();TP =FP =FN =0 
    with torch .no_grad ():
        for d in data :
            ops =_to_dev (d ["ops"],next (model .parameters ()).device )
            x =D ._model_input (ops ,meta )
            out =D ._forward (model ,ops ,x )
            pred =out .argmax (-1 ).cpu ().numpy ()
            pr =cluster (d ["V"],d ["F"],pred ,min_v )
            gt =d ["cp"]
            if len (pr )and len (gt ):
                m ,up ,ug =metrics .match_predictions (pr ,gt ,match );tp ,fp ,fn =len (m ),len (up ),len (ug )
            else :
                tp ,fp ,fn =0 ,len (pr ),len (gt )
            TP +=tp ;FP +=fp ;FN +=fn 
    f1 =2 *TP /max (2 *TP +FP +FN ,1 )
    return f1 ,TP /max (TP +FP ,1 ),TP /max (TP +FN ,1 ),(TP ,FP ,FN )


def main ():
    ap =argparse .ArgumentParser ()
    ap .add_argument ("--radius",type =float ,default =6.0 );ap .add_argument ("--match",type =float ,default =7.0 )
    ap .add_argument ("--epochs",type =int ,default =120 );ap .add_argument ("--limit",type =int ,default =0 )
    ap .add_argument ("--k-eig",type =int ,default =64 );ap .add_argument ("--min-v",type =int ,default =15 )
    ap .add_argument ("--lr",type =float ,default =1e-3 );ap .add_argument ("--cp-weight",type =float ,default =8.0 )
    a =ap .parse_args ()
    os .makedirs ("results/json_cp",exist_ok =True )
    dev ="cuda"if torch .cuda .is_available ()else "cpu"
    if not os .path .exists (B .SPLIT_FILE ):
        B .build_split ()
    tr =B .load_split ("train");va =B .load_split ("val")
    if a .limit :
        tr =tr [:a .limit ];va =va [:max (3 ,a .limit //3 )]
    print (f"prep operators: {len (tr )} train + {len (va )} val (k_eig={a .k_eig }) ...",flush =True )
    tr =prep (tr ,a .radius ,a .k_eig ,dev );va =prep (va ,a .radius ,a .k_eig ,dev )
    print (f"prepared {len (tr )} train {len (va )} val",flush =True )

    cfg ={"input_features":"xyz","loss":"nll","n_diffusion_blocks":3 ,"width":64 ,
    "n_eig":a .k_eig ,"dropout":0.3 }
    model ,meta =D .build_diffusionnet (cfg ,n_classes =2 )
    model =model .to (dev )
    w =torch .tensor ([1.0 ,a .cp_weight ],dtype =torch .float32 ,device =dev )
    opt =torch .optim .Adam (model .parameters (),lr =a .lr )
    best =-1 
    for ep in range (1 ,a .epochs +1 ):
        model .train ();tot =0.0 
        for di in np .random .permutation (len (tr )):
            d =tr [di ];opt .zero_grad ()
            ops =_to_dev (d ["ops"],dev );lab =d ["lab"].to (dev )
            x =D ._model_input (ops ,meta )
            out =D ._forward (model ,ops ,x )
            loss =D ._compute_loss (out ,lab ,w ,meta ,cfg )
            loss .backward ();opt .step ();tot +=float (loss )
        print (f"ep{ep :3d} loss{tot /max (len (tr ),1 ):.4f}",flush =True )
        if ep %10 ==0 or ep ==a .epochs :
            f1 ,pr ,rc ,cnt =evaluate (model ,meta ,va ,a .match ,a .min_v )
            print (f"  -> val F1={f1 :.3f} P={pr :.3f} R={rc :.3f} {cnt }",flush =True )
            if f1 >best :
                best =f1 ;torch .save ({"state":model .state_dict (),"cfg":cfg ,"meta":meta ,"val_f1":f1 },CKPT )
    print (f"DONE best val F1={best :.3f} -> {CKPT }",flush =True )


if __name__ =="__main__":
    main ()
