# -*- coding: utf-8 -*-
"""HUMAN-CP CONSISTENCY: model-predicted CPs vs CPs derived from HUMAN region labels.

WHY THIS EXISTS: the manufacturer arbiter (measure_cp_defs.py) has only 9 parts / 18 CPs -- too
small to tell a real gain from seed noise (+-0.10), and the JSON-mesh path to more parts FAILS the
domain check (scratchpad/json_domain_check: F1 0.520 -> 0.091). But we DO have human region labels
ten the queue parts (batch 1 _label_targets = 20, batch 2 _label_targets_2 as they arrive). Deriving
a CP POINT from each human-marked opening (the same v_o opening-midpoint the model uses) gives a
20-then-58-part measurement set -- not manufacturer-exact, but ~3x more parts and, crucially, in the
SAME mesh frame as the prediction (NO align_frames, so no frame-residual error).

WHAT IT MEASURES: per part, greedy-match predicted CPs to human-region CPs within a size-relative
tolerance -> P / R / F1, plus approach-direction agreement ten matched pairs.

HELD-OUT NOTE: the product `selftrain_120` was trained WITHOUT these human labels (71 corpus + 122
pseudo), so measuring it here IS held-out. Models trained WITH --partial-dir (human20*) have SEEN
these parts -> in-sample for them; the script flags that so we never quote an in-sample number as
generalisation.

Usage: PYTHONPATH=_diffusion_net_repo/src .venv/Scripts/python.exe measure_human_cp.py \
         [--ckpt results/seg_extra/selftrain_120.pt] [--dirs _label_targets _label_targets_2] [--outward-min 0.0]
"""
import os ,sys ,glob ,json ,argparse 
import numpy as np ,torch 
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
import diffusionnet as D ,connector3d ,cp_openings 
from region_label_helper import load_obj 
from infer_step_cp import load_any 

CE =int (connector3d .CABLE_ENTRY );CT =int (connector3d .CONTACT )
OP ="results/step_infer/ops"


def greedy_match (P ,G ,tol ):
    """Return (tp, fp, fn, pairs) where pairs are (pred_idx, gt_idx) matched within tol."""
    if not len (P )or not len (G ):
        return 0 ,len (P ),len (G ),[]
    dm =np .linalg .norm (P [:,None ,:]-G [None ,:,:],axis =2 )
    order =sorted ((dm [i ,j ],i ,j )for i in range (len (P ))for j in range (len (G )))
    up ,ug ,pairs =set (),set (),[]
    for d ,i ,j in order :
        if d >tol :
            break 
        if i in up or j in ug :
            continue 
        up .add (i );ug .add (j );pairs .append ((i ,j ))
    return len (pairs ),len (P )-len (pairs ),len (G )-len (pairs ),pairs 


def human_cps (V ,F ,L ):
    """Derive GT CP points+dirs from a human REGION label file. The annotator marked openings as
    class 3 (connection region); one CP per connected marked region. GT settings: no confidence
    mask (no probs), no per-terminal clustering (never cluster GT), small min_v to drop stray specks.
    Directions use the same axis-snap outward convention as prediction."""
    return cp_openings .connection_points (V ,F ,L ,min_v =20 ,classes =(CE ,),dedupe_mm =0.0 ,
    axis_snap =True ,cluster_mm =0.0 )


def main ():
    ap =argparse .ArgumentParser ()
    ap .add_argument ("--ckpt",nargs ="+",default =["results/seg_extra/selftrain_120.pt"],
    help ="one checkpoint, or several -> their softmax probabilities are AVERAGED "
    "(seed ensemble; only ensemble seeds trained on the SAME data -- mixing "
    "differently-trained models measured WORSE, arbiter 0.778 -> 0.500)")
    ap .add_argument ("--dirs",nargs ="+",default =["_label_targets","_label_targets_2"])
    ap .add_argument ("--cluster-mm",type =float ,default =10.0 ,
    help ="per-terminal merge radius for PREDICTIONS. 10.0 was tuned on the 9-part "
    "arbiter (simple, far-apart 2-CP terminals); on dense terminals whose "
    "openings are <10mm apart it MERGES distinct CPs into a midpoint that "
    "matches neither -> silently destroys recall. 0 = no clustering.")
    ap .add_argument ("--vertex-conf",type =float ,default =0.7 ,
    help ="per-vertex confidence mask. Its JOB is to erode the low-confidence bridge "
    "the model paints BETWEEN two adjacent openings so they split into separate "
    "CPs. 0.7 was tuned on the 9-part arbiter whose openings are far apart (no "
    "bridge to erode) -> may be far too low for dense terminals.")
    ap .add_argument ("--min-v",type =int ,default =60 ,
    help ="min vertices per predicted component; too high kills the small fragments "
    "that appear first a merged blob is split.")
    ap .add_argument ("--split-ratio",type =float ,default =0.0 ,
    help ="split a fragment that merged a ROW of adjacent openings into "
    "k=round(length/(ratio*width)) slabs. 0=off. Lower ratio = more splits.")
    ap .add_argument ("--outward-min",type =float ,default =0.0 ,
    help ="apply the candidate precision gate to predictions (0 = off = product)")
    ap .add_argument ("--device",default ="cuda"if torch .cuda .is_available ()else "cpu")
    ap .add_argument ("--out",default ="results/human_cp_eval.json")
    a =ap .parse_args ()

    # parts this ckpt may have been trained ten (batch-1 dir is used as --partial-dir for human20*)
    trained_dirs ={"_label_targets","_label_targets_dilated"}
    in_sample =any (os .path .basename (c ).startswith ("human20")for c in a .ckpt )

    models =[load_any (c ,dev =a .device )for c in a .ckpt ]
    if len (models )>1 :
        print (f"  ENSEMBLE of {len (models )}: {[os .path .basename (c )for c in a .ckpt ]}",flush =True )
    rows =[];TP =FP =FN =0 ;dirs_deg =[]
    for root in a .dirs :
        for d in sorted (glob .glob (os .path .join (root ,"*"))):
            if not os .path .isdir (d ):
                continue 
            pid =os .path .basename (os .path .normpath (d ))
            of =os .path .join (d ,f"{pid }.obj")
            lf =os .path .join (d ,f"{pid }.labels.txt")# the ACTUAL labels, not the template
            if not (os .path .exists (of )and os .path .exists (lf )):
                continue 
            L =np .array ([int (x )for x in open (lf ).read ().split ()],np .int64 )
            V ,F =load_obj (of )
            if len (L )!=len (V )or not (L ==CE ).any ():
                continue # unlabelled / all-Housing -> nothing to score
            V =np .ascontiguousarray (V ,np .float64 );F =np .ascontiguousarray (F ,np .int64 )
            G_cps =human_cps (V ,F ,L )
            G =np .array ([np .asarray (c ["point"])for c in G_cps ],float )if G_cps else np .zeros ((0 ,3 ))
            acc =None 
            for model ,meta ,_ in models :# softmax-average across seeds (1 model = plain predict)
                _ ,pb =D .predict (model ,meta ,V ,F ,device =a .device ,op_cache_dir =OP ,return_probs =True )
                pb =np .asarray (pb ,float );acc =pb if acc is None else acc +pb 
            probs =acc /len (models );plab =probs .argmax (-1 )
            P_cps =cp_openings .connection_points (V ,F ,np .asarray (plab ),min_v =a .min_v ,classes =(CE ,CT ),
            dedupe_mm =10.0 ,probs =np .asarray (probs ),vertex_conf =a .vertex_conf ,
            ct_depth_min_mm =1.0 ,cluster_mm =a .cluster_mm ,outward_min =a .outward_min ,
            split_ratio =a .split_ratio )
            P =np .array ([np .asarray (c ["point"])for c in P_cps ],float )if P_cps else np .zeros ((0 ,3 ))
            diag =float (np .linalg .norm (V .max (0 )-V .min (0 )));tol =max (3.0 ,0.06 *diag )
            tp ,fp ,fn ,pairs =greedy_match (P ,G ,tol )
            for pi ,gi in pairs :# direction agreement ten matched pairs
                dp =np .asarray (P_cps [pi ]["direction"],float );dg =np .asarray (G_cps [gi ]["direction"],float )
                c =float (np .clip (dp @dg /(np .linalg .norm (dp )*np .linalg .norm (dg )+1e-9 ),-1 ,1 ))
                dirs_deg .append (np .degrees (np .arccos (c )))
            TP +=tp ;FP +=fp ;FN +=fn 
            rows .append ({"part_id":pid ,"human_cps":len (G ),"pred_cps":len (P ),
            "tp":tp ,"fp":fp ,"fn":fn ,"tol_mm":round (tol ,1 )})
            print (f"  {pid :14s} human={len (G )} pred={len (P )}  tp{tp } fp{fp } fn{fn }",flush =True )

    pr =TP /max (TP +FP ,1 );rc =TP /max (TP +FN ,1 );f1 =2 *pr *rc /max (pr +rc ,1e-9 )
    md =float (np .median (dirs_deg ))if dirs_deg else None 
    print (f"\n=== HUMAN-CP CONSISTENCY ({len (rows )} parts, {TP +FN } human CPs) ===")
    print (f"  ckpt: {', '.join (os .path .basename (c )for c in a .ckpt )}  outward_min={a .outward_min }"
    +("   [IN-SAMPLE: ckpt trained on these -- not generalisation]"if in_sample else "   [held-out]"))
    print (f"  F1={f1 :.3f}  P={pr :.3f}  R={rc :.3f}  (tp{TP } fp{FP } fn{FN })"
    +(f"  | dir median {md :.1f} deg"if md is not None else ""))
    os .makedirs ("results",exist_ok =True )
    json .dump ({"ckpt":a .ckpt ,"in_sample":in_sample ,"outward_min":a .outward_min ,
    "n_parts":len (rows ),"n_human_cps":TP +FN ,
    "precision":round (pr ,3 ),"recall":round (rc ,3 ),"f1":round (f1 ,3 ),
    "dir_median_deg":md ,"per_part":rows },open (a .out ,"w"),indent =1 )
    print (f"  -> {a .out }")


if __name__ =="__main__":
    main ()
