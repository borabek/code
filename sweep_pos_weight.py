
# Hyperparameter sweep: pick heat_pos_weight AND the decode operating point from
# data, in one unattended run. For each pos_weight we train the knngraph backbone
# ten a fixed train/val split, keep the best-by-val-F1 checkpoint, then sweep the
# decode grid (shared with sweep_decode) ten that model to find its BEST achievable
# F1. The winner settles the flooding pos_weight and the decode operating point at
# before.
#
# IS THIS WORTH THE COMPUTE? Per the encoding-ceiling finding ([[cp-encoding-ceiling]]),
# the realistic F1 is ~0.90 and the decode is min_votes=1; pos_weight is likely a
# SECOND-ORDER lever now (centernet loss already handles the ~99%-background
# heatmap). The bigger levers -- vertex cap, augmentation, subsample -- are NOT
# swept here. Keep this script for a one-off pos_weight confirmation; pass a single
# value to --pos-weights to skip the retrain-in-loop and just do the decode sweep.
#
# Big parts (>--vert-cap verts) are skipped: they only make eval slow and are not
# needed to rank a hyperparameter. The chosen values are then locked in as the
# train_cp defaults.
#
#   python sweep_pos_weight.py "C:/Users/.../JSON" --device cuda
#   python sweep_pos_weight.py "..." --pos-weights 50 --min-precision 0.5   # decode-only

import os 
import time 
import argparse 
import logging 

import numpy as np 

import json_dataset as jd 
import cp_regressor as cpr 
from train_cp import split_part_ids ,evaluate ,aggregate 
from sweep_decode import (_build_infer ,build_grid ,sweep_grid ,rank_rows ,_score ,
_parse_floats ,_parse_ints )

logger =logging .getLogger (__name__ )


def _infer_builder (device ,max_gpu_verts ):
    def build (m ,mt ):
        def f (s ,pm ):
            return cpr .infer_knngraph (m ,mt ,s ["verts_norm"],device =device ,
            max_gpu_verts =max_gpu_verts ,
            offset_scale =s ["scale"],
            part_nr =s .get ("part_nr"))
        return f 
    return build 


def main (argv =None ):
    ap =argparse .ArgumentParser (description ="Sweep heat_pos_weight + decode for knngraph")
    ap .add_argument ("source")
    ap .add_argument ("--device",default ="cuda")
    ap .add_argument ("--max-parts",type =int ,default =120 )
    ap .add_argument ("--vert-cap",type =int ,default =50000 ,
    help ="skip parts above this many vertices (keeps the sweep fast)")
    ap .add_argument ("--epochs",type =int ,default =150 )
    ap .add_argument ("--eval-every",type =int ,default =25 )
    ap .add_argument ("--max-gpu-verts",type =int ,default =15000 )
    ap .add_argument ("--seed",type =int ,default =0 )
    ap .add_argument ("--val-frac",type =float ,default =0.2 )
    ap .add_argument ("--pos-weights",default ="8,15,30",
    help ="comma-separated heat_pos_weight values (pass one to skip the "
    "retrain-in-loop and just decode-sweep that model)")
    # in-training best-checkpoint tracking decode point -- MUST match deploy, else
    # the kept checkpoint is selected under a decode you have rejected.
    ap .add_argument ("--eval-thresh",type =float ,default =0.2 ,
    help ="heatmap thresh used for best-checkpoint tracking during training")
    ap .add_argument ("--eval-min-votes",type =int ,default =1 ,
    help ="min_votes used for best-checkpoint tracking during training. "
    "Default 1 to match deployment ([[cp-encoding-ceiling]]: GT "
    "recall 0.94 @1 vs 0.79 @2). Tracking at 2 would save a "
    "checkpoint optimised for a decode point you do not ship")
    # decode grid (shared shape with sweep_decode; previously hardcoded here)
    ap .add_argument ("--thresholds",default ="0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6",
    help ="comma-separated heatmap thresholds to try")
    ap .add_argument ("--min-votes-list",default ="1,2,3",
    help ="comma-separated min-votes values to try")
    ap .add_argument ("--nms-list",default ="5,10",
    help ="comma-separated nms clearance (mm) values to try")
    ap .add_argument ("--dist-list",default ="5",
    help ="comma-separated match radii (mm) for scoring (default 5)")
    ap .add_argument ("--metric",choices =["micro_f1","f1"],default ="micro_f1")
    ap .add_argument ("--min-precision",type =float ,default =0.0 ,
    help ="floor: ignore decode configs below this micro-precision when "
    "picking each model's best (guards a high-recall corner win)")
    ap .add_argument ("--heat-loss",choices =["bce","focal","centernet"],
    default ="centernet",dest ="heat_loss",
    help ="MUST match production (train_cp.py default 'centernet' -- "
    "bce/focal flood or collapse on this ~99%%-background "
    "heatmap). Without this, sweep_pos_weight silently trained "
    "under the OLD default 'bce', making its pos_weight "
    "conclusions inapplicable to actual centernet-loss runs")
    ap .add_argument ("--extra-source",action ="append",default =[],dest ="extra_sources",
    help ="additional corpus directory (repeatable)")
    ap .add_argument ("--keep-prefixes",default =None ,
    help ="scope filter: comma-separated PartNr prefixes to KEEP "
    "(e.g. 'wscaduniverse,PXC' for terminal blocks only)")
    ap .add_argument ("--max-bbox-mm",type =float ,default =None ,dest ="max_bbox_mm",
    help ="scope filter (mm): drop parts whose bbox diagonal exceeds "
    "this (use ~250 to match the connector-scoped production "
    "training, else large drives dominate this sweep)")
    ap .add_argument ("--min-recall",type =float ,default =0.0 ,
    help ="floor: ignore decode configs below this micro-recall")
    ap .add_argument ("--split-group",choices =["none","prefix","geometry"],
    default ="geometry",dest ="split_group",
    help ="leakage-safe train/val grouping, default 'geometry' "
    "(matches train_1h.py's production default). This sweep "
    "previously had NO grouping at all (per-part hash split), "
    "which can leak near-duplicate part-family siblings across "
    "train/val and inflate the F1 this sweep optimises against")
    args =ap .parse_args (argv )
    logging .basicConfig (level =logging .INFO ,format ="%(message)s")

    pos_weights =_parse_floats (args .pos_weights )

    # ---- load corpus ONCE, skipping no-CP and oversized parts ----------------
    t0 =time .time ()
    all_sources =[args .source ]+list (args .extra_sources or [])
    parts ,skipped_big =[],0 
    for src in all_sources :
        for p in jd .iter_parts (src ):
            if args .max_parts and len (parts )+skipped_big >=args .max_parts :
                break 
            if p .n_cps ==0 :
                continue 
            if p .n_vertices >args .vert_cap :
                skipped_big +=1 
                continue 
            parts .append (p )
    if args .keep_prefixes :
        prefs =tuple (s .strip ()for s in args .keep_prefixes .split (",")if s .strip ())
        before =len (parts )
        parts =[p for p in parts if str (p .part_nr ).startswith (prefs )]
        logger .info ("scope filter --keep-prefixes %s: kept %d of %d part(s)",
        args .keep_prefixes ,len (parts ),before )
    if args .max_bbox_mm :
        def _diag (p ):
            V =np .asarray (p .vertices ,dtype =np .float64 )
            return float (np .linalg .norm (V .max (0 )-V .min (0 )))if len (V )else 0.0 
        before =len (parts )
        parts =[p for p in parts if _diag (p )<=args .max_bbox_mm ]
        logger .info ("scope filter --max-bbox-mm %.0f: kept %d of %d part(s)",
        args .max_bbox_mm ,len (parts ),before )
    logger .info ("loaded %d usable parts (skipped %d oversized >%d verts) in %.0fs",
    len (parts ),skipped_big ,args .vert_cap ,time .time ()-t0 )

    ids =[p .part_nr for p in parts ]
    group_keys =jd .build_group_keys (parts ,mode =args .split_group )
    train_ids ,val_ids =split_part_ids (ids ,val_frac =args .val_frac ,seed =args .seed ,
    group_keys =group_keys )
    if not train_ids :
        train_ids ={ids [0 ]};val_ids .discard (ids [0 ])
        logger .warning ("tiny corpus: train split fell back to a SINGLE part (%s); "
        "raise --max-parts/--val-frac for a meaningful sweep",ids [0 ])

    train_s ,val_s ,val_m =[],[],[]
    for p in parts :
        s =cpr .prepare_sample (p ,dedup =True )
        if p .part_nr in val_ids :
            val_s .append (s )
            val_m .append ({"part_nr":p .part_nr ,"vertices":p .vertices })
        else :
            train_s .append (s )
    logger .info ("train=%d  val=%d  (val GT points=%d)",
    len (train_s ),len (val_s ),sum (len (s ["gt_points"])for s in val_s ))
    if not val_s :
        raise SystemExit ("no validation parts after the split -- raise --max-parts "
        "or --val-frac (small subsets can split to 0 val)")
    if not train_s :
        raise SystemExit ("no training parts after the split -- raise --max-parts "
        "or lower --val-frac")

    build =_infer_builder (args .device ,args .max_gpu_verts )
    grid =build_grid (_parse_floats (args .thresholds ),_parse_ints (args .min_votes_list ),
    _parse_floats (args .nms_list ),_parse_floats (args .dist_list ))
    logger .info ("decode grid: %d configs per model",len (grid ))

    results =[]
    for pw in pos_weights :
        logger .info ("\n========== heat_pos_weight = %g ==========",pw )
        ckpt =os .path .join (os .environ .get ("TEMP","."),f"sweep_pw{pw :g}.pt")

        def metrics_fn (m ,mt ):
        # track best by the SAME decode point we deploy (min_votes=1 by default)
            return aggregate (evaluate (val_s ,val_m ,build (m ,mt ),
            heatmap_thresh =args .eval_thresh ,
            min_votes =args .eval_min_votes ,
            nms_clearance_mm =5.0 ,progress =False ))

        t1 =time .time ()
        cpr .train_knngraph_regressor (
        train_s ,config ={"global_feat":True },epochs =args .epochs ,
        device =args .device ,max_gpu_verts =args .max_gpu_verts ,
        metrics_fn =metrics_fn ,eval_every =args .eval_every ,
        best_path =ckpt ,last_path =None ,history_path =None ,
        seed =args .seed ,heat_pos_weight =pw ,heat_loss =args .heat_loss ,
        log_every =max (1 ,args .epochs //6 ))
        logger .info ("  trained in %.0fs; decode-sweeping best checkpoint ...",
        time .time ()-t1 )

        # decode sweep ten the BEST checkpoint (not the last-epoch model)
        model ,meta ,_ =cpr .load_model (ckpt ,device =args .device )
        infer =_build_infer ("knngraph",model ,meta ,args .device ,None ,args .max_gpu_verts )
        arrs =[infer (s ,pm )for s ,pm in zip (val_s ,val_m )]
        rows =sweep_grid (arrs ,val_s ,val_m ,grid )
        if not rows :# no decode config scored -> skip this pw
            logger .warning ("  pw=%g: no valid decode configuration; skipping",pw )
            _rm (ckpt )
            continue 
        ranked ,floored =rank_rows (rows ,metric =args .metric ,
        min_precision =args .min_precision ,
        min_recall =args .min_recall )
        if (args .min_precision or args .min_recall )and not floored :
            logger .warning ("  pw=%g: no config met the precision/recall floor; "
            "ranking unfiltered",pw )
        best =ranked [0 ]
        # also report the OLD fixed default (thr=0.3, mv=2, nms=5, dist=5) for contrast
        old =_score (arrs ,val_s ,val_m ,0.3 ,2 ,5.0 ,5.0 )
        best =dict (best )
        best ["old_f1"]=old ["micro_f1"];best ["pw"]=pw 
        results .append (best )
        logger .info ("  pw=%g  BEST micro-F1=%.4f (thr=%.2f votes=%d nms=%.0f dist=%.0f  "
        "prec=%.3f recall=%.3f TP=%d FP=%d FN=%d)  |  old(0.3/2/5/5) F1=%.4f",
        pw ,best ["micro_f1"],best ["heatmap_thresh"],best ["min_votes"],
        best ["nms_clearance_mm"],best ["dist_thresh_mm"],
        best ["micro_precision"],best ["micro_recall"],
        best ["tp"],best ["fp"],best ["fn"],best ["old_f1"])
        _rm (ckpt )

    if not results :
        raise SystemExit ("no pos_weight produced a usable decode configuration")

    results .sort (key =lambda r :r [args .metric ],reverse =True )
    print ("\n================= SWEEP SUMMARY =================")
    print ("  pos_weight |  bestF1  prec   recall | thr votes nms dist | old(0.3)F1")
    for r in results :
        print ("  %9g | %.4f  %.3f  %.3f |%.2f  %d   %3.0f %3.0f | %.4f"
        %(r ["pw"],r ["micro_f1"],r ["micro_precision"],r ["micro_recall"],
        r ["heatmap_thresh"],r ["min_votes"],r ["nms_clearance_mm"],
        r ["dist_thresh_mm"],r ["old_f1"]))
    w =results [0 ]
    print ("\n  WINNER: heat_pos_weight=%g  decode thr=%.2f votes=%d nms=%.0f dist=%.0f  "
    "(val micro-F1=%.4f)"%(w ["pw"],w ["heatmap_thresh"],w ["min_votes"],
    w ["nms_clearance_mm"],w ["dist_thresh_mm"],w ["micro_f1"]))
    print ("\n  NOTE: this F1 is SELECTED-ON-VAL (best pw AND best decode picked on the "
    "same\n        val split it is scored on) -- treat it as an upper bound, not a "
    "held-out\n        number. Confirm the winner with sweep_decode.py --holdout-frac.")


def _rm (path ):
    try :
        os .remove (path )
    except OSError :
        pass 


if __name__ =="__main__":
    main ()
