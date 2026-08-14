"""Test-time augmentation (TTA) for the knngraph CP detector — no retraining.

Idea: a raw-xyz model is pose-sensitive, so its per-vertex heatmap is noisy under
rotation. Run inference ten K random rotations of the part, rotate each prediction
BACK to the original frame, and AVERAGE per vertex. A true connection point fires
in (almost) every rotation -> stays high; a spurious peak fires in only some ->
averages below threshold -> killed. So TTA lifts precision (kills FP) for free.

Because TTA sharpens the heatmap, its OPTIMAL decode threshold is LOWER than the
non-TTA model's (we can afford to fire more, FP is already filtered). So this
computes the TTA-averaged prediction ONCE per part, then sweeps decode thresholds
to find the best operating point — and prints the non-TTA baseline for reference.

Math (rotation R about the centroid c, NumPy row vectors): vertices x'=(x-c)@R.T+c;
a free vector v in the rotated frame maps back by v@R. Heatmap is per-vertex
(rotation-agnostic) -> average directly; offset/direction are vectors -> inverse-
rotate (@R) then average.

Usage:
  python tta_eval.py checkpoints/cp_knn_v8_best.ckpt corpus/ --k 8 --device cpu
  python tta_eval.py checkpoints/cp_knn_v3_best.ckpt corpus/ --k 4 --device cpu --max-gpu-verts 7000
"""
import argparse 
import logging 
import numpy as np 

import json_dataset as jd 
import cp_regressor as cpr 
import cp_targets as ct 
import metrics as mcp 
import train_cp as tc 
import augment as aug 

logging .getLogger ().setLevel (logging .ERROR )# silence per-part infer/subsample spam


def _parse_args ():
    ap =argparse .ArgumentParser (description ="TTA evaluation for the knngraph CP detector")
    ap .add_argument ("ckpt",help ="trained checkpoint (.ckpt)")
    ap .add_argument ("source",help ="corpus directory (same as training)")
    ap .add_argument ("--k",type =int ,default =8 ,help ="number of TTA rotations (default 8)")
    ap .add_argument ("--device",default ="cpu")
    ap .add_argument ("--max-gpu-verts",type =int ,default =7000 ,dest ="max_gpu_verts")
    ap .add_argument ("--val-frac",type =float ,default =0.2 ,dest ="val_frac")
    ap .add_argument ("--test-frac",type =float ,default =0.15 ,dest ="test_frac")
    ap .add_argument ("--seed",type =int ,default =0 )
    ap .add_argument ("--split-group",default ="geometry",dest ="split_group",
    choices =["none","prefix","geometry"],
    help ="MUST match the training run's --split-group or this "
    "reconstructs the wrong val set (default 'geometry' -- "
    "matches the current production training default)")
    ap .add_argument ("--nms",type =float ,default =5.0 ,help ="NMS clearance mm")
    ap .add_argument ("--dist",type =float ,default =5.0 ,help ="match radius mm")
    ap .add_argument ("--thresholds",default ="0.30,0.40,0.50,0.60,0.70",
    help ="comma-separated heatmap thresholds to sweep")
    ap .add_argument ("--votes-list",default ="1,2",dest ="votes_list",
    help ="comma-separated min-votes values to sweep")
    ap .add_argument ("--extra-source",action ="append",default =[],dest ="extra_sources",
    metavar ="DIR",help ="additional corpus directory (can be repeated)")
    ap .add_argument ("--max-bbox-mm",type =float ,default =None ,dest ="max_bbox_mm",
    help ="scope filter (mm): drop parts with bbox diagonal above this "
    "(use ~250 to evaluate ten connectors only, same as training)")
    ap .add_argument ("--keep-prefixes",default =None ,
    help ="scope filter: comma-separated PartNr prefixes to KEEP "
    "(e.g. 'wscaduniverse,PXC' for terminal blocks only)")
    ap .add_argument ("--batched",action ="store_true",
    help ="use batched TTA (all rotations in one GPU forward pass; "
    "faster for small parts, falls back for large)")
    return ap .parse_args ()


def main ():
    args =_parse_args ()
    thresholds =[float (x )for x in args .thresholds .split (",")]
    votes_list =[int (x )for x in args .votes_list .split (",")]

    model ,meta ,_ =cpr .load_model (args .ckpt ,device =args .device )

    parts =list (jd .iter_parts (args .source ))
    for extra in (args .extra_sources or []):
        parts .extend (jd .iter_parts (extra ))
    if args .keep_prefixes :
        prefs =tuple (s .strip ()for s in args .keep_prefixes .split (",")if s .strip ())
        before =len (parts )
        parts =[p for p in parts if str (p .part_nr ).startswith (prefs )]
        print (f"scope filter --keep-prefixes {','.join (prefs )}: kept {len (parts )} "
        f"of {before } parts",flush =True )
    if args .max_bbox_mm :
        before =len (parts )
        def _diag (p ):
            V =np .asarray (p .vertices ,dtype =np .float64 )
            return float (np .linalg .norm (V .max (0 )-V .min (0 )))if len (V )else 0.0 
        parts =[p for p in parts if _diag (p )<=args .max_bbox_mm ]
        print (f"scope filter --max-bbox-mm {args .max_bbox_mm :.0f}: kept {len (parts )} "
        f"of {before } parts",flush =True )
    ids =[p .part_nr for p in parts ]
    gk =jd .build_group_keys (parts ,mode =args .split_group )
    _ ,va ,_ =tc .three_way_split (ids ,val_frac =args .val_frac ,test_frac =args .test_frac ,
    seed =args .seed ,group_keys =gk )
    val =[p for p in parts if p .part_nr in va and p .n_cps >0 ]
    print (f"model={args .ckpt }  val parts={len (val )}  K={args .k }  device={args .device }",
    flush =True )

    rng =np .random .default_rng (0 )
    rots =[np .eye (3 )]+[aug ._random_rotation_matrix (rng )for _ in range (max (0 ,args .k -1 ))]

    def _infer_rotated (V ,R ,patch ,part_nr =None ):
        c =V .mean (0 )
        Vr =(V -c )@R .T +c 
        Vn ,_ ,scale =cpr .normalize_vertices (Vr )
        a =cpr .infer_knngraph (model ,meta ,Vn ,device =args .device ,
        max_gpu_verts =args .max_gpu_verts ,offset_scale =scale ,
        patch =patch ,part_nr =part_nr )
        a =a .copy ()
        a [:,ct .OFFSET ]=a [:,ct .OFFSET ]@R 
        a [:,ct .DIRECTION ]=a [:,ct .DIRECTION ]@R 
        return a 

    def _tta_arr (V ,patch ,part_nr =None ):
        if args .batched :
            return cpr .infer_knngraph_tta_batched (
            model ,meta ,V ,device =args .device ,
            max_gpu_verts =args .max_gpu_verts ,n_aug =args .k ,seed =0 ,patch =patch ,
            part_nr =part_nr )
        acc =None 
        for R in rots :
            a =_infer_rotated (V ,R ,patch ,part_nr =part_nr )
            acc =a if acc is None else (acc +a )
        arr =acc /len (rots )
        n =np .linalg .norm (arr [:,ct .DIRECTION ],axis =1 ,keepdims =True )
        arr [:,ct .DIRECTION ]=arr [:,ct .DIRECTION ]/np .where (n <1e-8 ,1.0 ,n )
        return arr 

    tta_mode ="batched"if args .batched else "serial"
    print (f"running inference (1x base + TTA/{tta_mode } per part) ...",flush =True )
    cache =[]
    for p in val :
        V =np .asarray (p .vertices ,dtype =np .float64 )
        _ ,bp ,bd =jd .dedup_connection_points (p )
        # match training: terminal blocks (wscad+PXC) spatial-patched, others subsampled
        patch =cpr .is_patch_part (p .part_nr )
        cache .append ((V ,bp ,bd ,_infer_rotated (V ,np .eye (3 ),patch ,part_nr =p .part_nr ),
        _tta_arr (V ,patch ,part_nr =p .part_nr )))

    def _score (which ,thr ,votes ):
        reps =[]
        for V ,bp ,bd ,base ,tta in cache :
            arr =base if which =="base"else tta 
            preds =ct .decode_predictions (V ,arr ,heatmap_thresh =thr ,
            nms_radius_mm =args .nms ,min_votes =votes )
            reps .append (mcp .keypoint_report (preds ,bp ,bd ,dist_thresh_mm =args .dist ))
        return tc .aggregate (reps )

    b =_score ("base",0.70 ,1 )
    print (f"\nNO-TTA baseline (thr=0.70, votes=1): F1={100 *b ['micro_f1']:.1f}%  "
    f"P={100 *b ['micro_precision']:.1f}%  R={100 *b ['micro_recall']:.1f}%",flush =True )

    print ("\n=== TTA decode sweep ===")
    print ("  thr  votes |   F1     P      R   |  TP   FP   FN")
    rows =[]
    for v in votes_list :
        for thr in thresholds :
            a =_score ("tta",thr ,v )
            rows .append ((a ["micro_f1"],thr ,v ,a ))
            print (f"  {thr :.2f}   {v }   | {100 *a ['micro_f1']:5.1f}  "
            f"{100 *a ['micro_precision']:5.1f}  {100 *a ['micro_recall']:5.1f} | "
            f"{a ['total_tp']:4d} {a ['total_fp']:4d} {a ['total_fn']:4d}")
    rows .sort (reverse =True )
    f1 ,thr ,v ,a =rows [0 ]
    print (f"\n>>> BEST TTA: thr={thr } votes={v }  F1={100 *f1 :.1f}%  "
    f"P={100 *a ['micro_precision']:.1f}%  R={100 *a ['micro_recall']:.1f}%"
    f"   (vs NO-TTA {100 *b ['micro_f1']:.1f}%)")


if __name__ =="__main__":
    main ()
