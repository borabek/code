"""Per-CP error analysis: WHY does the model miss certain connection points?

Runs inference ten the val set, then for every GT CP that was missed (FN) extracts
local geometry features to answer: "Is the recall ceiling from DATA (hard geometry the
model can never see) or from the MODEL (learnable but not yet learned)?"

Features per GT CP:
  local_density     -- vertices within 5 mm of the CP; low = CP region is sparse
  local_concavity   -- mean PCA curvature at k nearest vertices; high = deep recess
  normal_scatter    -- std of surface normals near CP; high = sharp discontinuity
  nearest_edge_mm   -- distance to nearest mesh boundary vertex (edge of the part)
  nearest_other_cp  -- distance to the next GT CP; isolated vs. cluster
  max_heat_near_cp  -- the model's peak heatmap value within 5 mm of the CP
                       (high = model saw it but threshold too high; low = blind spot)

Output: text report + optional JSON for deeper analysis.

Usage:
  python diag_cp_errors.py checkpoints/cp_knn_v15_best.ckpt corpus/ --device cuda
  python diag_cp_errors.py ckpt.ckpt corpus/ --extra-source wscad_corpus \\
      --keep-prefixes wscaduniverse,PXC --split-group geometry --json diag.json
"""
import argparse 
import json 
import logging 
import sys 

import numpy as np 

import json_dataset as jd 

logging .basicConfig (level =logging .ERROR ,format ="%(message)s")


def _parse_args ():
    ap =argparse .ArgumentParser (description ="Per-CP error analysis for the knngraph detector")
    ap .add_argument ("ckpt",help ="trained checkpoint (.ckpt)")
    ap .add_argument ("source",help ="corpus directory")
    ap .add_argument ("--device",default ="cpu")
    ap .add_argument ("--max-gpu-verts",type =int ,default =7000 ,dest ="max_gpu_verts")
    ap .add_argument ("--val-frac",type =float ,default =0.2 ,dest ="val_frac")
    ap .add_argument ("--test-frac",type =float ,default =0.15 ,dest ="test_frac")
    ap .add_argument ("--seed",type =int ,default =0 )
    ap .add_argument ("--split-group",default ="geometry",dest ="split_group",
    choices =["none","prefix","geometry"])
    ap .add_argument ("--extra-source",action ="append",default =[],dest ="extra_sources",
    metavar ="DIR")
    ap .add_argument ("--keep-prefixes",default =None )
    ap .add_argument ("--max-bbox-mm",type =float ,default =None ,dest ="max_bbox_mm",
    help ="scope filter (mm): drop parts with bbox diagonal above this "
    "(use ~250 to match a --max-bbox-mm training run)")
    ap .add_argument ("--thr",type =float ,default =0.50 ,help ="decode heatmap threshold")
    ap .add_argument ("--nms",type =float ,default =5.0 )
    ap .add_argument ("--dist",type =float ,default =5.0 ,help ="match radius mm")
    ap .add_argument ("--radius",type =float ,default =5.0 ,
    help ="local geometry neighbourhood radius mm")
    ap .add_argument ("--json",default =None ,dest ="json_out",
    help ="save per-CP records to this JSON file")
    ap .add_argument ("--split",default ="val",choices =["val","test","all"])
    return ap .parse_args ()


    # ---------------------------------------------------------------------------
    # geometry helpers
    # ---------------------------------------------------------------------------

def _local_density (V ,cp_pt ,radius ):
    """Number of vertices within `radius` mm of cp_pt."""
    return int (((V -cp_pt )**2 ).sum (1 ).pipe if False else 
    (np .linalg .norm (V -cp_pt ,axis =1 )<=radius ).sum ())


def _local_concavity (V ,cp_pt ,radius ,k =16 ):
    """Mean PCA curvature (λ0/Σλ) of the k nearest vertices to cp_pt."""
    dists =np .linalg .norm (V -cp_pt ,axis =1 )
    knn_idx =np .argsort (dists )[:min (k *3 ,len (V ))]
    Vk =V [knn_idx ]
    Q =Vk -Vk .mean (0 ,keepdims =True )
    cov =(Q .T @Q )/max (1 ,len (Vk ))
    w =np .linalg .eigvalsh (cov )# ascending
    return float (w [0 ]/(w .sum ()+1e-8 ))


def _normal_scatter (V ,cp_pt ,radius ,k =16 ):
    """Std of local PCA normals near cp_pt (normal discontinuity signal)."""
    dists =np .linalg .norm (V -cp_pt ,axis =1 )
    nbrs =np .where (dists <=radius )[0 ]
    if len (nbrs )<4 :
        return float ("nan")
    from cp_regressor import _knn_graph ,_pca_normals 
    try :
        import torch 
        Vn =V [nbrs ]
        nbr_idx =_knn_graph (Vn ,k =min (k ,len (Vn )-1 ))
        nrms =_pca_normals (Vn ,nbr_idx )# (n, 3)
        return float (nrms .std (0 ).mean ())
    except Exception :
        return float ("nan")


def _boundary_vertices (V ,F ):
    """Return indices of boundary (mesh-edge) vertices, or [] if no faces."""
    if F is None or len (F )==0 :
        return np .array ([],dtype =int )
    from collections import Counter 
    edge_count =Counter ()
    for tri in F :
        for a ,b in [(tri [0 ],tri [1 ]),(tri [1 ],tri [2 ]),(tri [2 ],tri [0 ])]:
            edge_count [tuple (sorted ((a ,b )))]+=1 
    boundary_verts =set ()
    for (a ,b ),cnt in edge_count .items ():
        if cnt ==1 :
            boundary_verts .add (a );boundary_verts .add (b )
    return np .array (list (boundary_verts ),dtype =int )


def _nearest_edge_dist (V ,F ,cp_pt ):
    """Distance from cp_pt to the nearest boundary vertex."""
    bv =_boundary_vertices (V ,F )
    if len (bv )==0 :
        return float ("nan")
    return float (np .linalg .norm (V [bv ]-cp_pt ,axis =1 ).min ())


def _max_heat_near (V_full ,heat_full ,cp_pt ,radius ):
    """Peak heatmap value the model produced within `radius` mm of a GT CP."""
    dists =np .linalg .norm (V_full -cp_pt ,axis =1 )
    near =np .where (dists <=radius )[0 ]
    if len (near )==0 :
        return float (heat_full .max ())# fallback: global max
    return float (heat_full [near ].max ())


    # ---------------------------------------------------------------------------
    # per-CP record extraction
    # ---------------------------------------------------------------------------

def _extract_cp_features (p ,arr ,gt_pts ,gt_dirs ,preds ,matched_gt ,radius ):
    """For each GT CP return a feature dict (TP / FN label + geometry)."""
    import cp_targets as ct 
    V =np .asarray (p .vertices ,dtype =np .float64 )
    heat =arr [:,ct .HEATMAP ]
    try :
        F =np .asarray (p .faces ,dtype =np .int64 )if hasattr (p ,"faces")else None 
    except Exception :
        F =None 

    records =[]
    for gi ,(gpt ,gdir )in enumerate (zip (gt_pts ,gt_dirs )):
        label ="TP"if gi in matched_gt else "FN"
        ld =_local_density (V ,gpt ,radius )
        lc =_local_concavity (V ,gpt ,radius )
        # nearest OTHER GT CP
        others =[j for j in range (len (gt_pts ))if j !=gi ]
        if others :
            nearest_cp =float (np .linalg .norm (
            gt_pts [np .array (others )]-gpt ,axis =1 ).min ())
        else :
            nearest_cp =float ("nan")
        ned =_nearest_edge_dist (V ,F ,gpt )
        mh =_max_heat_near (V ,heat ,gpt ,radius )
        records .append ({
        "part_nr":str (p .part_nr ),
        "cp_idx":gi ,
        "label":label ,
        "local_density":ld ,
        "local_concavity":lc ,
        "nearest_other_cp_mm":nearest_cp ,
        "nearest_edge_mm":ned ,
        "max_heat_near":mh ,
        "cp_x":float (gpt [0 ]),"cp_y":float (gpt [1 ]),"cp_z":float (gpt [2 ]),
        })
        # FP records
    for pi ,pred in enumerate (preds ):
        if pi not in [m [0 ]for m in []]:# all preds that are FP
            pass 
    return records 


    # ---------------------------------------------------------------------------
    # reporting
    # ---------------------------------------------------------------------------

def _pct (vals ,p ):
    v =[x for x in vals if x ==x ]# drop nan
    return float (np .percentile (v ,p ))if v else float ("nan")


def _report_group (label ,records ):
    n =len (records )
    if n ==0 :
        print (f"  {label }: 0 CPs");return 
    ld =[r ["local_density"]for r in records ]
    lc =[r ["local_concavity"]for r in records ]
    ncp =[r ["nearest_other_cp_mm"]for r in records ]
    ned =[r ["nearest_edge_mm"]for r in records ]
    mh =[r ["max_heat_near"]for r in records ]
    print (f"  {label } ({n } CPs):")
    print (f"    local_density      p25={_pct (ld ,25 ):.0f}  median={_pct (ld ,50 ):.0f}  p75={_pct (ld ,75 ):.0f} verts")
    print (f"    local_concavity    p25={_pct (lc ,25 ):.3f}  median={_pct (lc ,50 ):.3f}  p75={_pct (lc ,75 ):.3f}")
    print (f"    nearest_other_cp   p25={_pct (ncp ,25 ):.1f}  median={_pct (ncp ,50 ):.1f}  p75={_pct (ncp ,75 ):.1f} mm")
    print (f"    nearest_edge       p25={_pct (ned ,25 ):.1f}  median={_pct (ned ,50 ):.1f}  p75={_pct (ned ,75 ):.1f} mm")
    print (f"    max_heat_near_cp   p25={_pct (mh ,25 ):.3f}  median={_pct (mh ,50 ):.3f}  p75={_pct (mh ,75 ):.3f}")

    # Key diagnostic: high max_heat_near in FNs = threshold too high, not blind
    if label =="FN":
        blind =sum (1 for x in mh if x <0.20 )
        threshold_issue =sum (1 for x in mh if x >=0.20 )
        print (f"\n    *** FN diagnosis ***")
        print (f"    max_heat < 0.20 (MODEL BLIND):      {blind }/{n }  ({100 *blind /n :.0f}%)")
        print (f"    max_heat >= 0.20 (THRESHOLD ISSUE): {threshold_issue }/{n }  ({100 *threshold_issue /n :.0f}%)")
        print (f"    => {'DATA-limited'if blind >threshold_issue else 'THRESHOLD-limited'}: "
        f"{'more blind spots than rescuable with lower thr'if blind >threshold_issue else 'lower threshold could recover many FNs'}")


def main ():
    args =_parse_args ()
    import json_dataset as jd 
    import cp_regressor as cpr 
    import cp_targets as ct 
    import metrics as mcp 
    import train_cp as tc 

    model ,meta ,_ =cpr .load_model (args .ckpt ,device =args .device )

    parts =list (jd .iter_parts (args .source ))
    for extra in args .extra_sources :
        parts .extend (jd .iter_parts (extra ))
    if args .keep_prefixes :
        prefs =tuple (s .strip ()for s in args .keep_prefixes .split (",")if s .strip ())
        parts =[p for p in parts if str (p .part_nr ).startswith (prefs )]
    if args .max_bbox_mm :
        def _diag (p ):
            V =np .asarray (p .vertices ,dtype =np .float64 )
            return float (np .linalg .norm (V .max (0 )-V .min (0 )))if len (V )else 0.0 
        before =len (parts )
        parts =[p for p in parts if _diag (p )<=args .max_bbox_mm ]
        print (f"scope filter --max-bbox-mm {args .max_bbox_mm :.0f}: kept {len (parts )} "
        f"of {before } parts")
    ids =[p .part_nr for p in parts ]
    gk =jd .build_group_keys (parts ,mode =args .split_group )
    tr ,va ,te =tc .three_way_split (ids ,val_frac =args .val_frac ,
    test_frac =args .test_frac ,seed =args .seed ,
    group_keys =gk )
    if args .split =="val":
        keep =va 
    elif args .split =="test":
        keep =te 
    else :
        keep =set (tr )|set (va )|set (te )
    eval_parts =[p for p in parts if p .part_nr in keep and p .n_cps >0 ]
    print (f"model={args .ckpt }")
    print (f"split={args .split }  parts={len (eval_parts )}  thr={args .thr }  dist={args .dist }mm")
    print ("running inference ...",flush =True )

    all_records =[]
    total_tp =total_fp =total_fn =0 
    for p in eval_parts :
        V =np .asarray (p .vertices ,dtype =np .float64 )
        _ ,gt_pts ,gt_dirs =jd .dedup_connection_points (p )
        if len (gt_pts )==0 :
            continue 
        patch =cpr .is_patch_part (p .part_nr )
        Vn ,_ ,scale =cpr .normalize_vertices (V )
        arr =cpr .infer_knngraph (model ,meta ,Vn ,device =args .device ,
        max_gpu_verts =args .max_gpu_verts ,
        offset_scale =scale ,patch =patch ,
        part_nr =p .part_nr )
        preds =ct .decode_predictions (V ,arr ,heatmap_thresh =args .thr ,
        nms_radius_mm =args .nms ,min_votes =1 )
        rep =mcp .keypoint_report (preds ,gt_pts ,gt_dirs ,
        dist_thresh_mm =args .dist )
        total_tp +=rep ["tp"];total_fp +=rep ["fp"];total_fn +=rep ["fn"]

        # match bookkeeping
        from metrics import match_predictions 
        pred_pts_arr =(np .array ([p2 ["point"]for p2 in preds ],dtype =float ).reshape (-1 ,3 )
        if preds else np .zeros ((0 ,3 )))
        matches ,_ ,_ =match_predictions (pred_pts_arr ,gt_pts ,args .dist )
        matched_gt_set ={gj for _ ,gj ,_ in matches }
        all_records .extend (_extract_cp_features (
        p ,arr ,gt_pts ,gt_dirs ,preds ,matched_gt_set ,args .radius ))

    prec =total_tp /(total_tp +total_fp )if (total_tp +total_fp )else 0 
    rec =total_tp /(total_tp +total_fn )if (total_tp +total_fn )else 0 
    f1 =2 *prec *rec /(prec +rec )if (prec +rec )else 0 
    print (f"\nOverall: TP={total_tp } FP={total_fp } FN={total_fn }  "
    f"P={100 *prec :.1f}%  R={100 *rec :.1f}%  F1={100 *f1 :.1f}%\n")

    tp_rec =[r for r in all_records if r ["label"]=="TP"]
    fn_rec =[r for r in all_records if r ["label"]=="FN"]
    print ("=== Geometry comparison: TP (matched) vs FN (missed) ===\n")
    _report_group ("TP",tp_rec )
    print ()
    _report_group ("FN",fn_rec )

    # per-prefix breakdown (jd.prefix_family: single source of truth, also used
    # by tune_thresholds.py / predict.py for per-family decode thresholds)
    prefixes =sorted ({jd .prefix_family (r ["part_nr"])for r in all_records })
    if len (prefixes )>1 :
        print ("\n=== Per-prefix FN rate ===")
        for pfx in prefixes :
            pfx_recs =[r for r in all_records if jd .prefix_family (r ["part_nr"])==pfx ]
            pfx_fn =[r for r in pfx_recs if r ["label"]=="FN"]
            fn_rate =len (pfx_fn )/len (pfx_recs )if pfx_recs else 0 
            print (f"  {pfx :30s}  FN_rate={100 *fn_rate :.0f}%  "
            f"({len (pfx_fn )}/{len (pfx_recs )} missed)")

            # density buckets: are low-density CPs harder?
    print ("\n=== FN rate by local vertex density ===")
    buckets =[(0 ,5 ,"very sparse <5"),(5 ,20 ,"sparse 5-20"),
    (20 ,50 ,"medium 20-50"),(50 ,999999 ,"dense >50")]
    for lo ,hi ,name in buckets :
        bucket =[r for r in all_records if lo <=r ["local_density"]<hi ]
        fn_b =[r for r in bucket if r ["label"]=="FN"]
        rate =len (fn_b )/len (bucket )if bucket else 0 
        print (f"  {name :20s}: FN_rate={100 *rate :.0f}%  ({len (fn_b )}/{len (bucket )})")

    if args .json_out :
        with open (args .json_out ,"w",encoding ="utf-8")as fh :
            json .dump (all_records ,fh ,indent =2 )
        print (f"\nPer-CP records saved -> {args .json_out }")


if __name__ =="__main__":
    main ()
