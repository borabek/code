"""Rotation-invariance proof for the knngraph CP detector.

For a single part: rotates it N times, runs the model ten each view, rotates
predictions BACK to the original frame, overlays all results in one GLB.

  Tight clusters   → model is rotation-invariant  ✅
  Scattered clouds → model is pose-dependent       ❌

Outputs
-------
1. GLB file   — original grey mesh + one colour-per-rotation sphere cloud
2. Console    — per-rotation CP count, consistency mm, coverage %

Consistency = mean spread of matched CP clusters across rotations (mm).
Coverage    = % of GT CPs found in ALL N rotations (needs --gt from corpus).

Usage
-----
  # Quick: just the model, one part JSON:
  python rot_invariance.py checkpoints/cp_knn_v15_best.ckpt part.json \
      --n 8 --out rot_proof.glb --device cuda

  # With GT coverage (needs corpus):
  python rot_invariance.py checkpoints/cp_knn_v15_best.ckpt part.json \
      --n 8 --corpus "C:/Users/.../JSON" --part-nr wscaduniverse_0311087 \
      --out rot_proof.glb --device cuda

  # Pick a random corpus part automatically:
  python rot_invariance.py checkpoints/cp_knn_v15_best.ckpt \
      --corpus "C:/Users/.../JSON" --n 8 --out rot_proof.glb --device cuda
"""
import argparse 
import json 
import os 
import sys 

import numpy as np 

# ---------------------------------------------------------------------------
# rotation colours — one per rotation view
# ---------------------------------------------------------------------------

_ROT_COLOURS =[
[0.95 ,0.20 ,0.10 ,0.85 ],# red
[0.10 ,0.70 ,0.95 ,0.85 ],# cyan
[0.95 ,0.80 ,0.10 ,0.85 ],# yellow
[0.70 ,0.10 ,0.95 ,0.85 ],# purple
[0.10 ,0.95 ,0.40 ,0.85 ],# green
[0.95 ,0.50 ,0.10 ,0.85 ],# orange
[0.10 ,0.30 ,0.95 ,0.85 ],# blue
[0.95 ,0.10 ,0.70 ,0.85 ],# pink
]


def _colour (i ):
    return _ROT_COLOURS [i %len (_ROT_COLOURS )]


    # ---------------------------------------------------------------------------
    # matching: cluster predictions across rotations
    # ---------------------------------------------------------------------------

def _cluster_predictions (all_pred_pts ,match_dist_mm =8.0 ):
    """Group CP predictions from all rotations into clusters.

    all_pred_pts : list of (n_i, 3) arrays — one per rotation

    Returns list of clusters:
        cluster = {"centre": (3,), "per_rot": {rot_idx: (3,)}, "spread_mm": float}
    """
    # Start from rotation 0 as anchor
    clusters =[]
    for pt in all_pred_pts [0 ]:
        clusters .append ({"centre":pt .copy (),"per_rot":{0 :pt .copy ()}})

    for ri ,pts in enumerate (all_pred_pts [1 :],start =1 ):
        for pt in pts :
        # find nearest cluster centre
            if clusters :
                centres =np .array ([c ["centre"]for c in clusters ])
                dists =np .linalg .norm (centres -pt ,axis =1 )
                best =int (np .argmin (dists ))
                if dists [best ]<=match_dist_mm :
                    clusters [best ]["per_rot"][ri ]=pt .copy ()
                    # update centre as mean of all so far
                    pts_in =np .array (list (clusters [best ]["per_rot"].values ()))
                    clusters [best ]["centre"]=pts_in .mean (0 )
                    continue 
                    # new cluster
            clusters .append ({"centre":pt .copy (),"per_rot":{ri :pt .copy ()}})

            # compute spread per cluster
    n_rots =len (all_pred_pts )
    for c in clusters :
        pts_in =np .array (list (c ["per_rot"].values ()))
        if len (pts_in )>1 :
            dists =np .linalg .norm (pts_in -pts_in .mean (0 ),axis =1 )
            c ["spread_mm"]=float (dists .max ())
        else :
            c ["spread_mm"]=0.0 
        c ["n_rots_found"]=len (c ["per_rot"])
        c ["coverage_frac"]=len (c ["per_rot"])/n_rots 

    return clusters 


    # ---------------------------------------------------------------------------
    # main logic
    # ---------------------------------------------------------------------------

def run (args ):
    import cp_regressor as cpr 
    import cp_targets as ct 
    import augment as aug 
    from export_glb import _GlbBuilder ,build_scene ,_load_mesh ,_COL_GREY 

    # ---- load model ----
    model ,meta ,_ =cpr .load_model (args .ckpt ,device =args .device )
    print (f"Model: {args .ckpt }  k={meta .get ('k',16 )}  c_in={meta .get ('c_in','?')}")

    # ---- load mesh ----
    V ,F =None ,None 
    gt_pts =None 

    if args .part_json :
        V ,F =_load_mesh (part_json_path =args .part_json )
        part_nr =os .path .splitext (os .path .basename (args .part_json ))[0 ]
    elif args .corpus and args .part_nr :
        V ,F =_load_mesh (corpus_dir =args .corpus ,part_nr =args .part_nr )
        part_nr =args .part_nr 
    elif args .corpus :
    # pick first part with CPs from corpus
        import json_dataset as jd 
        picked =None 
        for p in jd .iter_parts (args .corpus ):
            if p .n_cps >=2 :
                picked =p ;break 
        if picked is None :
            sys .exit ("No part with >=2 CPs found in corpus")
        V =np .asarray (picked .vertices ,dtype =np .float32 )
        F =np .asarray (picked .faces ,dtype =np .uint32 )
        gt_pts =np .asarray (picked .cp_points ,dtype =np .float64 )
        part_nr =str (picked .part_nr )
        print (f"Auto-selected: {part_nr }  verts={len (V )}  GT CPs={len (gt_pts )}")
    else :
        sys .exit ("Provide a part JSON or --corpus [--part-nr]")

    if V is None or len (V )==0 :
        sys .exit (f"Could not load mesh for {part_nr }")

    V =np .asarray (V ,dtype =np .float64 )
    F =np .asarray (F ,dtype =np .uint32 )

    # load GT from corpus if available and not already loaded
    if gt_pts is None and args .corpus :
        import json_dataset as jd 
        for p in jd .iter_parts (args .corpus ):
            if str (p .part_nr )==part_nr :
                gt_pts =np .asarray (p .cp_points ,dtype =np .float64 )
                break 

    print (f"Part: {part_nr }  verts={len (V )}  faces={len (F )}"
    +(f"  GT CPs={len (gt_pts )}"if gt_pts is not None else ""))

    # ---- generate rotation matrices ----
    rng =np .random .default_rng (args .seed )
    # always include identity as rotation 0
    rots_90 =[
    np .eye (3 ),
    np .array ([[0 ,-1 ,0 ],[1 ,0 ,0 ],[0 ,0 ,1 ]],dtype =float ),# 90° z
    np .array ([[-1 ,0 ,0 ],[0 ,-1 ,0 ],[0 ,0 ,1 ]],dtype =float ),# 180° z
    np .array ([[0 ,1 ,0 ],[-1 ,0 ,0 ],[0 ,0 ,1 ]],dtype =float ),# 270° z
    np .array ([[1 ,0 ,0 ],[0 ,0 ,-1 ],[0 ,1 ,0 ]],dtype =float ),# 90° x
    np .array ([[0 ,0 ,1 ],[0 ,1 ,0 ],[-1 ,0 ,0 ]],dtype =float ),# 90° y
    np .array ([[1 ,0 ,0 ],[0 ,-1 ,0 ],[0 ,0 ,-1 ]],dtype =float ),# 180° x
    np .array ([[-1 ,0 ,0 ],[0 ,1 ,0 ],[0 ,0 ,-1 ]],dtype =float ),# 180° y
    ]
    if args .random_rots :
        rots =[np .eye (3 )]+[aug ._random_rotation_matrix (rng )
        for _ in range (args .n -1 )]
    else :
        rots =rots_90 [:args .n ]

    print (f"\nRunning {len (rots )} rotation views ...")

    # ---- inference per rotation ----
    patch =cpr .is_patch_part (part_nr )
    c =V .mean (0 )

    all_pred_pts =[]# one (n_i,3) per rotation
    all_pred_dirs =[]
    all_pred_scores =[]
    rot_labels =[]

    for ri ,R in enumerate (rots ):
        Vr =(V -c )@R .T +c 
        Vn ,_ ,scale =cpr .normalize_vertices (Vr )
        arr =cpr .infer_knngraph (model ,meta ,Vn ,device =args .device ,
        max_gpu_verts =args .max_gpu_verts ,
        offset_scale =scale ,patch =patch ,
        part_nr =part_nr )
        preds =ct .decode_predictions (V ,arr ,heatmap_thresh =args .thr ,
        nms_radius_mm =args .nms ,min_votes =1 )

        # rotate predictions BACK to original frame
        rot_pts =[]
        rot_dirs =[]
        rot_sc =[]
        for pr in preds :
            pt_rot =np .asarray (pr ["point"],dtype =np .float64 )
            d_rot =np .asarray (pr ["direction"],dtype =np .float64 )
            # back-rotate: pt_rot was computed with Vr coords (already in original
            # mm frame because decode_predictions uses V not Vr — only features rotate)
            # direction was predicted in rotated frame → back-rotate
            d_orig =d_rot @R # R^-1 = R^T for rotation; row-vec: v @ R
            rot_pts .append (pt_rot )
            rot_dirs .append (d_orig /(np .linalg .norm (d_orig )+1e-9 ))
            rot_sc .append (float (pr .get ("score",0.5 )))

        pts_arr =np .array (rot_pts ,dtype =np .float64 )if rot_pts else np .zeros ((0 ,3 ))
        all_pred_pts .append (pts_arr )
        all_pred_dirs .append (rot_dirs )
        all_pred_scores .append (rot_sc )
        rot_labels .append (f"rot{ri }")
        print (f"  rot {ri :2d}: {len (preds ):3d} CP(s) detected",flush =True )

        # ---- cluster analysis ----
    total_preds =sum (len (p )for p in all_pred_pts )
    clusters =_cluster_predictions (all_pred_pts ,match_dist_mm =args .match_dist )

    # stats
    spreads =[c ["spread_mm"]for c in clusters if c ["n_rots_found"]>1 ]
    mean_spread =float (np .mean (spreads ))if spreads else 0.0 
    max_spread =float (np .max (spreads ))if spreads else 0.0 
    n_rots =len (rots )
    full_coverage =[c for c in clusters if c ["n_rots_found"]==n_rots ]
    coverage_pct =100.0 *len (full_coverage )/max (len (clusters ),1 )

    # GT coverage (if available)
    gt_found_pct =None 
    if gt_pts is not None and len (gt_pts ):
    # how many GT CPs have at least one cluster within match_dist?
        centres =np .array ([c ["centre"]for c in clusters ])if clusters else np .zeros ((0 ,3 ))
        gt_found =0 
        for gp in gt_pts :
            if len (centres )and np .linalg .norm (centres -gp ,axis =1 ).min ()<=args .match_dist :
                gt_found +=1 
        gt_found_pct =100.0 *gt_found /len (gt_pts )

    print (f"\n=== Rotation-invariance report ===")
    print (f"  Part            : {part_nr }")
    print (f"  Rotations       : {n_rots }")
    print (f"  Total preds     : {total_preds }  ({total_preds /n_rots :.1f} avg/rot)")
    print (f"  Clusters found  : {len (clusters )}")
    print (f"  Mean spread     : {mean_spread :.2f} mm  (consistency — lower=better)")
    print (f"  Max spread      : {max_spread :.2f} mm")
    print (f"  Full coverage   : {len (full_coverage )}/{len (clusters )} clusters "
    f"found in ALL {n_rots } rotations ({coverage_pct :.0f}%)")
    if gt_found_pct is not None :
        n_found =int (round (gt_found_pct *len (gt_pts )/100 ))
        print (f"  GT recall       : {gt_found_pct :.0f}%  "
        f"({n_found }/{len (gt_pts )} GT CPs found by >=1 rotation)")

    verdict =("ROTATION-INVARIANT [OK]"if mean_spread <5.0 
    else "PARTIALLY POSE-DEPENDENT [WARN]"if mean_spread <15.0 
    else "POSE-DEPENDENT [FAIL]")
    print (f"\n  Verdict: {verdict }  (mean spread {mean_spread :.1f} mm)")

    # ---- build GLB ----
    glb =_GlbBuilder ()

    # grey mesh body
    if len (V )and len (F ):
        glb .add_triangle_mesh (V .astype (np .float32 ),F ,_COL_GREY ,"part_mesh")

        # GT CPs as white stars (large spheres)
    if gt_pts is not None and len (gt_pts ):
        from export_glb import _sphere_mesh ,_COL_CAD_GRN 
        diag =float (np .linalg .norm (V .max (0 )-V .min (0 )))or 1.0 
        sr =diag *0.020 # slightly larger than rot spheres
        sv_all ,sf_all ,off =[],[],0 
        for gp in gt_pts :
            sv ,sf =_sphere_mesh (gp ,sr )
            sf_all .append (sf +off );sv_all .append (sv );off +=len (sv )
        glb .add_triangle_mesh (np .concatenate (sv_all ),
        np .concatenate (sf_all ),
        _COL_CAD_GRN ,"gt_cps")

        # per-rotation coloured spheres
    diag =float (np .linalg .norm (V .max (0 )-V .min (0 )))or 1.0 
    sr =diag *0.013 
    from export_glb import _sphere_mesh ,_arrow_lines 

    for ri ,pts in enumerate (all_pred_pts ):
        if len (pts )==0 :
            continue 
        col =_colour (ri )
        sv_all ,sf_all ,off =[],[],0 
        lv_all ,ll_all ,loff =[],[],0 
        for pi ,pt in enumerate (pts ):
            sv ,sf =_sphere_mesh (pt ,sr )
            sf_all .append (sf +off );sv_all .append (sv );off +=len (sv )

            if pi <len (all_pred_dirs [ri ]):
                d =all_pred_dirs [ri ][pi ]
                lv ,ll =_arrow_lines (pt ,d ,length =diag *0.12 )
                if len (lv ):
                    ll_all .append (ll +loff );lv_all .append (lv );loff +=len (lv )

        glb .add_triangle_mesh (np .concatenate (sv_all ),
        np .concatenate (sf_all ),
        col ,f"rot{ri }_spheres")
        if lv_all :
            glb .add_line_set (np .concatenate (lv_all ),
            np .concatenate (ll_all ),
            col ,f"rot{ri }_arrows")

    out =args .out or f"{part_nr }_rot_invariance.glb"
    glb .write (out )

    print (f"\nGLB written: {out }")
    print ("Colour key:")
    for ri in range (min (n_rots ,len (_ROT_COLOURS ))):
        r ,g ,b ,_ =_ROT_COLOURS [ri %len (_ROT_COLOURS )]
        print (f"  rot {ri }: RGB({r :.0%}, {g :.0%}, {b :.0%})")
    if gt_pts is not None :
        print ("  GT CPs: green (large spheres)")
    print ("\nOpen the GLB in Windows 3D Viewer.")
    print ("Tight clusters across colours = rotation-invariant. "
    "Scattered clouds = pose-dependent.")

    return {
    "mean_spread_mm":mean_spread ,
    "max_spread_mm":max_spread ,
    "coverage_pct":coverage_pct ,
    "gt_recall_pct":gt_found_pct ,
    "n_clusters":len (clusters ),
    "n_rots":n_rots ,
    }


    # ---------------------------------------------------------------------------
    # CLI
    # ---------------------------------------------------------------------------

def _parse_args ():
    ap =argparse .ArgumentParser (description ="Rotation-invariance proof for CP detector")
    ap .add_argument ("ckpt",help ="trained checkpoint (.ckpt)")
    ap .add_argument ("part_json",nargs ="?",default =None ,
    help ="part mesh JSON (ABB format); omit to auto-pick from --corpus")
    ap .add_argument ("--n",type =int ,default =8 ,
    help ="number of rotations (default 8; uses 90° steps unless --random)")
    ap .add_argument ("--random-rots",action ="store_true",dest ="random_rots",
    help ="use random rotations instead of 90° axis steps")
    ap .add_argument ("--seed",type =int ,default =42 )
    ap .add_argument ("--device",default ="cpu")
    ap .add_argument ("--max-gpu-verts",type =int ,default =7000 ,dest ="max_gpu_verts")
    ap .add_argument ("--thr",type =float ,default =0.50 ,
    help ="heatmap threshold (default 0.50)")
    ap .add_argument ("--nms",type =float ,default =5.0 ,
    help ="NMS radius mm (default 5.0)")
    ap .add_argument ("--match-dist",type =float ,default =8.0 ,dest ="match_dist",
    help ="max mm to merge cross-rotation predictions into one cluster")
    ap .add_argument ("--corpus",default =None ,
    help ="corpus directory (to load mesh and/or GT CPs)")
    ap .add_argument ("--part-nr",default =None ,dest ="part_nr",
    help ="part_nr to find in --corpus")
    ap .add_argument ("--out",default =None ,
    help ="output GLB path (default: <part_nr>_rot_invariance.glb)")
    return ap .parse_args ()


if __name__ =="__main__":
    args =_parse_args ()
    run (args )
