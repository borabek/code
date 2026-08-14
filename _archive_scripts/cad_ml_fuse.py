"""CAD + ML hybrid CP prediction: fuse step_openings (geometric, exact where it
fires) with the ML detector (catches what the cylinder path can't see), and
score CAD-only vs ML-only vs FUSED against human GT.

Fusion rule: CAD detections have priority (when the B-rep says there is a
terminal-radius bore/slot at X, that is geometric fact); ML detections are
added only where they are FARTHER than the dedup radius from every CAD
detection -- i.e. ML fills CAD's blind spots (rectangular spring-clamp
openings, families with unusual geometry) instead of double-reporting the same
opening. Dedup radius = min(5mm, 4% bbox diagonal), the same size-adaptive rule
used by decode NMS, eval matching and step_openings merging.

Evaluation frame: CAD preds are transformed from the STEP frame into the
labeled-corpus frame with cad_eval.align_frames (measured: the frames are
permuted+translated); ML runs directly ten the corpus mesh. In PRODUCTION the
flow is simpler -- a new part arrives as STEP, gets tessellated before, and BOTH
detectors run in that one frame with no alignment step.

Usage:
  python cad_ml_fuse.py --cad-preds _final_cons.json --gt-dir "C:\\...\\JSON" \\
      --step-dir _cad_eval_pxc --ckpt checkpoints/cp_knn_v17_best.ckpt \\
      --device cpu
"""
import os 
import json 
import glob 
import argparse 
import logging 

import numpy as np 

import json_dataset as jd 
import metrics as mcp 
import cad_eval as ce 

logger =logging .getLogger (__name__ )


def fuse (cad_nodes ,ml_nodes ,dedup_mm ,ml_add_thr =0.30 ):
    """CAD nodes + HIGH-CONFIDENCE ML nodes farther than dedup_mm from every CAD
    node.

    ml_add_thr: confidence floor for ML detections ADDED beyond CAD. Asymmetric
    ten purpose: the ML-only operating point is a LOW threshold (~0.15, because
    the heat head is under-confident and recall craters above it), but at that
    threshold ML also emits many weak false peaks -- measured ten the 3-part
    human-GT eval, naive fusion (add every ML detection) dropped fused F1 to
    40% vs CAD-alone 72.7% purely from leaked ML FPs. Additions ten top of CAD
    should instead require the model's own high-PRECISION operating point
    (0.30: val P=75.4%, from the hp_v22 post-training sweep -- chosen from VAL,
    not tuned ten the tiny fusion eval), so fusion can only add detections the
    model is genuinely confident about and never degrades below CAD-alone by
    more than the rare confident-FP."""
    fused =list (cad_nodes )
    keep_ml =[n for n in ml_nodes 
    if float (n .get ("confidence_score",1.0 ))>=ml_add_thr ]
    if not cad_nodes :
        return fused +keep_ml 
    cad_pts =np .array ([n ["entry_point"]for n in cad_nodes ],float )
    for n in keep_ml :
        d =np .linalg .norm (cad_pts -np .asarray (n ["entry_point"],float ),axis =1 )
        if d .min ()>dedup_mm :
            fused .append (n )
    return fused 


def _score (nodes ,gt_pts ,gt_dirs ):
    preds =[{"point":np .asarray (n ["entry_point"],float ),
    "direction":np .asarray (n .get ("approach_vector",[0 ,0 ,1 ]),float )}
    for n in nodes ]
    return mcp .keypoint_report (preds ,gt_pts ,gt_dirs ,
    dist_thresh_mm =ce .DIST_THRESH_MM )


def main (argv =None ):
    ap =argparse .ArgumentParser (description ="fuse CAD-direct + ML CP predictions "
    "and score both + hybrid vs human GT")
    ap .add_argument ("--cad-preds",required =True ,dest ="cad_preds",
    help ="step_openings --out JSON (STEP frame)")
    ap .add_argument ("--gt-dir",required =True ,dest ="gt_dir")
    ap .add_argument ("--step-dir",required =True ,dest ="step_dir")
    ap .add_argument ("--ckpt",required =True ,help ="ML checkpoint (.ckpt)")
    ap .add_argument ("--device",default ="cpu",
    help ="cpu recommended while a training run owns the GPU")
    ap .add_argument ("--ml-thr",type =float ,default =None ,dest ="ml_thr",
    help ="override the checkpoint's embedded heatmap_thresh for the "
    "ML side (e.g. 0.15 for hp_v22, whose post-training decode "
    "sweep found the embedded 0.30 far from optimal -- the "
    "under-confident heat head needs a low threshold)")
    ap .add_argument ("--ml-add-thr",type =float ,default =0.30 ,dest ="ml_add_thr",
    help ="confidence floor for ML detections ADDED ten top of CAD "
    "(see fuse(); default 0.30 = hp_v22's high-precision val "
    "operating point). The ML-only column still uses --ml-thr")
    ap .add_argument ("--align-tol",type =float ,default =2.0 ,dest ="align_tol")
    ap .add_argument ("--out",default =None ,help ="optional fused-predictions JSON")
    args =ap .parse_args (argv )
    logging .basicConfig (level =logging .INFO ,format ="%(message)s")

    import step_to_json as sj 
    import cp_regressor as cpr 
    import predict as pd 

    model ,meta ,backbone ,decode =cpr .load_inference (args .ckpt ,device =args .device )
    if args .ml_thr is not None :
        decode =dict (decode ,heatmap_thresh =float (args .ml_thr ))
        logger .info ("ML decode heatmap_thresh overridden -> %.2f",args .ml_thr )
    with open (args .cad_preds ,encoding ="utf-8")as fh :
        cad_doc =json .load (fh )
    gt_index =ce .build_gt_index (args .gt_dir )

    rows ,fused_out =[],[]
    tot ={k :{"tp":0 ,"fp":0 ,"fn":0 }for k in ("CAD","ML","FUSED")}
    for part in cad_doc .get ("parts",[]):
        cat =ce .catalog_nr (part ["part_nr"])or str (part ["part_nr"])
        gt =gt_index .get (cat )
        if gt is None :
            continue 
        steps =glob .glob (os .path .join (args .step_dir ,f"*{cat }*.st*p"))
        if not steps :
            continue 
        V_step ,_ =sj .load_any_mesh (steps [0 ],deflection =0.3 )
        V_gt =np .asarray (gt .vertices ,float )
        R ,t ,res =ce .align_frames (V_step ,V_gt )
        if res >args .align_tol :
            logger .warning ("%s: alignment residual %.2fmm -- skipped",cat ,res )
            continue 
        cad_nodes =[{"entry_point":(R @np .asarray (cp ["entry_point"],float )+t ).tolist (),
        "approach_vector":(R @np .asarray (cp ["approach_vector"],float )).tolist (),
        "source":"cad"}
        for cp in part .get ("connection_points",[])]
        ml_raw ,_ =pd .predict_part (backbone ,model ,meta ,gt ,decode ,
        device =args .device )
        ml_nodes =[dict (n ,source ="ml")for n in ml_raw ]
        diag =float (np .linalg .norm (V_gt .max (0 )-V_gt .min (0 )))or 1.0 
        dedup_mm =min (5.0 ,0.04 *diag )
        fused_nodes =fuse (cad_nodes ,ml_nodes ,dedup_mm ,
        ml_add_thr =args .ml_add_thr )

        _ ,gt_pts ,gt_dirs =jd .dedup_connection_points (gt )
        reps ={"CAD":_score (cad_nodes ,gt_pts ,gt_dirs ),
        "ML":_score (ml_nodes ,gt_pts ,gt_dirs ),
        "FUSED":_score (fused_nodes ,gt_pts ,gt_dirs )}
        for k ,rep in reps .items ():
            tot [k ]["tp"]+=rep ["tp"];tot [k ]["fp"]+=rep ["fp"]
            tot [k ]["fn"]+=rep ["fn"]
        rows .append ((str (gt .part_nr ),len (gt_pts ),reps ))
        fused_out .append ({"part_nr":str (gt .part_nr ),
        "n_detected":len (fused_nodes ),
        "connection_points":fused_nodes })

    if not rows :
        raise SystemExit ("nothing scored (no GT/STEP overlap)")

    print (f"\n=== CAD vs ML vs FUSED (human GT, Hungarian, "
    f"{ce .DIST_THRESH_MM }mm) ===")
    print (f"{'part':18s} {'nGT':>3} | "+" | ".join (
    f"{k }: TP FP FN  F1 "for k in ("CAD","ML","FUSED")))
    for pn ,ngt ,reps in rows :
        cells =[]
        for k in ("CAD","ML","FUSED"):
            r =reps [k ]
            cells .append (f"{k }: {r ['tp']:2d} {r ['fp']:2d} {r ['fn']:2d} "
            f"{r ['f1']:.2f}")
        print (f"{pn :18s} {ngt :3d} | "+" | ".join (cells ))
    print ()
    for k in ("CAD","ML","FUSED"):
        tp ,fp ,fn =tot [k ]["tp"],tot [k ]["fp"],tot [k ]["fn"]
        p =tp /(tp +fp )if (tp +fp )else 0.0 
        r =tp /(tp +fn )if (tp +fn )else 0.0 
        f1 =2 *p *r /(p +r )if (p +r )else 0.0 
        print (f"POOLED {k :5s}: TP={tp } FP={fp } FN={fn }  "
        f"P={100 *p :.1f}%  R={100 *r :.1f}%  F1={100 *f1 :.1f}%")

    if args .out :
        with open (args .out ,"w",encoding ="utf-8")as fh :
            json .dump ({"detector":"cad+ml_fused","parts":fused_out },fh ,indent =1 )
        print (f"\nfused predictions -> {args .out }")


if __name__ =="__main__":
    main ()
