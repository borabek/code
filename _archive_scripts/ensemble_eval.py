"""Decode-level ensemble of N CP-detector checkpoints, scored honestly.

Union+dedup ensemble: every model predicts independently at its OWN operating
point; overlapping detections (within the adaptive dedup radius) are merged
keeping the highest-confidence node; survivors form the ensemble prediction.

LEAKAGE GUARD: pass --parts-file with an explicit newline-separated PartNr list
and evaluate ONLY ten parts that were held out from EVERY member's training --
e.g. for the hp_v22 x knngraph_v19 cross-architecture ensemble the clean set is
(hp_v22 val) INTERSECT (v19's reconstructed val): 13 parts; 6 of hp_v22's 22
val parts sat in v19's TRAINING set and would flatter any v19-involved number
(this also means the S11 v19-rebaseline 0.584 is partly optimistic, i.e. the
hierpoint gain is UNDERSTATED).

Usage:
  python ensemble_eval.py --gt-dir "C:\\...\\JSON" --extra-source wscad_corpus_v2 \\
      --parts-file _clean13.txt \\
      --member checkpoints/cp_hp_v22_best.ckpt:0.15 \\
      --member checkpoints/cp_knn_v19_best.ckpt:0.30 --device cpu
"""
import json 
import argparse 
import logging 

import numpy as np 

logging .basicConfig (level =logging .ERROR )


def dedup_union (node_lists ,dedup_mm ,min_members =1 ):
    """Merge N prediction lists: greedy by confidence, drop nodes within
    dedup_mm of an already-kept node (the kept one has >= confidence).

    min_members: consensus floor -- keep a merged detection only if at least
    this many DISTINCT members contributed a node within dedup_mm of it.
    Measured motivation: at a recall-heavy member threshold (0.15) a plain
    union (min_members=1) ACCUMULATES every member's false positives (3-seed
    union ten the shared val: 190 FP, F1 55.7% vs best single 64.4%), whereas
    2-of-3 consensus keeps only detections independent seeds agree ten."""
    tagged =[(mi ,n )for mi ,nodes in enumerate (node_lists )for n in nodes ]
    tagged .sort (key =lambda t :-float (t [1 ].get ("confidence_score",0.0 )))
    kept =[]# list of [node, {member_ids}]
    for mi ,n in tagged :
        p =np .asarray (n ["entry_point"],float )
        merged =False 
        for k in kept :
            if np .linalg .norm (p -np .asarray (k [0 ]["entry_point"],float ))<dedup_mm :
                k [1 ].add (mi )
                merged =True 
                break 
        if not merged :
            kept .append ([n ,{mi }])
    return [n for n ,members in kept if len (members )>=min_members ]


def main (argv =None ):
    ap =argparse .ArgumentParser (description ="decode-level N-model ensemble eval")
    ap .add_argument ("--gt-dir",required =True ,dest ="gt_dir")
    ap .add_argument ("--extra-source",action ="append",default =[],
    dest ="extra_sources")
    ap .add_argument ("--parts-file",required =True ,dest ="parts_file",
    help ="newline-separated PartNrs: the leakage-clean eval set")
    ap .add_argument ("--member",action ="append",required =True ,dest ="members",
    help ="ckpt_path:heatmap_thr (repeatable)")
    ap .add_argument ("--device",default ="cpu")
    ap .add_argument ("--max-gpu-verts",type =int ,default =14000 ,
    dest ="max_gpu_verts")
    ap .add_argument ("--consensus",type =int ,default =1 ,
    help ="min DISTINCT members that must agree (within the dedup "
    "radius) for a detection to survive; 1 = plain union")
    args =ap .parse_args (argv )

    import json_dataset as jd 
    import cp_regressor as cpr 
    import metrics as mcp 
    import predict as pd 

    want ={l .strip ()for l in open (args .parts_file ,encoding ="utf-8")
    if l .strip ()}
    parts =[p for p in jd .iter_parts (args .gt_dir )if str (p .part_nr )in want ]
    for src in args .extra_sources :
        parts +=[p for p in jd .iter_parts (src )if str (p .part_nr )in want ]
    print (f"clean eval parts: {len (parts )}/{len (want )}")

    members =[]
    for spec in args .members :
        path ,thr =spec .rsplit (":",1 )
        model ,meta ,backbone ,decode =cpr .load_inference (path ,
        device =args .device )
        decode =dict (decode ,heatmap_thresh =float (thr ))
        members .append ({"name":f"{path .split ('cp_')[-1 ].split ('_best')[0 ]}"
        f"@{thr }",
        "model":model ,"meta":meta ,"backbone":backbone ,
        "decode":decode })

    tot ={m ["name"]:{"tp":0 ,"fp":0 ,"fn":0 }for m in members }
    tot ["ENSEMBLE"]={"tp":0 ,"fp":0 ,"fn":0 }
    for p in parts :
        _ ,gt_pts ,gt_dirs =jd .dedup_connection_points (p )
        if not len (gt_pts ):
            continue 
        V =np .asarray (p .vertices ,float )
        diag =float (np .linalg .norm (V .max (0 )-V .min (0 )))or 1.0 
        dedup_mm =min (5.0 ,0.04 *diag )
        per_model =[]
        for m in members :
            nodes ,_ =pd .predict_part (m ["backbone"],m ["model"],m ["meta"],p ,
            m ["decode"],device =args .device ,
            max_gpu_verts =args .max_gpu_verts )
            per_model .append (nodes )
            rep =mcp .keypoint_report (
            [{"point":np .asarray (n ["entry_point"],float ),
            "direction":np .asarray (n ["approach_vector"],float )}
            for n in nodes ],gt_pts ,gt_dirs ,dist_thresh_mm =5.0 )
            tot [m ["name"]]["tp"]+=rep ["tp"]
            tot [m ["name"]]["fp"]+=rep ["fp"]
            tot [m ["name"]]["fn"]+=rep ["fn"]
        ens =dedup_union (per_model ,dedup_mm ,min_members =args .consensus )
        rep =mcp .keypoint_report (
        [{"point":np .asarray (n ["entry_point"],float ),
        "direction":np .asarray (n ["approach_vector"],float )}
        for n in ens ],gt_pts ,gt_dirs ,dist_thresh_mm =5.0 )
        tot ["ENSEMBLE"]["tp"]+=rep ["tp"]
        tot ["ENSEMBLE"]["fp"]+=rep ["fp"]
        tot ["ENSEMBLE"]["fn"]+=rep ["fn"]

    print (f"\n{'model':28s} {'TP':>4} {'FP':>4} {'FN':>4} {'P':>7} {'R':>7} {'F1':>7}")
    for name ,t in tot .items ():
        tp ,fp ,fn =t ["tp"],t ["fp"],t ["fn"]
        prec =tp /(tp +fp )if (tp +fp )else 0.0 
        rec =tp /(tp +fn )if (tp +fn )else 0.0 
        f1 =2 *prec *rec /(prec +rec )if (prec +rec )else 0.0 
        print (f"{name :28s} {tp :4d} {fp :4d} {fn :4d} {100 *prec :6.1f}% "
        f"{100 *rec :6.1f}% {100 *f1 :6.1f}%")


if __name__ =="__main__":
    main ()
