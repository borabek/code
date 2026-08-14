"""Per-checkpoint low-threshold decode sweep ten a manifest-pinned part list.

Runs inference ONCE per (checkpoint, part) -- the expensive step -- then decodes
the cached (N,7) array at every threshold (cheap), so a 7-point sweep costs the
same as a single evaluation. No TTA (dead lever for hierpoint, RESULTS 11b).

Usage:
  python sweep_seed_thresholds.py --gt-dir "C:\\...\\JSON" \\
      --extra-source wscad_corpus_v2 --parts-file _shared_val22.txt \\
      --ckpt checkpoints/cp_hp_v22_best.ckpt \\
      --ckpt checkpoints/cp_hp_v22_seed1_best.ckpt \\
      --thresholds 0.10,0.12,0.15,0.18,0.20,0.25,0.30 --device cpu
"""
import argparse 
import logging 

import numpy as np 

logging .basicConfig (level =logging .ERROR )


def main (argv =None ):
    ap =argparse .ArgumentParser (description ="infer-once decode-many threshold "
    "sweep per checkpoint")
    ap .add_argument ("--gt-dir",required =True ,dest ="gt_dir")
    ap .add_argument ("--extra-source",action ="append",default =[],
    dest ="extra_sources")
    ap .add_argument ("--parts-file",required =True ,dest ="parts_file")
    ap .add_argument ("--ckpt",action ="append",required =True ,dest ="ckpts")
    ap .add_argument ("--thresholds",default ="0.10,0.12,0.15,0.18,0.20,0.25,0.30")
    ap .add_argument ("--device",default ="cpu")
    ap .add_argument ("--max-gpu-verts",type =int ,default =14000 ,
    dest ="max_gpu_verts")
    args =ap .parse_args (argv )
    thresholds =[float (t )for t in args .thresholds .split (",")]

    import json_dataset as jd 
    import cp_regressor as cpr 
    import cp_targets as ct 
    import metrics as mcp 
    import train_cp as tc 

    want ={l .strip ()for l in open (args .parts_file ,encoding ="utf-8")
    if l .strip ()}
    parts =[p for p in jd .iter_parts (args .gt_dir )if str (p .part_nr )in want ]
    for src in args .extra_sources :
        parts +=[p for p in jd .iter_parts (src )if str (p .part_nr )in want ]
    print (f"parts: {len (parts )}/{len (want )}")

    for ckpt in args .ckpts :
        model ,meta ,backbone =cpr .load_model (ckpt ,device =args .device )
        cache =[]
        for p in parts :
            _ ,gt_pts ,gt_dirs =jd .dedup_connection_points (p )
            if not len (gt_pts ):
                continue 
            V =np .asarray (p .vertices ,float )
            Vn ,_ ,scale =cpr .normalize_vertices (V )
            patch =cpr .is_patch_part (p .part_nr )
            arr =cpr .infer_knngraph (model ,meta ,Vn ,device =args .device ,
            max_gpu_verts =args .max_gpu_verts ,
            offset_scale =scale ,patch =patch ,
            part_nr =p .part_nr )
            cache .append ((p ,V ,gt_pts ,gt_dirs ,arr ))
        name =ckpt .split ("cp_")[-1 ].replace ("_best.ckpt","")
        print (f"\n=== {name } ===")
        print (f"{'thr':>5} {'TP':>4} {'FP':>4} {'FN':>4} {'P':>7} {'R':>7} {'F1':>7}")
        best =(0 ,None )
        for thr in thresholds :
            tp =fp =fn =0 
            for p ,V ,gt_pts ,gt_dirs ,arr in cache :
                preds =ct .decode_predictions (V ,arr ,heatmap_thresh =thr ,
                nms_radius_mm =tc ._nms_radius (None ,5.0 ),
                min_votes =1 )
                rep =mcp .keypoint_report (preds ,gt_pts ,gt_dirs ,
                dist_thresh_mm =5.0 )
                tp +=rep ["tp"];fp +=rep ["fp"];fn +=rep ["fn"]
            prec =tp /(tp +fp )if (tp +fp )else 0.0 
            rec =tp /(tp +fn )if (tp +fn )else 0.0 
            f1 =2 *prec *rec /(prec +rec )if (prec +rec )else 0.0 
            if f1 >best [0 ]:
                best =(f1 ,thr )
            print (f"{thr :5.2f} {tp :4d} {fp :4d} {fn :4d} {100 *prec :6.1f}% "
            f"{100 *rec :6.1f}% {100 *f1 :6.1f}%")
        print (f"  -> best thr {best [1 ]:.2f} = {100 *best [0 ]:.1f}%")


if __name__ =="__main__":
    main ()
