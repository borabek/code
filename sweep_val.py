# -*- coding: utf-8 -*-
"""ADIM 4 step 1: lock the decode threshold ten the FROZEN real val list.

Runs inference ONCE per val part (the expensive bit), then decodes at many
heatmap thresholds (cheap) and reports F1/accuracy per threshold. Uses the exact
eval_real.py machinery (jd.dedup_connection_points + cpr.infer_knngraph +
ct.decode_predictions + mcp.keypoint_report) so the number is comparable to the
single-shot test. min_votes stays 1 (ADIM 1 ceiling). NO test data is touched.

Usage:
  .venv/Scripts/python.exe sweep_val.py checkpoints/cp_real_v1_best.ckpt
"""
import argparse 
import numpy as np 
import cp_regressor as cpr ,cp_targets as ct ,json_dataset as jd ,metrics as mcp 
import train_cp as tc 

ap =argparse .ArgumentParser ()
ap .add_argument ("ckpt")
ap .add_argument ("--list",default ="_real_val.txt",dest ="lst")
ap .add_argument ("--corpus",default =r"C:\Users\DE00024082\Desktop\JSON")
ap .add_argument ("--device",default ="cuda")
ap .add_argument ("--thresholds",default ="0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50")
a =ap .parse_args ()

thrs =[float (x )for x in a .thresholds .split (",")]
want =set (l .strip ()for l in open (a .lst ,encoding ="utf-8")if l .strip ())
model ,meta ,_ =cpr .load_model (a .ckpt ,device =a .device )
dedup =tc ._nms_radius (None ,5.0 )

# one inference pass per part; keep (V, arr, gt, gd)
cache =[]
for p in jd .iter_parts (a .corpus ):
    pn =str (p .part_nr )
    if pn not in want :
        continue 
    _ ,gt ,gd =jd .dedup_connection_points (p )
    if not len (gt ):
        continue 
    V =np .asarray (p .vertices ,float )
    Vn ,_ ,scale =cpr .normalize_vertices (V )
    arr =cpr .infer_knngraph (model ,meta ,Vn ,device =a .device ,max_gpu_verts =14000 ,
    offset_scale =scale ,patch =cpr .is_patch_part (pn ),part_nr =pn )
    cache .append ((V ,arr ,gt ,gd ))
print (f"inferred {len (cache )} val parts before; sweeping {len (thrs )} thresholds\n")

print (f"  {'thr':>5} {'F1':>8} {'accuracy':>9} {'prec':>7} {'recall':>7}   TP/FP/FN")
best =None 
for thr in thrs :
    TP =FP =FN =0 
    for V ,arr ,gt ,gd in cache :
        preds =ct .decode_predictions (V ,arr ,heatmap_thresh =thr ,
        nms_radius_mm =dedup ,min_votes =1 )
        r =mcp .keypoint_report (preds ,gt ,gd ,dist_thresh_mm =5.0 )
        TP +=r ["tp"];FP +=r ["fp"];FN +=r ["fn"]
    f1 =2 *TP /(2 *TP +FP +FN )if TP else 0.0 
    jac =TP /(TP +FP +FN )if TP else 0.0 
    prec =TP /(TP +FP )if TP +FP else 0.0 
    rec =TP /(TP +FN )if TP +FN else 0.0 
    mark =""
    if best is None or f1 >best [1 ]:
        best =(thr ,f1 ,jac );mark ="  <=="
    print (f"  {thr :>5.2f} {f1 :>8.4f} {jac :>9.4f} {prec :>7.4f} {rec :>7.4f}   {TP }/{FP }/{FN }{mark }")

print (f"\n>>> BEST threshold = {best [0 ]:.2f}  ->  F1 {best [1 ]:.4f}  accuracy {best [2 ]:.4f}")
print ("DONE")
