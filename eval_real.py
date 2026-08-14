# -*- coding: utf-8 -*-
"""Score a checkpoint against the REAL Desktop\\JSON labels ten the frozen real-val.

The single source of truth: mesh = Graphic3d.Points, GT = ConnectionPoints
(manufacturer data, terminal names). No pseudo-labels anywhere.

Usage:
  .venv/Scripts/python.exe eval_real.py checkpoints/cp_hp_v31_ftc6_best.ckpt
  .venv/Scripts/python.exe eval_real.py checkpoints/cp_hp_v28_ftc3_best.ckpt --list _real_test.txt
"""
import argparse ,os 
import numpy as np 
import cp_regressor as cpr ,cp_targets as ct ,json_dataset as jd ,metrics as mcp 
import train_cp as tc 

ap =argparse .ArgumentParser ()
ap .add_argument ("ckpt")
ap .add_argument ("--list",default ="_real_val.txt",dest ="lst")
ap .add_argument ("--corpus",default =r"C:\Users\DE00024082\Desktop\JSON")
ap .add_argument ("--thr",type =float ,default =0.30 )
ap .add_argument ("--device",default ="cuda")
a =ap .parse_args ()

want =set (l .strip ()for l in open (a .lst ,encoding ="utf-8")if l .strip ())
model ,meta ,_ =cpr .load_model (a .ckpt ,device =a .device )
dedup =tc ._nms_radius (None ,5.0 )

TP =FP =FN =0 
scored =0 
per =[]
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
    preds =ct .decode_predictions (V ,arr ,heatmap_thresh =a .thr ,
    nms_radius_mm =dedup ,min_votes =1 )
    r =mcp .keypoint_report (preds ,gt ,gd ,dist_thresh_mm =5.0 )
    TP +=r ["tp"];FP +=r ["fp"];FN +=r ["fn"];scored +=1 
    per .append ((r ["tp"]/max (r ["tp"]+r ["fp"]+r ["fn"],1 ),pn ))

f1 =2 *TP /(2 *TP +FP +FN )if TP else 0.0 
jac =TP /(TP +FP +FN )if TP else 0.0 
print (f"\nckpt : {a .ckpt }")
print (f"list : {a .lst }  ({scored } part scored, GT total = {TP +FN })")
print (f"TP={TP }  FP={FP }  FN={FN }")
print (f"F1 = {f1 :.4f}    Jaccard (accuracy) = {jac :.4f}")
per .sort ()
print ("en kotu 5 part:",[(round (j ,2 ),pn )for j ,pn in per [:5 ]])
print ("en iyi 5 part :",[(round (j ,2 ),pn )for j ,pn in per [-5 :]])
print ("DONE")
