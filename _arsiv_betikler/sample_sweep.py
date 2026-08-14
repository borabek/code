# -*- coding: utf-8 -*-
"""E2 sampling-parity sweep (TRAINING_JSON_WSCAD_90 plan, section 6.2).

WITHOUT retraining, run the SAME checkpoint ten the 9 STEP twins under different
STRUCTURE-PRESERVING resampling policies, and score each against the Desktop-JSON GT
(frame-aligned). The paired probe already showed RANDOM subsample-to-JSON-count -> 0
(destroys kNN structure). Voxel (fixed mm spacing) and FPS (farthest-point) preserve
structure and bring STEP toward the JSON density the model was trained ten. If a policy
lifts STEP F1 toward the JSON ceiling, mesh-density parity is a real lever (=> bake it
into train+inference). Leaky diagnostic ten the 9 seen bridge parts, not a reported score.

Usage:
  .venv/Scripts/python.exe sample_sweep.py checkpoints/cp_m0_overfit_last.ckpt
"""
import sys ,glob ,os 
import numpy as np 
from scipy .spatial import cKDTree 
import cp_regressor as cpr ,cp_targets as ct ,json_dataset as jd ,metrics as mcp 
import train_cp as tc ,step_to_json as sj ,cad_eval as ce 

CK =sys .argv [1 ]if len (sys .argv )>1 else "checkpoints/cp_m0_overfit_last.ckpt"
TEACHER =r"C:\Users\DE00024082\Desktop\JSON"
STEPDIR ="_cad_eval_pxc"
THR =0.30 ;DEFL =0.3 ;DIST =5.0 
dedup =tc ._nms_radius (None ,5.0 )
bridge =[l .strip ()for l in open ("_bridge_test.txt",encoding ="utf-8")if l .strip ()]
model ,meta ,_ =cpr .load_model (CK ,device ="cuda")


def voxel (V ,cell ):
    key =np .floor (V /cell ).astype (np .int64 )
    _ ,idx =np .unique (key ,axis =0 ,return_index =True )
    return V [np .sort (idx )]


def fps (V ,n ):
    if len (V )>30000 :# cap input so FPS stays fast
        V =V [np .random .RandomState (0 ).choice (len (V ),30000 ,replace =False )]
    n =min (n ,len (V ))
    sel =np .empty (n ,int );sel [0 ]=0 
    d =np .sum ((V -V [0 ])**2 ,axis =1 )
    for i in range (1 ,n ):
        j =int (np .argmax (d ));sel [i ]=j 
        d =np .minimum (d ,np .sum ((V -V [j ])**2 ,axis =1 ))
    return V [sel ]


def infer (V ,pn ,patch ):
    V =np .asarray (V ,float )
    Vn ,_ ,scale =cpr .normalize_vertices (V )
    arr =cpr .infer_knngraph (model ,meta ,Vn ,device ="cuda",max_gpu_verts =14000 ,
    offset_scale =scale ,patch =patch ,part_nr =pn )
    return ct .decode_predictions (V ,arr ,heatmap_thresh =THR ,nms_radius_mm =dedup ,min_votes =1 )


def score (preds ,R ,t ,gt ,gd ):
    P =[{"point":R @np .asarray (q ["point"],float )+t ,
    "direction":R @np .asarray (q .get ("direction",[0 ,0 ,1 ]),float )}for q in preds ]
    r =mcp .keypoint_report (P ,gt ,gd ,dist_thresh_mm =DIST )
    return r ["tp"],r ["fp"],r ["fn"]


POLICIES =["patch","uni14k","voxel0.5","voxel1.0","voxel2.0","fps512","fps1000","fps2000"]


def resample (V ,pol ):
    if pol in ("patch","uni14k"):
        return V ,(pol =="patch")
    if pol .startswith ("voxel"):
        return voxel (V ,float (pol [5 :])),False 
    if pol .startswith ("fps"):
        return fps (V ,int (pol [3 :])),False 
    return V ,False 


jparts ={str (p .part_nr ):p for p in jd .iter_parts (TEACHER )if str (p .part_nr )in set (bridge )}
agg ={p :[0 ,0 ,0 ]for p in POLICIES }
nverts ={p :[]for p in POLICIES }
for pn in bridge :
    p =jparts .get (pn )
    if p is None :
        continue 
    cat =pn .split (".")[-1 ]
    _ ,gt ,gd =jd .dedup_connection_points (p )
    steps =glob .glob (os .path .join (STEPDIR ,f"*{cat }*.stp"))
    if not steps :
        continue 
    Vs ,_ =sj .load_any_mesh (steps [0 ],deflection =DEFL )
    Vs =np .asarray (Vs ,float )
    R ,t ,res =ce .align_frames (Vs ,np .asarray (p .vertices ,float ))
    for pol in POLICIES :
        Vr ,patch =resample (Vs ,pol )
        nverts [pol ].append (len (Vr ))
        tp ,fp ,fn =score (infer (Vr ,pn ,patch ),R ,t ,gt ,gd )
        agg [pol ][0 ]+=tp ;agg [pol ][1 ]+=fp ;agg [pol ][2 ]+=fn 

print (f"JSON native ceiling on these 9 = F1 0.94 (memorised)\n")
print (f"  {'policy':<10}{'~nverts':>9}{'F1':>8}{'Jacc':>7}{'prec':>7}{'recall':>7}   TP/FP/FN")
best =None 
for pol in POLICIES :
    tp ,fp ,fn =agg [pol ]
    f1 =2 *tp /max (2 *tp +fp +fn ,1 );jac =tp /max (tp +fp +fn ,1 )
    pr =tp /max (tp +fp ,1 );rc =tp /max (tp +fn ,1 )
    nv =int (np .median (nverts [pol ]))
    mark =""
    if best is None or f1 >best [1 ]:
        best =(pol ,f1 );mark ="  <=="
    print (f"  {pol :<10}{nv :>9}{f1 :>8.3f}{jac :>7.3f}{pr :>7.3f}{rc :>7.3f}   {tp }/{fp }/{fn }{mark }")
print (f"\n>>> best sampling policy = {best [0 ]}  (STEP F1 {best [1 ]:.3f})   [JSON ceiling 0.94]")
print ("DONE")
