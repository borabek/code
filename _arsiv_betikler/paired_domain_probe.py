# -*- coding: utf-8 -*-
"""P1 paired-domain probe (TRAINING_JSON_WSCAD_90 plan, section 5).

The crux question: the SAME physical part scores F1~0.21 ten its coarse Desktop-JSON
mesh but F1~0.09 ten its dense WSCAD STEP twin. WHERE does it break? This runs the
winner checkpoint ten three inputs per catalog and scores all against the SAME
Desktop-JSON GT (frame-aligned for STEP):

  JSON      : native Desktop-JSON mesh (~157-965 verts)         [source domain]
  STEP      : WSCAD STEP tessellation (~26k-97k verts)          [target domain]
  STEP->den : STEP randomly subsampled to ~JSON vertex count    [density-parity test]

If STEP->den recovers the JSON F1, the break is MESH DENSITY (=> plan E2 sampling
parity). If not, it is frame/axis-permutation (E1) or local-feature (E4). This is a
LEAKY diagnostic gate, NOT a reported accuracy (uses the seen 9 bridge catalogs).

Usage:
  .venv/Scripts/python.exe paired_domain_probe.py
"""
import glob ,os ,sys 
import numpy as np 
from scipy .spatial import cKDTree 
import cp_regressor as cpr ,cp_targets as ct ,json_dataset as jd ,metrics as mcp 
import train_cp as tc ,step_to_json as sj ,cad_eval as ce 

TEACHER =r"C:\Users\DE00024082\Desktop\JSON"
STEPDIR ="_cad_eval_pxc"
CK =sys .argv [1 ]if len (sys .argv )>1 else "checkpoints/cp_real_v1_best.ckpt"
THR =0.30 ;DEFL =0.3 ;DIST =5.0 
dedup =tc ._nms_radius (None ,5.0 )

bridge =[l .strip ()for l in open ("_bridge_test.txt",encoding ="utf-8")if l .strip ()]
model ,meta ,_ =cpr .load_model (CK ,device ="cuda")


FORCE_NOPATCH ="nopatch"in sys .argv 


def run (V ,pn ):
    V =np .asarray (V ,float )
    Vn ,_ ,scale =cpr .normalize_vertices (V )
    use_patch =False if FORCE_NOPATCH else cpr .is_patch_part (pn )
    arr =cpr .infer_knngraph (model ,meta ,Vn ,device ="cuda",max_gpu_verts =14000 ,
    offset_scale =scale ,patch =use_patch ,part_nr =pn )
    preds =ct .decode_predictions (V ,arr ,heatmap_thresh =THR ,nms_radius_mm =dedup ,min_votes =1 )
    return preds 


def nn_spacing (V ):
    V =np .asarray (V ,float )
    if len (V )<2 :
        return 0.0 
    d ,_ =cKDTree (V ).query (V ,k =2 )
    return float (np .median (d [:,1 ]))


def score (preds ,R ,t ,gt ,gd ):
    """transform preds' points+dirs by (R,t) [identity for JSON], score vs GT."""
    P =[]
    for q in preds :
        if R is not None :
            pt =R @np .asarray (q ["point"],float )+t 
            dv =R @np .asarray (q .get ("direction",[0 ,0 ,1 ]),float )
        else :
            pt =q ["point"];dv =q .get ("direction",[0 ,0 ,1 ])
        P .append ({"point":pt ,"direction":dv })
    r =mcp .keypoint_report (P ,gt ,gd ,dist_thresh_mm =DIST )
    f1 =2 *r ["tp"]/max (2 *r ["tp"]+r ["fp"]+r ["fn"],1 )
    return r ["tp"],r ["fp"],r ["fn"],f1 ,len (P )


jparts ={str (p .part_nr ):p for p in jd .iter_parts (TEACHER )if str (p .part_nr )in set (bridge )}
rng =np .random .RandomState (0 )

print (f"{'catalog':<12}{'n_json':>7}{'n_step':>8}{'sp_json':>8}{'sp_step':>8}"
f"{'res':>6}  {'JSON f1':>8}{'STEP f1':>8}{'S>den f1':>9}   J/S/Sden pred")
agg ={"J":[0 ,0 ,0 ],"S":[0 ,0 ,0 ],"D":[0 ,0 ,0 ]}
for pn in bridge :
    p =jparts .get (pn )
    if p is None :
        continue 
    cat =pn .split (".")[-1 ]
    _ ,gt ,gd =jd .dedup_connection_points (p )
    Vj =np .asarray (p .vertices ,float )
    pj =run (Vj ,pn )
    tj ,fpj ,fnj ,f1j ,npj =score (pj ,None ,None ,gt ,gd )

    steps =glob .glob (os .path .join (STEPDIR ,f"*{cat }*.stp"))
    if not steps :
        print (f"{cat :<12}{len (Vj ):>7}   no STEP")
        continue 
    Vs ,_Fs =sj .load_any_mesh (steps [0 ],deflection =DEFL )
    Vs =np .asarray (Vs ,float )
    R ,t ,res =ce .align_frames (Vs ,Vj )
    if "prealign"in sys .argv :# rotate STEP into JSON frame first (isolate density)
        ps =run (Vs @R .T +t ,pn )
        ts ,fps ,fns ,f1s ,nps =score (ps ,None ,None ,gt ,gd )
    else :
        ps =run (Vs ,pn )
        ts ,fps ,fns ,f1s ,nps =score (ps ,R ,t ,gt ,gd )

    idx =rng .choice (len (Vs ),min (len (Vj ),len (Vs )),replace =False )
    Vd =Vs [idx ]
    pd =run (Vd ,pn )
    td ,fpd ,fnd ,f1d ,npd =score (pd ,R ,t ,gt ,gd )

    for k ,(tp ,fp ,fn )in zip (("J","S","D"),((tj ,fpj ,fnj ),(ts ,fps ,fns ),(td ,fpd ,fnd ))):
        agg [k ][0 ]+=tp ;agg [k ][1 ]+=fp ;agg [k ][2 ]+=fn 
    print (f"{cat :<12}{len (Vj ):>7}{len (Vs ):>8}{nn_spacing (Vj ):>8.2f}{nn_spacing (Vs ):>8.2f}"
    f"{res :>6.1f}  {f1j :>8.2f}{f1s :>8.2f}{f1d :>9.2f}   {npj }/{nps }/{npd }")

print ()
for k ,lab in (("J","JSON native"),("S","STEP native"),("D","STEP->JSON-density")):
    tp ,fp ,fn =agg [k ]
    f1 =2 *tp /max (2 *tp +fp +fn ,1 );jac =tp /max (tp +fp +fn ,1 )
    print (f"  {lab :<20} TP={tp :3d} FP={fp :4d} FN={fn :3d}  F1={f1 :.3f}  Jaccard={jac :.3f}")
print ("\nDONE")
