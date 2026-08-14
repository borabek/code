"""Diagnostic: is the project even *trainable* to good numbers?
Two questions, answered ten the real corpus WITHOUT any model:
  (1) Ceiling: if we decode the GROUND-TRUTH target, do we recover the CPs?
      (F1 well below 1.0 => the encode/decode loses CPs; no model can beat it.)
  (2) Sparsity: how many vertices actually light up (heat>thr) per CP, before
      and after the knngraph subsample? (~1 => min_votes>=2 must collapse recall.)
"""
import sys ,numpy as np 
import json_dataset as jd 
import cp_targets as ct 
import metrics as mcp 

SRC =sys .argv [1 ]if len (sys .argv )>1 else r"C:\Users\DE00024082\Desktop\JSON"
THR =0.3 ;NMS =5.0 ;DIST =5.0 ;CAP =6000 

def match (gt ,pred ,r =DIST ):
    """Optimal (Hungarian) nearest match -- SINGLE SOURCE OF TRUTH shared with
    metrics.match_predictions (used everywhere else in the pipeline), so this
    ceiling number can't be pessimistic relative to a greedy re-implementation
    (a greedy match can orphan a perfect match behind a mediocre one processed
    first, undercounting TP ten ambiguous/dense-CP parts)."""
    if len (gt )==0 :return 0 ,len (pred ),0 
    if len (pred )==0 :return 0 ,0 ,len (gt )
    P =np .array ([p ["point"]for p in pred ])
    matches ,unmatched_pred ,unmatched_gt =mcp .match_predictions (P ,np .asarray (gt ),r )
    return len (matches ),len (unmatched_pred ),len (unmatched_gt )

def agg (rows ):
    tp =sum (r [0 ]for r in rows );fp =sum (r [1 ]for r in rows );fn =sum (r [2 ]for r in rows )
    p =tp /(tp +fp )if tp +fp else 0 ;r =tp /(tp +fn )if tp +fn else 0 
    f1 =2 *p *r /(p +r )if p +r else 0 
    return tp ,fp ,fn ,p ,r ,f1 

parts =[p for p in jd .iter_parts (SRC )if p .n_cps >0 ]
print (f"{len (parts )} parts with CPs\n")

for mv in (1 ,2 ,3 ):
    rows =[]
    for part in parts :
        _ ,bp ,bd =jd .dedup_connection_points (part )
        tgt ,mask ,sigma =ct .encode_targets (part .vertices ,bp ,bd )
        preds =ct .decode_predictions (part .vertices ,tgt ,heatmap_thresh =THR ,
        nms_radius_mm =NMS ,min_votes =mv )
        rows .append (match (bp ,preds ))
    tp ,fp ,fn ,p ,r ,f1 =agg (rows )
    print (f"GT-decode ceiling  min_votes={mv }: F1={f1 :.3f}  P={p :.3f} R={r :.3f}  "
    f"TP={tp } FP={fp } FN={fn }")

    # sparsity: hot vertices per CP, full vs subsampled
full_hot ,sub_hot ,n_cp_tot ,cps_per_part =[],[],0 ,[]
rng =np .random .RandomState (0 )
for part in parts :
    _ ,bp ,bd =jd .dedup_connection_points (part )
    cps_per_part .append (len (bp ))
    tgt ,mask ,sigma =ct .encode_targets (part .vertices ,bp ,bd )
    h =tgt [:,ct .HEATMAP ]
    nearest ,_ =ct ._nearest_cp (part .vertices ,bp )
    for j in range (len (bp )):
        full_hot .append (int (((h >THR )&(nearest ==j )).sum ()))
        # subsample like the knngraph trainer (force-keep exact peaks)
    n =len (part .vertices )
    if n >CAP :
        peaks =np .where (h >=1 -1e-4 )[0 ]
        rest =np .setdiff1d (np .arange (n ),peaks ,assume_unique =True )
        extra =rng .choice (rest ,CAP -len (peaks ),replace =False )
        idx =np .sort (np .concatenate ([peaks ,extra ]))
        hs ,ns =h [idx ],nearest [idx ]
    else :
        hs ,ns =h ,nearest 
    for j in range (len (bp )):
        sub_hot .append (int (((hs >THR )&(ns ==j )).sum ()))
    n_cp_tot +=len (bp )

fh ,sh =np .array (full_hot ),np .array (sub_hot )
print (f"\nCPs/part: median={int (np .median (cps_per_part ))} mean={np .mean (cps_per_part ):.1f} "
f"total={n_cp_tot }")
print (f"hot verts/CP (heat>{THR }) FULL mesh : median={int (np .median (fh ))} "
f"min={fh .min ()} %with<2={100 *(fh <2 ).mean ():.0f}%")
print (f"hot verts/CP (heat>{THR }) SUB {CAP }  : median={int (np .median (sh ))} "
f"min={sh .min ()} %with<2={100 *(sh <2 ).mean ():.0f}%  %with==0={100 *(sh ==0 ).mean ():.0f}%")
