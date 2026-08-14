# -*- coding: utf-8 -*-
"""Aligned-STEP overfit corpus builder (domain-gap fork, plan section 5).

For each of the 9 bridge catalogs: tessellate the WSCAD STEP twin, transfer the
Desktop-JSON GT onto the STEP mesh's frame (via cad_eval.align_frames), and write a
json_dataset-format part file whose MESH is the STEP tessellation but whose LABELS are
the real manufacturer CPs. Training an overfit model ten these and scoring it ten the same
answers: is the STEP input LEARNABLE for these CPs? (Coverage says yes -- CPs sit <=3.6mm
from a STEP vertex.) If yes, the whole gap is train/infer domain mismatch.

Frame: align_frames gives (R,t) with x_json = R @ x_step + t, so
       x_step = R.T @ (x_json - t)  ->  row form: (x_json - t) @ R.

Usage:  .venv/Scripts/python.exe make_step_corpus.py
"""
import glob ,os ,json 
import numpy as np 
import json_dataset as jd ,step_to_json as sj ,cad_eval as ce 

TEACHER =r"C:\Users\DE00024082\Desktop\JSON"
STEPDIR ="_cad_eval_pxc"
OUT ="_m0_step9"
DEFL =0.3 
CAP =80000 # uniform cap so the 4GB GPU/prebuild stay sane
os .makedirs (OUT ,exist_ok =True )
rng =np .random .RandomState (0 )

bridge =[l .strip ()for l in open ("_bridge_test.txt",encoding ="utf-8")if l .strip ()]
jparts ={str (p .part_nr ):p for p in jd .iter_parts (TEACHER )if str (p .part_nr )in set (bridge )}

for pn in bridge :
    p =jparts .get (pn )
    if p is None :
        continue 
    cat =pn .split (".")[-1 ]
    steps =glob .glob (os .path .join (STEPDIR ,f"*{cat }*.stp"))
    if not steps :
        print (f"  {pn }: no STEP, skip");continue 
    _ ,gt ,gd =jd .dedup_connection_points (p )
    gt =np .asarray (gt ,float );gd =np .asarray (gd ,float )
    Vj =np .asarray (p .vertices ,float )
    Vs ,Fs =sj .load_any_mesh (steps [0 ],deflection =DEFL )
    Vs =np .asarray (Vs ,float );Fs =np .asarray (Fs ,int )
    R ,t ,res =ce .align_frames (Vs ,Vj )

    if len (Vs )>CAP :# uniform cap; keep faces referencing kept verts
        keep =np .sort (rng .choice (len (Vs ),CAP ,replace =False ))
        remap =-np .ones (len (Vs ),int );remap [keep ]=np .arange (len (keep ))
        Vs =Vs [keep ]
        Fs =Fs [np .all (np .isin (Fs ,keep ),axis =1 )]
        Fs =remap [Fs ]

    gt_step =(gt -t )@R # JSON-frame GT -> STEP frame
    gd_step =gd @R 
    cps =[{"Index":i ,"Name":str (i ),
    "Point":{"X":float (a [0 ]),"Y":float (a [1 ]),"Z":float (a [2 ])},
    "InsertDirection":{"X":float (b [0 ]),"Y":float (b [1 ]),"Z":float (b [2 ])}}
    for i ,(a ,b )in enumerate (zip (gt_step ,gd_step ))]
    lo =Vs .min (0 );hi =Vs .max (0 )
    obj ={"PartNr":pn ,
    "Graphic3d":{"Points":[{"X":float (v [0 ]),"Y":float (v [1 ]),"Z":float (v [2 ])}for v in Vs ],
    "Indices":Fs .reshape (-1 ).astype (int ).tolist ()},
    "BoundingBox":{"Dimension":{"X":float (hi [0 ]-lo [0 ]),"Y":float (hi [1 ]-lo [1 ]),"Z":float (hi [2 ]-lo [2 ])},
    "Location":{"X":float (lo [0 ]),"Y":float (lo [1 ]),"Z":float (lo [2 ])}},
    "ConnectionPoints":cps }
    json .dump (obj ,open (os .path .join (OUT ,pn +".json"),"w"))
    print (f"  {pn }: STEP verts={len (Vs )} faces={len (Fs )} CP={len (cps )} align_res={res :.2f}mm")

print (f"DONE -> {OUT }/")
