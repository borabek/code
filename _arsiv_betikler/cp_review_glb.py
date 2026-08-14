# -*- coding: utf-8 -*-
"""Colour-numbered CP review GLB (real 3D, colour + proper z-buffer in Windows 3D Viewer).
Each CP ball gets a DISTINCT colour from a fixed palette so it maps to the numbered entry in
<pid>.json (CP1=green, CP2=red, CP3=cyan, ...). This lets the colour GLB alone drive the
sealed-benchmark confirmation -- no PNG needed. Balls offset outward so they sit proud of the
opening. Isometric orientation so it opens ten the large face.

Per part -> _cp_confirm/<pid>_review.glb  (+ the existing <pid>.json legend/coords).
Usage: PYTHONPATH=_diffusion_net_repo/src .venv/Scripts/python.exe cp_review_glb.py --split test_locked
"""
import argparse ,os 
import numpy as np 
import trimesh 
import scheffler_dataset as dataset 
import cp_openings ,viz_semantic_full 

# GREY mesh + big bright BALLS that pop (user: a colour-coded mesh hid the markers). Ball colour =
# class: green = CableEntry (the CP), blue = Contact (auxiliary). Both shown so nothing looks empty.
import connector3d as _c3 
CE =int (_c3 .CABLE_ENTRY );CT =int (_c3 .CONTACT )
CLASS_COL ={CE :(25 ,235 ,60 ),CT :(40 ,120 ,245 )}# green=CableEntry, blue=Contact
CLASS_NAME ={CE :"CableEntry",CT :"Contact"}


def review_glb (V ,F ,cps ,labels ,out ):
    R ,ctr =viz_semantic_full .reorient (V )
    Vt =(V -ctr )@R .T 
    scene =trimesh .Scene ()
    grey =np .tile ((165 ,165 ,165 ,255 ),(len (Vt ),1 )).astype (np .uint8 )
    scene .add_geometry (trimesh .Trimesh (Vt ,F ,vertex_colors =grey ,process =False ),node_name ="mesh")
    diag =float (np .linalg .norm (Vt .max (0 )-Vt .min (0 )))or 1.0 
    # AUTO ball size so markers never visually overlap ten dense parts (shrink to a fraction of the
    # nearest CP-CP gap, capped/floored for visibility).
    P =np .array ([c ["point"]for c in cps ],float )
    if len (P )>1 :
        from scipy .spatial .distance import pdist 
        mind =float (pdist (P ).min ())
    else :
        mind =diag 
    r =min (diag *0.022 ,0.30 *mind );r =max (r ,diag *0.011 )# small enough to see the opening
    legend =[]
    for i ,c in enumerate (cps ):
        p =(np .asarray (c ["point"],float )-ctr )@R .T 
        d =np .asarray (c ["direction"],float )@R .T 
        n =np .linalg .norm (d );d =d /n if n >1e-6 else (p -Vt .mean (0 ))
        lbl =int (c ["source_label"]);rgb =CLASS_COL .get (lbl ,(200 ,200 ,200 ))
        s =trimesh .creation .icosphere (subdivisions =3 ,radius =r )
        s .apply_translation (p +d *r *0.15 )
        s .visual .vertex_colors =np .tile (rgb +(255 ,),(len (s .vertices ),1 )).astype (np .uint8 )
        scene .add_geometry (s ,node_name =f"CP{i +1 }_{CLASS_NAME .get (lbl ,'?')}")
        legend .append ((i +1 ,CLASS_NAME .get (lbl ,"?"),lbl ))
    scene .export (out )
    return legend 


def main ():
    ap =argparse .ArgumentParser ()
    ap .add_argument ("--split",default ="test_locked");ap .add_argument ("--out",default ="_cp_confirm")
    ap .add_argument ("--min-v",type =int ,default =20 )
    a =ap .parse_args ()
    os .makedirs (a .out ,exist_ok =True )
    samples =dataset .load_split ("wscad_corpus_scheffler_exact",a .split ,
    allow_locked =(a .split =="test_locked"),verify_hashes =False )
    for s in samples :
        pid =s ["part_id"];V =np .asarray (s ["verts"],float );F =np .asarray (s ["faces"],int )
        # BOTH classes shown (thesis): CableEntry (green) + Contact (blue). No cross-class merge
        # (dedupe_mm=0) so every feature is visible; the CP COUNT metric stays CableEntry-only.
        L =np .asarray (s ["labels"])
        cps =cp_openings .connection_points (V ,F ,L ,min_v =a .min_v ,classes =(CE ,CT ),dedupe_mm =0.0 )
        leg =review_glb (V ,F ,cps ,L ,os .path .join (a .out ,f"{pid }_review.glb"))
        nce =sum (1 for _ ,nm ,_ in leg if nm =="CableEntry");nct =len (leg )-nce 
        print (f"  {pid }: CableEntry(green)={nce }  Contact(blue)={nct } -> {pid }_review.glb")
    print (f"\n{len (samples )} parts -> {a .out }/  (grey mesh; big green=CableEntry CP, blue=Contact)")
    print ("DONE")


if __name__ =="__main__":
    main ()
