# -*- coding: utf-8 -*-
"""Export the CP review as STL (real 3D, opens in any CAD / Windows 3D Viewer with proper
z-buffering). NOTE: STL carries NO colour -- each CP is a protruding SPHERE (a bump) ten the
model, not a green dot. Balls are offset outward so they clearly stick out of the surface.
Per part -> <out>/<pid>_CP.stl  (mesh + CP spheres merged into one solid).

human labels by default; --model adds the model-predicted CP STL too.
Usage: PYTHONPATH=_diffusion_net_repo/src .venv/Scripts/python.exe cp_export_stl.py --split test_locked
"""
import argparse ,os 
import numpy as np 
import trimesh 
import scheffler_dataset as dataset 
import cp_openings ,viz_semantic_full 


def build (V ,F ,cps ):
    diag =float (np .linalg .norm (V .max (0 )-V .min (0 )))or 1.0 
    r =diag *0.030 
    parts =[trimesh .Trimesh (V ,F ,process =False )]
    for c in cps :
        p =np .asarray (c ["point"],float );d =np .asarray (c ["direction"],float )
        n =np .linalg .norm (d );d =d /n if n >1e-6 else (p -V .mean (0 ))
        s =trimesh .creation .icosphere (subdivisions =3 ,radius =r )
        s .apply_translation (p +d *r *0.55 )# half-embedded -> clear bump that stays attached
        parts .append (s )
    m =trimesh .util .concatenate (parts )
    R ,c =viz_semantic_full .reorient (V )# isometric so it opens ten the large face
    T =np .eye (4 );T [:3 ,:3 ]=R ;T [:3 ,3 ]=-R @V .mean (0 )
    m .apply_transform (T )
    return m 


def main ():
    ap =argparse .ArgumentParser ()
    ap .add_argument ("--split",default ="test_locked");ap .add_argument ("--out",default ="_cp_stl")
    ap .add_argument ("--min-v",type =int ,default =20 );ap .add_argument ("--model",action ="store_true")
    a =ap .parse_args ()
    os .makedirs (a .out ,exist_ok =True )
    model =meta =None 
    if a .model :
        import diffusionnet 
        model ,meta ,_ =diffusionnet .load_checkpoint ("results/scheffler_semantic/refit91.pt",device ="cuda")
    samples =dataset .load_split ("wscad_corpus_scheffler_exact",a .split ,
    allow_locked =(a .split =="test_locked"),verify_hashes =False )
    for s in samples :
        pid =s ["part_id"];V =np .asarray (s ["verts"],float );F =np .asarray (s ["faces"],int )
        cps =cp_openings .connection_points (V ,F ,np .asarray (s ["labels"]),min_v =a .min_v )
        build (V ,F ,cps ).export (os .path .join (a .out ,f"{pid }_CP.stl"))
        line =f"  {pid }: {len (cps )} CP -> {pid }_CP.stl"
        if a .model :
            import diffusionnet 
            plab ,probs =diffusionnet .predict (model ,meta ,V ,F ,device ="cuda",
            op_cache_dir ="results/scheffler_semantic/operators",return_probs =True )
            mcps =cp_openings .connection_points (V ,F ,np .asarray (plab ),min_v =a .min_v ,probs =probs ,vertex_conf =0.9 )
            build (V ,F ,mcps ).export (os .path .join (a .out ,f"{pid }_CP_model.stl"))
            line +=f" | model {len (mcps )} -> {pid }_CP_model.stl"
        print (line )
    print (f"\n{len (samples )} parts -> {a .out }/  (open <pid>_CP.stl; CP = protruding sphere, no colour in STL)")
    print ("DONE")


if __name__ =="__main__":
    main ()
