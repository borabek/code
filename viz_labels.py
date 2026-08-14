# -*- coding: utf-8 -*-
"""Trusted, dependency-free label/prediction visualiser. Writes ONE OBJ per part
with the component mesh PLUS a small marker cube at each connection point, all in
the SAME raw coordinate frame — no transforms, no separate objects, so the markers
CANNOT drift from the mesh (unlike the retired export_glb, whose frame bug detached
them). Verified 2026-07-15 by eye ten ABB/drive parts: markers sit ten the terminals.

Green (default) = ground-truth CPs from the teacher JSON.  Red = model predictions
(pass --pred a predict-style JSON). Open the .obj in Windows 3D Viewer.

Usage:
  .venv/Scripts/python.exe viz_labels.py ABB.2CSR255180R1105
  .venv/Scripts/python.exe viz_labels.py --list _real_val.txt --out-dir _viz_val
  .venv/Scripts/python.exe viz_labels.py ABB.2CSR255180R1105 --pred _preds.json
"""
import argparse ,json ,os 
import numpy as np 

TEACHER =r"C:\Users\DE00024082\Desktop\JSON"
_CUBE =np .array ([[x ,y ,z ]for x in (-1 ,1 )for y in (-1 ,1 )for z in (-1 ,1 )],float )
_CT =[(0 ,1 ,3 ),(0 ,3 ,2 ),(4 ,6 ,7 ),(4 ,7 ,5 ),(0 ,4 ,5 ),(0 ,5 ,1 ),
(2 ,3 ,7 ),(2 ,7 ,6 ),(0 ,2 ,6 ),(0 ,6 ,4 ),(1 ,5 ,7 ),(1 ,7 ,3 )]


def _load_teacher (pn ):
    d =json .load (open (os .path .join (TEACHER ,pn +".json"),encoding ="utf-8"))
    V =np .array ([[p ["X"],p ["Y"],p ["Z"]]for p in d ["Graphic3d"]["Points"]],float )
    F =np .array (d ["Graphic3d"]["Indices"],int ).reshape (-1 ,3 )
    P =np .array ([[c ["Point"]["X"],c ["Point"]["Y"],c ["Point"]["Z"]]
    for c in d .get ("ConnectionPoints",[])],float )
    return V ,F ,P 


def _cubes (pts ,r ,base ):
    """vertex lines + face lines for a cube at each point, 1-indexed from `base`."""
    vlines ,flines =[],[]
    b =base 
    for p in pts :
        for c in _CUBE :
            vlines .append (f"v {p [0 ]+c [0 ]*r :.3f} {p [1 ]+c [1 ]*r :.3f} {p [2 ]+c [2 ]*r :.3f}")
        for t in _CT :
            flines .append (f"f {b +t [0 ]+1 } {b +t [1 ]+1 } {b +t [2 ]+1 }")
        b +=8 
    return vlines ,flines ,b 


def write_obj (pn ,out ,preds =None ):
    V ,F ,P =_load_teacher (pn )
    diag =float (np .linalg .norm (V .max (0 )-V .min (0 )))if len (V )else 1.0 
    r =diag *0.015 
    lines =[f"# {pn }: {len (V )} verts, {len (F )} tris, GT {len (P )} CP"
    +(f", pred {len (preds )}"if preds is not None else "")]
    # mesh
    lines .append ("o part_mesh")
    for v in V :
        lines .append (f"v {v [0 ]:.3f} {v [1 ]:.3f} {v [2 ]:.3f}")
    off =len (V )
    # GT markers (green)
    lines .append ("o CP_ground_truth")
    vl ,fl ,off =_cubes (P ,r ,off )
    lines +=vl 
    # prediction markers (red), slightly larger so both are visible
    pred_fl =[]
    if preds is not None and len (preds ):
        lines .append ("o CP_prediction")
        vl2 ,pred_fl ,off =_cubes (np .asarray (preds ,float ),r *1.3 ,off )
        lines +=vl2 
        # faces (mesh, then GT cubes, then pred cubes)
    for t in F :
        lines .append (f"f {t [0 ]+1 } {t [1 ]+1 } {t [2 ]+1 }")
    lines +=fl 
    lines +=pred_fl 
    open (out ,"w").write ("\n".join (lines ))
    from scipy .spatial import cKDTree 
    med =float (np .median (cKDTree (V ).query (P )[0 ]))if len (P )and len (V )else 0.0 
    return len (P ),med 


def main ():
    ap =argparse .ArgumentParser ()
    ap .add_argument ("part",nargs ="?",help ="a single teacher PartNr, e.g. ABB.2CSR255180R1105")
    ap .add_argument ("--list",dest ="lst",help ="file of PartNrs, one per line")
    ap .add_argument ("--pred",help ="predict-style JSON to overlay model CPs (red)")
    ap .add_argument ("--out-dir",default ="_viz")
    a =ap .parse_args ()

    preds_by_part ={}
    if a .pred :
        pj =json .load (open (a .pred ,encoding ="utf-8"))
        for p in pj .get ("parts",[pj ]):
            preds_by_part [p .get ("part_nr")]=[
            c .get ("point",c .get ("entry_point"))for c in p .get ("connection_points",[])]

    parts =([a .part ]if a .part else [])+(
    [l .strip ()for l in open (a .lst ,encoding ="utf-8")if l .strip ()]if a .lst else [])
    os .makedirs (a .out_dir ,exist_ok =True )
    for pn in parts :
        out =os .path .join (a .out_dir ,pn +".obj")
        n ,med =write_obj (pn ,out ,preds_by_part .get (pn ))
        print (f"  {pn :<26} {n } CP (surface {med :.1f}mm) -> {out }")
    print ("DONE")


if __name__ =="__main__":
    main ()
