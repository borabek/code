# Overlay detected connection points ten a part mesh as an OBJ scene you can open
# in Blender / MeshLab to EYEBALL whether the points land ten real connection
# features (openings / terminals). Without ground-truth labels ten real parts,
# this visual check is the only way to judge a detection's correctness.
#
# Each detected point becomes a small octahedron marker; its approach vector is
# drawn as a short line. The part mesh and the markers are separate OBJ groups
# so a viewer can colour them differently.
#
# Usage:
#   python viz_preds.py real_json/PART.json preds_real.json --out PART_scene.obj
#   (the part_nr in the preds file is matched to the mesh's PartNr)
import os 
import json 
import argparse 

import numpy as np 

import step_to_json as sj # mesh loader that also accepts raw .stp/.stl/.obj


def _octahedron (c ,r ):
    """6 verts / 8 faces around centre c (radius r)."""
    v =np .array ([[r ,0 ,0 ],[-r ,0 ,0 ],[0 ,r ,0 ],
    [0 ,-r ,0 ],[0 ,0 ,r ],[0 ,0 ,-r ]],float )+np .asarray (c )
    f =[[0 ,2 ,4 ],[2 ,1 ,4 ],[1 ,3 ,4 ],[3 ,0 ,4 ],
    [2 ,0 ,5 ],[1 ,2 ,5 ],[3 ,1 ,5 ],[0 ,3 ,5 ]]
    return v ,np .array (f )


def write_scene (mesh_json ,preds_json ,out_obj ,marker_frac =0.012 ,arrow_len_frac =0.06 ):
    part =next (sj .iter_parts_any (mesh_json ))# .json OR raw .stp/.stl/.obj
    V ,F =part .vertices ,part .faces 
    diag =float (np .linalg .norm (V .max (0 )-V .min (0 )))
    r =marker_frac *diag 
    L =arrow_len_frac *diag 

    # find this part's predictions (match PartNr, else take the only/first part)
    pred =json .load (open (preds_json ,encoding ="utf-8"))
    nodes =[]
    for p in pred .get ("parts",[]):
        if p .get ("part_nr")==part .part_nr or len (pred ["parts"])==1 :
            nodes =p .get ("connection_points",[])
            break 

    lines =["# part mesh + %d detected connection point(s)"%len (nodes ),
    "o part_mesh"]
    for x ,y ,z in V :
        lines .append ("v %.5f %.5f %.5f"%(x ,y ,z ))
    for a ,b ,c in F :
        lines .append ("f %d %d %d"%(a +1 ,b +1 ,c +1 ))

    off =len (V )# running vertex offset (1-based in OBJ)
    arrows =[]
    lines .append ("o detected_points")
    for nd in nodes :
        c =nd .get ("entry_point")or nd .get ("position")
        if c is None :
            continue 
        mv ,mf =_octahedron (c ,r )
        for x ,y ,z in mv :
            lines .append ("v %.5f %.5f %.5f"%(x ,y ,z ))
        for a ,b ,cc in mf :
            lines .append ("f %d %d %d"%(off +a +1 ,off +b +1 ,off +cc +1 ))
        ap =nd .get ("approach_vector")
        if ap is not None :
            tip =np .asarray (c )+L *np .asarray (ap )
            arrows .append ((c ,tip ))
        off +=len (mv )

    if arrows :
        lines .append ("o approach_vectors")
        for c ,tip in arrows :
            lines .append ("v %.5f %.5f %.5f"%tuple (c ))
            lines .append ("v %.5f %.5f %.5f"%tuple (tip ))
            lines .append ("l %d %d"%(off +1 ,off +2 ))
            off +=2 

    os .makedirs (os .path .dirname (os .path .abspath (out_obj )),exist_ok =True )
    with open (out_obj ,"w",encoding ="utf-8")as fh :
        fh .write ("\n".join (lines )+"\n")
    print ("wrote %s  (%d verts, %d detected point(s))"%(out_obj ,len (V ),len (nodes )))
    return out_obj 


def main (argv =None ):
    ap =argparse .ArgumentParser (description ="Overlay detected CPs on a part mesh -> OBJ scene")
    ap .add_argument ("mesh_json",help ="the part JSON (mesh)")
    ap .add_argument ("preds_json",help ="predictions JSON from predict.py")
    ap .add_argument ("--out",required =True ,help ="output .obj scene")
    args =ap .parse_args (argv )
    write_scene (args .mesh_json ,args .preds_json ,args .out )


def _selftest ():
    """Write a scene for a tiny tetra mesh + one fake detection (no CAD/torch)."""
    import tempfile 
    V =[[0 ,0 ,0 ],[10 ,0 ,0 ],[0 ,10 ,0 ],[0 ,0 ,10 ]]
    F =[[0 ,1 ,2 ],[0 ,1 ,3 ],[0 ,2 ,3 ],[1 ,2 ,3 ]]
    mesh ={"PartNr":"VIZTEST",
    "Graphic3d":{"Points":[{"X":x ,"Y":y ,"Z":z }for x ,y ,z in V ],
    "Indices":[i for f in F for i in f ]},
    "ConnectionPoints":[]}
    preds ={"parts":[{"part_nr":"VIZTEST","connection_points":[
    {"entry_point":[3.0 ,3.0 ,0.0 ],"approach_vector":[0 ,0 ,1 ]}]}]}
    d =tempfile .mkdtemp ()
    mj ,pj ,oo =(os .path .join (d ,n )for n in ("m.json","p.json","s.obj"))
    json .dump (mesh ,open (mj ,"w"));json .dump (preds ,open (pj ,"w"))
    write_scene (mj ,pj ,oo )
    assert os .path .getsize (oo )>0 
    print ("viz_preds selftest OK: scene written with 1 marker")


if __name__ =="__main__":
    import sys 
    if len (sys .argv )==1 :
        _selftest ()
    else :
        main ()
