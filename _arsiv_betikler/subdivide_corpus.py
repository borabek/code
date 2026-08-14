# -*- coding: utf-8 -*-
"""Train-side densification (domain-gap fix). hierpoint/knngraph is a POINT network
(_pca_normals_curvature computes features from the kNN neighbourhood, faces unused), so
midpoint-subdividing the coarse Desktop-JSON meshes toward STEP-like density gives the
model the dense-input regime it fails ten at STEP inference -- while keeping valid faces
for the loader and the CPs unchanged. Skips already-dense parts; caps iterations so the
corpus stays disk-sane. Output = a new corpus dir; the read-only Desktop\\JSON is untouched.

Usage:  .venv/Scripts/python.exe subdivide_corpus.py --target 10000 --out _densified
"""
import argparse ,os ,json 
import numpy as np 
import json_dataset as jd 

ap =argparse .ArgumentParser ()
ap .add_argument ("--src",default =r"C:\Users\DE00024082\Desktop\JSON")
ap .add_argument ("--out",default ="_densified")
ap .add_argument ("--target",type =int ,default =10000 ,help ="subdivide until >= this many verts")
ap .add_argument ("--max-iters",type =int ,default =3 )
a =ap .parse_args ()
os .makedirs (a .out ,exist_ok =True )


def subdivide (V ,F ):
    E =np .sort (np .vstack ([F [:,[0 ,1 ]],F [:,[1 ,2 ]],F [:,[2 ,0 ]]]),axis =1 )
    Eu ,inv =np .unique (E ,axis =0 ,return_inverse =True )
    mid =0.5 *(V [Eu [:,0 ]]+V [Eu [:,1 ]])
    Vn =np .vstack ([V ,mid ])
    m =inv .reshape (3 ,len (F )).T +len (V )# midpoints of edges 01,12,20 per face
    a_ ,b_ ,c_ =F [:,0 ],F [:,1 ],F [:,2 ]
    m01 ,m12 ,m20 =m [:,0 ],m [:,1 ],m [:,2 ]
    Fn =np .vstack ([np .column_stack ([a_ ,m01 ,m20 ]),
    np .column_stack ([m01 ,b_ ,m12 ]),
    np .column_stack ([m20 ,m12 ,c_ ]),
    np .column_stack ([m01 ,m12 ,m20 ])])
    return Vn ,Fn 


n =0 
for p in jd .iter_parts (a .src ):
    V =np .asarray (p .vertices ,float );F =np .asarray (p .faces ,int )
    n0 =len (V )
    it =0 
    while len (V )<a .target and it <a .max_iters and len (F ):
        V ,F =subdivide (V ,F );it +=1 
    _ ,gt ,gd =jd .dedup_connection_points (p )
    # keep ALL raw CPs (not dedup) so the file round-trips like the source
    lo =V .min (0 );hi =V .max (0 )
    cps =[]
    for i ,c in enumerate (p .cp_points ):
        d =p .cp_directions [i ]
        cps .append ({"Index":i ,"Name":str (p .cp_names [i ])if i <len (p .cp_names )else str (i ),
        "Point":{"X":float (c [0 ]),"Y":float (c [1 ]),"Z":float (c [2 ])},
        "InsertDirection":{"X":float (d [0 ]),"Y":float (d [1 ]),"Z":float (d [2 ])}})
    obj ={"PartNr":str (p .part_nr ),
    "Graphic3d":{"Points":[{"X":float (v [0 ]),"Y":float (v [1 ]),"Z":float (v [2 ])}for v in V ],
    "Indices":F .reshape (-1 ).astype (int ).tolist ()},
    "BoundingBox":{"Dimension":{"X":float (hi [0 ]-lo [0 ]),"Y":float (hi [1 ]-lo [1 ]),"Z":float (hi [2 ]-lo [2 ])},
    "Location":{"X":float (lo [0 ]),"Y":float (lo [1 ]),"Z":float (lo [2 ])}},
    "ConnectionPoints":cps }
    json .dump (obj ,open (os .path .join (a .out ,str (p .part_nr )+".json"),"w"))
    n +=1 
    if n %50 ==0 :
        print (f"  {n } parts... last {p .part_nr }: {n0 } -> {len (V )} verts ({it } subdiv)",flush =True )

print (f"DONE: {n } parts -> {a .out }/")
