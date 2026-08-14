# -*- coding: utf-8 -*-
"""Render the user-approved CP set (cp_openings.connection_points) ten a part with the
semantic-coloured mesh + a GREEN ball at each CP (small outward offset so it sits ON the
opening, not floating). render->Read to confirm balls land ten the physical openings."""
import sys ,numpy as np 
import matplotlib ;matplotlib .use ("Agg");import matplotlib .pyplot as plt 
from mpl_toolkits .mplot3d .art3d import Poly3DCollection 
import scheffler_dataset as dataset ,connector3d ,cp_openings 
COL ={int (connector3d .HOUSING ):(185 ,185 ,185 ),int (connector3d .CONTACT ):(55 ,110 ,235 ),
int (connector3d .SNAP_POINT ):(240 ,215 ,40 ),int (connector3d .CABLE_ENTRY ):(25 ,225 ,55 ),
int (connector3d .LABEL_SURFACE ):(235 ,140 ,40 )}
pid =sys .argv [1 ];split =sys .argv [2 ]if len (sys .argv )>2 else "val"
elev =float (sys .argv [3 ])if len (sys .argv )>3 else 18.0 
azim =float (sys .argv [4 ])if len (sys .argv )>4 else -60.0 
s ={x ["part_id"]:x for x in dataset .load_split ("wscad_corpus_scheffler_exact",split ,verify_hashes =False )}[pid ]
V =np .asarray (s ["verts"],float );F =np .asarray (s ["faces"],int );L =np .asarray (s ["labels"])
cps =cp_openings .connection_points (V ,F ,L )
by ={}
for c in cps :by [c ["source_label"]]=by .get (c ["source_label"],0 )+1 
print (f"{pid }: {len (cps )} CP  by-class {by }")
fig =plt .figure (figsize =(7 ,7 ));ax =fig .add_subplot (111 ,projection ="3d")
fc =np .array ([COL .get (int (x ),(150 ,150 ,150 ))for x in L ],np .uint8 )[F ][:,:,:3 ].mean (1 )/255.0 
ax .add_collection3d (Poly3DCollection (V [F ],facecolors =fc ,edgecolors ="none"))
diag =float (np .linalg .norm (V .max (0 )-V .min (0 )))
for c in cps :
    q =c ["point"]+c ["direction"]*diag *0.012 # small offset -> sits ten opening
    ax .scatter ([q [0 ]],[q [1 ]],[q [2 ]],c ="lime",s =230 ,edgecolors ="k",linewidths =0.8 ,depthshade =False )
ax .set_xlim (V [:,0 ].min (),V [:,0 ].max ());ax .set_ylim (V [:,1 ].min (),V [:,1 ].max ());ax .set_zlim (V [:,2 ].min (),V [:,2 ].max ())
try :ax .set_box_aspect (V .max (0 )-V .min (0 ))
except Exception :pass 
ax .view_init (elev =elev ,azim =azim );ax .set_axis_off ()
out =f"_vcp_{pid }.png";plt .tight_layout ();plt .savefig (out ,dpi =92 ,bbox_inches ="tight");print ("wrote",out )
