# -*- coding: utf-8 -*-
"""Kendi teshisim: mesh baglanti-segmentasyonuyla boyali + CP noktalari, 4 acidan. Hangi CP real
opening bolgesinde (mavi) hangisi duz govdede (gri) -- durustce goreyim."""
import os ,sys ,glob ,json 
import numpy as np ,torch 
import matplotlib ;matplotlib .use ("Agg");import matplotlib .pyplot as plt 
from mpl_toolkits .mplot3d .art3d import Poly3DCollection 
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
import diffusionnet as D ,connector3d ,thesis_remesh ,robot_cp 
from infer_step_cp import step_to_mesh ,load_any 

CE =int (connector3d .CABLE_ENTRY );CT =int (connector3d .CONTACT );OP ="results/step_infer/ops"
dev ="cuda"if torch .cuda .is_available ()else "cpu"
STEP ={os .path .basename (s ).split ("_")[1 ]:s for s in glob .glob ("all_wscad_stp/*.stp")}
pid =sys .argv [1 ]if len (sys .argv )>1 else "3001381"

cks =json .load (open ("cp_config.json"))["robot_vote2_checkpoints"]
models =[load_any (c ,dev =dev )[:2 ]for c in cks ]
Vr ,Fr =step_to_mesh (STEP [pid ])
V ,F =thesis_remesh .remesh_uniform (Vr ,Fr ,target =6000 )
V =np .ascontiguousarray (V ,np .float64 );F =np .ascontiguousarray (F ,np .int64 )
acc =None 
for model ,meta in models :
    _ ,pb =D .predict (model ,meta ,V ,F ,device =dev ,op_cache_dir =OP ,return_probs =True )
    pb =np .asarray (pb ,float );acc =pb if acc is None else acc +pb 
probs =acc /len (models );conn =probs [:,CE ]+probs [:,CT ]
cps =robot_cp .extract (models ,STEP [pid ],dev ,0.5 ,3 )
P =np .array ([c ["point"]for c in cps ],float );PD =np .array ([c ["direction"]for c in cps ],float )

tri =V [F ];fconn =conn [F ].mean (1 )
col =np .zeros ((len (F ),3 ));col [:]=[0.72 ,0.72 ,0.72 ]
m =fconn >=0.5 
col [m ]=np .stack ([0.1 *np .ones (m .sum ()),0.35 *np .ones (m .sum ()),(0.6 +0.4 *fconn [m ]).clip (0 ,1 )],1 )
ext =V .max (0 )-V .min (0 );c0 =V .mean (0 );rr =ext .max ()*0.55 ;L =ext .max ()*0.14 
fig =plt .figure (figsize =(16 ,4.3 ))
angs =[(20 ,-60 ),(20 ,30 ),(20 ,120 ),(75 ,-90 )]
for k ,(ev ,az )in enumerate (angs ):
    ax =fig .add_subplot (1 ,4 ,k +1 ,projection ="3d")
    ax .add_collection3d (Poly3DCollection (tri ,facecolors =np .clip (col ,0 ,1 ),edgecolors ="none",alpha =0.95 ))
    for i in range (len (P )):
        p =P [i ];d =PD [i ]/(np .linalg .norm (PD [i ])+1e-9 )*L 
        ax .quiver (*p ,*d ,color ="red",linewidth =2.6 ,arrow_length_ratio =0.4 ,zorder =10 )
        ax .scatter (*p ,s =55 ,c ="red",edgecolors ="black",linewidths =0.7 ,depthshade =False ,zorder =11 )
        ax .text (p [0 ],p [1 ],p [2 ],f"{i +1 }",color ="black",fontsize =9 ,zorder =12 )
    ax .set_xlim (c0 [0 ]-rr ,c0 [0 ]+rr );ax .set_ylim (c0 [1 ]-rr ,c0 [1 ]+rr );ax .set_zlim (c0 [2 ]-rr ,c0 [2 ]+rr )
    try :ax .set_box_aspect (ext )
    except Exception :pass 
    ax .view_init (elev =ev ,azim =az );ax .set_axis_off ();ax .set_title (f"aci {k +1 }",fontsize =9 )
fig .suptitle (f"{pid }: MAVI=model baglanti-segmentasyonu, kirmizi nokta+ok=robot CP  (CP mavide=makul, gride=wrong)",fontsize =12 )
plt .tight_layout (rect =[0 ,0 ,1 ,0.94 ])
plt .savefig (f"results/_inspect_{pid }.png",dpi =105 ,bbox_inches ="tight")
print (f"-> results/_inspect_{pid }.png | {len (cps )} CP")
for i ,c in enumerate (cps ):
# CP'nin 4mm cevresindeki mean baglanti olasiligi
    near =np .linalg .norm (V -np .asarray (c ["point"],float ),axis =1 )<=4.0 
    mp =float (conn [near ].mean ())if near .any ()else 0.0 
    print (f"  CP{i +1 }: confidence {c ['confidence']} oy {c ['votes']} | cevre baglanti-olasiligi {mp :.2f} {'(MAVI-makul)'if mp >=0.4 else '(GRI-supheli)'}")
