# -*- coding: utf-8 -*-
"""URUN DEMOSU: random WSCAD .stp -> robot CP'leri finds. Tam kullanicinin istedigi sey.

Uretici ground-truth GEREKMEZ -- ham, never unseen a part ver, robot (vote>=2 urun) CP listesini
and gorseli produces. Her CP: point + direction (kirmizi ok) + size + depth + confidence + tier.

Kullanim: PYTHONPATH=_diffusion_net_repo/src .venv/Scripts/python.exe demo_random.py <pid1> <pid2> ...
"""
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


def main ():
    pids =sys .argv [1 :]or open ("_demo_parts.txt").read ().split ()
    cfg =json .load (open ("cp_config.json"))
    cks =cfg ["robot_vote2_checkpoints"]
    ca =float (cfg .get ("robot_conf_auto",0.75 ));mav =int (cfg .get ("robot_min_auto_votes",2 ))
    models =[load_any (c ,dev =dev )[:2 ]for c in cks ]
    print (f"URUN: {len (models )} model vote>=2 | conf_auto {ca } min_auto_votes {mav }\n",flush =True )

    n =len (pids );cols =3 ;rows =(n +cols -1 )//cols 
    fig =plt .figure (figsize =(5.0 *cols ,5.2 *rows ))
    allout ={}
    for k ,pid in enumerate (pids ):
        if pid not in STEP :
            print (f"  {pid }: STEP none");continue 
        cps =robot_cp .extract (models ,STEP [pid ],dev ,ca ,mav )# <-- URUN CIKTISI
        allout [pid ]=cps 
        na =sum (1 for c in cps if c ["tier"]=="auto")
        print (f"  {pid }: {len (cps )} CP  ({na } auto / {len (cps )-na } review)")
        for c in cps :
            print (f"     nokta {c ['point']}  direction {c ['direction']}  boy {c ['size_mm']}mm  "
            f"depth {c ['depth_mm']}mm  confidence {c ['confidence']}  oy {c ['votes']}  [{c ['tier']}]")
            # gorsel
        Vr ,Fr =step_to_mesh (STEP [pid ])
        V ,F =thesis_remesh .remesh_uniform (Vr ,Fr ,target =6000 )
        V =np .ascontiguousarray (V ,np .float64 );F =np .ascontiguousarray (F ,np .int64 )
        tri =V [F ];nrm =np .cross (tri [:,1 ]-tri [:,0 ],tri [:,2 ]-tri [:,0 ])
        nrm =nrm /(np .linalg .norm (nrm ,axis =1 ,keepdims =True )+1e-9 )
        lt =np .array ([.4 ,.3 ,1. ]);lt =lt /np .linalg .norm (lt )
        col =np .stack ([0.6 +0.35 *np .abs (nrm @lt )]*3 ,1 )
        ax =fig .add_subplot (rows ,cols ,k +1 ,projection ="3d")
        ax .add_collection3d (Poly3DCollection (tri ,facecolors =np .clip (col ,0 ,1 ),edgecolors ="none",alpha =0.9 ))
        ext =V .max (0 )-V .min (0 );L =ext .max ()*0.20 ;c0 =V .mean (0 );rr =ext .max ()*0.55 
        for c in cps :
        # STEP frame == remesh frame, dogrudan ciz
            p =np .asarray (c ["point"],float );d =np .asarray (c ["direction"],float )
            d =d /(np .linalg .norm (d )+1e-9 )*L 
            color ="red"if c ["tier"]=="auto"else "orange"
            ax .quiver (p [0 ],p [1 ],p [2 ],d [0 ],d [1 ],d [2 ],color =color ,linewidth =2.4 ,
            arrow_length_ratio =0.35 ,zorder =10 )
            ax .scatter (*p ,s =45 ,c =color ,depthshade =False ,zorder =10 )
        ax .set_xlim (c0 [0 ]-rr ,c0 [0 ]+rr );ax .set_ylim (c0 [1 ]-rr ,c0 [1 ]+rr );ax .set_zlim (c0 [2 ]-rr ,c0 [2 ]+rr )
        try :ax .set_box_aspect (ext )
        except Exception :pass 
        thin =int (np .argmin (ext ));ev ,az ={0 :(0 ,0 ),1 :(0 ,-90 ),2 :(90 ,-90 )}[thin ]
        ax .view_init (elev =ev ,azim =az );ax .set_axis_off ()
        ax .set_title (f"{pid }  ->  {len (cps )} CP ({na } otonom)",fontsize =10 )

    json .dump (allout ,open ("results/_demo_random.json","w"),indent =1 )
    fig .suptitle ("ROBOT: random WSCAD .stp -> CP'ler (kirmizi=otonom, turuncu=insan-kontrol; ok=point+direction)  --  never unseen parts",
    fontsize =13 )
    plt .tight_layout (rect =[0 ,0 ,1 ,0.96 ])
    plt .savefig ("results/_demo_random.png",dpi =95 ,bbox_inches ="tight")
    print ("\n-> results/_demo_random.png + results/_demo_random.json")


if __name__ =="__main__":
    main ()
