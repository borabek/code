# Render a part mesh + detected connection points to PNG images from a few
# angles, so detections can be eyeballed WITHOUT any 3D software (Blender/MeshLab).
# The mesh is drawn as a light point cloud; detected CPs are big red markers.
#
#   python render_scene.py real_json/PART.json preds_real.json --out 3d_models/PART
import sys 
import json 
import argparse 

import numpy as np 

import step_to_json as sj # mesh loader that also accepts raw .stp/.stl/.obj

import matplotlib 
matplotlib .use ("Agg")# no display needed -> writes PNG files
import matplotlib .pyplot as plt # noqa: E402


def _set_equal (ax ,V ):
    mins ,maxs =V .min (0 ),V .max (0 )
    ctr =(mins +maxs )/2.0 
    r =(maxs -mins ).max ()/2.0 or 1.0 
    ax .set_xlim (ctr [0 ]-r ,ctr [0 ]+r )
    ax .set_ylim (ctr [1 ]-r ,ctr [1 ]+r )
    ax .set_zlim (ctr [2 ]-r ,ctr [2 ]+r )


def render (mesh_json ,preds_json ,out_prefix ,max_pts =9000 ,deflection =0.6 ):
    part =next (sj .iter_parts_any (mesh_json ,deflection =deflection ))# .json OR raw .stp/.stl/.obj
    V =part .vertices 

    pred =json .load (open (preds_json ,encoding ="utf-8"))
    nodes =[]
    for p in pred .get ("parts",[]):
        if p .get ("part_nr")==part .part_nr or len (pred ["parts"])==1 :
            nodes =p .get ("connection_points",[])
            break 
    cps =np .array ([(n .get ("entry_point")or n .get ("position"))for n in nodes ])if nodes else np .zeros ((0 ,3 ))

    rng =np .random .RandomState (0 )
    idx =rng .choice (len (V ),min (max_pts ,len (V )),replace =False )
    P =V [idx ]
    # colour the cloud by the axis with the LEAST spread (the part's "thickness"
    # axis) so the relief -- raised terminals vs recessed openings -- shows up.
    depth_axis =int (np .argmin (P .max (0 )-P .min (0 )))
    cval =P [:,depth_axis ]

    views =[(18 ,-60 ),(18 ,60 ),(88 ,-90 )]# 3 angles incl. top-down
    written =[]
    for vi ,(el ,az )in enumerate (views ,1 ):
        fig =plt .figure (figsize =(8 ,8 ))
        ax =fig .add_subplot (111 ,projection ="3d")
        ax .scatter (P [:,0 ],P [:,1 ],P [:,2 ],s =3 ,c =cval ,cmap ="viridis",
        alpha =0.55 ,linewidths =0 )
        if len (cps ):
            ax .scatter (cps [:,0 ],cps [:,1 ],cps [:,2 ],s =160 ,c ="red",
            marker ="that",edgecolors ="black",depthshade =False ,zorder =5 )
            for k ,c in enumerate (cps ):
                ax .text (c [0 ],c [1 ],c [2 ],f"  CP{k }",color ="darkred",fontsize =11 )
        ax .view_init (elev =el ,azim =az )
        ax .set_title (f"{part .part_nr }   —   {len (cps )} detected connection point(s)")
        _set_equal (ax ,V )
        ax .set_axis_off ()
        out =f"{out_prefix }_view{vi }.png"
        fig .savefig (out ,dpi =110 ,bbox_inches ="tight")
        plt .close (fig )
        written .append (out )
        print ("wrote",out )
    return written 


def main (argv =None ):
    ap =argparse .ArgumentParser (description ="Render mesh + detected CPs to PNGs")
    ap .add_argument ("mesh_json")
    ap .add_argument ("preds_json")
    ap .add_argument ("--out",required =True ,help ="output PNG path prefix")
    ap .add_argument ("--max-pts",type =int ,default =9000 )
    ap .add_argument ("--deflection",type =float ,default =0.6 ,
    help ="gmsh mesh size (mm) for a STEP input (coarser avoids "
    "tessellation failures ten complex parts)")
    args =ap .parse_args (argv )
    render (args .mesh_json ,args .preds_json ,args .out ,max_pts =args .max_pts ,
    deflection =args .deflection )


if __name__ =="__main__":
    main ()
