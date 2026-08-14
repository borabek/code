# -*- coding: utf-8 -*-
"""Re-render the batch-2 NEUTRAL reference images from the saved OBJs (no model, no GPU).

The first pass used cavity shading, which came out near-white ten a white background -> invisible.
This uses Lambert (normal.light) grey shading that always shows form, three views per part.
"""
import os ,sys ,glob 
import numpy as np 
import matplotlib ;matplotlib .use ("Agg");import matplotlib .pyplot as plt 
from mpl_toolkits .mplot3d .art3d import Poly3DCollection 
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
from region_label_helper import load_obj 

LIGHT =np .array ([0.4 ,0.3 ,1.0 ]);LIGHT =LIGHT /np .linalg .norm (LIGHT )


def render (V ,F ,out_png ):
    tri =V [F ]
    n =np .cross (tri [:,1 ]-tri [:,0 ],tri [:,2 ]-tri [:,0 ])
    n =n /(np .linalg .norm (n ,axis =1 ,keepdims =True )+1e-9 )
    ext =V .max (0 )-V .min (0 );ctr =V .mean (0 );rr =ext .max ()*0.55 
    fig =plt .figure (figsize =(13 ,4.2 ))
    for i ,(ev ,az )in enumerate ([(18 ,-60 ),(18 ,30 ),(90 ,-90 )]):
        ax =fig .add_subplot (1 ,3 ,i +1 ,projection ="3d")
        # view-dependent light so faces toward the camera are lit; abs so back-faces aren't black
        shade =np .abs (n @LIGHT )
        grey =0.30 +0.60 *shade 
        fc =np .clip (np .stack ([grey ]*3 ,1 ),0 ,1 )
        pc =Poly3DCollection (tri ,facecolors =fc ,edgecolors =(0 ,0 ,0 ,0.06 ),linewidths =0.1 )
        ax .add_collection3d (pc )
        ax .set_xlim (ctr [0 ]-rr ,ctr [0 ]+rr );ax .set_ylim (ctr [1 ]-rr ,ctr [1 ]+rr );ax .set_zlim (ctr [2 ]-rr ,ctr [2 ]+rr )
        try :ax .set_box_aspect (ext )
        except Exception :pass 
        ax .view_init (elev =ev ,azim =az );ax .set_axis_off ()
        ax .set_facecolor ((0.93 ,0.94 ,0.96 ))
    fig .patch .set_facecolor ((0.93 ,0.94 ,0.96 ))
    plt .tight_layout ();plt .savefig (out_png ,dpi =95 ,bbox_inches ="tight",facecolor =fig .get_facecolor ())
    plt .close (fig )


def main ():
    root =sys .argv [1 ]if len (sys .argv )>1 else "_label_targets_2"
    n =0 
    for d in sorted (glob .glob (os .path .join (root ,"*"))):
        if not os .path .isdir (d ):continue 
        pid =os .path .basename (os .path .normpath (d ))
        of =os .path .join (d ,f"{pid }.obj")
        if not os .path .exists (of ):continue 
        V ,F =load_obj (of )
        render (V ,F ,os .path .join (d ,f"{pid }.NEUTRAL.png"))
        n +=1 ;print (f"  rerendered {pid }",flush =True )
    print (f"\n{n } neutral renders -> {root }/*/*.NEUTRAL.png")


if __name__ =="__main__":
    main ()
