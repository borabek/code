# TODO 3: WHY do these 6 parts yield ZERO CPs? Hypothesis from the FBI numbers: their openings are
# SMALL (CE 7.0% vs 10.1%) and the bodies THICK (8.9 vs 4.6mm), so at a 6000-vertex remesh the wire
# mouths are under-resolved -> the segmented region's shape is wrong -> no CP survives derivation.
# Green = the annotator's opening (what MUST be found). Blue = what the model calls a connection.
import os ,glob 
import numpy as np ,torch 
import matplotlib ;matplotlib .use ("Agg");import matplotlib .pyplot as plt 
from mpl_toolkits .mplot3d .art3d import Poly3DCollection 
from region_label_helper import load_obj 
import diffusionnet as D ,connector3d ,cp_openings 
from infer_step_cp import load_any 
CE =int (connector3d .CABLE_ENTRY );CT =int (connector3d .CONTACT );OP ="results/step_infer/ops"
dev ="cuda"if torch .cuda .is_available ()else "cpu"
CK =["results/seg_extra/human77c_s0.pt","results/seg_extra/human77c_s1.pt","results/seg_extra/human77c_s2.pt"]
models =[load_any (c ,dev =dev )for c in CK ]
pids =["280-612","3076031","3212095","3249017","3271323","3281118"]
fig =plt .figure (figsize =(17 ,8.6 ))
for k ,pid in enumerate (pids ):
    V ,F =load_obj (f"_label_targets_4/{pid }/{pid }.obj")
    L =np .array ([int (x )for x in open (f"_label_targets_4/{pid }/{pid }.labels.txt").read ().split ()])
    Vc =np .ascontiguousarray (V ,np .float64 );Fc =np .ascontiguousarray (F ,np .int64 )
    acc =None 
    for model ,meta ,_ in models :
        _ ,pb =D .predict (model ,meta ,Vc ,Fc ,device =dev ,op_cache_dir =OP ,return_probs =True )
        pb =np .asarray (pb ,float );acc =pb if acc is None else acc +pb 
    probs =acc /len (models );lab =probs .argmax (-1 )
    cps =cp_openings .connection_points (Vc ,Fc ,lab ,min_v =60 ,classes =(CE ,CT ),dedupe_mm =10.0 ,
    probs =probs ,vertex_conf =0.7 ,ct_depth_min_mm =1.0 ,cluster_mm =10.0 )
    tri =V [F ];n =np .cross (tri [:,1 ]-tri [:,0 ],tri [:,2 ]-tri [:,0 ])
    n =n /(np .linalg .norm (n ,axis =1 ,keepdims =True )+1e-9 )
    lt =np .array ([0.4 ,0.3 ,1.0 ]);lt =lt /np .linalg .norm (lt )
    col =np .stack ([0.35 +0.55 *np .abs (n @lt )]*3 ,1 )
    mconn =np .isin (lab ,(CE ,CT ))
    col [mconn [F ].any (1 )]=[0.15 ,0.45 ,0.95 ]# model says connection
    col [(L ==CE )[F ].any (1 )]=[0.1 ,0.9 ,0.2 ]# human opening (drawn ten top)
    ext =V .max (0 )-V .min (0 );ctr =V .mean (0 );rr =ext .max ()*0.55 
    thin =int (np .argmin (ext ));ev ,az ={0 :(0 ,0 ),1 :(0 ,-90 ),2 :(90 ,-90 )}[thin ]
    ax =fig .add_subplot (2 ,3 ,k +1 ,projection ="3d")
    ax .add_collection3d (Poly3DCollection (tri ,facecolors =np .clip (col ,0 ,1 ),edgecolors ="none"))
    ax .set_xlim (ctr [0 ]-rr ,ctr [0 ]+rr );ax .set_ylim (ctr [1 ]-rr ,ctr [1 ]+rr );ax .set_zlim (ctr [2 ]-rr ,ctr [2 ]+rr )
    try :ax .set_box_aspect (ext )
    except Exception :pass 
    ax .view_init (elev =ev ,azim =az );ax .set_axis_off ()
    ax .set_title (f"{pid }  insanCE {100 *(L ==CE ).mean ():.1f}%  modelConn {100 *mconn .mean ():.1f}%  CP={len (cps )}",fontsize =9 )
fig .suptitle ("SIFIR CP veren 6 part -- yesil=insan acikligi, mavi=modelin gordugu baglanti",fontsize =12 )
plt .tight_layout (rect =[0 ,0 ,1 ,0.96 ]);plt .savefig ("results/_fail6.png",dpi =95 ,bbox_inches ="tight")
print ("-> results/_fail6.png",flush =True )
