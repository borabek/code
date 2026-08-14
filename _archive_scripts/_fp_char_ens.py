# TODO B-4: the ensemble leaves 4 FP / 4 FN (P=R=0.778). Killing the 4 FPs would lift F1 to 0.875.
# Is there ANY per-CP signal (confidence, size, area, outwardness) that separates the 4 FPs from the
# 14 TPs? If yes -> a principled robotic-validation gate; if no -> precision is at its floor too.
import os ,glob ,json 
import numpy as np ,torch 
import diffusionnet as D ,connector3d ,cp_openings ,thesis_remesh 
from cad_eval import align_frames 
from infer_step_cp import step_to_mesh ,load_any 
CE =int (connector3d .CABLE_ENTRY );CT =int (connector3d .CONTACT )
OP ="results/step_infer/ops";JDIR ="C:/Users/DE00024082/Desktop/JSON"
dev ="cuda"if torch .cuda .is_available ()else "cpu"
CK =["results/seg_extra/human77c_s0.pt","results/seg_extra/human77c_s1.pt","results/seg_extra/human77c_s2.pt"]
models =[load_any (c ,dev =dev )for c in CK ]


def matched_idx (P ,G ,tol ):
    if not len (P )or not len (G ):return set ()
    dm =np .linalg .norm (P [:,None ,:]-G [None ,:,:],axis =2 )
    order =sorted ((dm [i ,j ],i ,j )for i in range (len (P ))for j in range (len (G )))
    up ,ug ,m =set (),set (),set ()
    for d ,i ,j in order :
        if d >tol :break 
        if i in up or j in ug :continue 
        up .add (i );ug .add (j );m .add (i )
    return m 


TP ,FP =[],[]
for stp in sorted (glob .glob ("_cad_eval_pxc/*.stp")):
    pid =os .path .basename (stp ).split ("_")[1 ]
    jf =os .path .join (JDIR ,f"PXC.{pid }.json")
    if not os .path .exists (jf ):continue 
    j =json .load (open (jf ,encoding ="utf-8-sig"))
    Vj =np .array ([[p ["X"],p ["Y"],p ["Z"]]for p in j ["Graphic3d"]["Points"]],float )
    G =np .array ([[c ["Point"]["X"],c ["Point"]["Y"],c ["Point"]["Z"]]for c in j .get ("ConnectionPoints",[])],float )
    Vr ,Fr =step_to_mesh (stp );V ,F =thesis_remesh .remesh_uniform (Vr ,Fr ,target =6000 )
    V =np .ascontiguousarray (V ,np .float64 );F =np .ascontiguousarray (F ,np .int64 )
    acc =None 
    for model ,meta ,_ in models :
        _ ,pb =D .predict (model ,meta ,V ,F ,device =dev ,op_cache_dir =OP ,return_probs =True )
        pb =np .asarray (pb ,float );acc =pb if acc is None else acc +pb 
    probs =acc /len (models );lab =probs .argmax (-1 )
    cps =cp_openings .connection_points (V ,F ,lab ,min_v =60 ,classes =(CE ,CT ),dedupe_mm =10.0 ,
    probs =probs ,vertex_conf =0.7 ,ct_depth_min_mm =1.0 ,cluster_mm =10.0 )
    R ,t ,_ =align_frames (Vr ,Vj )
    P =(np .array ([np .asarray (c ["point"])for c in cps ],float )@R .T +t )if cps else np .zeros ((0 ,3 ))
    tol =max (3.0 ,0.06 *float (np .linalg .norm (Vj .max (0 )-Vj .min (0 ))))
    m =matched_idx (P ,G ,tol );bc =V .mean (0 )
    for i ,c in enumerate (cps ):
        p =np .asarray (c ["point"],float )
        u =p -bc ;nu =np .linalg .norm (u )+1e-9 ;u =u /nu 
        outward =float ((p -bc )@u )/(float (((V -bc )@u ).max ())+1e-9 )
        row =dict (conf =float (c .get ("confidence",0 )),nv =float (c .get ("n_verts",0 )),
        area =float (c .get ("area",0 )),outward =outward ,
        src ="CE"if int (c .get ("source_label",CE ))==CE else "CT")
        (TP if i in m else FP ).append (row )

print (f"\n=== ENSEMBLE per-CP: TP {len (TP )} / FP {len (FP )} ===",flush =True )
for name ,rows in [("TP",TP ),("FP",FP )]:
    if not rows :continue 
    print (f"  {name }: "+"  ".join (f"{k }={np .mean ([r [k ]for r in rows ]):.3f}"for k in ("conf","nv","area","outward"))
    +f"  | CE {sum (r ['src']=='CE'for r in rows )}/CT {sum (r ['src']=='CT'for r in rows )}")
print ("\n=== tum TP'yi koruyan single-threshold FP kirimi (F1 etkisi) ===")
nT =len (TP );nF =len (FP );FN =18 -nT 
for k in ("conf","nv","area","outward"):
    if not TP or not FP :break 
    tmin =min (r [k ]for r in TP )
    drop =sum (1 for r in FP if r [k ]<tmin )
    fp2 =nF -drop 
    f1 =2 *nT /(2 *nT +fp2 +FN )
    print (f"  {k } >= {tmin :.3f}: {drop }/{nF } FP duser -> F1 {f1 :.3f}")
