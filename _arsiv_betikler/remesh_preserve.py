# -*- coding: utf-8 -*-
"""EVDE-3 ON-TESHISI: bizim remesh (pymeshlab, 6000v) WEI acikliklarini MUHURLUYOR mu?

Manifold/ManifoldPlus derlemeye girismeden before hipotezi ucuza test et (teshis-before-tedavi).
Her WEI model-FN acikligi for: acikligin ekseni along [0,15]mm iceri, perp<r silindirde
ORIJINAL STEP tessellation'daki (Vr) verteks ORANI vs REMESH'teki (V) ratio.
  remesh_orani << orijinal_orani  -> remesh tuneli MUHURLEDI (Manifold denemeye value)
  oranlar benzer                  -> opening korunuyor, remesh SUCLU DEGIL (Manifold detourunu ATLA)
Karsilastirma yogunluk-normalize (two mesh'in total verteks count very different).
GPU is required (model FN'lerini bulmak for).
"""
import os ,sys ,json ,time 
import numpy as np ,torch 
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
import diffusionnet as D ,cp_openings ,thesis_remesh 
from cad_eval import align_frames 
from infer_step_cp import step_to_mesh ,load_any 
from big_arbiter import eligible ,CE ,CT ,OP 
from fbi_recall import match_fn 

CKPT ="results/seg_extra/recall_hard_s2.pt"
HELD =set (open ("_hw_r3.txt").read ().split ())


def frac_in_cyl (P ,origin ,axis ,lo ,hi ,rad ):
    """verteslerin ORANI: axis silindiri inside kalanlar / total."""
    rel =np .asarray (P ,float )-np .asarray (origin ,float )[None ,:]
    al =rel @axis 
    perp =np .linalg .norm (rel -al [:,None ]*axis [None ,:],axis =1 )
    return float (((al >=lo )&(al <=hi )&(perp <=rad )).sum ())/max (len (P ),1 )


def main ():
    dev ="cuda"if torch .cuda .is_available ()else "cpu"
    model ,meta ,_ =load_any (CKPT ,dev =dev )
    os .environ ["BA_ALLOW_SEEN"]="1"
    raw =[p for p in eligible ()if p [0 ]=="WEI"and p [1 ]in HELD ]
    print (f"{len (raw )} WEI held-out | remesh koruma testi",flush =True )

    ratios =[]# remesh_orani / orijinal_orani  (1.0 = korunmus, <<1 = muhurlenmis)
    n_fn =0 ;t0 =time .time ()
    for k ,(mfg ,pid ,jf ,stp )in enumerate (raw ,1 ):
        try :
            j =json .load (open (jf ,encoding ="utf-8-sig"))
            Vj =np .array ([[p ["X"],p ["Y"],p ["Z"]]for p in j ["Graphic3d"]["Points"]],float )
            G =np .array ([[c ["Point"]["X"],c ["Point"]["Y"],c ["Point"]["Z"]]for c in j .get ("ConnectionPoints",[])],float )
            Gd =np .array ([[c ["InsertDirection"]["X"],c ["InsertDirection"]["Y"],c ["InsertDirection"]["Z"]]for c in j .get ("ConnectionPoints",[])],float )
            if not len (G ):continue 
            Gd =Gd /(np .linalg .norm (Gd ,axis =1 ,keepdims =True )+1e-9 )
            Vr ,Fr =step_to_mesh (stp )
            V ,F =thesis_remesh .remesh_uniform (Vr ,Fr ,target =6000 )
            V =np .ascontiguousarray (V ,np .float64 );F =np .ascontiguousarray (F ,np .int64 )
            _ ,pb =D .predict (model ,meta ,V ,F ,device =dev ,op_cache_dir =OP ,return_probs =True )
            probs =np .asarray (pb ,float );lab =probs .argmax (-1 )
            cps =cp_openings .connection_points (V ,F ,lab ,min_v =30 ,classes =(CE ,CT ),dedupe_mm =10.0 ,
            probs =probs ,vertex_conf =0.5 ,ct_depth_min_mm =1.0 ,cluster_mm =5.0 )
            R ,t ,_ =align_frames (Vr ,Vj )
            Pm =(np .array ([np .asarray (c ["point"])for c in cps ],float )@R .T +t )if cps else np .zeros ((0 ,3 ))
            tol =max (3.0 ,0.06 *float (np .linalg .norm (Vj .max (0 )-Vj .min (0 ))))
            matched =match_fn (Pm ,G ,Gd ,tol )
            Gm =(G -t )@R ;Gd_m =Gd @R # mesh/STEP frame
            for gi in range (len (G )):
                if gi in matched :continue 
                n_fn +=1 
                ax =Gd_m [gi ]/(np .linalg .norm (Gd_m [gi ])+1e-9 )
                # opening AGZI ~15mm disarda (manufacturer CP kontakt) -> agizdan iceri [0,15]mm bak
                mouth =Gm [gi ]+15.0 *ax 
                f_raw =frac_in_cyl (Vr ,mouth ,-ax ,0.0 ,15.0 ,4.0 )
                f_rem =frac_in_cyl (V ,mouth ,-ax ,0.0 ,15.0 ,4.0 )
                if f_raw >1e-6 :
                    ratios .append (f_rem /f_raw )
        except Exception :
            continue 
        if k %25 ==0 :print (f"  {k }/{len (raw )}  {time .time ()-t0 :.0f}s",flush =True )

    r =np .array (ratios )
    print (f"\n=== REMESH KORUMA ({n_fn } FN, {len (r )} olculebilir) ===")
    if len (r ):
        print (f"  remesh/orijinal verteks-orani: medyan {np .median (r ):.2f} | ort {r .mean ():.2f}")
        print (f"  MUHURLENMIS (<0.5):  {(r <0.5 ).sum ()} ({100 *(r <0.5 ).mean ():.0f}%)")
        print (f"  KORUNMUS   (>=0.5):  {(r >=0.5 ).sum ()} ({100 *(r >=0.5 ).mean ():.0f}%)")
        print ("\n  YORUM: medyan ~1.0 -> remesh acikligi KORUYOR (Manifold detourunu ATLA).")
        print ("         medyan <<0.5 -> remesh MUHURLUYOR (Manifold/ManifoldPlus denemeye deger).")
    json .dump ({"n_fn":n_fn ,"ratios":ratios },open ("results/remesh_preserve.json","w"),indent =1 )


if __name__ =="__main__":
    main ()
