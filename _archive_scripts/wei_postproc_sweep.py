# -*- coding: utf-8 -*-
"""WEI BEDAVA-RECALL sweep: teshis dedi ki FN'lerin %11'i turetme-kacisi (min_v cozer) + a kismi
threshold-six (FN max-probability medyan 0.36 -> vconf dusurmek yakalar). Sizinti-siz 145 WEI held-out'ta
2D grid (min_v x vertex_conf) tara; each hucrede WEI P/R/F1. Urunu (min_v30/vc0.5) referans al.
Robot tier mantigi: low-vconf CP'ler low-confidence -> REVIEW'e duser, AUTO precision korunur.
"""
import os ,sys ,json ,itertools ,time 
import numpy as np ,torch 
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
import diffusionnet as D ,cp_openings ,thesis_remesh 
from cad_eval import align_frames 
from infer_step_cp import step_to_mesh ,load_any 
from big_arbiter import greedy ,eligible ,CE ,CT ,OP 

CKPT ="results/seg_extra/recall_hard_s2.pt"
HELD =set (open ("_hw_r3.txt").read ().split ())


def main ():
    dev ="cuda"if torch .cuda .is_available ()else "cpu"
    model ,meta ,_ =load_any (CKPT ,dev =dev )
    os .environ ["BA_ALLOW_SEEN"]="1"
    raw =[p for p in eligible ()if p [0 ]=="WEI"and p [1 ]in HELD ]
    print (f"{len (raw )} WEI held-out part | {os .path .basename (CKPT )}",flush =True )

    cache =[];t0 =time .time ()
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
            _ ,probs =D .predict (model ,meta ,V ,F ,device =dev ,op_cache_dir =OP ,return_probs =True )
            R ,t ,_ =align_frames (Vr ,Vj )
            tol =max (3.0 ,0.06 *float (np .linalg .norm (Vj .max (0 )-Vj .min (0 ))))
            cache .append ({"V":V ,"F":F ,"probs":np .asarray (probs ,float ),"Vr":Vr ,"R":R ,"t":t ,"G":G ,"Gd":Gd ,"tol":tol })
        except Exception :
            continue 
        if k %25 ==0 :print (f"  {k }/{len (raw )} cache {time .time ()-t0 :.0f}s",flush =True )
    print (f"  {len (cache )} part onbellekte\n",flush =True )

    grid =list (itertools .product ([5.0 ],[30 ,20 ,10 ],[0.50 ,0.40 ,0.35 ,0.30 ]))
    print (f"{'min_v':>5s} {'vconf':>5s} | {'P':>6s} {'R':>6s} {'F1':>6s}   (urun = min_v30 vconf0.50)")
    best =None 
    for cl ,mv ,vc in grid :
        T =Fp =Fn =0 
        for c in cache :
            lab =c ["probs"].argmax (-1 )
            cps =cp_openings .connection_points (c ["V"],c ["F"],lab ,min_v =mv ,classes =(CE ,CT ),dedupe_mm =10.0 ,
            probs =c ["probs"],vertex_conf =vc ,ct_depth_min_mm =1.0 ,cluster_mm =cl )
            P =(np .array ([np .asarray (x ["point"])for x in cps ],float )@c ["R"].T +c ["t"])if cps else np .zeros ((0 ,3 ))
            tp ,fp ,fn =greedy (P ,c ["G"],c ["tol"],c ["Gd"],40.0 )
            T +=tp ;Fp +=fp ;Fn +=fn 
        p =T /max (T +Fp ,1 );r =T /max (T +Fn ,1 );f =2 *p *r /max (p +r ,1e-9 )
        star =" <-- urun"if (mv ==30 and vc ==0.50 )else ""
        print (f"{mv :5d} {vc :5.2f} | {p :6.3f} {r :6.3f} {f :6.3f}{star }",flush =True )
        if best is None or f >best [0 ]:best =(f ,p ,r ,mv ,vc )
    print (f"\n  en iyi WEI F1={best [0 ]:.3f} (P{best [1 ]:.3f} R{best [2 ]:.3f}) @ min_v{best [3 ]} vconf{best [4 ]}")


if __name__ =="__main__":
    main ()
