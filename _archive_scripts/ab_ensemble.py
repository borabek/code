# -*- coding: utf-8 -*-
"""A/B: TOPLULUK BILESIMI -- urunun 4 ckpt'i most iyi secim mi?

WHY SORULUYOR
  Urundeki 4 ckpt'in 3'u AYNI tarif (recall_hard_keig96 s0/s1/s2), biri keig64. Cesitlilik
  low; oysa union'in (vote>=1) butun kazanci, different modellerin FARKLI acikliklari
  kacirmasindan geliyor. Ayrica `best_full.pt` segmentasyonda 68.1% IoU with urundeki
  recall_hard_s2'yi (63.3%) 4.8 score geciyor and CP-F1 acisindan HIC denenmedi.

  Onemli uyari: high IoU otomatik as high CP-F1 demek DEGIL. Topluluk CP-F1'e according to
  secilmisti, IoU'ya according to not. Bu yuzden olculmeden degistirilmez.

KOLLAR
  A) urun          : cp_config.robot_vote2_checkpoints (4 ckpt)
  B) +best_full    : urun + best_full.pt (5 ckpt, cesitlilik artar)
  C) cesitli-4     : recall_hard_s2 + best_full + human103c_s0 + adj_s0 (4 different tarif)

KILL (olcumden before yazildi)
  corpus-agirlikli CP-F1 katkisi < +0.02 -> reddedilir (this night diger kaldiraclarla same bar)

MALIYET: each arm for full inference is required (segmentasyon ciktisi degisiyor), cache
kullanilamaz. Bu yuzden ORNEKLEM with kosulur and regime-agirlikli raporlanir.
"""
import os ,sys ,json 
import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))

W ={"low":0.895 ,"very":0.105 }
ARMS ={
"urun (4 ckpt)":None ,# cp_config'ten okunur
"+best_full (5)":["results/seg_extra/recall_hard_s2.pt",
"results/seg_extra/recall_hard_keig96_s0.pt",
"results/seg_extra/recall_hard_keig96_s1.pt",
"results/seg_extra/recall_hard_keig96_s2.pt",
"results/seg_extra/best_full.pt"],
"cesitli-4":["results/seg_extra/recall_hard_s2.pt",
"results/seg_extra/best_full.pt",
"results/seg_extra/human103c_s0.pt",
"results/seg_extra/adj_s0.pt"],
}


def run_arm (cks ,parts ,dev ):
    import torch ,thesis_remesh ,cp_openings ,robot_cp ,wire_gate ,diffusionnet as D 
    from cad_eval import align_frames 
    from infer_step_cp import step_to_mesh ,load_any 
    from connector_constants import CABLE_ENTRY as CE ,CONTACT as CT 
    cfg =json .load (open ("cp_config.json"))
    pp =cfg ["prediction_postproc"]
    models =[load_any (c ,dev =dev )[:2 ]for c in cks ]
    agg ={"low":[0 ,0 ,0 ],"very":[0 ,0 ,0 ]}
    for mfg ,pid ,jf ,stp ,n in parts :
        try :
            Vr ,Fr =step_to_mesh (stp )
            V ,F =thesis_remesh .remesh_uniform (Vr ,Fr ,target =6000 )
            V =np .ascontiguousarray (V ,np .float64 );F =np .ascontiguousarray (F ,np .int64 )
            acc ,per =None ,[]
            hi =n >=8 
            for model ,meta in models :
                _ ,pb =D .predict (model ,meta ,V ,F ,device =dev ,
                op_cache_dir =f"results/step_infer/ops_k{int (meta .get ('k_eig',64 ))}",
                return_probs =True )
                pb =np .asarray (pb ,float )
                acc =pb if acc is None else acc +pb 
                per .append (cp_openings .connection_points (
                V ,F ,pb .argmax (-1 ),min_v =int (pp ["min_vertices"]),classes =(CE ,CT ),
                dedupe_mm =10.0 ,probs =pb ,vertex_conf =float (pp ["vertex_confidence_mask"]),
                ct_depth_min_mm =1.0 ,cluster_mm =float (pp ["cluster_mm"]),
                conn_promote =(0.25 if hi else 0.0 )))
            cps =robot_cp ._vote2 (per ,min_votes =1 )
            probs =acc /len (per )
            thr =float (cfg .get ("robot_wire_gate_threshold_highcp",0.25 ))if hi else float (cfg .get ("robot_wire_gate_threshold",0.35 ))
            if cps :
                cps =wire_gate .apply (V ,F ,probs ,cps ,CE ,CT ,threshold =thr ,top_n =None )
            P =np .array ([c ["point"]for c in cps ],float )if cps else np .zeros ((0 ,3 ))

            j =json .load (open (jf ,encoding ="utf-8-sig"))
            Vj =np .array ([[q [c ]for c in "XYZ"]for q in j ["Graphic3d"]["Points"]],float )
            G =np .array ([[c ["Point"][q ]for q in "XYZ"]for c in j ["ConnectionPoints"]],float )
            Gd =np .array ([[c ["InsertDirection"][q ]for q in "XYZ"]for c in j ["ConnectionPoints"]],float )
            Gd /=np .linalg .norm (Gd ,axis =1 ,keepdims =True )+1e-9 
            R ,t ,_ =align_frames (Vr ,Vj )
            Gm =(G -t )@R ;Gdm =Gd @R 
            tol =max (3.0 ,0.06 *float (np .linalg .norm (V .max (0 )-V .min (0 ))))

            hit =np .zeros (len (Gm ),bool );used =set ()
            if len (P )and len (Gm ):
                diff =P [:,None ,:]-Gm [None ,:,:]
                al =(diff *Gdm [None ,:,:]).sum (-1 )
                pe =np .linalg .norm (diff -al [...,None ]*Gdm [None ,:,:],axis =-1 )
                pe =np .where (np .abs (al )<=40.0 ,pe ,np .inf )
                for d_ ,a_ ,b_ in sorted ((pe [a ,b ],a ,b )
                for a in range (len (P ))for b in range (len (Gm ))):
                    if d_ >tol or a_ in used or hit [b_ ]:
                        continue 
                    hit [b_ ]=True ;used .add (a_ )
            k ="very"if hi else "low"
            tp =int (hit .sum ())
            agg [k ][0 ]+=tp ;agg [k ][1 ]+=len (P )-tp ;agg [k ][2 ]+=len (Gm )-tp 
        except Exception :
            continue 
    out ={}
    for k ,(T ,Fp ,Fn )in agg .items ():
        p =T /max (T +Fp ,1 );r =T /max (T +Fn ,1 )
        out [k ]=2 *p *r /max (p +r ,1e-9 )
    out ["weighted"]=sum (W [k ]*out [k ]for k in W )
    return out 


def main ():
    import torch 
    from big_arbiter import eligible 
    LOCK =set (json .load (open ("results/split_lock.json"))["locked_parts"])
    parts =[]
    for m ,p ,jf ,s in eligible ():
        if p in LOCK :
            continue 
        try :
            n =len (json .load (open (jf ,encoding ="utf-8-sig"))["ConnectionPoints"])
        except Exception :
            continue 
        if n >0 :
            parts .append ((m ,p ,jf ,s ,n ))
    rng =np .random .RandomState (0 )
    lo =[x for x in parts if x [4 ]<8 ];hi =[x for x in parts if x [4 ]>=8 ]
    sel =([lo [i ]for i in rng .choice (len (lo ),min (30 ,len (lo )),replace =False )]+
    [hi [i ]for i in rng .choice (len (hi ),min (18 ,len (hi )),replace =False )])
    dev ="cuda"if torch .cuda .is_available ()else "cpu"
    cfg =json .load (open ("cp_config.json"))
    print (f"{len (sel )} part (30 dusuk / 18 very)\n",flush =True )
    print (f"{'arm':<18}{'low-CP':>10}{'very-CP':>10}{'agirlikli':>12}")
    res ={}
    base =None 
    for name ,cks in ARMS .items ():
        cks =cks or cfg ["robot_vote2_checkpoints"]
        missing =[c for c in cks if not os .path .exists (c )]
        if missing :
            print (f"{name :<18}  ATLANDI (missing: {[os .path .basename (m )for m in missing ]})")
            continue 
        r =run_arm (cks ,sel ,dev )
        res [name ]=r 
        if base is None :
            base =r ["weighted"]
        delta =""if r ["weighted"]==base else f"   ({r ['weighted']-base :+.4f})"
        print (f"{name :<18}{r ['low']:>10.4f}{r ['very']:>10.4f}{r ['weighted']:>12.4f}{delta }",
        flush =True )
    if res :
        best =max (res .items (),key =lambda kv :kv [1 ]["weighted"])
        gain =best [1 ]["weighted"]-base 
        print (f"\nEN IYI: {best [0 ]}  ({gain :+.4f})")
        print (f"KAPI (>= +0.02): {'GECTI'if gain >=0.02 else 'OLU'}")
        json .dump ({"arms":res ,"best":best [0 ],"gain":gain ,
        "kill_passed":bool (gain >=0.02 )},
        open ("results/ab_ensemble.json","w"),indent =1 )
        print ("receipt -> results/ab_ensemble.json")


if __name__ =="__main__":
    main ()
