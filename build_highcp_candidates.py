# -*- coding: utf-8 -*-
"""Op 3: high-CP PXC candidate genisletme. KILIT SORU: multi-res (6k+9k) + low vertex_conf + cluster_mm=0
candidate havuzunun CANDIDATE RECALL'ini (havuzdaki real-tel / manufacturer-CP) yukseltiyor mu?
Yukseltmezse top-N secimin sececek seyi absent. High-CP PXC parcalarda (>=11 CP) olc: 6k-standart vs
6k+9k-genisletilmis. GPU. High-CP-ONLY (global urune uygulanmaz)."""
import os ,sys ,json ,glob ,time 
import numpy as np ,torch 
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
import diffusionnet as D ,cp_openings ,thesis_remesh ,connector3d 
from cad_eval import align_frames 
from infer_step_cp import step_to_mesh ,load_any 
from big_arbiter import eligible ,CE ,CT ,OP 
from robot_cp import _vote2 

dev ="cuda"if torch .cuda .is_available ()else "cpu"
HICP =11 # high-CP esigi


def cand_recall (cps ,V ,Vr ,Vj ,G ,Gd ):
    """havuzdaki CP'lerin kaci manufacturer-CP'ye eslesiyor (candidate recall)."""
    if not cps or not len (G ):return 0 ,len (G )
    R ,t ,_ =align_frames (Vr ,Vj )
    P =np .array ([np .asarray (c ["point"])for c in cps ],float )@R .T +t 
    tol =max (3.0 ,0.06 *float (np .linalg .norm (Vj .max (0 )-Vj .min (0 ))))
    diff =P [:,None ,:]-G [None ,:,:];al =(diff *Gd [None ,:,:]).sum (-1 )
    pe =np .where (np .abs (al )<=40.0 ,np .linalg .norm (diff -al [...,None ]*Gd [None ,:,:],axis =-1 ),np .inf )
    order =sorted ((pe [a ,b ],a ,b )for a in range (len (P ))for b in range (len (G ))if pe [a ,b ]<=tol )
    ug =set ()
    for dd ,a ,b in order :
        ug .add (b )if b not in ug else None 
    return len (ug ),len (G )


def derive (V ,F ,pb ,mv ,vc ,cl ):
    return cp_openings .connection_points (V ,F ,pb .argmax (-1 ),min_v =mv ,classes =(CE ,CT ),dedupe_mm =10.0 ,
    probs =pb ,vertex_conf =vc ,ct_depth_min_mm =1.0 ,cluster_mm =cl )


def main ():
    cks =json .load (open ("cp_config.json"))["robot_vote2_checkpoints"]
    models =[load_any (c ,dev =dev )[:2 ]for c in cks ]
    os .environ ["BA_ALLOW_SEEN"]="1"
    parts =[p for p in eligible ()if p [0 ]=="PXC"]
    # high-CP olanlari sec
    hi =[]
    for mfg ,pid ,jf ,stp in parts :
        try :
            n =len (json .load (open (jf ,encoding ="utf-8-sig")).get ("ConnectionPoints",[]))
            if n >=HICP :hi .append ((pid ,jf ,stp ,n ))
        except Exception :pass 
    print (f"high-CP PXC ({HICP }+): {len (hi )} part",flush =True )

    std_tp =std_gt =exp_tp =exp_gt =0 ;t0 =time .time ()
    for k ,(pid ,jf ,stp ,n )in enumerate (hi ,1 ):
        try :
            j =json .load (open (jf ,encoding ="utf-8-sig"))
            Vj =np .array ([[p ["X"],p ["Y"],p ["Z"]]for p in j ["Graphic3d"]["Points"]],float )
            G =np .array ([[c ["Point"]["X"],c ["Point"]["Y"],c ["Point"]["Z"]]for c in j .get ("ConnectionPoints",[])],float )
            Gd =np .array ([[c ["InsertDirection"]["X"],c ["InsertDirection"]["Y"],c ["InsertDirection"]["Z"]]for c in j .get ("ConnectionPoints",[])],float )
            if not len (G ):continue 
            Gd =Gd /(np .linalg .norm (Gd ,axis =1 ,keepdims =True )+1e-9 )
            Vr ,Fr =step_to_mesh (stp )
            # 6k standart
            V6 ,F6 =thesis_remesh .remesh_uniform (Vr ,Fr ,target =6000 )
            V6 =np .ascontiguousarray (V6 ,np .float64 );F6 =np .ascontiguousarray (F6 ,np .int64 )
            per6 =[]
            for model ,meta in models :
                _ ,pb =D .predict (model ,meta ,V6 ,F6 ,device =dev ,op_cache_dir =f"{OP }_k{int (meta .get ('k_eig',64 ))}",return_probs =True )
                per6 .append (derive (V6 ,F6 ,np .asarray (pb ,float ),30 ,0.5 ,5.0 ))
            std =_vote2 (per6 )
            # 9k + low vc + cluster0 (ekstra candidate)
            V9 ,F9 =thesis_remesh .remesh_uniform (Vr ,Fr ,target =9000 )
            V9 =np .ascontiguousarray (V9 ,np .float64 );F9 =np .ascontiguousarray (F9 ,np .int64 )
            per9 =[]
            for model ,meta in models :
                _ ,pb =D .predict (model ,meta ,V9 ,F9 ,device =dev ,op_cache_dir =f"{OP }_k{int (meta .get ('k_eig',64 ))}_9k",return_probs =True )
                per9 .append (derive (V9 ,F9 ,np .asarray (pb ,float ),15 ,0.30 ,0.0 ))
                # genisletilmis: 6k std + 9k ekstra (union, ayri frame but ikisi de STEP frame -> Vr uzerinden)
            exp =_vote2 (per6 +per9 )# same cluster mantigi
            s_tp ,gt =cand_recall (std ,V6 ,Vr ,Vj ,G ,Gd )
            # exp for 9k CP'ler V9 frame'inde, 6k V6 frame'inde -- ikisi de STEP frame (thesis_remesh recenter absent)
            e_tp ,_ =cand_recall (exp ,None ,Vr ,Vj ,G ,Gd )
            std_tp +=s_tp ;std_gt +=gt ;exp_tp +=e_tp ;exp_gt +=gt 
        except Exception as e :
            continue 
        if k %5 ==0 :print (f"  {k }/{len (hi )}  {time .time ()-t0 :.0f}s",flush =True )

    print (f"\n=== high-CP PXC CANDIDATE RECALL ({len (hi )} part) ===")
    print (f"  6k standart:            {std_tp }/{std_gt } = {std_tp /max (std_gt ,1 ):.3f}")
    print (f"  6k+9k genisletilmis:    {exp_tp }/{exp_gt } = {exp_tp /max (exp_gt ,1 ):.3f}")
    print (f"  -> genisletme candidate recall'i {'YUKSELTTI'if exp_tp >std_tp else 'YUKSELTMEDI'} "
    f"(+{exp_tp -std_tp } CP). >0.60 ise top-N with F1 kazanci mumkun.")


if __name__ =="__main__":
    main ()
