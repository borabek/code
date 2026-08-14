# -*- coding: utf-8 -*-
"""Op 3 F1: high-CP PXC genisletilmis pool (6k+9k) + RF gate + top-N -> F1 yukseliyor mu?
SIZINTISIZ: RF gate high-CP parts DISINDA (standart f1_sweep_data) egitilir, high-CP'ye uygulanir.
Karsilastir: standart-6k top-N vs genisletilmis-6k9k top-N (ikisi de same gate)."""
import os ,sys ,json ,time 
import numpy as np ,torch 
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
import diffusionnet as D ,cp_openings ,thesis_remesh ,connector3d ,wire_gate 
from cad_eval import align_frames 
from infer_step_cp import step_to_mesh ,load_any 
from big_arbiter import eligible ,CE ,CT ,OP 
from robot_cp import _vote2 
from sklearn .ensemble import RandomForestClassifier 

dev ="cuda"if torch .cuda .is_available ()else "cpu"
HICP =11 


def derive (V ,F ,pb ,mv ,vc ,cl ):
    return cp_openings .connection_points (V ,F ,pb .argmax (-1 ),min_v =mv ,classes =(CE ,CT ),dedupe_mm =10.0 ,
    probs =pb ,vertex_conf =vc ,ct_depth_min_mm =1.0 ,cluster_mm =cl )


def feats_and_match (cps ,V ,avg ,Vr ,Vj ,G ,Gd ):
    X =wire_gate .feats_for (V ,None ,avg ,cps ,CE ,CT )
    R ,t ,_ =align_frames (Vr ,Vj )
    P =np .array ([np .asarray (c ["point"])for c in cps ],float )@R .T +t 
    tol =max (3.0 ,0.06 *float (np .linalg .norm (Vj .max (0 )-Vj .min (0 ))))
    diff =P [:,None ,:]-G [None ,:,:];al =(diff *Gd [None ,:,:]).sum (-1 )
    pe =np .where (np .abs (al )<=40.0 ,np .linalg .norm (diff -al [...,None ]*Gd [None ,:,:],axis =-1 ),np .inf )
    yy =np .zeros (len (P ),int );order =sorted ((pe [a ,b ],a ,b )for a in range (len (P ))for b in range (len (G ))if pe [a ,b ]<=tol )
    up ,ug =set (),set ()
    for dd ,a ,b in order :
        if a in up or b in ug :continue 
        up .add (a );ug .add (b );yy [a ]=1 
    return X ,yy 


def main ():
# gate'i high-CP DISI standart veride egit (sizintisiz)
    d =np .load ("results/f1_sweep_data.npz",allow_pickle =True )
    Xs ,ys ,grp =d ["X"],d ["y"],d ["groups"];ngt_of =dict (zip (d ["grp_ids"].tolist (),d ["ngt"].tolist ()))
    hicp_grp ={g for g in ngt_of if ngt_of [g ]>=HICP }
    keep =~np .isin (grp ,list (hicp_grp ))
    clf =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,random_state =0 ).fit (Xs [keep ],ys [keep ])
    print (f"gate egitildi: {keep .sum ()} CP (high-CP {len (hicp_grp )} part HARIC)",flush =True )

    cks =json .load (open ("cp_config.json"))["robot_vote2_checkpoints"]
    models =[load_any (c ,dev =dev )[:2 ]for c in cks ]
    os .environ ["BA_ALLOW_SEEN"]="1"
    hi =[]
    for mfg ,pid ,jf ,stp in [p for p in eligible ()if p [0 ]=="PXC"]:
        try :
            n =len (json .load (open (jf ,encoding ="utf-8-sig")).get ("ConnectionPoints",[]))
            if n >=HICP :hi .append ((pid ,jf ,stp ,n ))
        except Exception :pass 
    print (f"high-CP PXC: {len (hi )} part",flush =True )

    def topN (cps ,V ,avg ,Vr ,Vj ,G ,Gd ,N ):
        if not cps :return 0 ,0 
        X ,yy =feats_and_match (cps ,V ,avg ,Vr ,Vj ,G ,Gd )
        s =clf .predict_proba (X )[:,1 ]
        keepidx =np .argsort (-s )[:N ]
        return int ((yy [keepidx ]==1 ).sum ()),len (keepidx )

    sT =sN =eT =eN =GT =0 ;t0 =time .time ()
    for k ,(pid ,jf ,stp ,n )in enumerate (hi ,1 ):
        try :
            j =json .load (open (jf ,encoding ="utf-8-sig"))
            Vj =np .array ([[p ["X"],p ["Y"],p ["Z"]]for p in j ["Graphic3d"]["Points"]],float )
            G =np .array ([[c ["Point"]["X"],c ["Point"]["Y"],c ["Point"]["Z"]]for c in j .get ("ConnectionPoints",[])],float )
            Gd =np .array ([[c ["InsertDirection"]["X"],c ["InsertDirection"]["Y"],c ["InsertDirection"]["Z"]]for c in j .get ("ConnectionPoints",[])],float )
            if not len (G ):continue 
            Gd =Gd /(np .linalg .norm (Gd ,axis =1 ,keepdims =True )+1e-9 );N =len (G );GT +=N 
            Vr ,Fr =step_to_mesh (stp )
            V6 ,F6 =thesis_remesh .remesh_uniform (Vr ,Fr ,target =6000 );V6 =np .ascontiguousarray (V6 ,np .float64 );F6 =np .ascontiguousarray (F6 ,np .int64 )
            acc6 =None ;per6 =[]
            for model ,meta in models :
                _ ,pb =D .predict (model ,meta ,V6 ,F6 ,device =dev ,op_cache_dir =f"{OP }_k{int (meta .get ('k_eig',64 ))}",return_probs =True )
                pb =np .asarray (pb ,float );acc6 =pb if acc6 is None else acc6 +pb ;per6 .append (derive (V6 ,F6 ,pb ,30 ,0.5 ,5.0 ))
            avg6 =acc6 /len (per6 );std =_vote2 (per6 )
            V9 ,F9 =thesis_remesh .remesh_uniform (Vr ,Fr ,target =9000 );V9 =np .ascontiguousarray (V9 ,np .float64 );F9 =np .ascontiguousarray (F9 ,np .int64 )
            per9 =[]
            for model ,meta in models :
                _ ,pb =D .predict (model ,meta ,V9 ,F9 ,device =dev ,op_cache_dir =f"{OP }_k{int (meta .get ('k_eig',64 ))}_9k",return_probs =True )
                per9 .append (derive (V9 ,F9 ,np .asarray (pb ,float ),15 ,0.30 ,0.0 ))
            exp =_vote2 (per6 +per9 )
            st ,sn =topN (std ,V6 ,avg6 ,Vr ,Vj ,G ,Gd ,N )
            et ,en =topN (exp ,V6 ,avg6 ,Vr ,Vj ,G ,Gd ,N )# avg6 features; 9k CP'ler for de V6-avg yeterli (approximately)
            sT +=st ;sN +=sn ;eT +=et ;eN +=en 
        except Exception :
            continue 
        if k %5 ==0 :print (f"  {k }/{len (hi )}  {time .time ()-t0 :.0f}s",flush =True )

    def f1 (tp ,nk ,gt ):p =tp /max (nk ,1 );r =tp /max (gt ,1 );return p ,r ,2 *p *r /max (p +r ,1e-9 )
    ps =f1 (sT ,sN ,GT );pe =f1 (eT ,eN ,GT )
    print (f"\n=== high-CP PXC top-N F1 (sizintisiz gate, {len (hi )} part, {GT } CP) ===")
    print (f"  6k standart pool:      P{ps [0 ]:.3f} R{ps [1 ]:.3f} F1 {ps [2 ]:.3f}")
    print (f"  6k+9k genisletilmis:    P{pe [0 ]:.3f} R{pe [1 ]:.3f} F1 {pe [2 ]:.3f}")
    print (f"  -> genisletme high-CP F1'i {'YUKSELTTI'if pe [2 ]>ps [2 ]else 'YUKSELTMEDI'}")


if __name__ =="__main__":
    main ()
